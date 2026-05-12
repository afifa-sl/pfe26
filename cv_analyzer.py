"""
Module d'analyse de CV via pipeline RAG.
Extraction CV + recherche exigences + analyse LLM.
"""

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Extraction texte
# ─────────────────────────────────────────────────────────────

def extract_text_from_pdf(content: bytes) -> str:
    """Extraction texte PDF."""
    try:
        import fitz
        doc = fitz.open(stream=content, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()

    except ImportError:
        logger.warning("PyMuPDF absent, fallback pdfplumber")

        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                return "\n".join(
                    page.extract_text() or ""
                    for page in pdf.pages
                ).strip()

        except ImportError:
            raise RuntimeError(
                "Aucune librairie PDF disponible. "
                "Installez : pip install pymupdf"
            )


def extract_text_from_docx(content: bytes) -> str:
    """Extraction DOCX."""
    try:
        from docx import Document
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs).strip()

    except ImportError:
        raise RuntimeError(
            "python-docx non installé. "
            "Installez : pip install python-docx"
        )


def extract_cv_text(content: bytes, filename: str) -> str:
    """Dispatcher extraction."""
    name = filename.lower()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(content)

    elif name.endswith(".docx"):
        return extract_text_from_docx(content)

    elif name.endswith(".txt"):
        return content.decode("utf-8", errors="replace").strip()

    raise ValueError(
        f"Format non supporté : {filename}. "
        "Formats acceptés : PDF, DOCX, TXT"
    )


# ─────────────────────────────────────────────────────────────
# Prompt analyse
# ─────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """
Tu es un expert RH chez Sonatrach.

Analyse le CV ci-dessous par rapport au poste "{poste}"
en utilisant les exigences récupérées depuis la base documentaire interne.

=== EXIGENCES DU POSTE ===
{job_context}

=== CV DU CANDIDAT ===
{cv_text}

=== BARÈME STRICT ===
- 0-2 : Profil hors domaine
- 3-4 : Faible adéquation
- 5-6 : Adéquation moyenne
- 7-8 : Bonne adéquation
- 9-10 : Excellente adéquation

RÈGLES :
- Formation non technique pour poste technique => score max 3
- Sans expérience industrielle/pétrole/gaz => pénalité
- Qualités personnelles seules ≠ bon score

Réponds STRICTEMENT en français :

**SCORE DE CORRESPONDANCE** : [0-10]/10

**POINTS FORTS**
- ...

**POINTS FAIBLES / MANQUANTS**
- ...

**RECOMMANDATION FINALE**
Recommandé / À étudier / Non recommandé avec justification.

**REMARQUES**
Observations supplémentaires.

**POSTE RECOMMANDÉ** : [intitulé du poste Sonatrach le PLUS ADAPTÉ au profil réel du candidat, 
indépendamment du poste visé — basé uniquement sur sa formation et son expérience]
"""


def build_analysis_prompt(cv_text: str, poste: str, job_context: str) -> str:
    return ANALYSIS_PROMPT.format(
        poste=poste or "poste généraliste Sonatrach",
        job_context=job_context or (
            "Aucun document spécifique trouvé. "
            "Utiliser bonnes pratiques RH générales."
        ),
        cv_text=cv_text[:4000],
    )


# ─────────────────────────────────────────────────────────────
# Analyse principale
# ─────────────────────────────────────────────────────────────

def analyze_cv_with_pipeline(pipeline, cv_text: str, poste: str) -> dict:
    import time
    t0 = time.time()

    search_poste = poste.strip() if poste else ""

    if search_poste:
    search_query = f"exigences compétences diplômes requis poste {search_poste}"
else:
    # Utiliser le contenu réel du CV pour chercher les postes correspondants
    # → évite de biaiser vers un poste fixe comme "recrutement"
    cv_hint = " ".join(cv_text[:600].split())[:300]
    search_query = f"poste Sonatrach requis diplôme expérience {cv_hint}"

    try:
        query_embedding = pipeline.embedder.embed_single(search_query)

        dense_results = pipeline.vector_store.search(
            query_embedding,
            k=pipeline.config.top_k_dense,
        )

        sparse_results = []
        if pipeline.bm25:
            sparse_results = pipeline.bm25.search(
                search_query,
                k=pipeline.config.top_k_sparse,
            )

        from src.retrieval.hybrid_search import reciprocal_rank_fusion
        fused = reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            k=pipeline.config.rrf_k,
        )

        top_chunks = fused[:pipeline.config.top_k_after_rerank]

        if pipeline.reranker and top_chunks:
            pairs = [(search_query, c["content"]) for c in top_chunks]
            scores = pipeline.reranker.model.predict(pairs)

            for c, s in zip(top_chunks, scores):
                c["rerank_score"] = float(s)

            top_chunks = sorted(
                top_chunks,
                key=lambda x: x.get("rerank_score", 0),
                reverse=True,
            )

        job_context = "\n\n---\n\n".join(
            f"[{c['metadata'].get('source', '?')}]\n{c['content']}"
            for c in top_chunks
        )

        sources = list({
            c["metadata"].get("source", "?")
            for c in top_chunks
        })

    except Exception as e:
        logger.warning("Erreur recherche RAG : %s", e)
        job_context = ""
        sources = []

    # Prompt final
    prompt = build_analysis_prompt(
        cv_text=cv_text,
        poste=search_poste,
        job_context=job_context,
    )

    try:
        answer = pipeline.llm.generate(
            prompt=prompt,
            system="Tu es un expert RH chez Sonatrach. Réponds uniquement en français.",
            temperature=0.0,
            max_tokens=pipeline.config.llm_max_tokens_long,
        )

    except Exception as e:
        raise RuntimeError(f"Erreur analyse LLM : {e}")

    score = _extract_score(answer)
    recommended_poste = _extract_recommended_poste(answer)

    
    elapsed = round(time.time() - t0, 2)

    return {
    "answer": answer,
    "score": score,
    "poste": search_poste or recommended_poste or "Non précisé",
    "recommended_poste": recommended_poste or "Non précisé",  # ← ajouter
    "sources": sources,
    "elapsed_seconds": elapsed,
}

# ─────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────

def _extract_score(text: str) -> Optional[int]:
    import re

    patterns = [
        r"SCORE[^:\n]*:\s*\**\s*(\d{1,2})\s*\**\s*/\s*10",
        r"SCORE[^:\n]*:\s*\[?(\d{1,2})\]?\s*/\s*10",
        r"\*\*(\d{1,2})\*\*\s*/\s*10",
        r"(\d{1,2})\s*/\s*10",
    ]

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 10:
                return val

    return None


def _extract_recommended_poste(text: str) -> Optional[str]:
    import re
    patterns = [
        # Inline avec deux-points : **POSTE RECOMMANDÉ** : Ingénieur forage
        r"\*{0,2}POSTE\s+RECOMMAND[EÉ]\*{0,2}\s*[:\-]\s*([^\n\[\]]+)",
        # Ligne suivante : **POSTE RECOMMANDÉ**\nIngénieur forage
        r"\*{0,2}POSTE\s+RECOMMAND[EÉ]\*{0,2}\s*\n+\s*([^\n\[\]\*]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            value = m.group(1).strip().strip("*•[] \t")
            # Rejeter si c'est le texte de template (crochets, trop court)
            if value and "[" not in value and len(value) > 3:
                return value
   
    return None
