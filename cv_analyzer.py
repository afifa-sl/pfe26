"""
Module d'analyse de CV via le pipeline RAG.
Extrait le texte du CV, interroge les documents indexés pour les exigences du poste,
puis utilise le LLM pour évaluer la correspondance.
"""
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Extraction de texte ──────────────────────────────────────────────────────

def extract_text_from_pdf(content: bytes) -> str:
    """Extrait le texte d'un PDF via PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=content, filetype="pdf")
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages).strip()
    except ImportError:
        logger.warning("PyMuPDF non disponible, tentative avec pdfplumber")
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                return "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                ).strip()
        except ImportError:
            raise RuntimeError(
                "Aucune librairie PDF disponible. "
                "Installez PyMuPDF: pip install pymupdf"
            )


def extract_text_from_docx(content: bytes) -> str:
    """Extrait le texte d'un DOCX via python-docx."""
    try:
        from docx import Document
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except ImportError:
        raise RuntimeError(
            "python-docx non disponible. "
            "Installez-le: pip install python-docx"
        )


def extract_cv_text(content: bytes, filename: str) -> str:
    """Dispatche l'extraction selon l'extension du fichier."""
    name = filename.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(content)
    elif name.endswith(".docx"):
        return extract_text_from_docx(content)
    elif name.endswith(".txt"):
        return content.decode("utf-8", errors="replace").strip()
    else:
        raise ValueError(
            f"Format non supporté : {filename}. "
            "Formats acceptés : PDF, DOCX, TXT"
        )


# ── Prompt d'analyse ─────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """Tu es un expert RH chez Sonatrach. Analyse le CV ci-dessous par rapport au poste "{poste}" en t'appuyant sur les critères et exigences extraits de la base documentaire interne.

=== EXIGENCES DU POSTE (base documentaire interne) ===
{job_context}

=== CV DU CANDIDAT ===
{cv_text}

=== BARÈME DE NOTATION STRICT ===
- 0-2 : Aucune formation technique, aucune expérience pertinente, profil totalement hors domaine
- 3-4 : Formation marginalement liée, très peu d'expérience pertinente
- 5-6 : Formation partiellement liée OU quelques expériences pertinentes
- 7-8 : Bonne adéquation formation + expérience avec les exigences du poste
- 9-10 : Excellente adéquation, toutes les exigences remplies

⚠️ RÈGLES STRICTES :
- Une licence en lettres, sciences humaines ou toute formation non technique = score MAX 3 pour un poste technique Sonatrach
- Sans expérience dans le secteur pétrolier/gazier/industriel = pénalité importante
- Les "qualités transversales" seules ne justifient JAMAIS un score supérieur à 3
- Soyez rigoureux : un mauvais profil doit avoir un score bas, pas moyen

=== INSTRUCTIONS ===
Réponds UNIQUEMENT en français, de façon structurée :

**SCORE DE CORRESPONDANCE** : [chiffre entre 0 et 10]/10

**POINTS FORTS**
- ...

**POINTS FAIBLES / MANQUANTS**
- ...

**RECOMMANDATION FINALE**
Recommandé / À étudier / Non recommandé — avec justification.

**REMARQUES**
Observations supplémentaires."""

def build_analysis_prompt(cv_text: str, poste: str, job_context: str) -> str:
    return ANALYSIS_PROMPT.format(
        poste=poste or "poste généraliste Sonatrach (ingénierie, technique ou administratif)",
        job_context=job_context or "Aucun document spécifique au poste n'a été trouvé. Basez-vous sur les bonnes pratiques générales.",
        cv_text=cv_text[:4000],  # limite pour ne pas dépasser le contexte LLM
    )


# ── Analyse principale ────────────────────────────────────────────────────────

def analyze_cv_with_pipeline(pipeline, cv_text: str, poste: str) -> dict:
    """
    Analyse un CV en utilisant :
    1. La recherche RAG pour trouver les exigences du poste dans les documents
    2. Le LLM pour évaluer la correspondance

    Retourne un dict avec : score, analysis, sources, elapsed_seconds
    """
    import time
    t0 = time.time()

    # 1. Requête RAG sur le poste pour récupérer les exigences documentées
    search_query = f"exigences compétences diplômes requis poste {poste}" if poste else "profil compétences recrutement Sonatrach"

    try:
        # Récupération des chunks pertinents via le vector store
        query_embedding = pipeline.embedder.embed_single(search_query) 
        dense_results = pipeline.vector_store.search(query_embedding, k=pipeline.config.top_k_dense)

        # BM25 si disponible
        sparse_results = []
        if pipeline.bm25:
            sparse_results = pipeline.bm25.search(search_query, k=pipeline.config.top_k_sparse)

        # Fusion RRF
        from src.retrieval.hybrid_search import reciprocal_rank_fusion
        fused = reciprocal_rank_fusion(dense_results, sparse_results, k=pipeline.config.rrf_k)

        # Reranking
        top_chunks = fused[:pipeline.config.top_k_after_rerank]
        if pipeline.reranker and top_chunks:
            pairs = [(search_query, c["content"]) for c in top_chunks]
            scores = pipeline.reranker.model.predict(pairs)
            for c, s in zip(top_chunks, scores):
                c["rerank_score"] = float(s)
            top_chunks = sorted(top_chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)

        job_context = "\n\n---\n\n".join(
            f"[{c['metadata'].get('source', '?')}]\n{c['content']}"
            for c in top_chunks
        )
        sources = list({c["metadata"].get("source", "?") for c in top_chunks})

    except Exception as e:
        logger.warning("Erreur lors de la recherche RAG : %s", e)
        job_context = ""
        sources = []

    # 2. Construction du prompt
    prompt = build_analysis_prompt(cv_text, poste, job_context)

    # 3. Appel LLM
    try:
        messages = [{"role": "user", "content": prompt}]
        answer = pipeline.llm.generate(
    prompt=prompt,
    system="Tu es un expert RH chez Sonatrach. Réponds uniquement en français.",
    temperature=0.0,
    max_tokens=pipeline.config.llm_max_tokens_long,
)
    except Exception as e:
        raise RuntimeError(f"Erreur LLM lors de l'analyse : {e}")

    # 4. Extraction du score (robuste)
    score = _extract_score(answer)
    
    elapsed = round(time.time() - t0, 2)

    return {
        "answer": answer,
        "score": score,
        "poste": poste or "Non précisé",
        "sources": sources,
        "elapsed_seconds": elapsed,
    }


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
