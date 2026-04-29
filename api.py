#!/usr/bin/env python3
"""
API REST FastAPI pour le RAG local.

Démarrage:
    python api.py
    # ou
    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health          → Statut du système
    GET  /stats           → Statistiques d'indexation
    POST /query           → Interroger le RAG
    POST /ingest          → Déclencher l'ingestion
    POST /upload          → Uploader un document
    POST /reset           → Réinitialiser l'index
    POST /lien            → Ajouter des URLs et scraper tous les liens
    GET  /lien            → Lister les liens enregistrés
    POST /lien/scrape     → Scraper tous les liens sans en ajouter
"""
import logging
import sys
import os
import shutil
import json
import time
from collections import defaultdict
from src.cv_analyzer import analyze_cv_with_rag

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, Field, field_validator
from typing import List, Optional

from config import config
from src.pipeline import RAGPipeline
from src.ingestion.loader import scrape_url

app = FastAPI(
    title="RAG Local API",
    description="Retrieval-Augmented Generation 100% local (HuggingFace + ChromaDB + sentence-transformers)",
    version="1.0.0",
)

# CORS — restreint aux origines connues (ajouter les domaines autorisés)
ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:8001").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Rate Limiter simple en mémoire ───────────────────────────────────────

_rate_limit_store: dict = defaultdict(list)
RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("RATE_LIMIT_MAX", "30"))  # par minute
RATE_LIMIT_WINDOW = 60  # secondes


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiter par IP — bloque au-delà de RATE_LIMIT_MAX requêtes/minute."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # Nettoie les entrées expirées
    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if now - t < RATE_LIMIT_WINDOW
    ]

    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        return JSONResponse(
            status_code=429,
            content={"detail": "Trop de requêtes. Réessayez dans une minute."},
        )

    _rate_limit_store[client_ip].append(now)
    return await call_next(request)

pipeline: Optional[RAGPipeline] = None
ingestion_status: dict = {"running": False, "last_result": None, "error": None}
lien_status: dict = {"running": False, "last_result": None, "error": None}

LINKS_STORE_PATH = os.path.join(".", "data", "links.json")


def _load_links() -> List[str]:
    if os.path.exists(LINKS_STORE_PATH):
        with open(LINKS_STORE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_links(links: List[str]) -> None:
    os.makedirs(os.path.dirname(LINKS_STORE_PATH), exist_ok=True)
    with open(LINKS_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(links, f, indent=2)


@app.on_event("startup")
async def startup():
    global pipeline
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.docs_dir, exist_ok=True)
    pipeline = RAGPipeline(config)


# ─── Schemas ────────────────────────────────────────────────────────────────

class LienRequest(BaseModel):
    urls: List[HttpUrl]


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    use_query_transform: bool = False
    stream: bool = False
    history: List[ChatMessage] = Field(default=[], max_length=20)

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("La question ne peut pas être vide")
        return v


class QueryResponse(BaseModel):
    question: str
    search_query: str
    answer: str
    sources: List[str]
    chunks_used: int
    elapsed_seconds: float


# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "chunks_indexed": pipeline.vector_store.count() if pipeline else 0,
        "llm_model": config.llm_model,
        "embedding_model": config.embedding_model,
    }


@app.get("/stats")
async def stats():
    if not pipeline:
        raise HTTPException(503, "Pipeline non initialisé")
    return {
        "chunks_indexed": pipeline.vector_store.count(),
        "embedding_model": config.embedding_model,
        "embedding_device": config.embedding_device,
        "llm_model": config.llm_model,
        "reranker_model": config.reranker_model,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "top_k_dense": config.top_k_dense,
        "top_k_sparse": config.top_k_sparse,
        "top_k_after_rerank": config.top_k_after_rerank,
        "docs_dir": config.docs_dir,
        "ingestion_status": ingestion_status,
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not pipeline:
        raise HTTPException(503, "Pipeline non initialisé")
    if pipeline.vector_store.count() == 0:
        raise HTTPException(400, "Aucun document indexé. Appelez POST /ingest d'abord.")
    try:
        history = [{"role": m.role, "content": m.content} for m in request.history]
        result = pipeline.query(
            question=request.question,
            use_query_transform=request.use_query_transform,
            stream=False,
            history=history if history else None,
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"Erreur pipeline: {e}")


@app.post("/ingest")
async def ingest(background_tasks: BackgroundTasks, reset: bool = False):
    """Déclenche l'ingestion en arrière-plan."""
    if not pipeline:
        raise HTTPException(503, "Pipeline non initialisé")
    if ingestion_status["running"]:
        raise HTTPException(409, "Ingestion déjà en cours")

    def run_ingest():
        ingestion_status["running"] = True
        ingestion_status["error"] = None
        try:
            result = pipeline.ingest(reset=reset)
            ingestion_status["last_result"] = result
        except Exception as e:
            ingestion_status["error"] = str(e)
        finally:
            ingestion_status["running"] = False

    background_tasks.add_task(run_ingest)
    return {
        "message": f"Ingestion démarrée (reset={reset}) depuis '{config.docs_dir}'",
        "hint": "GET /stats pour suivre la progression",
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload un fichier dans le dossier documents."""
    os.makedirs(config.docs_dir, exist_ok=True)
    dest = os.path.join(config.docs_dir, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    size = os.path.getsize(dest)
    return {
        "message": f"Fichier '{file.filename}' uploadé",
        "path": dest,
        "size_bytes": size,
        "hint": "Appelez POST /ingest pour l'indexer",
    }


@app.post("/lien")
async def add_liens(request: LienRequest, background_tasks: BackgroundTasks):
    """Ajoute des URLs au store et scrape tous les liens enregistrés."""
    if not pipeline:
        raise HTTPException(503, "Pipeline non initialisé")
    if lien_status["running"]:
        raise HTTPException(409, "Scraping de liens déjà en cours")

    # Ajout des nouvelles URLs au store persistant
    existing = _load_links()
    new_urls = [str(u) for u in request.urls]
    added = [u for u in new_urls if u not in existing]
    all_links = existing + added
    _save_links(all_links)

    def run_scrape():
        lien_status["running"] = True
        lien_status["error"] = None
        links = _load_links()
        docs = []
        errors = []
        for url in links:
            try:
                doc = scrape_url(url)
                docs.append(doc)
                print(f"  Scrappé: {url} ({len(doc.content):,} chars)")
            except Exception as e:
                errors.append({"url": url, "error": str(e)})
                print(f"  Erreur scraping {url}: {e}")
        try:
            result = pipeline.ingest_documents(docs)
            result["errors"] = errors
            lien_status["last_result"] = result
        except Exception as e:
            lien_status["error"] = str(e)
        finally:
            lien_status["running"] = False

    background_tasks.add_task(run_scrape)
    return {
        "message": f"{len(added)} lien(s) ajouté(s), scraping de {len(all_links)} lien(s) démarré",
        "added": added,
        "total_links": len(all_links),
        "hint": "GET /stats pour suivre la progression",
    }


@app.get("/lien")
async def list_liens():
    """Liste tous les liens enregistrés."""
    links = _load_links()
    return {"links": links, "total": len(links), "status": lien_status}


@app.post("/lien/scrape")
async def scrape_liens(background_tasks: BackgroundTasks):
    """Scrape et indexe tous les liens déjà enregistrés."""
    if not pipeline:
        raise HTTPException(503, "Pipeline non initialisé")
    if lien_status["running"]:
        raise HTTPException(409, "Scraping de liens déjà en cours")
    links = _load_links()
    if not links:
        raise HTTPException(400, "Aucun lien enregistré. Appelez POST /lien d'abord.")

    def run_scrape():
        lien_status["running"] = True
        lien_status["error"] = None
        docs = []
        errors = []
        for url in links:
            try:
                doc = scrape_url(url)
                docs.append(doc)
                print(f"  Scrappé: {url} ({len(doc.content):,} chars)")
            except Exception as e:
                errors.append({"url": url, "error": str(e)})
                print(f"  Erreur scraping {url}: {e}")
        try:
            result = pipeline.ingest_documents(docs)
            result["errors"] = errors
            lien_status["last_result"] = result
        except Exception as e:
            lien_status["error"] = str(e)
        finally:
            lien_status["running"] = False

    background_tasks.add_task(run_scrape)
    return {
        "message": f"Scraping de {len(links)} lien(s) démarré",
        "links": links,
        "hint": "GET /lien pour suivre le statut",
    }


@app.post("/reset")
async def reset_index():
    """Supprime et réinitialise l'index complet."""
    if not pipeline:
        raise HTTPException(503, "Pipeline non initialisé")
    pipeline.vector_store.reset()
    bm25_path = config.bm25_index_path
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    ingestion_status["last_result"] = None
    return {"message": "Index réinitialisé"}

"""
=============================================================
AJOUT À api.py — Endpoint d'analyse de CV
=============================================================

1. Ajoute ces imports en haut de api.py (après les imports existants) :
   from src.cv_analyzer import analyze_cv_with_rag

2. Colle ce bloc d'endpoints AVANT la ligne `if __name__ == "__main__":`
=============================================================
"""

# ─── CV Analyzer ────────────────────────────────────────────────────────────

import PyPDF2
import io
from pydantic import BaseModel as _BaseModel


class CVAnalysisResponse(_BaseModel):
    cv_name: str
    detected_profile: str
    detected_skills: list
    post_exists_in_sonatrach: bool
    confidence: str            # "haute" | "moyenne" | "faible"
    matching_department: str
    matching_post: str
    recommendation: str        # "Recommandé" | "À étudier" | "Non recommandé"
    justification: str
    rag_sources: list
    elapsed_seconds: float


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extrait le texte brut d'un PDF uploadé."""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()


@app.post("/cv/analyze", response_model=CVAnalysisResponse)
async def analyze_cv(file: UploadFile = File(...)):
    """
    Analyse un CV (PDF ou texte) et détermine :
    - Le profil détecté
    - Si le poste correspondant existe chez Sonatrach
    - Le département concerné
    - Une recommandation GTP (Gestion des Talents et du Personnel)
    """
    import time as _time

    if not pipeline:
        raise HTTPException(503, "Pipeline non initialisé")
    if pipeline.vector_store.count() == 0:
        raise HTTPException(400, "Aucun document indexé. Appelez POST /ingest d'abord.")

    # ── 1. Lecture du fichier ──────────────────────────────────────────────
    content = await file.read()
    filename = file.filename or "cv_inconnu"

    if filename.lower().endswith(".pdf"):
        try:
            cv_text = _extract_text_from_pdf(content)
        except Exception as e:
            raise HTTPException(400, f"Impossible de lire le PDF : {e}")
    else:
        try:
            cv_text = content.decode("utf-8", errors="ignore")
        except Exception:
            raise HTTPException(400, "Format non supporté. Envoyez un PDF ou un fichier texte.")

    if len(cv_text) < 50:
        raise HTTPException(400, "CV trop court ou illisible. Vérifiez le fichier.")

    cv_text_truncated = cv_text[:3000]  # Limite pour le prompt

    # ── 2. Première requête RAG : extraction du profil ────────────────────
    t0 = _time.time()

    extraction_q = f"""
Voici le contenu d'un CV :
---
{cv_text_truncated}
---
Identifie et liste :
1. Le titre du poste ou la spécialité principale de ce candidat
2. Les compétences techniques clés (max 8)
3. Le niveau d'expérience (junior <3ans / confirmé 3-7ans / senior >7ans)
Réponds en JSON strict avec les clés : titre_poste, competences, niveau_experience.
"""

    try:
        extraction_result = pipeline.query(
            question=extraction_q,
            use_query_transform=False,
            stream=False,
        )
        profile_answer = extraction_result.get("answer", "")
        rag_sources = extraction_result.get("sources", [])
    except Exception as e:
        raise HTTPException(500, f"Erreur analyse profil : {e}")

    # Parse JSON souple
    import json, re
    detected_profile = "Profil non déterminé"
    detected_skills = []
    try:
        json_match = re.search(r'\{.*\}', profile_answer, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            detected_profile = parsed.get("titre_poste", detected_profile)
            detected_skills = parsed.get("competences", [])
            if isinstance(detected_skills, str):
                detected_skills = [s.strip() for s in detected_skills.split(",")]
    except Exception:
        # Fallback : extraire le profil depuis le texte libre
        lines = profile_answer.split("\n")
        if lines:
            detected_profile = lines[0][:100]

    # ── 3. Deuxième requête RAG : vérification poste Sonatrach ────────────
    verification_q = f"""
Est-ce que le poste ou la spécialité "{detected_profile}" existe dans les offres d'emploi ou l'organigramme de Sonatrach ?
Si oui, précise :
- Le département ou la direction concernée (ex: Direction Exploration, DRH, Direction Informatique...)
- L'intitulé exact du poste chez Sonatrach
- Si ce profil correspond aux besoins actuels
Sois précis et base-toi uniquement sur les documents indexés.
"""

    try:
        verification_result = pipeline.query(
            question=verification_q,
            use_query_transform=False,
            stream=False,
        )
        verification_answer = verification_result.get("answer", "")
        rag_sources += [s for s in verification_result.get("sources", []) if s not in rag_sources]
    except Exception as e:
        raise HTTPException(500, f"Erreur vérification poste : {e}")

    elapsed = round(_time.time() - t0, 2)

    # ── 4. Analyse sémantique de la réponse ──────────────────────────────
    answer_lower = verification_answer.lower()

    # Détection existence du poste
    negative_keywords = ["n'existe pas", "pas de poste", "aucun poste", "introuvable",
                         "ne figure pas", "non trouvé", "pas trouvé", "pas mentionné"]
    positive_keywords = ["existe", "correspond", "disponible", "recrute", "recherche",
                         "département", "direction", "poste de", "offre"]

    neg_score = sum(1 for kw in negative_keywords if kw in answer_lower)
    pos_score = sum(1 for kw in positive_keywords if kw in answer_lower)

    post_exists = pos_score > neg_score

    # Confiance
    if abs(pos_score - neg_score) >= 3:
        confidence = "haute"
    elif abs(pos_score - neg_score) >= 1:
        confidence = "moyenne"
    else:
        confidence = "faible"

    # Extraction département
    dept_patterns = [
        r"direction\s+[\w\s\-]+",
        r"département\s+[\w\s\-]+",
        r"division\s+[\w\s\-]+",
        r"DRH|DSI|DEX|DG|DP\b",
    ]
    matching_department = "Non déterminé"
    for pattern in dept_patterns:
        match = re.search(pattern, verification_answer, re.IGNORECASE)
        if match:
            matching_department = match.group().strip()[:80]
            break

    # Extraction poste Sonatrach
    post_patterns = [
        r"poste de\s+[\w\s\-]+",
        r"intitulé[:\s]+[\w\s\-]+",
        r"en tant que\s+[\w\s\-]+",
    ]
    matching_post = detected_profile
    for pattern in post_patterns:
        match = re.search(pattern, verification_answer, re.IGNORECASE)
        if match:
            matching_post = match.group().strip()[:80]
            break

    # Recommandation GTP
    if post_exists and confidence == "haute":
        recommendation = "Recommandé"
    elif post_exists and confidence == "moyenne":
        recommendation = "À étudier"
    else:
        recommendation = "Non recommandé"

    # Justification courte
    justification_lines = [l.strip() for l in verification_answer.split("\n") if len(l.strip()) > 30]
    justification = " ".join(justification_lines[:2])[:300] if justification_lines else verification_answer[:300]

    return CVAnalysisResponse(
        cv_name=filename,
        detected_profile=detected_profile,
        detected_skills=detected_skills[:8],
        post_exists_in_sonatrach=post_exists,
        confidence=confidence,
        matching_department=matching_department,
        matching_post=matching_post,
        recommendation=recommendation,
        justification=justification,
        rag_sources=list(set(rag_sources))[:5],
        elapsed_seconds=elapsed,
    )

if __name__ == "__main__":
    import uvicorn
    # Pour un tunnel ngrok, utiliser: python run_tunnel.py
    uvicorn.run(app, host="0.0.0.0", port=8001)
