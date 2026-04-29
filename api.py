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

if __name__ == "__main__":
    import uvicorn
    # Pour un tunnel ngrok, utiliser: python run_tunnel.py
    uvicorn.run(app, host="0.0.0.0", port=8001)