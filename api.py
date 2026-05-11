#!/usr/bin/env python3
import sys, os, json, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from config import config
from auth import init_db, login, verify_token, revoke_token, create_user, list_users, update_user, delete_user ,save_history, get_history, delete_history 
from src.pipeline import RAGPipeline
from src.ingestion.loader import scrape_url

init_db()

app = FastAPI(title="Assistant Sonatrach", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

pipeline = None
ingestion_status = {"running": False, "last_result": None, "error": None}
lien_status      = {"running": False, "last_result": None, "error": None}
LINKS_FILE = "./data/links.json"

def _load_links():
    if os.path.exists(LINKS_FILE):
        with open(LINKS_FILE) as f: return json.load(f)
    return []

def _save_links(links):
    os.makedirs("./data", exist_ok=True)
    with open(LINKS_FILE, "w") as f: json.dump(links, f, indent=2)

@app.on_event("startup")
async def startup():
    global pipeline
    os.makedirs("./data", exist_ok=True)
    os.makedirs(config.docs_dir, exist_ok=True)
    pipeline = RAGPipeline(config)

def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Token manquant ou invalide")
    user = verify_token(authorization.split(" ", 1)[1])
    if not user: raise HTTPException(401, "Session expirée ou invalide")
    return user

def require_superadmin(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != "superadmin": raise HTTPException(403, "Accès réservé aux administrateurs")
    return user

class LoginRequest(BaseModel):
    email: str
    password: str

class CreateUserRequest(BaseModel):
    email: str; nom: str; prenom: str; role: str = "employee"; password: str

class UpdateUserRequest(BaseModel):
    nom: Optional[str] = None; prenom: Optional[str] = None
    role: Optional[str] = None; password: Optional[str] = None; active: Optional[int] = None

class QuestionRequest(BaseModel):
    question: str; history: Optional[List[dict]] = []

class LienRequest(BaseModel):
    urls: List[str]

@app.post("/auth/login")
def auth_login(req: LoginRequest):
    result = login(req.email, req.password)
    if not result: raise HTTPException(401, "Email ou mot de passe incorrect")
    return result

@app.post("/auth/logout")
def auth_logout(authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        revoke_token(authorization.split(" ", 1)[1])
    return {"message": "Déconnecté"}

@app.get("/auth/me")
def auth_me(user: dict = Depends(get_current_user)):
    return {k: user[k] for k in ("id","email","nom","prenom","role")}

@app.get("/users")
def get_users(admin: dict = Depends(require_superadmin)): return list_users()

@app.post("/users")
def add_user(req: CreateUserRequest, admin: dict = Depends(require_superadmin)):
    try: return {"message": "Utilisateur créé", "user": create_user(req.email, req.nom, req.prenom, req.role, req.password)}
    except ValueError as e: raise HTTPException(400, str(e))

@app.put("/users/{user_id}")
def edit_user(user_id: int, req: UpdateUserRequest, admin: dict = Depends(require_superadmin)):
    update_user(user_id, **{k: v for k, v in req.dict().items() if v is not None})
    return {"message": "Utilisateur mis à jour"}

@app.delete("/users/{user_id}")
def remove_user(user_id: int, admin: dict = Depends(require_superadmin)):
    delete_user(user_id); return {"message": "Utilisateur désactivé"}

@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": pipeline.vector_store.count() if pipeline else 0, "llm_model": config.llm_model}

@app.get("/stats")
def stats(user: dict = Depends(get_current_user)):
    return {"chunks_indexed": pipeline.vector_store.count() if pipeline else 0,
            "llm_model": config.llm_model, "embedding_model": config.embedding_model,
            "reranker_model": config.reranker_model, "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap, "top_k_dense": config.top_k_dense,
            "top_k_sparse": config.top_k_sparse, "top_k_after_rerank": config.top_k_after_rerank,
            "docs_dir": config.docs_dir, "ingestion_status": ingestion_status}

@app.post("/query")
def query_endpoint(req: QuestionRequest, user: dict = Depends(get_current_user)):
    if not req.question.strip():
        raise HTTPException(400, "Question vide")
    try:
        result = pipeline.query(
            question=req.question,
            use_query_transform=False,
            stream=False,
            history=req.history or None,
        )
        result["source"] = "rag"
        result.setdefault("search_query", req.question)

        # ✅ Sauvegarde dans l'historique
        save_history(
            user_id=user["id"],
            question=req.question,
            answer=result["answer"],
            source=result["source"],
        )
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Historique par utilisateur ────────────────────────────────────────────────
@app.get("/historique")
def get_my_history(user: dict = Depends(get_current_user)):
    """Retourne l'historique de l'utilisateur connecté."""
    return get_history(user["id"])

@app.delete("/historique")
def clear_my_history(user: dict = Depends(get_current_user)):
    """Supprime l'historique de l'utilisateur connecté."""
    delete_history(user["id"])
    return {"message": "Historique supprimé"}

@app.get("/historique/{user_id}")
def get_user_history(user_id: int, admin: dict = Depends(require_superadmin)):
    """Superadmin peut voir l'historique de n'importe quel utilisateur."""
    return get_history(user_id)

@app.post("/upload")
def upload(file: UploadFile = File(...), admin: dict = Depends(require_superadmin)):
    os.makedirs(config.docs_dir, exist_ok=True)
    dest = os.path.join(config.docs_dir, file.filename)
    with open(dest, "wb") as f: shutil.copyfileobj(file.file, f)
    return {"message": f"Fichier '{file.filename}' uploadé", "hint": "Appelez POST /ingest pour l'indexer"}

@app.post("/ingest")
def ingest(background_tasks: BackgroundTasks, reset: bool = False, admin: dict = Depends(require_superadmin)):
    if ingestion_status["running"]: raise HTTPException(409, "Ingestion déjà en cours")
    def run():
        ingestion_status["running"] = True; ingestion_status["error"] = None
        try: ingestion_status["last_result"] = pipeline.ingest(reset=reset)
        except Exception as e: ingestion_status["error"] = str(e)
        finally: ingestion_status["running"] = False
    background_tasks.add_task(run)
    return {"message": f"Ingestion démarrée (reset={reset})"}

@app.post("/reset")
def reset_index(admin: dict = Depends(require_superadmin)):
    pipeline.vector_store.reset()
    if os.path.exists(config.bm25_index_path): os.remove(config.bm25_index_path)
    ingestion_status["last_result"] = None
    return {"message": "Index réinitialisé"}

@app.post("/lien")
def add_liens(req: LienRequest, background_tasks: BackgroundTasks, admin: dict = Depends(require_superadmin)):
    existing = _load_links(); added = [u for u in req.urls if u not in existing]; _save_links(existing + added)
    def run():
        lien_status["running"] = True; docs, errors = [], []
        for url in _load_links():
            try: docs.append(scrape_url(url))
            except Exception as e: errors.append({"url": url, "error": str(e)})
        try: r = pipeline.ingest_documents(docs); r["errors"] = errors; lien_status["last_result"] = r
        except Exception as e: lien_status["error"] = str(e)
        finally: lien_status["running"] = False
    background_tasks.add_task(run)
    return {"message": f"{len(added)} lien(s) ajouté(s)", "added": added}

@app.get("/lien")
def list_liens(user: dict = Depends(get_current_user)):
    return {"links": _load_links(), "status": lien_status}

@app.post("/lien/scrape")
def scrape_liens(background_tasks: BackgroundTasks, admin: dict = Depends(require_superadmin)):
    links = _load_links()
    if not links: raise HTTPException(400, "Aucun lien enregistré")
    def run():
        lien_status["running"] = True; docs, errors = [], []
        for url in links:
            try: docs.append(scrape_url(url))
            except Exception as e: errors.append({"url": url, "error": str(e)})
        try: r = pipeline.ingest_documents(docs); r["errors"] = errors; lien_status["last_result"] = r
        except Exception as e: lien_status["error"] = str(e)
        finally: lien_status["running"] = False
    background_tasks.add_task(run)
    return {"message": f"Scraping de {len(links)} lien(s) démarré"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=False)
