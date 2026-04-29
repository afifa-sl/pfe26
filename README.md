# RAG Local

Système de Retrieval-Augmented Generation 100% local — aucune clé API, aucune donnée envoyée à l'extérieur.
---

## Prérequis

- Python 3.10+
- [Ollama](https://ollama.com) installé sur la machine

---

## Installation

### 1. Créer et activer un environnement virtuel

```bash
cd rag_local

# Créer le venv
python -m venv .venv

# Activer (Windows)
.venv\Scripts\activate

# Activer (macOS / Linux)
source .venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

> Le premier lancement télécharge automatiquement les modèles d'embedding (~80 MB) et le reranker (~100 MB) depuis HuggingFace. Une seule fois.

### 3. Démarrer Ollama et télécharger un modèle LLM

```bash
# Dans un terminal séparé
ollama serve

# Télécharger le modèle (une seule fois, ~2 GB)
ollama pull llama3.2

# Alternatives plus légères
ollama pull phi3        # ~2 GB, très rapide
ollama pull mistral     # ~4 GB, meilleure qualité
```

---

## Utilisation

### Étape 1 — Ajouter des documents

Copie tes fichiers dans le dossier `documents/`. Formats supportés :

```
documents/
├── rapport_annuel.pdf
├── documentation_technique.docx
├── notes.md
└── ...
```

Formats acceptés : `.pdf`, `.docx`, `.txt`, `.md`, `.html`, `.csv`, `.json`

### Étape 2 — Ingérer les documents

```bash
python ingest.py
```

Options disponibles :

```bash
python ingest.py --docs-dir ./mes_docs     # Dossier personnalisé
python ingest.py --reset                   # Réindexe depuis zéro
python ingest.py --model mistral           # Autre modèle Ollama
```

### Étape 3 — Interroger le RAG

**Mode interactif :**
```bash
python query.py
>>> Quelle est la politique de remboursement ?
>>> quit
```

**Question directe :**
```bash
python query.py -q "Résume les points clés du rapport"
```

**Options :**
```bash
python query.py --no-transform    # Désactive la réécriture de requête (plus rapide)
python query.py --no-stream       # Réponse complète d'un coup (pas de streaming)
python query.py --model mistral   # Utilise un autre modèle
```

---

## Lancer l'API REST

```bash
python api.py
```

L'API tourne sur `http://localhost:8000`. La documentation interactive est disponible sur `http://localhost:8000/docs`.

---

## Configuration

Tous les paramètres sont dans [config.py](config.py). Les plus utiles :

| Paramètre | Défaut | Description |
|---|---|---|
| `llm_model` | `llama3.2` | Modèle Ollama |
| `embedding_model` | `all-MiniLM-L6-v2` | Modèle d'embedding (léger) |
| `chunk_size` | `512` | Taille des chunks en tokens |
| `top_k_after_rerank` | `5` | Nombre de chunks envoyés au LLM |

Pour de **meilleurs résultats multilingues**, change l'embedding model pour `BAAI/bge-m3` dans `config.py` :
```python
embedding_model: str = "BAAI/bge-m3"  # ~570 MB, qualité supérieure
```

> Si tu changes de modèle d'embedding, relance `python ingest.py --reset` pour réindexer.

---

---

## Intégration Frontend

L'API REST expose des endpoints JSON simples. Voici tout ce dont l'équipe front a besoin.

### Base URL

```
http://localhost:8000
```

---

### Endpoint principal — Poser une question

```
POST /query
Content-Type: application/json
```

**Body :**
```json
{
  "question": "Quelle est la politique de congés ?",
  "use_query_transform": true
}
```

**Réponse :**
```json
{
  "question": "Quelle est la politique de congés ?",
  "search_query": "Politique de congés payés et RTT dans l'entreprise",
  "answer": "Selon le document [reglement_interieur.pdf], les employés bénéficient de 25 jours de congés payés par an...",
  "sources": ["reglement_interieur.pdf", "guide_rh.pdf"],
  "chunks_used": 4,
  "elapsed_seconds": 3.2
}
```

**Exemple fetch (JavaScript) :**

```javascript
async function askRAG(question) {
  const response = await fetch("http://localhost:8000/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    throw new Error(`Erreur API: ${response.status}`);
  }

  return response.json();
}

// Usage
const result = await askRAG("Quels sont les avantages du RAG ?");
console.log(result.answer);
console.log("Sources :", result.sources);
```

---

### Upload d'un document

```
POST /upload
Content-Type: multipart/form-data
```

**Exemple :**
```javascript
async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("http://localhost:8000/upload", {
    method: "POST",
    body: formData,
  });

  return response.json();
  // { "message": "Fichier 'rapport.pdf' uploadé", "hint": "Appelez POST /ingest pour l'indexer" }
}
```

---

### Déclencher l'ingestion après upload

```
POST /ingest
```

L'ingestion tourne en arrière-plan. Surveille la progression avec `GET /stats`.

```javascript
// 1. Upload du fichier
await uploadDocument(monFichier);

// 2. Lancer l'ingestion
await fetch("http://localhost:8000/ingest", { method: "POST" });

// 3. Attendre la fin (polling simple)
async function waitForIngestion() {
  while (true) {
    const stats = await fetch("http://localhost:8000/stats").then(r => r.json());
    if (!stats.ingestion_status.running) break;
    await new Promise(resolve => setTimeout(resolve, 2000)); // check toutes les 2s
  }
}
```

---

### Vérifier l'état du système

```
GET /health
```

```json
{
  "status": "ok",
  "chunks_indexed": 142,
  "llm_model": "llama3.2",
  "embedding_model": "all-MiniLM-L6-v2"
}
```

Utile pour afficher un indicateur de chargement ou désactiver le champ de saisie si `chunks_indexed === 0`.

---

### Indexer des URLs (scraping web)

```
POST /lien
Content-Type: application/json
```

**Body :**
```json
{
  "urls": ["https://github.com/Greyma/CV", "https://example.com/page"]
}
```

Les URLs sont stockées dans `data/links.json` (persistant) et immédiatement scrapées + indexées en arrière-plan. Les doublons sont ignorés automatiquement.

**Réponse :**
```json
{
  "message": "1 lien(s) ajouté(s), scraping de 2 lien(s) démarré",
  "added": ["https://example.com/page"],
  "total_links": 2,
  "hint": "GET /stats pour suivre la progression"
}
```

**Re-scraper tous les liens sans en ajouter :**
```
POST /lien/scrape
```

**Lister les liens enregistrés + statut :**
```
GET /lien
```

**Exemple JavaScript :**
```javascript
// Ajouter et indexer une URL
await fetch("http://localhost:8000/lien", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ urls: ["https://github.com/Greyma/CV"] }),
});

// Re-scraper tous les liens existants
await fetch("http://localhost:8000/lien/scrape", { method: "POST" });
```

**PowerShell :**
```powershell
# Ajouter un lien
Invoke-RestMethod -Method POST -Uri http://localhost:8000/lien `
  -ContentType "application/json" `
  -Body '{"urls": ["https://github.com/Greyma/CV"]}'

# Re-scraper
Invoke-RestMethod -Method POST -Uri http://localhost:8000/lien/scrape
```

---

### Tous les endpoints

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/health` | Statut + nombre de documents indexés |
| `GET` | `/stats` | Configuration complète + statut ingestion |
| `POST` | `/query` | **Poser une question** |
| `POST` | `/upload` | Uploader un fichier |
| `POST` | `/ingest` | Lancer l'ingestion (arrière-plan) |
| `POST` | `/lien` | Ajouter des URLs et les indexer |
| `GET` | `/lien` | Lister les URLs enregistrées |
| `POST` | `/lien/scrape` | Re-scraper tous les liens stockés |
| `POST` | `/reset` | Vider l'index |

Documentation interactive complète : `http://localhost:8000/docs`

---

### Gestion des erreurs côté front

| Code HTTP | Signification | Action suggérée |
|---|---|---|
| `400` | Aucun document indexé | Afficher "Base de connaissance vide" |
| `409` | Ingestion déjà en cours | Désactiver le bouton, afficher un loader |
| `500` | Erreur pipeline | Afficher le message d'erreur |
| `503` | API non prête | Retry après 2s |

```javascript
try {
  const result = await askRAG(question);
  // afficher result.answer
} catch (error) {
  if (error.status === 400) {
    showMessage("Aucun document n'est encore indexé.");
  } else {
    showMessage("Une erreur est survenue, réessayez.");
  }
}
```

---

## Structure du projet

```
rag_local/
├── config.py              ← Paramètres (modèles, chunk size, etc.)
├── requirements.txt       ← Dépendances Python
├── ingest.py              ← CLI ingestion
├── query.py               ← CLI requêtes interactives
├── api.py                 ← API REST FastAPI
├── documents/             ← Dépôt des fichiers à indexer
├── data/                  ← Index ChromaDB + BM25 + links.json (généré automatiquement)
└── src/
    ├── ingestion/         ← Chargement, chunking, embeddings
    ├── retrieval/         ← ChromaDB, BM25, fusion RRF
    ├── reranking/         ← Cross-encoder
    └── generation/        ← Client Ollama, transformation de requêtes
```
