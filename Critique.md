# Critique technique du RAG Myassa

Ce document recense les faiblesses du projet, classées par criticité :
- **[BLOQUANT]** — défaut majeur de qualité, sécurité, ou fiabilité
- **[IMPORTANT]** — dégrade la qualité des réponses ou la maintenabilité
- **[MINEUR]** — amélioration utile, non-urgente

---

## Verdict global

Le pipeline est **fonctionnel et bien structuré en couches** (ingestion → retrieval → reranking → génération), mais il souffre de **trois problèmes de fond** :

1. **Mauvais usage du RAG** sur des données structurées. Les fichiers `DIRECTION/DEPARTEMENT/SERVICE/POSTE.xlsx` sont des **tables relationnelles** — y appliquer un RAG vectoriel + reranker pour répondre à "liste les services" est un anti-pattern. Une requête SQL sur DuckDB ou pandas donnerait 100% de précision en <50ms. Le RAG vectoriel ne devrait servir que pour `Explications_F.docx`.
2. **Aucune mesure de qualité**. Pas un seul test, pas de jeu d'évaluation. Vous n'avez aucun moyen objectif de dire si une modification améliore ou dégrade le système.
3. **Sécurité du tunnel ngrok** : exposition publique sans authentification.

Le reste relève du polissage et de l'ingénierie.

---

## 1. Architecture & approche RAG

### [BLOQUANT] Mauvais paradigme pour les données structurées
Excel = données tabulaires. Le pipeline les transforme en texte (`[SOURCE] CHANTIER: X | DEPARTEMENT: Y`) puis les indexe en vector + BM25. C'est :
- **Coûteux** : 1841 lignes POSTE.xlsx → 1841 chunks → 1841 embeddings → 1841 entrées ChromaDB.
- **Imprécis** : le retriever rate des éléments à cause de scores bas, le reranker tronque, le LLM hallucine.
- **Non-déterministe** : la même question peut donner 2 réponses différentes.

**Fix** : ajouter une couche `StructuredQueryEngine` qui charge les Excel dans **DuckDB** (ou pandas), et que l'IntentRouter route vers SQL pour les questions tabulaires. Le RAG vectoriel reste pour les `.docx`/`.pdf` uniquement.

### [IMPORTANT] Couplage fort dans `pipeline.py`
La classe `RAGPipeline` orchestre tout : ingestion, classification, retrieval, validation, génération. ~500 lignes. Difficile à tester unitairement. Devrait être 4-5 classes : `Ingestor`, `Retriever`, `IntentRouter` (déjà extrait), `Generator`, `Pipeline` (façade).

### [IMPORTANT] Pas de tests
Zéro test automatisé. Pour un PFE c'est défendable, mais pour défendre la qualité du système devant un jury, c'est un point faible majeur. **Minimum vital** :
- Un fichier `eval/questions.json` avec 30 questions et réponses attendues.
- Un script `evaluate.py` qui calcule la précision/rappel sur ce jeu.
- Tests unitaires sur les fonctions pures : `_fold`, `_parse_kv_chunk`, `_detect_header_row`, `IntentRouter._parse_json`, `_parse_kept_indices`.

---

## 2. Ingestion (`loader.py`, `chunker.py`)

### [IMPORTANT] Pas d'ingestion incrémentale
Ajouter un fichier = `--reset` complet (ré-embed tout). Pour un projet en prod ce serait un blocker. Solution : hash du fichier en metadata, ne ré-indexer que les nouveaux/modifiés.

### [IMPORTANT] Chunk size unique
`chunk_size=256` tokens pour TOUT (Excel mono-ligne, docx longs, PDF). Idéalement :
- Excel : pas de chunking (1 ligne = 1 chunk déjà petit).
- Docx/PDF : 512-1024 tokens avec overlap.

### [MINEUR] Workbook openpyxl jamais fermé
Dans `_load_excel`, le `wb` est ouvert mais pas fermé. Sur 1000 fichiers Excel, leak mémoire. Utiliser un `try/finally` ou `with closing(wb)`.

### [MINEUR] Tokenizer rechargé à chaque appel
`_make_token_counter` télécharge/charge le tokenizer à chaque appel. Le code essaie de l'éviter (passage de `count_fn` à `chunk_documents`), mais `chunk_document` (sans s) appelé seul recrée un fallback `len()/4` non cohérent.

### [MINEUR] Estimation 1 token = 4 chars
Pour le français, c'est plutôt 3-3.5 chars/token. Le fallback sous-estime de ~15%.

---

## 3. Retrieval (`bm25_search.py`, `vector_store.py`, `hybrid_search.py`)

### [BLOQUANT] Pickle BM25
`bm25_search.py:99` sérialise tout en pickle. Si un attaquant remplace le fichier `bm25_index.pkl`, **exécution arbitraire** au chargement (`pickle.load`). Solution : sauver uniquement les données (documents + tokens) en JSON, reconstruire `BM25Okapi` au chargement.

### [IMPORTANT] Tokenisation BM25 primitive
- Regex `[a-zàâäéèêëîïôùûüçœæ]{2,}` → ne capture pas `l'`, `d'`, `qu'`, contractions, chiffres.
- Stemmer "maison" (lignes 36-53) très basique. **Solution** : utiliser `nltk.stem.SnowballStemmer('french')` ou `spacy fr_core_news_sm` (déjà standard).
- Les stopwords incluent `"être"`, `"avoir"`, `"faire"` : pour des fiches RH où "Chef de" est partout, ok. Mais ils sont aussi dans des fonctions comme "Faire le suivi" → perte d'info.

### [IMPORTANT] Embeddings choisis sub-optimaux
`paraphrase-multilingual-MiniLM-L12-v2` (384 dim) — correct mais loin du SOTA pour le français en 2025-2026 :
- **`BAAI/bge-m3`** (1024 dim) : meilleur multi-lingual, gère long context.
- **`intfloat/multilingual-e5-large`** (1024 dim) : excellent pour FR.
- Coût : ~3x plus de RAM, ~2x plus lent à indexer, mais qualité de retrieval nettement supérieure.

### [MINEUR] `vector_store.count()` à chaque requête exhaustive
`pipeline.py` appelle `self.vector_store.count()` qui fait un appel ChromaDB. À cacher.

### [MINEUR] RRF k=60 hardcodé dans `hybrid_search.py`
Configurable via `config.rrf_k` mais la valeur par défaut 60 vient d'un papier de 2009. À calibrer empiriquement.

---

## 4. Reranking

### [IMPORTANT] Cross-encoder mmarco distillé
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` est correct mais distillé. Pour 30-50 candidats, le coût d'un meilleur reranker est négligeable. Alternative : `BAAI/bge-reranker-v2-m3`.

### [IMPORTANT] `max_length=512` tronque les longs chunks
Si un chunk fait 800 tokens, le reranker n'en voit que les 512 premiers → score biaisé. Solution : tronquer le chunk avant rerank ou utiliser un reranker à long contexte.

---

## 5. Génération LLM (`llm.py`, `query_transform.py`)

### [IMPORTANT] Latence inacceptable pour interactif
Qwen2.5-3B sur CPU = 5-10s par appel. Avec :
- 1 appel IntentRouter (~2s)
- 1 appel par batch de validation (3 batches × 2s = 6s)
- 1 appel génération finale (~5s)

→ **15-20s par requête** dans le pire cas. Solutions par ordre de simplicité :
1. **GPU obligatoire** (Colab T4 = gratuit). 5x plus rapide.
2. **Modèle plus petit pour la classification** (Qwen 0.5B sur CPU = 1s).
3. **Paralléliser les batches de validation** (asyncio.gather + threading).
4. **Quantification GGUF** (llama.cpp) pour CPU performant.

### [IMPORTANT] Pas de timeout sur `generate()`
Si le modèle entre en boucle (cas réel sur petits modèles), blocage indéfini. Ajouter un `timeout` côté client (threading.Timer ou async).

### [MINEUR] Détection de boucle (3 lignes identiques) trop stricte
Stoppe à tort les listes répétitives légitimes ("Chef de Section X / Chef de Section Y / Chef de Section Z").

### [MINEUR] `query_transform` jamais utilisé par défaut
`use_query_transform=False` par défaut. Mort code en prod.

---

## 6. IntentRouter (notre nouveau module)

### [IMPORTANT] Point de panne unique
Tout le routing dépend du LLM. Si Qwen retourne un JSON corrompu → fallback `qa` silencieux → on retombe sur le retrieval hybride classique (qualité dégradée). Le `confidence: "low"` est loggé mais n'est jamais utilisé pour alerter l'utilisateur.

### [IMPORTANT] Schéma figé au démarrage
`SchemaDiscovery.scan()` est appelé une fois dans `__init__`. Si `documents/` change après démarrage, le router ne le sait pas. Pour un PFE acceptable, en prod il faudrait recalculer après chaque ingestion.

### [MINEUR] Cache non persisté
Le cache LRU de l'IntentRouter est en mémoire. Au redémarrage, on perd toutes les classifications déjà calculées.

### [MINEUR] Few-shot consomme ~400 tokens à chaque requête
Les 4 exemples du prompt sont rejoués à chaque appel. Sur un petit modèle c'est nécessaire mais cher en latence.

---

## 7. Validation par batches (notre nouveau module)

### [IMPORTANT] Séquentiel, pas parallèle
`_llm_validate_batch` traite les batches en série. Pour 25 items / batch=10 → 3 appels séquentiels = ~6s. **Parallélisable** trivialement avec `concurrent.futures.ThreadPoolExecutor` (les appels HF sont bloquants mais relâchent le GIL pendant le forward pass) → 2s.

### [IMPORTANT] Failsafe trop permissif
Si un batch échoue (parse, exception), on **conserve tous les éléments**. Cela peut laisser passer du bruit que la validation devrait justement filtrer. Un compromis : tenter 1 retry avant de garder.

### [MINEUR] Pas de fusion sémantique des doublons
"SOCIALE" / "SOCIALES" / "SOCIOAL" sont des typos d'un même intitulé. La validation LLM peut en garder 2-3 versions. Avant validation, ajouter une dédup **fuzzy** (`rapidfuzz.process.dedupe` avec seuil 85) qui réduirait la liste avant les appels LLM.

---

## 8. API & Déploiement (`run_tunnel.py`, `requirements.txt`)

### [BLOQUANT] ngrok sans authentification
`run_tunnel.py` expose l'API publiquement sans aucune auth. N'importe qui scannant ngrok peut hammer votre LLM (= votre quota Colab) ou injecter des requêtes. **Fix obligatoire** :
- Ajouter `ngrok config add-authtoken <token>` (token personnel).
- Header `Authorization: Bearer <secret>` validé côté FastAPI.
- Rate limit (slowapi).

### [IMPORTANT] `requirements.txt` non pinné
Aucune version. Une mise à jour de `transformers` ou `chromadb` peut casser le projet sans préavis. **Fix** : `pip freeze > requirements.txt` ou utiliser un `requirements.lock`.

### [IMPORTANT] Pas de conteneurisation
Pas de Dockerfile. Pour un PFE évalué, un `docker compose up` qui démarre tout serait un gros plus.

### [MINEUR] Pas de fichier `.env`
La config est dans `config.py` codée en dur. Devrait pouvoir override par variable d'env (`os.getenv("LLM_MODEL", default)`).

---

## 9. Observabilité

### [IMPORTANT] Logs non-structurés
Tout est en texte libre via `logging.info`. Difficile à parser/agréger. Pour démontrer la rigueur, passer à JSON :
```python
logger.info("intent_classified", extra={"intent": "list", "source": "SERVICE", "ms": 1234})
```

### [IMPORTANT] Pas de métriques
On ne sait pas mesurer : latence p50/p95, taux de bypass, taux de cache hit, taux d'erreur classification. **Fix minimal** : un dict `self.stats` mis à jour à chaque requête, exposé via un endpoint `/stats`.

---

## 10. Qualité des données (hors RAG)

### [IMPORTANT] Données "sales" non nettoyées en amont
Les Excel contiennent des **typos** ("SOCIOAL"), des **duplicatas** ("SOCIALE" / "SOCIALES"), des variations d'espacement. Le RAG hérite de tout ça. **Aucun pipeline ne peut compenser ça à 100%**. Recommandation : un script `clean_data.py` exécuté avant l'ingestion qui :
- Trim + normalise espaces.
- Détecte les valeurs proches (rapidfuzz) et propose des fusions au user.
- Logge les anomalies dans un rapport.

---

## Top 8 actions prioritaires

| # | Action | Coût | Impact |
|---|---|---|---|
| 1 | Ajouter `StructuredQueryEngine` (DuckDB) pour les Excel | 1 jour | **Énorme** — fiabilité 100% sur les listes |
| 2 | Sécuriser ngrok (authtoken + bearer header) | 1h | **Critique** — sécurité |
| 3 | Créer `eval/questions.json` (30 Q+A) + `evaluate.py` | 0.5 jour | **Énorme** — mesure objective |
| 4 | Pinner `requirements.txt` | 5 min | Reproductibilité |
| 5 | Remplacer pickle BM25 par JSON + reconstruction | 30 min | Sécurité |
| 6 | Paralléliser `_llm_validate_batch` (ThreadPoolExecutor) | 30 min | Latence ÷ 2-3 |
| 7 | Passer aux embeddings `bge-m3` ou `multilingual-e5-large` | 1h + ré-ingestion | Qualité retrieval +20% |
| 8 | Script `clean_data.py` pour normaliser les Excel | 0.5 jour | Réduit le bruit en amont |

---

## Ce qui est BIEN dans le projet

Pour ne pas être uniquement négatif, voici les bons points :

- **Architecture en couches propre** : ingestion / retrieval / reranking / génération bien séparées.
- **Hybrid search (dense + BM25 + RRF)** : pattern moderne et solide.
- **Cross-encoder reranker** : étape souvent oubliée dans les RAG amateurs.
- **Choix de Qwen2.5** : excellent rapport qualité/taille pour FR.
- **Logging détaillé** par étape (avec compteurs) : très utile pour déboguer.
- **Auto-détection du header Excel** : élégant.
- **Le pivot vers IntentRouter** : suppression du hardcode FR, vraie amélioration.
- **Cache LRU sur retrieval et classification** : bon réflexe perf.
- **README détaillé** avec exemples d'API, codes d'erreur, frontend.

Le projet a une bonne fondation. Les critiques ci-dessus sont du polissage de qualité industrielle, pas un re-design from scratch.
