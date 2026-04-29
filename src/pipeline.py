"""Pipeline RAG principal — orchestration de toutes les couches."""
import hashlib
import logging
import os
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

from .ingestion.loader import load_directory, Document
from .ingestion.chunker import chunk_documents
from .ingestion.embedder import Embedder
from .retrieval.vector_store import VectorStore
from .retrieval.bm25_search import BM25Search, BM25Document
from .retrieval.hybrid_search import reciprocal_rank_fusion
from .reranking.reranker import CrossEncoderReranker
from .generation.llm import HFClient
from .generation.query_transform import QueryTransformer

# ── Prompts optimisés pour petit modèle (qwen2.5:0.5b) ──────────────────────
# Règle d'or : prompt court + instruction simple = meilleure réponse

_SYSTEM_PROMPT = """Tu es un assistant RH. Réponds en français uniquement à partir du contexte fourni.
Chaque entrée du contexte est préfixée par sa source entre crochets (ex: [DIRECTION], [DEPARTEMENT], [SERVICE], [POSTE]).
Utilise UNIQUEMENT les données de la source pertinente à la question. Ignore les entrées provenant de sources non pertinentes.
Si l'information n'est pas dans le contexte, réponds : "Je ne trouve pas cette information dans les documents."
N'invente JAMAIS de données. Cite uniquement ce qui apparaît explicitement dans le contexte."""

_GENERATION_PROMPT = """Contexte:
{context}

{history}Question: {question}

Réponse:"""

_GENERATION_PROMPT_LIST = """Contexte:
{context}

{history}Question: {question}

IMPORTANT: La question demande une liste. Tu dois citer TOUS les éléments présents dans le contexte, sans en omettre.
Formatte la réponse sous forme de liste numérotée. Ne résume pas, ne regroupe pas, liste chaque élément.

Réponse:"""

# Mots-clés qui indiquent une question de type "liste exhaustive"
_LIST_KEYWORDS = [
    # Pluriels explicites — "les directeurs", "les chefs", "les noms"
    "donne moi les", "donne-moi les", "donnes moi les",
    "donne les", "donne la liste",
    "quels sont", "quelles sont",
    "qui sont les",
    "affiche les", "montre les", "cite les",
    # Exhaustivité explicite
    "liste", "lister", "tous les", "toutes les", "tout le", "toute la",
    "combien", "énumère", "énumérer", "ensemble des", "totalité",
    "chaque", "l'ensemble", "récapitulatif", "récapitule",
    "affiche tous", "affiche toutes", "montre tous", "montre toutes",
    # Disponibilité / existence
    "disponible", "disponibles", "existant", "existants",
    # Formations (pluriel uniquement — évite "la formation X" qui est une requête ciblée)
    "les formations", "toutes les formations", "plan de formation",
    "obligatoire", "obligatoires",
    "facultatif", "facultatifs", "facultatives", "facultative",
]

# Mots-clés qui annulent le mode exhaustif même si un keyword liste est présent.
# Ex: "donne moi des détails pour la formation X" → requête ciblée, pas une liste.
_LIST_OVERRIDE_KEYWORDS = [
    "détail", "détails", "detail", "details",
    "explique", "expliquer", "explications", "explication",
    "décris", "décrire", "description",
    "présente", "présenter", "présentation",
    "qu'est-ce que", "c'est quoi", "kesako",
    "parle moi de", "dis moi",
    "en quoi consiste", "que couvre", "que comprend",
]


class RAGPipeline:
    def __init__(self, config):
        self.config = config

        logger.info("Initialisation du pipeline RAG local...")
        logger.info("=" * 50)

        self.embedder = Embedder(
            model_name=config.embedding_model,
            device=config.embedding_device,
        )

        self.vector_store = VectorStore(
            persist_dir=config.chroma_persist_dir,
            collection_name=config.collection_name,
        )

        self.bm25 = BM25Search()
        self.bm25.load(config.bm25_index_path)

        self.reranker = CrossEncoderReranker(model_name=config.reranker_model)

        self.llm = HFClient(
            model=config.llm_model,
        )

        self.query_transformer = QueryTransformer(llm=self.llm)

        # Cache LRU
        self._retrieval_cache: Dict[tuple, List[Dict]] = {}
        self._cache_max_size: int = 128

        # Formations en mémoire pour bypass LLM
        self._formations_obligatoires, self._formations_facultatives = \
            self._load_formations_from_excel(config.docs_dir)

        logger.info("=" * 50)
        logger.info("Pipeline prêt.")

    @staticmethod
    def _normalize_stem(fname: str) -> str:
        """SERVICE (1).xlsx → SERVICE"""
        import re as _re
        stem = fname.rsplit(".", 1)[0] if "." in fname else fname
        stem = stem.upper().strip()
        stem = _re.sub(r"\s*\(\d+\)\s*$", "", stem)
        stem = _re.sub(r"\s*_\d+\s*$", "", stem)
        return stem.strip()

    @staticmethod
    def _load_formations_from_excel(docs_dir: str):
        """
        Lit KAM_Formations_GTP.xlsx directement depuis le dossier documents.
        Retourne (obligatoires: list[str], facultatives: list[str]).
        Nécessaire car load_excel_as_documents ne gère pas ce fichier correctement
        (le header réel N°/Intitulé/Statut est en ligne 4, pas en ligne 1).
        """
        import glob, openpyxl
        patterns = ["*FORMATION*GTP*.xlsx","*KAM*FORMATION*.xlsx",
                    "*formation*gtp*.xlsx","*GTP*.xlsx"]
        found = []
        for pat in patterns:
            found.extend(glob.glob(f"{docs_dir}/**/{pat}", recursive=True))
            found.extend(glob.glob(f"{docs_dir}/{pat}"))
        if not found:
            return [], []
        path = found[0]
        try:
            wb = openpyxl.load_workbook(path, data_only=True)
            ws = wb.active
            obligatoires, facultatives, current_statut = [], [], None
            for row in ws.iter_rows(values_only=True):
                c0 = str(row[0]).strip() if row[0] else ""
                c1 = str(row[1]).strip() if len(row)>1 and row[1] else ""
                c2 = str(row[2]).strip() if len(row)>2 and row[2] else ""
                joined = (c0+c1+c2).upper()
                if "OBLIGATOIRE" in joined and not c0.isdigit():
                    current_statut = "Obligatoire"; continue
                if "FACULTATIV" in joined and not c0.isdigit():
                    current_statut = "Facultative"; continue
                if not c0.isdigit(): continue
                statut = c2 if c2 else current_statut or "Inconnue"
                if "obligatoire" in statut.lower():
                    obligatoires.append(c1.strip())
                else:
                    facultatives.append(c1.strip())
            wb.close()
            logger.info("  ✓ %d formations chargées (%d oblig. / %d facult.)",
                        len(obligatoires)+len(facultatives), len(obligatoires), len(facultatives))
            return obligatoires, facultatives
        except Exception as e:
            logger.warning("  ⚠ Formations non chargées: %s", e)
            return [], []

    @staticmethod
    def _is_list_question(question: str) -> bool:
        """Détecte si la question demande une liste exhaustive.
        Les questions avec des mots-clés de détail (ex: 'détails pour la formation X')
        sont des requêtes ciblées et ne doivent PAS déclencher le mode exhaustif."""
        q = RAGPipeline._normalize_accents(question.lower())
        # Si la question demande un détail/explication → jamais exhaustif
        if any(RAGPipeline._normalize_accents(kw) in q for kw in _LIST_OVERRIDE_KEYWORDS):
            return False
        norm_kws = [RAGPipeline._normalize_accents(kw) for kw in _LIST_KEYWORDS]
        return any(kw in q for kw in norm_kws)

    # ── SOURCE-KEYWORD MAPPING ──────────────────────────────────────────────
    # Mapping entre mots-clés dans la question et fichiers sources pertinents.
    # Permet de filtrer les chunks non pertinents (ex: POSTE.xlsx quand on
    # demande des "directeurs" → DIRECTION.xlsx est plus pertinent).
    _SOURCE_KEYWORDS = {
        "DIRECTION": ["directeur", "directeurs", "directrice", "directrices", "direction", "directions"],
        "DEPARTEMENT": ["departement", "departements", "département", "départements", "chef de departement", "chefs de departement"],
        "SERVICE": ["service", "services", "chef de service", "chefs de service"],
        # Questions sur le contenu d'une formation → Explications_F.docx
        "EXPLICATIONS_F": [
            "détail", "détails", "detail", "details",
            "explique", "explication", "explications",
            "décris", "description", "présente",
            "en quoi consiste", "que couvre", "que comprend",
            "parle moi de", "qu'est-ce que",
        ],
    }
    # Mots-clés qui excluent POSTE.xlsx (données organisationnelles vs fiches de poste)
    # "chantier" est une colonne dans DIRECTION/DEPARTEMENT/SERVICE, pas dans POSTE
    _EXCLUDE_POSTE_KEYWORDS = [
        "chantier", "chantiers", "affectation", "affectations",
        "matricule", "matricules", "nom", "prenom", "prénom",
        "observation", "fonction",
        # Questions sur une personne → DEPARTEMENT/SERVICE, pas POSTE
        "qui est le chef", "qui est la chef", "qui est le directeur",
        "qui est le responsable", "chef de departement", "chef de département",
        "chef de service", "directeur de", "responsable de",
    ]

    @staticmethod
    def _normalize_accents(text: str) -> str:
        """Normalise les accents français pour un matching robuste."""
        for src, dst in [("é","e"),("è","e"),("ê","e"),("ë","e"),
                         ("à","a"),("â","a"),("ä","a"),
                         ("î","i"),("ï","i"),
                         ("ô","o"),("ö","o"),
                         ("ù","u"),("û","u"),("ü","u"),
                         ("ç","c"),("œ","oe"),("æ","ae")]:
            text = text.replace(src, dst)
        return text

    @staticmethod
    def _detect_relevant_sources(question: str) -> set:
        """Détecte les sources pertinentes à partir des mots-clés de la question."""
        q = RAGPipeline._normalize_accents(question.lower())
        relevant = set()
        for source, keywords in RAGPipeline._SOURCE_KEYWORDS.items():
            norm_kws = [RAGPipeline._normalize_accents(kw) for kw in keywords]
            if any(kw in q for kw in norm_kws):
                relevant.add(source)
        return relevant

    @staticmethod
    def _should_exclude_poste(question: str) -> bool:
        """Détecte si la question porte sur des données organisationnelles
        (colonnes de DIRECTION/DEPARTEMENT/SERVICE) qui n'existent pas dans POSTE."""
        q = RAGPipeline._normalize_accents(question.lower())
        exclude_kws = [RAGPipeline._normalize_accents(kw) for kw in RAGPipeline._EXCLUDE_POSTE_KEYWORDS]
        return any(kw in q for kw in exclude_kws)

    @staticmethod
    def _filter_by_source(chunks: list, relevant_sources: set, exclude_poste: bool = False) -> list:
        """Filtre les chunks pour privilégier les sources pertinentes.
        Si des sources pertinentes sont détectées, garde uniquement les chunks
        provenant de ces sources + un petit nombre de chunks d'autres sources."""

        filtered = chunks

        # Exclure POSTE.xlsx si la question porte sur des données organisationnelles
        if exclude_poste and not relevant_sources:
            filtered = [c for c in filtered
                        if c["metadata"].get("filename", "").upper().rsplit(".", 1)[0] != "POSTE"]
            if filtered:
                logger.info("  → Exclusion POSTE: %d/%d chunks retenus", len(filtered), len(chunks))
                return filtered
            return chunks  # Fallback si tout filtré

        if not relevant_sources:
            return chunks  # Pas de filtre si aucune source détectée

        from_relevant = []
        from_other = []
        for chunk in filtered:
            fname = chunk["metadata"].get("filename", "")
            source_stem = RAGPipeline._normalize_stem(fname)
            if source_stem in relevant_sources:
                from_relevant.append(chunk)
            else:
                from_other.append(chunk)

        if not from_relevant:
            return chunks  # Aucun chunk de la source pertinente → ne pas filtrer

        # Garder tous les chunks pertinents + max 3 chunks d'autres sources (contexte)
        result = from_relevant + from_other[:3]
        logger.info("  → Filtre source: %d/%d chunks retenus (sources: %s)",
                     len(result), len(chunks), ", ".join(relevant_sources))
        return result

    # ── EXTRACTION DIRECTE (bypass retrieval pour listes exhaustives) ──────
    # Patterns question → colonne à extraire
    _DIRECT_EXTRACT_PATTERNS = [
        # (mots-clés question, colonne à extraire, sources à scanner)
        # colonne=None         → retourner toute la ligne
        # colonne="CHANTIER"   → extraire les valeurs de la colonne CHANTIER
        # colonne="FORMATION_ALL"    → lire Excel formations (toutes)
        # colonne="FORMATION_OBLIG"  → lire Excel formations obligatoires uniquement
        # colonne="FORMATION_FACULT" → lire Excel formations facultatives uniquement
        # ⚠ Ordre important : patterns spécifiques AVANT les génériques
        (["chef de departement", "chefs de departement",
          "chef de département", "chefs de département"],         None,               ["DEPARTEMENT"]),
        (["chef de service", "chefs de service"],                  None,               ["SERVICE"]),
        (["directeur", "directeurs", "directrice"],                None,               ["DIRECTION"]),
        (["chantier", "chantiers"],                                "CHANTIER",         ["DIRECTION", "DEPARTEMENT", "SERVICE"]),
        (["service"],                                              "CHANTIER",         ["SERVICE"]),
        (["departement", "département"],                           "CHANTIER",         ["DEPARTEMENT"]),
        (["direction"],                                            "CHANTIER",         ["DIRECTION"]),
        # Formations — lu directement depuis Excel (loader ne gère pas ce format)
        (["formations obligatoires", "formations obligatoire",
          "formation obligatoire", "obligatoires", "obligatoire"], "FORMATION_OBLIG",  ["KAM_FORMATIONS_GTP"]),
        (["formations facultatives", "formations facultative",
          "formation facultative", "facultatif", "facultatifs",
          "facultatives", "facultative"],                          "FORMATION_FACULT", ["KAM_FORMATIONS_GTP"]),
        (["formations disponibles", "formations disponible",
          "quelles sont les formations", "liste des formations",
          "toutes les formations", "plan de formation",
          "formations existantes"],                                "FORMATION_ALL",    ["KAM_FORMATIONS_GTP"]),
    ]

    def _try_direct_extract(self, question: str) -> Optional[List[Dict]]:
        """Pour les questions de type liste, extrait directement sans passer par le retriever.
        Gère 4 types de colonnes :
          None              → toute la ligne (chefs, directeurs)
          "CHANTIER"        → valeur de la colonne CHANTIER (noms services/depts/dirs)
          "FORMATION_ALL"   → toutes les formations depuis Excel
          "FORMATION_OBLIG" → formations obligatoires depuis Excel
          "FORMATION_FACULT"→ formations facultatives depuis Excel
        """
        q = question.lower()
        # Normaliser les accents pour le matching
        for src, dst in [("é","e"),("è","e"),("ê","e"),("à","a"),
                         ("â","a"),("î","i"),("ô","o"),("ù","u"),("û","u")]:
            q = q.replace(src, dst)

        for keywords, column, sources in self._DIRECT_EXTRACT_PATTERNS:
            if not any(kw in q for kw in keywords):
                continue

            # ── Cas spécial : formations lues depuis Excel ─────────────────
            if column in ("FORMATION_ALL", "FORMATION_OBLIG", "FORMATION_FACULT"):
                if column == "FORMATION_OBLIG":
                    items = [(f, "Obligatoire") for f in self._formations_obligatoires]
                elif column == "FORMATION_FACULT":
                    items = [(f, "Facultative") for f in self._formations_facultatives]
                else:  # FORMATION_ALL
                    items = ([(f, "Obligatoire") for f in self._formations_obligatoires]
                           + [(f, "Facultative") for f in self._formations_facultatives])
                if not items:
                    logger.warning("  ⚠ Formations vides — vérifiez KAM_Formations_GTP.xlsx")
                    continue
                results = [{"content": f"{intitule} ({statut})",
                            "metadata": {"filename": "KAM_Formations_GTP.xlsx"}}
                           for intitule, statut in items]
                logger.info("  ⟹ Extraction directe: %d formations (colonne=%s)",
                            len(results), column)
                return results

            # ── Cas général : scanner les documents BM25 ──────────────────
            results = []
            seen_values = set()
            for doc in self.bm25.documents:
                fname       = doc.metadata.get("filename", "")
                source_stem = self._normalize_stem(fname)
                if source_stem not in sources:
                    continue

                if column:
                    for part in doc.content.split("|"):
                        part = part.strip()
                        if ":" in part:
                            key, val = part.split(":", 1)
                            key = key.strip().lstrip("[").rstrip("]").strip()
                            val = val.strip()
                            if key.upper() == column and val and val.lower() not in seen_values:
                                seen_values.add(val.lower())
                                results.append({"content": val, "metadata": doc.metadata})
                else:
                    results.append({"content": doc.content, "metadata": doc.metadata})

            if results:
                logger.info("  ⟹ Extraction directe: %d résultats (colonne=%s, sources=%s)",
                            len(results), column or "ALL", ",".join(sources))
                return results

        return None

    # ── DÉDUPLICATION ────────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate_chunks(chunks: list) -> list:
        """Supprime les chunks avec un contenu identique (hash SHA256)."""
        seen = set()
        unique = []
        for chunk in chunks:
            h = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(chunk)
        removed = len(chunks) - len(unique)
        if removed > 0:
            logger.info("  → %d chunk(s) dupliqué(s) supprimé(s)", removed)
        return unique

    # ── INGESTION ────────────────────────────────────────────────────────────

    def ingest(self, docs_dir: Optional[str] = None, reset: bool = False) -> Dict[str, Any]:
        """Ingère les documents du dossier dans le RAG."""
        docs_dir = docs_dir or self.config.docs_dir

        logger.info("=" * 50)
        logger.info("INGESTION PIPELINE")
        logger.info("=" * 50)

        if reset:
            logger.info("Réinitialisation du vector store...")
            self.vector_store.reset()
            if os.path.exists(self.config.bm25_index_path):
                os.remove(self.config.bm25_index_path)

        # 1. Chargement
        logger.info("[1/4] Chargement des documents depuis '%s'...", docs_dir)
        documents = load_directory(docs_dir)
        if not documents:
            raise ValueError(f"Aucun document trouvé dans: {docs_dir}")
        logger.info("  → %d documents chargés", len(documents))

        # 2. Chunking
        logger.info("[2/4] Découpage (chunk_size=%d tokens, overlap=%d)...", self.config.chunk_size, self.config.chunk_overlap)
        chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            embedding_model=self.config.embedding_model,
        )
        logger.info("  → %d chunks créés", len(chunks))
        chunks = self._deduplicate_chunks(chunks)

        # 3. Embeddings
        logger.info("[3/4] Génération des embeddings (%s)...", self.config.embedding_model)
        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed(
            texts,
            batch_size=self.config.embedding_batch_size,
            show_progress=True,
        )
        logger.info("  → Shape: %s", embeddings.shape)

        # 4. Indexation
        logger.info("[4/4] Indexation (ChromaDB + BM25)...")
        self.vector_store.add(
            ids=[c.id for c in chunks],
            embeddings=list(embeddings),
            texts=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

        bm25_docs = [
            BM25Document(id=c.id, content=c.content, metadata=c.metadata)
            for c in chunks
        ]
        self.bm25.add_documents(bm25_docs)
        self.bm25.save(self.config.bm25_index_path)

        self._retrieval_cache.clear()

        logger.info("=" * 50)
        logger.info("Ingestion terminée: %d chunks indexés", len(chunks))
        logger.info("=" * 50)

        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "embedding_dim": int(embeddings.shape[1]),
        }

    def ingest_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Ingère une liste de Documents directement (sans lire un dossier)."""
        chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            embedding_model=self.config.embedding_model,
        )
        if not chunks:
            return {"documents": len(documents), "chunks": 0, "embedding_dim": 0}
        chunks = self._deduplicate_chunks(chunks)

        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed(
            texts,
            batch_size=self.config.embedding_batch_size,
            show_progress=False,
        )
        self.vector_store.add(
            ids=[c.id for c in chunks],
            embeddings=list(embeddings),
            texts=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        bm25_docs = [
            BM25Document(id=c.id, content=c.content, metadata=c.metadata)
            for c in chunks
        ]
        self.bm25.add_documents(bm25_docs)
        self.bm25.save(self.config.bm25_index_path)
        self._retrieval_cache.clear()

        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "embedding_dim": int(embeddings.shape[1]),
        }

    # ── QUERY ────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        use_query_transform: bool = False,   # Désactivé par défaut pour petits modèles
        stream: bool = False,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Interroge le RAG et retourne la réponse avec ses sources."""
        start = time.time()

        # Détection automatique : question de type "liste exhaustive" ?
        exhaustive = self._is_list_question(question)
        if exhaustive:
            logger.info("  ⟹ Mode exhaustif détecté (question de type liste)")

        # ── Extraction directe (bypass retrieval pour les listes) ───────
        if exhaustive:
            direct = self._try_direct_extract(question)
            if direct:
                # ✅ BYPASS COMPLET DU LLM — formatage Python direct
                seen: set = set()
                unique_items = []
                for d in direct:
                    item = d["content"].rstrip(".").strip()
                    key  = item.lower()
                    if key not in seen and len(item) > 2:
                        seen.add(key)
                        unique_items.append(item)
                answer  = (f"Il y a {len(unique_items)} résultats :\n"
                           + "\n".join(f"{i+1}. {item}"
                                       for i, item in enumerate(unique_items)))
                elapsed = round(time.time() - start, 2)
                sources = list({d["metadata"].get("filename","?") for d in direct})
                logger.info("  ✅ Réponse directe (sans LLM): %d éléments en %.2fs",
                            len(unique_items), elapsed)
                return {
                    "question":        question,
                    "search_query":    question,
                    "answer":          answer,
                    "sources":         sources,
                    "chunks_used":     len(unique_items),
                    "elapsed_seconds": elapsed,
                }

        # Étape 1 : Transformation de la requête (optionnelle)
        if use_query_transform:
            logger.info("  [1/4] Transformation de la requête...")
            search_query = self.query_transformer.rewrite(question)
            if search_query != question:
                logger.info("        → %s", search_query)
        else:
            search_query = question

        # Détection des sources pertinentes (fait ICI pour inclure dans le cache key)
        relevant_sources = self._detect_relevant_sources(question)
        exclude_poste    = self._should_exclude_poste(question)

        # Étape 2+3 : Recherche hybride + Reranking (avec cache)
        cache_key = (search_query.strip().lower(), exhaustive,
                     frozenset(relevant_sources), exclude_poste)
        if cache_key in self._retrieval_cache:
            logger.info("  [2/4] Recherche hybride... (cache hit)")
            logger.info("  [3/4] Reranking... (cache hit)")
            reranked = self._retrieval_cache[cache_key]
        else:
            logger.info("  [2/4] Recherche hybride%s...", " (mode exhaustif)" if exhaustive else "")
            query_emb = self.embedder.embed_single(search_query)
            if exhaustive:
                # Mode exhaustif : remonter beaucoup plus de candidats
                k_dense = min(self.config.max_chunks_exhaustive * 5, self.vector_store.count())
                k_sparse = self.config.max_chunks_exhaustive * 5
            else:
                k_dense = self.config.top_k_dense
                k_sparse = self.config.top_k_sparse
            dense  = self.vector_store.search(query_emb, k=k_dense)
            sparse = self.bm25.search(search_query, k=k_sparse)
            hybrid = reciprocal_rank_fusion(dense, sparse, k=self.config.rrf_k)
            logger.info("        Dense: %d | Sparse: %d | RRF: %d", len(dense), len(sparse), len(hybrid))

            logger.info("  [3/4] Reranking...")
            if exhaustive:
                # Mode exhaustif : top-k élargi pour récupérer un max de résultats
                reranked = self.reranker.rerank(
                    query=search_query,
                    documents=hybrid[:self.config.max_chunks_exhaustive * 3],
                    top_k=self.config.max_chunks_exhaustive,
                )
                if reranked:
                    logger.info("        Scores reranker: [%.2f ... %.2f]",
                                reranked[0].get("rerank_score", 0),
                                reranked[-1].get("rerank_score", 0))
            else:
                # Mode classique : top-k fixe
                reranked = self.reranker.rerank(
                    query=search_query,
                    documents=hybrid[:20],
                    top_k=self.config.top_k_after_rerank,
                )
            # Mise en cache (LRU simple)
            if len(self._retrieval_cache) >= self._cache_max_size:
                oldest_key = next(iter(self._retrieval_cache))
                del self._retrieval_cache[oldest_key]
            self._retrieval_cache[cache_key] = reranked
        logger.info("        → %d chunks retenus", len(reranked))

        # Étape 3.5 : Filtre par source pertinente (évite la pollution POSTE.xlsx)
        if relevant_sources or exclude_poste:
            reranked = self._filter_by_source(reranked, relevant_sources, exclude_poste)

        # Étape 4 : Génération
        logger.info("  [4/4] Génération LLM...")
        context      = self._format_context(reranked)
        history_text = self._format_history(history) if history else ""
        template = _GENERATION_PROMPT_LIST if exhaustive else _GENERATION_PROMPT
        prompt = template.format(
            context=context,
            question=question,
            history=history_text,
        )

        # Adapter max_tokens : plus de tokens pour les listes exhaustives
        max_tokens = self.config.llm_max_tokens_long if exhaustive else self.config.llm_max_tokens

        if stream:
            answer_parts = []
            for token in self.llm.generate_stream(
                prompt=prompt, system=_SYSTEM_PROMPT, max_tokens=max_tokens,
            ):
                print(token, end="", flush=True)
                answer_parts.append(token)
            print()
            answer = "".join(answer_parts)
        else:
            answer = self.llm.generate(
                prompt=prompt,
                system=_SYSTEM_PROMPT,
                temperature=self.config.llm_temperature,
                max_tokens=max_tokens,
            )

        elapsed = round(time.time() - start, 2)

        return {
            "question":        question,
            "search_query":    search_query,
            "answer":          answer,
            "sources":         self._extract_sources(reranked),
            "chunks_used":     len(reranked),
            "elapsed_seconds": elapsed,
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        if not history:
            return ""
        lines = []
        # Garde seulement les 4 derniers échanges pour ne pas surcharger le contexte
        for msg in history[-4:]:
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "Historique:\n" + "\n".join(lines) + "\n\n"

    def _format_context(self, chunks: List[Dict]) -> str:
        """Formate le contexte de façon compacte pour les petits modèles."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            # Le contenu est déjà préfixé par [SOURCE] grâce au loader
            parts.append(chunk['content'])
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, chunks: List[Dict]) -> List[str]:
        seen: set = set()
        sources = []
        for chunk in chunks:
            src = chunk["metadata"].get("filename", "inconnu")
            if src not in seen:
                sources.append(src)
                seen.add(src)
        return sources
