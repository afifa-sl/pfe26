"""Recherche sparse BM25 (Okapi BM25) — optimisé français.

Format de persistance : JSON (jamais pickle).
    pickle.load() permet l'exécution arbitraire de code si l'index est altéré
    par un attaquant. JSON est purement déclaratif — aucun code exécuté.
    L'objet BM25Okapi est reconstruit à chaque load() à partir du tokenized_corpus
    (rapide : ~ms pour quelques milliers de docs).
"""
import json
import logging
import os
import re
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class BM25Document:
    id: str
    content: str
    metadata: Dict[str, Any]


# Mots vides français à ignorer dans la tokenisation
_STOPWORDS_FR = {
    "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux",
    "et", "ou", "est", "en", "à", "ce", "se", "sa", "son", "ses",
    "il", "elle", "ils", "elles", "je", "tu", "nous", "vous", "on",
    "que", "qui", "quoi", "dont", "où", "par", "sur", "sous", "dans",
    "avec", "pour", "sans", "entre", "vers", "chez", "plus", "très",
    "tout", "tous", "cette", "cet", "ces", "mon", "ton", "ma", "ta",
    "leur", "leurs", "même", "aussi", "mais", "donc", "or", "ni",
    "car", "si", "ne", "pas", "plus", "bien", "être", "avoir", "faire",
    # Mots de requête fréquents (ne servent pas au matching BM25)
    "donne", "donner", "donnez", "moi", "quel", "quelle", "quels", "quelles",
    "comment", "existe", "existant", "existants", "sont",
    "affiche", "afficher", "montre", "montrer", "cite", "citer",
}


def _stem_fr(token: str) -> str:
    """Stemming minimal français — réduit pluriels et féminins courants.
    Pas un vrai stemmer (Snowball) mais suffisant pour le matching BM25."""
    if len(token) <= 3:
        return token
    # Pluriels : -aux → -al, -eaux → -eau, -s final
    if token.endswith("eaux"):
        return token[:-1]
    if token.endswith("aux"):
        return token[:-2] + "l"
    if token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    # Féminins courants : -trice → -teur, -euse → -eur
    if token.endswith("trice"):
        return token[:-4] + "eur"
    if token.endswith("euse"):
        return token[:-3] + "ur"
    return token


class BM25Search:
    def __init__(self):
        self.documents: List[BM25Document] = []
        self._tokenized_corpus: List[List[str]] = []
        self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        """Tokenisation avec gestion des accents français, stopwords et stemming."""
        text = text.lower()
        tokens = re.findall(r"\b[a-zàâäéèêëîïôùûüçœæ]{2,}\b", text)
        return [_stem_fr(t) for t in tokens if t not in _STOPWORDS_FR]

    def add_documents(self, documents: List[BM25Document]) -> None:
        from rank_bm25 import BM25Okapi
        self.documents = documents
        self._tokenized_corpus = [self._tokenize(d.content) for d in documents]
        self.bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, k: int = 15) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.documents:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                results.append({
                    "id":       self.documents[idx].id,
                    "content":  self.documents[idx].content,
                    "metadata": self.documents[idx].metadata,
                    "score":    float(scores[idx]),
                    "rank":     rank + 1,
                })
        return results

    # ── Persistance JSON (pas de pickle : sécurité) ──────────────────────

    _FORMAT_VERSION = 2  # bump à chaque changement de format

    def save(self, path: str) -> None:
        """Sauvegarde l'index au format JSON.
        Ne sauve PAS l'objet BM25Okapi (ne se sérialise pas en JSON proprement) ;
        il sera reconstruit à partir du tokenized_corpus au load()."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "format_version": self._FORMAT_VERSION,
            "documents": [asdict(d) for d in self.documents],
            "tokenized_corpus": self._tokenized_corpus,
        }
        # Écriture atomique : on écrit dans un .tmp puis on rename
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)
        logger.info("Index BM25 sauvegardé (JSON): %s (%d docs)", path, len(self.documents))

    def load(self, path: str) -> bool:
        """Charge l'index JSON et reconstruit BM25Okapi.
        Refuse les anciens formats pickle (.pkl avec contenu binaire) — sécurité."""
        if not os.path.exists(path):
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.warning(
                "Index BM25 illisible (%s) — probablement un ancien format pickle. "
                "Refus de chargement par sécurité. Supprimez %s puis relancez l'ingestion.",
                e, path,
            )
            return False

        if not isinstance(data, dict) or "documents" not in data or "tokenized_corpus" not in data:
            logger.warning("Index BM25 corrompu (format inattendu) — ignoré: %s", path)
            return False

        version = data.get("format_version", 1)
        if version != self._FORMAT_VERSION:
            logger.warning("Index BM25 version %s ≠ %s attendue — réindexation requise.",
                           version, self._FORMAT_VERSION)
            return False

        self.documents = [BM25Document(**d) for d in data["documents"]]
        self._tokenized_corpus = data["tokenized_corpus"]
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("Index BM25 chargé (JSON): %d documents", len(self.documents))
        return True