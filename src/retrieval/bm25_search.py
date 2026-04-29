"""Recherche sparse BM25 (Okapi BM25) — optimisé français."""
import logging
import os
import pickle
import re
from typing import List, Dict, Any
from dataclasses import dataclass

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

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "documents":         self.documents,
                "tokenized_corpus":  self._tokenized_corpus,
                "bm25":              self.bm25,
            }, f)
        logger.info("Index BM25 sauvegardé: %s", path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.documents           = data["documents"]
        self._tokenized_corpus   = data["tokenized_corpus"]
        # Charge l'objet BM25 directement s'il existe, sinon reconstruit (rétrocompat)
        if "bm25" in data and data["bm25"] is not None:
            self.bm25 = data["bm25"]
        else:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("Index BM25 chargé: %d documents", len(self.documents))
        return True