"""Transformation de requêtes: réécriture et expansion multi-query."""
import re
from typing import List
from .llm import HFClient

_REWRITE_PROMPT = """Reformule cette question de recherche en français, de façon claire et précise.
Réponds avec UNIQUEMENT la question reformulée, rien d'autre.

Question: {query}
Reformulation:"""

_EXPANSION_PROMPT = """Génère 3 reformulations alternatives de cette question en français.
Réponds avec UNIQUEMENT les 3 questions, une par ligne, sans numérotation.

Question: {query}
Reformulations:"""


class QueryTransformer:
    def __init__(self, llm: HFClient):
        self.llm = llm

    def rewrite(self, query: str) -> str:
        """Réécrit la requête pour améliorer la récupération."""
        try:
            result = self.llm.generate(
                prompt=_REWRITE_PROMPT.format(query=query),
                temperature=0.1,
                max_tokens=100,
            )
            # Nettoie les artefacts courants des petits modèles
            rewritten = result.strip()
            # Supprime les préfixes parasites type "**Question reformulée:**"
            rewritten = re.sub(r"^\*{0,2}[^:]*:\*{0,2}\s*", "", rewritten).strip()
            # Prend uniquement la première ligne non vide
            lines = [l.strip() for l in rewritten.split("\n") if l.strip()]
            rewritten = lines[0] if lines else query
            # Si le résultat est trop long ou bizarre, retourne l'original
            if len(rewritten) > 3 * len(query) or len(rewritten) < 5:
                return query
            return rewritten
        except Exception:
            return query

    def expand(self, query: str) -> List[str]:
        """Génère 3 formulations alternatives de la requête."""
        try:
            result = self.llm.generate(
                prompt=_EXPANSION_PROMPT.format(query=query),
                temperature=0.4,
                max_tokens=300,
            )
            lines = [l.strip() for l in result.strip().split("\n") if l.strip()]
            # Nettoie les numérotations et préfixes
            cleaned = []
            for line in lines:
                line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
                if line:
                    cleaned.append(line)
            return cleaned[:3] if cleaned else [query]
        except Exception:
            return [query]