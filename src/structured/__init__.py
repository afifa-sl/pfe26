"""Module de requêtage structuré (DuckDB) pour les données tabulaires Excel.

Le RAG vectoriel (embeddings + reranker + LLM) est inadapté aux données
structurées : il est lent, non-déterministe, et hallucine sur les listes
exhaustives. Pour ce type de données, une simple requête SQL donne 100% de
précision en quelques millisecondes.

Ce module est utilisé en bypass du pipeline RAG pour toutes les questions
dont l'IntentRouter détecte qu'elles ciblent une source Excel.
"""
from .query_engine import StructuredQueryEngine

__all__ = ["StructuredQueryEngine"]
