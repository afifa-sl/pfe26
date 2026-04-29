#!/usr/bin/env python3
"""
CLI — Ingestion de documents dans le RAG local.

Usage:
    python ingest.py
    python ingest.py --docs-dir ./mes_docs
    python ingest.py --reset          # Réindexe depuis zéro
    python ingest.py --model Qwen/Qwen2.5-7B-Instruct  # Modèle HuggingFace différent
"""
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from config import config
from src.pipeline import RAGPipeline


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ingestion de documents dans le RAG local")
    parser.add_argument("--docs-dir", default=config.docs_dir, help="Dossier des documents à indexer")
    parser.add_argument("--model", default=config.llm_model, help="Modèle HuggingFace (ex: Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--reset", action="store_true", help="Réinitialise l'index avant d'ingérer")
    parser.add_argument(
        "--embedding-model",
        default=config.embedding_model,
        help="Modèle d'embedding (ex: all-MiniLM-L6-v2, BAAI/bge-m3)",
    )
    args = parser.parse_args()

    config.docs_dir = args.docs_dir
    config.llm_model = args.model
    config.embedding_model = args.embedding_model

    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.docs_dir, exist_ok=True)

    pipeline = RAGPipeline(config)
    result = pipeline.ingest(docs_dir=args.docs_dir, reset=args.reset)

    print(f"Résumé:")
    print(f"  Documents traités : {result['documents']}")
    print(f"  Chunks indexés    : {result['chunks']}")
    print(f"  Dimension embed.  : {result['embedding_dim']}")
    print(f"\nLancez maintenant: python query.py")


if __name__ == "__main__":
    main()
