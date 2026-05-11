#!/usr/bin/env python3
"""
CLI — Interrogation interactive du RAG local.

Usage:
    python query.py                          # Mode interactif
    python query.py -q "Ma question"         # Question directe
    python query.py --no-transform           # Sans réécriture de requête
    python query.py --model mistral          # Modèle différent
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

    parser = argparse.ArgumentParser(description="Interroger le RAG local")
    parser.add_argument("-q", "--question", help="Question directe (non interactif)")
    parser.add_argument("--model", default=config.llm_model, help="Modèle HuggingFace (ex: Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--transform", action="store_true", help="Active la réécriture de requête (désactivé par défaut)")
    parser.add_argument("--no-stream", action="store_true", help="Désactive le streaming de la réponse")
    args = parser.parse_args()

    config.llm_model = args.model

    print("Chargement du pipeline RAG...")
    pipeline = RAGPipeline(config)

    if pipeline.vector_store.count() == 0:
        print("\nERREUR: Aucun document indexé.")
        print("Lancez d'abord: python ingest.py")
        sys.exit(1)

    print(f"Chunks indexés : {pipeline.vector_store.count()}")
    print(f"LLM            : {config.llm_model}")
    print(f"Embedding      : {config.embedding_model}")

    use_transform = args.transform
    use_stream = False  # désactivé — évite le double affichage

    def ask(question: str) -> None:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        result = pipeline.query(
            question=question,
            use_query_transform=use_transform,
            stream=use_stream,
        )
        # Toujours afficher (stream ou bypass LLM direct)
        print(f"\n{result['answer']}", flush=True)
        print(f"\n{'─'*60}")
        print(f"Sources   : {', '.join(result['sources']) or 'aucune'}")
        print(f"Chunks    : {result['chunks_used']}")
        print(f"Durée     : {result['elapsed_seconds']}s")
        print(f"{'─'*60}")

    if args.question:
        ask(args.question)
        return

    # Mode interactif
    print("\nTapez votre question (ou 'quit' pour quitter):\n")
    while True:
        try:
            question = input(">>> ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q", ":q"):
                print("Au revoir!")
                break
            ask(question)
        except KeyboardInterrupt:
            print("\nAu revoir!")
            break
        except Exception as e:
            print(f"Erreur: {e}")


if __name__ == "__main__":
    main()
