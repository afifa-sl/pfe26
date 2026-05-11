#!/usr/bin/env python3
"""Outil de diagnostic du pipeline RAG.

Vérifie que :
  1. Les fichiers Excel sont correctement chargés dans DuckDB (tables, colonnes,
     échantillons de lignes).
  2. L'IntentRouter classifie correctement quelques questions types.
  3. Le StructuredQueryEngine retourne bien les bons résultats SQL.

Usage :
    python diagnose.py                    # diagnostic complet
    python diagnose.py --tables-only      # juste lister les tables DuckDB
    python diagnose.py --question "Quels sont les services ?"   # test ciblé
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("diagnose")


SAMPLE_QUESTIONS = [
    "Donne moi la liste des Services",
    "Quels sont les services existants dans le domaine de l'électricité",
    "Donne moi la liste des Chef de Département",
    "Liste des directeurs",
    "Toutes les formations obligatoires",
    "Qui est le chef du département RH ?",
]


def section(title: str):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def check_duckdb(structured):
    """Affiche pour chaque table : nom, colonnes, 5 lignes d'exemple."""
    section("1. CONTENU DE DUCKDB (lecture des Excel)")
    if not structured.tables:
        print("  ⚠ AUCUNE table chargée.")
        print("  → Vérifiez que le dossier 'documents/' contient des .xlsx")
        print("  → Vérifiez les logs au démarrage : 'StructuredQueryEngine: N table(s)...'")
        return False

    for name, info in structured.tables.items():
        print(f"\n  Table : {name}")
        print(f"    Fichier source  : {info['filename']}")
        print(f"    Lignes          : {info['row_count']}")
        print(f"    Colonnes ({len(info['columns'])}) : {', '.join(info['columns'])}")

        # 5 lignes d'exemple
        sql_table = info["sql_table"]
        rows = structured.conn.execute(
            f'SELECT * FROM "{sql_table}" LIMIT 5'
        ).fetchall()
        col_names = [d[0] for d in structured.conn.description]

        if not rows:
            print("    ⚠ Table vide !")
            continue

        print("    Échantillon (5 premières lignes) :")
        for i, row in enumerate(rows, 1):
            preview = []
            for col, val in zip(col_names, row):
                if val is None:
                    continue
                v = str(val)[:30]
                preview.append(f"{col}={v}")
                if len(preview) >= 4:  # limite l'affichage
                    break
            print(f"      {i}. {' | '.join(preview)}")

        # Compter les valeurs distinctes pour les 3 premières colonnes
        print("    Valeurs distinctes (top 3 colonnes) :")
        for col in info["columns"][:3]:
            try:
                cnt = structured.conn.execute(
                    f'SELECT COUNT(DISTINCT "{col}") FROM "{sql_table}" '
                    f'WHERE "{col}" IS NOT NULL'
                ).fetchone()[0]
                print(f"      • {col} : {cnt} valeurs uniques")
            except Exception as e:
                print(f"      • {col} : erreur ({e})")
    return True


def check_intent_router(pipeline, questions):
    """Pour chaque question, affiche la classification de l'IntentRouter."""
    section("2. CLASSIFICATION INTENT (IntentRouter)")
    print(f"\n  Schéma exposé au LLM ({len(pipeline.schema)} sources) :")
    for name, info in pipeline.schema.items():
        kind = "DOC" if info.get("is_doc") else "TABLE"
        cols = info.get("columns", [])
        print(f"    [{kind}] {name} : {len(cols)} colonnes")

    print()
    for q in questions:
        print(f"  Q : {q!r}")
        try:
            intent = pipeline.intent_router.classify(q)
            print(f"    → intent={intent['intent']}  source={intent['source']}  "
                  f"column={intent['column']}  filter={intent.get('filter')}  "
                  f"exhaustive={intent['exhaustive']}  conf={intent['confidence']}")
            # diagnostic : pourquoi le bypass ne s'est pas déclenché ?
            issues = []
            if not intent["exhaustive"]:
                issues.append("intent ≠ list (bypass désactivé)")
            if not intent["source"]:
                issues.append("source = null (LLM n'a pas trouvé de table pertinente)")
            if intent["source"] and intent["source"] not in pipeline.schema:
                issues.append(f"source '{intent['source']}' INCONNUE dans le schéma")
            if intent["source"] and pipeline.schema.get(intent["source"], {}).get("is_doc"):
                issues.append(f"source '{intent['source']}' est un DOC, pas une table")
            if issues:
                print(f"      ⚠ Bypass DuckDB désactivé : {' ; '.join(issues)}")
            else:
                print("      ✓ Bypass DuckDB devrait se déclencher")
        except Exception as e:
            print(f"    ✗ ERREUR classify() : {e}")
        print()


def check_structured_query(pipeline, questions):
    """Pour chaque question dont l'intent est exhaustif + table, exécute le SQL."""
    section("3a. EXÉCUTION SQL DIRECTE — listes exhaustives (StructuredQueryEngine)")
    for q in questions:
        intent = pipeline.intent_router.classify(q)
        if not (intent["exhaustive"] and intent["source"]
                and intent["source"] in pipeline.schema
                and not pipeline.schema[intent["source"]].get("is_doc")):
            continue
        print(f"\n  Q : {q!r}")
        print(f"    Intent : source={intent['source']} column={intent['column']} "
              f"filter={intent.get('filter')}")
        try:
            results = pipeline.structured.list_values(
                table=intent["source"],
                column=intent["column"],
                filters=intent.get("filter") or {},
                distinct=True,
            )
            print(f"    → {len(results)} résultat(s)")
            for w in pipeline.structured.last_warnings:
                print(f"    ⚠ {w}")
            for r in results[:5]:
                print(f"      • {r['content'][:120]}")
            if len(results) > 5:
                print(f"      ... ({len(results) - 5} autres)")
            if not results:
                print("    ⚠ AUCUN résultat ! Le bypass va retomber sur le RAG.")
        except Exception as e:
            print(f"    ✗ ERREUR list_values() : {e}")

    section("3b. EXÉCUTION SQL — QA structuré (intent=qa/detail + filtre)")
    for q in questions:
        intent = pipeline.intent_router.classify(q)
        if not (not intent["exhaustive"]
                and intent["source"]
                and intent.get("filter")
                and intent["source"] in pipeline.schema
                and not pipeline.schema[intent["source"]].get("is_doc")):
            continue
        print(f"\n  Q : {q!r}")
        print(f"    Intent : intent={intent['intent']} source={intent['source']} "
              f"filter={intent.get('filter')}")
        try:
            results = pipeline.structured.list_values(
                table=intent["source"],
                column=None,          # toutes les colonnes pour QA
                filters=intent.get("filter") or {},
                distinct=False,
            )
            print(f"    → {len(results)} ligne(s) SQL retournée(s)")
            for w in pipeline.structured.last_warnings:
                print(f"    ⚠ {w}")
            for r in results[:3]:
                print(f"      • {r['content'][:150]}")
            if len(results) > 3:
                print(f"      ... ({len(results) - 3} autres)")
            if results:
                print("      ✓ Structured QA va s'activer (SQL → LLM focalisé)")
            else:
                print("      ⚠ 0 résultat SQL → fallback RAG classique")
        except Exception as e:
            print(f"    ✗ ERREUR list_values() : {e}")


def main():
    parser = argparse.ArgumentParser(description="Diagnostic du pipeline RAG")
    parser.add_argument("--tables-only", action="store_true",
                        help="Affiche uniquement les tables DuckDB (rapide)")
    parser.add_argument("--question", type=str, default=None,
                        help="Teste une question spécifique (sinon utilise un set par défaut)")
    args = parser.parse_args()

    print("Initialisation du pipeline (peut prendre 30-60s la 1ère fois)...")
    from config import config
    from src.pipeline import RAGPipeline
    pipeline = RAGPipeline(config)

    ok = check_duckdb(pipeline.structured)
    if not ok:
        sys.exit(1)
    if args.tables_only:
        return

    questions = [args.question] if args.question else SAMPLE_QUESTIONS
    check_intent_router(pipeline, questions)
    check_structured_query(pipeline, questions)

    section("RÉSUMÉ")
    print()
    print("  ── Bypass liste (exhaustif) ─────────────────────────────────────")
    print("  Toute question en section 3a avec résultats > 0 répond en < 1s.")
    print("  Si lente / hallucinée : vérifier l'intent en section 2 (✓ ou ⚠).")
    print()
    print("  ── Structured QA (ciblé) ────────────────────────────────────────")
    print("  Section 3b : questions du type 'Qui est...' / 'Quel est...' sur")
    print("  une table DuckDB avec un filtre. Le SQL retourne 1-N lignes qui")
    print("  sont passées directement au LLM comme contexte focalisé → rapide")
    print("  et précis (pas de retrieval vectoriel).")
    print()
    print("  ── Fallback RAG ─────────────────────────────────────────────────")
    print("  Questions sans source DuckDB claire → pipeline vectoriel classique.")
    print()


if __name__ == "__main__":
    main()
