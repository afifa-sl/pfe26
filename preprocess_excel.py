#!/usr/bin/env python3
"""
Script de prétraitement : Convertit les Excel RH en documents Markdown lisibles par le RAG
"""

from pathlib import Path
import pandas as pd

# Chemins
RAW_DIR = Path("documents/raw")
STRUCTURED_DIR = Path("documents/structured")
STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)

def clean_name(name: str) -> str:
    """Nettoie les noms pour créer des noms de fichiers corrects"""
    if not isinstance(name, str):
        name = str(name)
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip().replace(" ", "_")

def save_markdown(filename: str, content: str):
    """Sauvegarde le contenu dans un fichier .md"""
    path = STRUCTURED_DIR / filename
    path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"  ✓ Créé : {filename}")

# ====================== DIRECTION ======================
print("Conversion des Directions...")

df_dir = pd.read_excel(RAW_DIR / "DIRECTION.xlsx")
for _, row in df_dir.iterrows():
    content = f"""# Direction : {row.get('SHORT_LIBELLE_DIRECTION', '')}

**Code Affectation** : {row.get('AFFECTATION', '')}
**Chantier** : {row.get('CHANTIER', '')}
**Responsable** : {row.get('NOM', '')} {row.get('PRENOM', '')}
**Fonction** : {row.get('FONCTION', '')}
**Observation** : {row.get('OBSERVATION', '')}
"""
    filename = f"direction_{clean_name(row.get('SHORT_LIBELLE_DIRECTION', 'inconnue'))}.md"
    save_markdown(filename, content)

# ====================== DEPARTEMENT ======================
print("\nConversion des Départements...")

df_dept = pd.read_excel(RAW_DIR / "DEPARTEMENT.xlsx")
for _, row in df_dept.iterrows():
    content = f"""# Département : {row.get('CHANTIER', '')}

**Code** : {row.get('AFFECTATION', '')}
**Direction rattachée** : {row.get('SHORT_LIBELLE_DIRECTION', '')}
**Responsable** : {row.get('NOM', '')} {row.get('PRENOM', '')}
**Fonction** : {row.get('FONCTION', '')}
**Observation** : {row.get('OBSERVATION', '')}
"""
    filename = f"departement_{row.get('ID')}_{clean_name(row.get('CHANTIER', 'inconnu'))}.md"
    save_markdown(filename, content)

# ====================== SERVICE ======================
print("\nConversion des Services...")

df_serv = pd.read_excel(RAW_DIR / "SERVICE.xlsx")
for _, row in df_serv.iterrows():
    content = f"""# Service : {row.get('CHANTIER', '')}

**Code** : {row.get('AFFECTATION', '')}
**Direction / Département** : {row.get('SHORT_LIBELLE_DIRECTION', '')}
**Responsable** : {row.get('NOM', '')} {row.get('PRENOM', '')}
**Fonction** : {row.get('FONCTION', '')}
**Observation** : {row.get('OBSERVATION', '')}
"""
    filename = f"service_{row.get('ID')}_{clean_name(row.get('CHANTIER', 'inconnu'))}.md"
    save_markdown(filename, content)

# ====================== POSTE ======================
print("\nConversion des Postes (cela peut prendre un peu de temps)...")

df_poste = pd.read_excel(RAW_DIR / "POSTE.xlsx")
for _, row in df_poste.iterrows():
    content = f"""# Poste : {row.get('LIBELLE_POSTE', '')}

**ID Poste** : {row.get('ID', '')}
**Libellé de base** : {row.get('LIBELLE_POSTE_BASE', '')}
**Filière** : {row.get('LIBELLE_FILIERE', '')}
**Sous-filière** : {row.get('LIBELLE_SOUS_FILIERE', '')}
**Activité** : {row.get('LIBELLE_ACTIVITE', '')}
**Catégorie** : {row.get('CATEGORIE', '')}
"""
    filename = f"poste_{row.get('ID')}_{clean_name(row.get('LIBELLE_POSTE', 'inconnu'))[:60]}.md"
    save_markdown(filename, content)

print(f"\n✅ Conversion terminée !")
print(f"   → {len(list(STRUCTURED_DIR.glob('*.md')))} fichiers Markdown créés dans documents/structured/")
print("\nMaintenant lance l'ingestion :")
print("   python ingest.py --reset")
