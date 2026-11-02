"""Télécharge le jeu de données public SKAB (Skoltech Anomaly Benchmark, licence GPL-3.0).

Les données ne sont pas incluses dans ce dépôt. Usage : python scripts/get_data.py
"""
import subprocess
import sys
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "data" / "skab_raw"
URL = "https://github.com/waico/SKAB"

if (DEST / "data" / "valve1").exists():
    print(f"SKAB déjà présent dans {DEST}")
    sys.exit(0)
DEST.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(["git", "clone", "--depth", "1", URL, str(DEST)], check=True)
print(f"SKAB téléchargé dans {DEST}")
