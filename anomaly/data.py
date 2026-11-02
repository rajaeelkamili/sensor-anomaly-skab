"""Chargement du jeu SKAB.

SKAB : 8 capteurs (vibrations, courant, pression, température, tension, débit) sur un banc d'essai
de circulation d'eau, 1 mesure par seconde. Les fichiers valve1 et valve2 contiennent des anomalies
injectées (colonne `anomaly`) ; `anomaly-free.csv` contient uniquement du fonctionnement normal.
"""
from pathlib import Path

import pandas as pd

SENSORS = [
    "Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
    "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS",
]
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "skab_raw" / "data"


def load_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", index_col="datetime", parse_dates=True).sort_index()
    for col in ("anomaly", "changepoint"):
        if col not in df.columns:
            df[col] = 0.0
    return df


def load_normal(root=DEFAULT_ROOT) -> pd.DataFrame:
    return load_csv(Path(root) / "anomaly-free" / "anomaly-free.csv")


def load_valve(valve: str = "valve1", root=DEFAULT_ROOT) -> dict:
    """Retourne {nom_de_fichier: DataFrame} pour valve1 ou valve2."""
    folder = Path(root) / valve
    files = sorted(folder.glob("*.csv"), key=lambda p: int(p.stem))
    return {f"{valve}/{p.stem}": load_csv(p) for p in files}
