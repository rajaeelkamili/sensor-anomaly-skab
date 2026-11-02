"""Construction des entrées des modèles."""
import numpy as np
import pandas as pd

from .data import SENSORS


def rolling_features(df: pd.DataFrame, window: int = 10) -> np.ndarray:
    """Valeurs brutes + moyenne et écart-type glissants (par capteur)."""
    x = df[SENSORS]
    mean = x.rolling(window, min_periods=1).mean()
    std = x.rolling(window, min_periods=1).std().fillna(0.0)
    return np.hstack([x.values, mean.values, std.values]).astype(np.float32)


def sliding_windows(x: np.ndarray, window: int = 10) -> np.ndarray:
    """Fenêtres glissantes aplaties, une par pas de temps (début complété par la première ligne)."""
    pad = np.repeat(x[:1], window - 1, axis=0)
    xp = np.vstack([pad, x])
    idx = np.arange(len(x))[:, None] + np.arange(window)[None, :]
    return xp[idx].reshape(len(x), -1).astype(np.float32)
