"""Trois détecteurs non supervisés, entraînés uniquement sur du fonctionnement normal.

Protocole : le fichier `anomaly-free` est coupé dans le temps ; les 80 % premiers servent à
entraîner, les 20 % derniers à calibrer le seuil (quantile des scores normaux). Aucune étiquette
d'anomalie n'est utilisée pour entraîner ou régler le seuil.
"""
import joblib
import numpy as np
import pandas as pd
import torch
from pyod.models.ecod import ECOD
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch import nn

from . import SEED
from .data import SENSORS
from .features import rolling_features, sliding_windows

SMOOTH = 5  # lissage des scores (moyenne glissante, en secondes)


class Detector:
    name = "base"

    def __init__(self, quantile: float = 0.995):
        self.quantile = quantile
        self.threshold = None

    # --- à définir par chaque modèle
    def _fit(self, df: pd.DataFrame): ...
    def _raw_score(self, df: pd.DataFrame) -> np.ndarray: ...

    # --- commun
    def fit(self, normal: pd.DataFrame):
        cut = int(len(normal) * 0.8)
        train, calib = normal.iloc[:cut], normal.iloc[cut:]
        self._fit(train)
        self.set_quantile(self.quantile, calib)
        return self

    def set_quantile(self, quantile: float, calib: pd.DataFrame = None):
        if calib is not None:
            self._calib_scores = self.score(calib)
        self.quantile = quantile
        self.threshold = float(np.quantile(self._calib_scores, quantile))

    def score(self, df: pd.DataFrame) -> np.ndarray:
        s = pd.Series(self._raw_score(df))
        return s.rolling(SMOOTH, min_periods=1).mean().values

    def predict(self, df: pd.DataFrame, threshold: float = None) -> np.ndarray:
        return self.score(df) > (self.threshold if threshold is None else threshold)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "Detector":
        return joblib.load(path)


class IsolationForestDetector(Detector):
    name = "isolation_forest"

    def _fit(self, df):
        self.scaler = StandardScaler().fit(rolling_features(df))
        self.model = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1)
        self.model.fit(self.scaler.transform(rolling_features(df)))

    def _raw_score(self, df):
        return -self.model.score_samples(self.scaler.transform(rolling_features(df)))


class ECODDetector(Detector):
    """ECOD (PyOD) : détection par fonctions de répartition empiriques, sans hyperparamètre."""
    name = "ecod"

    def _fit(self, df):
        self.scaler = StandardScaler().fit(rolling_features(df))
        self.model = ECOD()
        self.model.fit(self.scaler.transform(rolling_features(df)))

    def _raw_score(self, df):
        return self.model.decision_function(self.scaler.transform(rolling_features(df)))


class AutoencoderDetector(Detector):
    """Autoencodeur PyTorch sur fenêtres glissantes ; score = erreur de reconstruction."""
    name = "autoencoder"

    def __init__(self, quantile=0.995, window=10, epochs=60, hidden=64, latent=16):
        super().__init__(quantile)
        self.window, self.epochs, self.hidden, self.latent = window, epochs, hidden, latent
        self.net = None

    def _build(self, dim):
        return nn.Sequential(
            nn.Linear(dim, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.latent), nn.ReLU(),
            nn.Linear(self.latent, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, dim),
        )

    def _windows(self, df):
        return sliding_windows(self.scaler.transform(df[SENSORS].values), self.window)

    def _fit(self, df):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.scaler = StandardScaler().fit(df[SENSORS].values)
        x = torch.tensor(self._windows(df))
        self.net = self._build(x.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        gen = torch.Generator().manual_seed(SEED)
        self.history = []
        for _ in range(self.epochs):
            perm = torch.randperm(len(x), generator=gen)
            total = 0.0
            for i in range(0, len(x), 128):
                batch = x[perm[i:i + 128]]
                opt.zero_grad()
                loss = loss_fn(self.net(batch), batch)
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            self.history.append(total / len(x))

    def _raw_score(self, df):
        x = torch.tensor(self._windows(df))
        self.net.eval()
        with torch.no_grad():
            return ((self.net(x) - x) ** 2).mean(dim=1).numpy()

    # torch.nn.Module se sérialise mal avec joblib : on passe par des tableaux numpy
    def __getstate__(self):
        state = self.__dict__.copy()
        if self.net is not None:
            state["net"] = {k: v.numpy() for k, v in self.net.state_dict().items()}
            state["_dim"] = self.window * len(SENSORS)
        return state

    def __setstate__(self, state):
        weights, dim = state.get("net"), state.pop("_dim", None)
        self.__dict__.update(state)
        if weights is not None:
            self.net = self._build(dim)
            self.net.load_state_dict({k: torch.tensor(v) for k, v in weights.items()})


DETECTORS = {c.name: c for c in (IsolationForestDetector, ECODDetector, AutoencoderDetector)}


def detect_changepoints(df: pd.DataFrame, penalty: float = 80.0, min_size: int = 20) -> list:
    """Détection de ruptures (ruptures, Binseg) sur les capteurs standardisés."""
    import ruptures as rpt
    x = StandardScaler().fit_transform(df[SENSORS].values)
    bkps = rpt.Binseg(model="l2", min_size=min_size, jump=5).fit(x).predict(pen=penalty)
    return [b for b in bkps if b < len(x)]


BASELINE = 540  # secondes de référence "normale" au début de chaque essai (SKAB : première anomalie après ~560 s)


def run_detector(name: str, df: pd.DataFrame, baseline: int = BASELINE, quantile: float = 0.995, epochs: int = 60):
    """Protocole par essai : entraîne sur le début de `df` (supposé normal), puis note tout l'essai.

    Les `baseline` premiers points sont coupés en 80 % (entraînement) et 20 % (calibration du seuil).
    Retourne (scores, seuil).
    """
    cls = DETECTORS[name]
    det = cls(quantile=quantile, epochs=epochs) if name == "autoencoder" else cls(quantile=quantile)
    det.fit(df.iloc[:baseline])
    return det.score(df), det.threshold
