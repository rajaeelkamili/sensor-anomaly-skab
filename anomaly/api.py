"""API FastAPI : détection d'anomalies sur une série de capteurs envoyée en JSON.

Lancer :  uvicorn anomaly.api:app --port 8000   puis ouvrir http://127.0.0.1:8000/docs (Swagger)
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import SENSORS
from .detectors import BASELINE, DETECTORS, run_detector

app = FastAPI(title="Détection d'anomalies capteurs (SKAB)", version="1.0.0",
              description="Entraîne un détecteur non supervisé sur le début de la série (référence normale), "
                          "puis note toute la série. Prototype, non destiné à la production.")


class DetectRequest(BaseModel):
    model: str = Field("autoencoder", description=f"Un de : {', '.join(DETECTORS)}")
    rows: List[Dict[str, float]] = Field(..., description=f"Lignes avec les capteurs : {SENSORS}")
    baseline: int = Field(BASELINE, ge=50, description="Nombre de premières lignes supposées normales")
    quantile: float = Field(0.995, gt=0.5, lt=1.0)
    epochs: int = Field(30, ge=1, le=200, description="Époques (autoencodeur seulement)")


class DetectResponse(BaseModel):
    model: str
    threshold: float
    n_alarms: int
    scores: List[float]
    alarms: List[bool]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    return {"models": list(DETECTORS), "sensors": SENSORS}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    if req.model not in DETECTORS:
        raise HTTPException(422, f"Modèle inconnu : {req.model}. Choix : {list(DETECTORS)}")
    if len(req.rows) <= req.baseline:
        raise HTTPException(422, f"Il faut plus de {req.baseline} lignes (reçu {len(req.rows)}).")
    df = pd.DataFrame(req.rows)
    missing = [c for c in SENSORS if c not in df.columns]
    if missing:
        raise HTTPException(422, f"Capteurs manquants : {missing}")
    scores, thr = run_detector(req.model, df[SENSORS], req.baseline, req.quantile, req.epochs)
    alarms = scores > thr
    return DetectResponse(model=req.model, threshold=float(thr), n_alarms=int(alarms.sum()),
                          scores=[float(x) for x in scores], alarms=[bool(a) for a in alarms])
