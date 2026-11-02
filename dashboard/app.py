"""Tableau de bord Streamlit.  Lancer :  streamlit run dashboard/app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from anomaly.data import DEFAULT_ROOT, SENSORS, load_csv, load_valve
from anomaly.detectors import BASELINE, DETECTORS, detect_changepoints, run_detector
from anomaly.metrics import event_metrics, point_metrics

st.set_page_config(page_title="Anomalies capteurs (SKAB)", layout="wide")
st.title("Détection d'anomalies sur données de banc d'essai")
st.caption("Prototype : jeu public SKAB, apprentissage non supervisé sur le début de chaque essai.")


@st.cache_data
def get_files():
    return load_valve("valve1", DEFAULT_ROOT)


@st.cache_data(show_spinner="Entraînement du détecteur...")
def run(name, df, baseline, quantile, epochs):
    return run_detector(name, df, baseline, quantile, epochs)


if not (DEFAULT_ROOT / "valve1").exists():
    st.error("Données absentes. Lance d'abord : `python scripts/get_data.py`")
    st.stop()

with st.sidebar:
    st.header("Paramètres")
    source = st.radio("Source", ["Essai SKAB", "Fichier CSV"])
    model = st.selectbox("Modèle", list(DETECTORS), index=2)
    quantile = st.slider("Quantile du seuil (sur la référence normale)", 0.90, 0.9999, 0.995, format="%.4f")
    baseline = st.number_input("Durée de référence normale (s)", 100, 1000, BASELINE, step=10)
    epochs = st.slider("Époques (autoencodeur)", 10, 120, 60)

if source == "Essai SKAB":
    files = get_files()
    name = st.selectbox("Essai", list(files))
    df = files[name]
else:
    up = st.file_uploader("CSV séparé par ';' avec colonne datetime et les 8 capteurs", type="csv")
    if up is None:
        st.info("Charge un fichier pour continuer.")
        st.stop()
    df = load_csv(up)

scores, thr = run(model, df, int(baseline), float(quantile), int(epochs))
alarms = scores > thr
has_truth = df["anomaly"].sum() > 0

c1, c2, c3 = st.columns(3)
c1.metric("Alarmes", int(alarms.sum()))
c2.metric("Seuil", f"{thr:.4g}")
if has_truth:
    y = df["anomaly"].values.astype(int)[int(baseline):]
    pm = point_metrics(y, alarms[int(baseline):])
    em = event_metrics(y, alarms[int(baseline):])
    c3.metric("F1 / FPR (hors référence)", f"{pm['f1']:.2f} / {pm['fpr']:.2f}")
    st.caption(f"Précision {pm['precision']:.2f}, rappel {pm['recall']:.2f}, événements détectés {em['events_detected']}/{em['events']}, "
               f"séries de fausses alarmes {em['false_alarm_runs']}.")

t = np.arange(len(df))
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06,
                    subplot_titles=("Capteurs (standardisés)", "Score d'anomalie"))
std = (df[SENSORS] - df[SENSORS].iloc[:int(baseline)].mean()) / (df[SENSORS].iloc[:int(baseline)].std() + 1e-9)
for c in SENSORS:
    fig.add_trace(go.Scatter(x=t, y=std[c], name=c, mode="lines", line=dict(width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=scores, name="score", line=dict(color="black", width=1)), row=2, col=1)
fig.add_hline(y=thr, line_dash="dash", line_color="red", row=2, col=1)
if has_truth:
    for a, b in __import__("anomaly.metrics", fromlist=["_runs"])._runs(df["anomaly"].values):
        fig.add_vrect(x0=a, x1=b, fillcolor="orange", opacity=0.2, line_width=0)
fig.add_vline(x=int(baseline), line_dash="dot", line_color="gray", annotation_text="fin de la référence")
fig.update_layout(height=650, legend=dict(orientation="h"), margin=dict(t=40, b=10))
st.plotly_chart(fig, use_container_width=True)

with st.expander("Ruptures (ruptures, Binseg)"):
    cps = detect_changepoints(df)
    st.write(f"Ruptures détectées aux instants (s) : {cps}")
