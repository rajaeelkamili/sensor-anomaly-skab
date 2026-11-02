"""Métriques : au niveau du point, au niveau de l'événement, et pour les ruptures."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _runs(mask: np.ndarray):
    """Liste des intervalles [debut, fin] (inclus) où mask est vrai."""
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return []
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def point_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    tp = int((y_true & y_pred).sum()); fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum()); tn = int((~y_true & ~y_pred).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def event_metrics(y_true, y_pred) -> dict:
    """Un événement vrai est détecté si au moins une alarme tombe dedans.
    Fausse alerte = série d'alarmes qui ne touche aucun événement vrai."""
    true_runs, pred_runs = _runs(y_true), _runs(y_pred)
    detected = sum(1 for a, b in true_runs if np.asarray(y_pred)[a:b + 1].any())
    false_alarms = sum(1 for a, b in pred_runs if not np.asarray(y_true)[a:b + 1].any())
    return {
        "events": len(true_runs), "events_detected": detected,
        "event_recall": detected / len(true_runs) if true_runs else 0.0,
        "false_alarm_runs": false_alarms,
    }


def ranking_metrics(y_true, score) -> dict:
    y_true = np.asarray(y_true).astype(int)
    if y_true.min() == y_true.max():
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {"auroc": roc_auc_score(y_true, score), "auprc": average_precision_score(y_true, score)}


def changepoint_metrics(true_idx, pred_idx, margin: int = 15) -> dict:
    """Précision et rappel des ruptures avec une tolérance de `margin` pas de temps."""
    true_idx, pred_idx = list(true_idx), list(pred_idx)
    hit_true = sum(any(abs(t - p) <= margin for p in pred_idx) for t in true_idx)
    hit_pred = sum(any(abs(t - p) <= margin for t in true_idx) for p in pred_idx)
    return {
        "cp_precision": hit_pred / len(pred_idx) if pred_idx else 0.0,
        "cp_recall": hit_true / len(true_idx) if true_idx else 0.0,
    }
