"""Pipeline de bout en bout : entraînement, évaluation, figures et rapport reproductible.

Usage :  python -m anomaly.evaluate [--root data/skab_raw/data] [--out results] [--quantile 0.995]
Produit : results/metrics.csv, results/per_file.csv, results/changepoints.csv,
          results/report.md, results/*.png, models/*.joblib
"""
import argparse
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import DEFAULT_ROOT, load_normal, load_valve
from .detectors import BASELINE, DETECTORS, detect_changepoints, run_detector
from .metrics import changepoint_metrics, event_metrics, point_metrics, ranking_metrics

ROOT = Path(__file__).resolve().parent.parent
EVAL_START = BASELINE  # on évalue uniquement les points hors référence


def evaluate(root=DEFAULT_ROOT, out=ROOT / "results", models_dir=None, quantile=0.995, epochs=60):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    normal = load_normal(root)
    files = load_valve("valve1", root)

    # Protocole A (principal) : référence au début de chaque essai
    # Protocole B (transfert) : un seul modèle entraîné sur le fichier `anomaly-free`
    transfer = {}
    for name, cls in DETECTORS.items():
        det = cls(quantile=quantile, epochs=epochs) if name == "autoencoder" else cls(quantile=quantile)
        transfer[name] = det.fit(normal)
        print(f"[ok] transfert : {name} entraîné sur anomaly-free")

    results = {"per_file_baseline": {}, "transfer": {}}
    scores_cache, thr_cache = {}, {}
    for name in DETECTORS:
        for fname, df in files.items():
            s, thr = run_detector(name, df, BASELINE, quantile, epochs)
            results["per_file_baseline"][(name, fname)] = (s, thr)
            scores_cache[(name, fname)], thr_cache[(name, fname)] = s, thr
            results["transfer"][(name, fname)] = (transfer[name].score(df), transfer[name].threshold)
        print(f"[ok] protocole par essai : {name}")

    all_rows, per_file = [], []
    for proto, res in results.items():
        for name in DETECTORS:
            y_all, s_all, p_all = [], [], []
            pf = []
            for fname, df in files.items():
                s, thr = res[(name, fname)]
                y = df["anomaly"].values.astype(int)[EVAL_START:]
                s = s[EVAL_START:]
                p = s > thr
                pf.append({"protocol": proto, "model": name, "file": fname,
                           **point_metrics(y, p), **event_metrics(y, p)})
                y_all.append(y); s_all.append(s); p_all.append(p)
            per_file += pf
            pfd = pd.DataFrame(pf)
            y_all, s_all, p_all = map(np.concatenate, (y_all, s_all, p_all))
            all_rows.append({
                "protocol": proto, "model": name, **point_metrics(y_all, p_all),
                **ranking_metrics(y_all, s_all),
                "events": int(pfd.events.sum()), "events_detected": int(pfd.events_detected.sum()),
                "event_recall": pfd.events_detected.sum() / pfd.events.sum(),
                "false_alarm_runs": int(pfd.false_alarm_runs.sum()),
            })
    metrics = pd.DataFrame(all_rows).drop(columns=["tp", "fp", "fn", "tn"]).round(3)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(per_file).round(3).to_csv(out / "per_file.csv", index=False)

    cp_rows = []
    for fname, df in files.items():
        t = np.where(df["changepoint"].values > 0)[0]
        cp_rows.append({"file": fname, **changepoint_metrics(t, detect_changepoints(df))})
    cp = pd.DataFrame(cp_rows)
    cp_summary = cp[["cp_precision", "cp_recall"]].mean().round(3)
    cp.round(3).to_csv(out / "changepoints.csv", index=False)

    main = metrics[metrics.protocol == "per_file_baseline"].reset_index(drop=True)
    _figures(out, main, files, scores_cache, thr_cache)
    _report(out, metrics, cp_summary, normal, files, quantile)
    return metrics, cp_summary


def _figures(out, metrics, files, scores_cache, thr_cache):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(metrics)); w = 0.2
    for i, col in enumerate(["precision", "recall", "f1", "event_recall"]):
        ax.bar(x + (i - 1.5) * w, metrics[col], w, label=col)
    ax.set_xticks(x); ax.set_xticklabels(metrics["model"]); ax.set_ylim(0, 1.05)
    ax.set_title("Comparaison des détecteurs (valve1, 16 fichiers)"); ax.legend(ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "comparison.png", dpi=140); plt.close(fig)

    fname = list(files)[1]
    df = files[fname]; y = df["anomaly"].values
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 2.2 * len(metrics)), sharex=True)
    for ax, name in zip(np.atleast_1d(axes), metrics["model"]):
        ax.plot(scores_cache[(name, fname)], lw=0.9)
        ax.axhline(thr_cache[(name, fname)], color="red", ls="--", lw=0.9, label="seuil")
        ax.fill_between(range(len(y)), 0, 1, where=y > 0, color="orange", alpha=0.25,
                        transform=ax.get_xaxis_transform(), label="anomalie réelle")
        ax.set_ylabel(name, fontsize=8)
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper left")
    np.atleast_1d(axes)[-1].set_xlabel("temps (s)")
    np.atleast_1d(axes)[0].set_title(f"Scores d'anomalie sur {fname}")
    fig.tight_layout(); fig.savefig(out / "scores_example.png", dpi=140); plt.close(fig)


def _report(out, metrics, cp_summary, normal, files, quantile):
    import sklearn, torch
    n_events = int(metrics.events.iloc[0])
    cols = ["model", "precision", "recall", "f1", "fpr", "auroc", "auprc", "event_recall", "false_alarm_runs"]

    def tab(df):
        df = df[cols]
        return df.to_markdown(index=False) if _has_tabulate() else df.to_string(index=False)

    main = metrics[metrics.protocol == "per_file_baseline"]
    tr = metrics[metrics.protocol == "transfer"]
    best = main.sort_values("f1", ascending=False).iloc[0]
    prev = float(np.concatenate([d["anomaly"].values[EVAL_START:] for d in files.values()]).mean())
    f1_all = 2 * prev / (1 + prev)
    text = f"""# Rapport d'évaluation : détection d'anomalies sur le banc SKAB

Rapport généré automatiquement par `python -m anomaly.evaluate` (graine 42).

## Données et protocole
- SKAB, `valve1` : {len(files)} essais, {n_events} événements anormaux. Chaque essai commence par ~560 s de fonctionnement normal.
- Apprentissage **non supervisé** : aucune étiquette d'anomalie n'est utilisée pour entraîner ni pour régler un seuil.
- **Protocole principal (référence par essai)** : les {BASELINE} premières secondes de chaque essai servent de référence normale (80 % entraînement, 20 % calibration du seuil au quantile {quantile}). L'évaluation porte sur les points **après** cette référence. C'est le fonctionnement d'une campagne de banc : on compare un essai à son propre début.
- **Protocole de transfert** : un seul modèle entraîné sur `anomaly-free` ({len(normal)} points) puis appliqué à tous les essais.
- Après la référence, {prev:.0%} des points sont anormaux. **Un détecteur qui déclenche l'alarme en permanence obtient donc une précision de {prev:.2f}, un rappel de 1 et un F1 de {f1_all:.2f}** : le F1 seul ne suffit pas, il faut lire `fpr` et `auroc` avec lui.
- Scores lissés sur 5 s. Entrées : valeurs brutes + moyenne et écart-type glissants sur 10 s (isolation forest, ECOD) ; fenêtres de 10 s (autoencodeur).

## Résultats : protocole principal (référence par essai)
{tab(main)}

Meilleur F1 par point : **{best['model']}** ({best['f1']:.3f}), avec un `fpr` de {best['fpr']:.2f} et {int(best['false_alarm_runs'])} séries de fausses alarmes sur {len(files)} essais. L'isolation forest et ECOD ne déclenchent aucune alarme avec ce seuil, et leur `auroc` ({main.set_index('model').loc['isolation_forest','auroc']:.2f} et {main.set_index('model').loc['ecod','auroc']:.2f}) montre qu'ils ordonnent mal les points. Hypothèse, non vérifiée ici : sur des variables en niveau, une dérive lente (la température moyenne diffère déjà d'un essai à l'autre) domine le signal de la vanne. L'autoencodeur, qui apprend les relations entre capteurs, ordonne nettement mieux (`auroc` {main.set_index('model').loc['autoencoder','auroc']:.2f}), mais son F1 est proche de celui de l'alarme permanente : son vrai gain se lit dans `fpr` et `auroc`, et il produit encore beaucoup de fausses alarmes. Piste suivante : des variables stationnarisées (différences, résidus par rapport à une moyenne mobile longue).

## Résultats : transfert depuis `anomaly-free`
{tab(tr)}

Ce protocole échoue (le F1 de {tr.set_index('model').loc['isolation_forest','f1']:.2f} correspond à une alarme permanente, `fpr` = {tr.set_index('model').loc['isolation_forest','fpr']:.2f}) : le fichier `anomaly-free` a une température moyenne d'environ 89 °C, contre environ 70 °C dans les essais `valve1`. Les modèles considèrent presque tout comme anormal (ou, pour ECOD, presque rien). Un modèle entraîné sur un autre régime de fonctionnement ne se transfère pas sans normalisation par essai.

## Lecture des colonnes
- `precision`, `recall`, `f1`, `fpr` : par point, seuil fixé sur le normal uniquement.
- `auroc`, `auprc` : indépendantes du seuil.
- `event_recall` : part des événements avec au moins une alarme ; `false_alarm_runs` : séries d'alarmes hors de tout événement.

## Ruptures (ruptures, Binseg)
Précision {cp_summary['cp_precision']}, rappel {cp_summary['cp_recall']} (tolérance 15 s, moyenne sur les essais). La pénalité (80) a été choisie après avoir regardé une petite grille sur ces mêmes fichiers : résultat légèrement optimiste.

## Limites
- Banc de circulation d'eau, pas un banc moteur : les résultats ne se transposent pas tels quels.
- Le seuil par quantile règle le compromis faux positifs / rappel ; il n'est pas optimisé.
- Évaluation par point sévère pour des anomalies progressives ; l'évaluation par événement est plus indulgente.
- Les anomalies de `valve1` sont injectées par un opérateur (fermeture de vanne), pas des pannes naturelles.
- `valve2` n'est pas utilisé ici.

![comparaison](comparison.png)
![scores](scores_example.png)

Environnement : Python {platform.python_version()}, scikit-learn {sklearn.__version__}, torch {torch.__version__.split('+')[0]}.
"""
    (out / "report.md").write_text(text, encoding="utf-8")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--quantile", type=float, default=0.995)
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    m, cp = evaluate(a.root, a.out, quantile=a.quantile, epochs=a.epochs)
    print(m.to_string(index=False)); print(cp.to_string())
