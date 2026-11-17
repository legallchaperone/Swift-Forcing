"""Train the learned start-time model and export it as JSON for the app.

usage:  <venv>/bin/python -m qmon.learn <features.pkl> <out.json>
Needs numpy, pandas and scikit-learn (the app only reads the JSON, with the standard library).

The model (version 2) is a discrete-time hazard model: boosted trees give the chance that a
job starts in each time bin (0-15 min, 15-30 min, ..., 10-14 days) if it has not started yet.
Jobs still waiting at the fetch time count as "not started" up to that time, so long waits are
not lost. The product of the bin chances is the predicted wait distribution. Each (point, user)
pair has total weight 1, so one user's large array does not dominate.

The app shows the median and the 24-hour window with the most probability (the 10-90% range
when that is already narrower). Chosen on a strict time-split backtest over 60 days of DeltaAI
history (research/eval2.py, model "haz+").
"""
import json, pickle, sys, time
import numpy as np, pandas as pd

from qmon import features

EDGES = [0, .25, .5, 1, 2, 3, 4.5, 6, 8, 10, 12, 15, 18, 21, 24, 30, 36, 42, 48, 60, 72, 96, 120, 168, 240, 336]
EXTRA = ["cut_q10", "cut_q50", "below_cut", "n_started6", "d_prio_q10", "d_prio_q50", "hour", "wday"]
FEATS = list(features.FEATS) + EXTRA


def export_hgb(m):
    """Trees of a fitted HistGradientBoosting model as plain lists."""
    trees = []
    for (pred,) in m._predictors:
        nd = pred.nodes
        trees.append([[int(n["feature_idx"]), float(n["num_threshold"]), int(n["missing_go_to_left"]),
                       int(n["left"]), int(n["right"]), int(n["is_leaf"]), float(n["value"])] for n in nd])
    return {"base": float(np.ravel(m._baseline_prediction)[0]), "trees": trees}


def expand(X, dur, ev, w):
    """One row per (job, time bin it was still waiting in); label 1 in the bin where it started."""
    E = np.array(EDGES)
    NB = len(E) - 1
    b = np.clip(np.searchsorted(E, dur, side="right") - 1, 0, NB - 1)
    nb = np.where(ev == 1, b + 1, b)
    rep = np.repeat(np.arange(len(dur)), nb)
    kk = np.concatenate([np.arange(n) for n in nb])
    y = np.zeros(len(rep), int)
    last = np.cumsum(nb) - 1
    y[last[(ev == 1) & (nb > 0)]] = 1
    return np.column_stack([X[rep], kk, E[kk]]), y, w[rep]


def train(path, per_point=400, seed=0, rounds=300):
    from sklearn.ensemble import HistGradientBoostingClassifier as C
    with open(path, "rb") as f:
        d = pickle.load(f)
    F = d["fetched"]
    df = pd.DataFrame(d["rows"])
    rng = np.random.default_rng(seed)
    keep = []
    for T, g in df.groupby("T"):
        idx = g.index.values
        keep.extend(rng.choice(idx, per_point, replace=False) if len(idx) > per_point else idx)
    df = df.loc[sorted(keep)]
    started = df.truth.notna() & (df.truth <= F)
    canc = ~started & df.end.notna() & (df.end <= F)
    df = df[~canc & (df.rule_ty > 0.25)]
    started = started[df.index]
    if "u_eta" not in df.columns:      # tables built before these were part of features.compute
        df["u_eta"] = (df.u_ahead_n + 1) / (df.u_st24 / 24.0 + 0.05)
        df["a_eta"] = (df.a_ahead + 1) / (df.a_st24 / 24.0 + 0.05)
    for c in FEATS:
        if c not in df.columns:
            df[c] = np.nan
    dur = np.where(started, (df.truth - df["T"]) / 3600.0, (F - df["T"]) / 3600.0)
    ev = started.values.astype(int)
    w = 1.0 / df.groupby(["T", "user"])["id"].transform("count").values
    X = df[FEATS].values.astype(float)
    Xe, y, we = expand(X, dur, ev, w)
    m = C(learning_rate=0.05, max_iter=rounds, max_leaf_nodes=31, min_samples_leaf=500, l2_regularization=10.0,
          random_state=0).fit(Xe, y, sample_weight=we)
    out = {"version": 2, "generated": time.time(),
           "trained_on": {"rows": int(len(df)), "points": int(df["T"].nunique()), "from": float(df["T"].min()),
                          "to": float(df["T"].max()), "fetched": F},
           "feats": FEATS, "haz": {"edges": EDGES, "gbm": export_hgb(m)}, "cap_h": 24.0}
    # self-check: the exported trees must reproduce scikit-learn's probabilities
    from qmon import mlmodel
    ref = m.predict_proba(Xe[:2000:4])[:, 1]
    mine = np.array([mlmodel.sigmoid(mlmodel.gbm_predict(out["haz"]["gbm"], list(x))) for x in Xe[:2000:4]])
    assert np.allclose(ref, mine, atol=1e-6), float(np.abs(ref - mine).max())
    return out


if __name__ == "__main__":
    m = train(sys.argv[1])
    with open(sys.argv[2], "w") as f:
        json.dump(m, f)
    print("trained on %d rows, %d points" % (m["trained_on"]["rows"], m["trained_on"]["points"]))
