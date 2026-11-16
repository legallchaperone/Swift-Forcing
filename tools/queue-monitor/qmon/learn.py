"""Train the learned start-time model and export it as JSON for the app.

usage:  <venv>/bin/python -m qmon.learn <features.pkl> <out.json>
Needs numpy, pandas and scikit-learn (the app only reads the JSON, with the standard library).

The model ("mix3") averages three predictions in log(hours + 1) space, then widens the
10-90% range (lower side x1.3, upper side x2). The three parts:
  calws  rule simulation x Kaplan-Meier ratio quantiles per cell
         (predicted-wait bucket, time limit, size, time already waited, fair share)
  two    a boosted-tree score of log(real wait / rule wait); Kaplan-Meier ratio quantiles
         per (score bin, predicted-wait bucket)
  twod   the same, but on the wait itself instead of the ratio to the rule simulation
Chosen on a strict time-split backtest over 55 days of DeltaAI history (research/evaluate.py).
"""
import json, pickle, sys, time
import numpy as np, pandas as pd

from qmon import calib, features

QS = (0.1, 0.5, 0.9)
CALWS = [("b", "lc3", "sc", "wb", "fsb"), ("b", "lc3", "sc", "fsb"), ("b", "lc3", "sc", "wb"),
         ("b", "lc3", "sc"), ("b", "lc3"), ("b",)]
WIDEN = (1.3, 2.0)


def km_table(keys_df, levels, ratio, ev, min_events):
    out = []
    for lev in levels:
        for k, ix in keys_df.groupby(list(lev)).indices.items():
            if ev[ix].sum() < min_events:
                continue
            q = calib.km_quantiles(list(zip(ratio[ix], ev[ix])), QS)
            if q and None not in q:
                kk = k if isinstance(k, tuple) else (k,)
                out.append([list(lev), [int(x) for x in kk], [float(x) for x in q]])
    return out


def export_hgb(m):
    """Trees of a fitted HistGradientBoostingRegressor as plain lists."""
    trees = []
    for (pred,) in m._predictors:
        nd = pred.nodes
        trees.append([[int(n["feature_idx"]), float(n["num_threshold"]), int(n["missing_go_to_left"]),
                       int(n["left"]), int(n["right"]), int(n["is_leaf"]), float(n["value"])] for n in nd])
    return {"base": float(np.ravel(m._baseline_prediction)[0]), "trees": trees}


def two_stage(tr, dur, ev, raw, nbins=20, rounds=200, min_events=40):
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB

    def mk():
        return HGB(loss="absolute_error", learning_rate=0.05, max_iter=rounds, max_leaf_nodes=15,
                   min_samples_leaf=1000, l2_regularization=10.0, random_state=0)
    rule = np.ones(len(tr)) if raw else tr.rule_ty.values
    lr = np.log((dur + 0.5) / (rule + 0.5))
    fit_m = (ev == 1) | (dur > 24)
    X = tr[features.FEATS].values.astype(float)
    gid = pd.util.hash_array(tr.id.values.astype(str)) % 3
    oof = np.zeros(len(tr))
    for g in range(3):
        a = (gid != g) & fit_m
        oof[gid == g] = mk().fit(X[a], lr[a]).predict(X[gid == g])
    m = mk().fit(X[fit_m], lr[fit_m])
    edges = np.quantile(oof, np.linspace(0, 1, nbins + 1)[1:-1])
    keys = pd.DataFrame({"b": tr.b.values, "s": np.searchsorted(edges, oof)})
    levels = [("s",)] if raw else [("b", "s"), ("s",)]
    return {"raw": bool(raw), "edges": [float(x) for x in edges], "levels": [list(l) for l in levels],
            "table": km_table(keys, levels, dur / rule, ev, min_events), "gbm": export_hgb(m), "_model": m}


def train(path, per_point=400, seed=0):
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
    dur = np.where(started, (df.truth - df["T"]) / 3600.0, (F - df["T"]) / 3600.0)
    ev = started.values.astype(int)
    if "u_eta" not in df.columns:      # tables built before these were part of features.compute
        df["u_eta"] = (df.u_ahead_n + 1) / (df.u_st24 / 24.0 + 0.05)
        df["a_eta"] = (df.a_ahead + 1) / (df.a_st24 / 24.0 + 0.05)
    kd = pd.DataFrame([features.keys(r) for r in df[["rule_ty", "limit_h", "eq", "waited_h", "fs"]].to_dict("records")],
                      index=df.index)
    df = pd.concat([df, kd], axis=1)
    out = {"version": 1, "generated": time.time(), "trained_on": {"rows": int(len(df)), "points": int(df["T"].nunique()),
           "from": float(df["T"].min()), "to": float(df["T"].max()), "fetched": F},
           "feats": features.FEATS, "widen": list(WIDEN),
           "calws": {"levels": [list(l) for l in CALWS], "table": km_table(df, CALWS, dur / df.rule_ty.values, ev, 30)}}
    parts = {}
    for name, raw in (("two", False), ("twod", True)):
        parts[name] = two_stage(df, dur, ev, raw)
    # self-check: the exported trees must reproduce scikit-learn's predictions
    from qmon import mlmodel
    X = df[features.FEATS].values.astype(float)[:500]
    for name, p in parts.items():
        ref = p.pop("_model").predict(X)
        mine = np.array([mlmodel.gbm_predict(p["gbm"], list(x)) for x in X])
        assert np.allclose(ref, mine, atol=1e-6), (name, float(np.abs(ref - mine).max()))
        out[name] = p
    return out


if __name__ == "__main__":
    m = train(sys.argv[1])
    with open(sys.argv[2], "w") as f:
        json.dump(m, f)
    print("trained on %d rows, %d points" % (m["trained_on"]["rows"], m["trained_on"]["points"]))
