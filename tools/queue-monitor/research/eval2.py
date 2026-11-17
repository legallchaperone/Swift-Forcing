"""Start-time models judged with a narrow range: at most 24 hours wide (median +-12 h, or the
24-hour window that holds the most probability).

Each model gives a full predicted distribution of the wait as quantiles on a grid
(5%, 10%, ..., 95%). From it we report:
  * point error of the median: median and mean absolute error (hours)
  * the range: share of real starts inside it, and its mean width (hours)
Fold k tests on points T in [D, D + fold_days), D = last point - 7 days * k; models train only on
points before D, with outcomes as known at D (jobs still waiting at D are censored).

usage: python research/eval2.py <table.pkl> [--folds 2,3,4,5,6,7] [--models ...] [--hgb]
"""
import argparse, os, pickle, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmon import calib, features  # noqa

GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)          # 19 levels
I10, I50, I90 = 1, 9, 17
CAP = 24.0
FEATS = list(features.FEATS)
EXTRA = ["cut_q10", "cut_q50", "below_cut", "n_started6", "d_prio_q10", "d_prio_q50", "hour", "wday"]


# ------------------------------------------------------------------ data
def load(path, per_point=400, seed=0):
    with open(path, "rb") as f:
        d = pickle.load(f)
    df = pd.DataFrame(d["rows"])
    rng = np.random.default_rng(seed)
    df["sel"] = False
    for T, g in df.groupby("T"):
        idx = g.index.values
        if len(idx) > per_point:
            idx = rng.choice(idx, per_point, replace=False)
        df.loc[idx, "sel"] = True
    df["ucnt"] = df.groupby(["T", "user"])["id"].transform("count")
    if "u_eta" not in df:
        df["u_eta"] = (df.u_ahead_n + 1) / (df.u_st24 / 24.0 + 0.05)
        df["a_eta"] = (df.a_ahead + 1) / (df.a_st24 / 24.0 + 0.05)
    k = pd.DataFrame([features.keys(r) for r in df[["rule_ty", "limit_h", "eq", "waited_h", "fs"]].to_dict("records")],
                     index=df.index)
    for c in k:
        df[c] = k[c]
    return d["fetched"], df


def outcome(d, D):
    started = d.truth.notna() & (d.truth <= D)
    canc = ~started & d.end.notna() & (d.end <= D)
    dur = np.where(started, (d.truth - d["T"]) / 3600.0, (D - d["T"]) / 3600.0)
    return dur, started.values.astype(int), (~canc).values


# ------------------------------------------------------------------ distributions
def km_grid(obs, top):
    """KM quantiles on GRID; levels beyond the last event get `top` (no information)."""
    q = calib.km_quantiles(obs, tuple(GRID))
    top = 1.2 * max(v for v, _ in obs)          # beyond the data: just above the longest wait seen
    return np.array([top if v is None else v for v in q], float)


def km_cells(keys_df, levels, val, ev, min_events, top):
    table = {}
    for lev in levels:
        for k, ix in keys_df.groupby(list(lev)).indices.items():
            if ev[ix].sum() >= min_events:
                table[(lev, k if isinstance(k, tuple) else (k,))] = km_grid(list(zip(val[ix], ev[ix])), top)
    return table


def km_apply(te, table, levels, mult, fallback):
    G = fallback.copy()
    cols = {c: te[c].values for lev in levels for c in lev}
    for i in range(len(te)):
        for lev in levels:
            q = table.get((lev, tuple(cols[c][i] for c in lev)))
            if q is not None:
                G[i] = q * mult[i]
                break
    return G


def rule_grid(te):
    """The rule simulation's slow / typical / fast runs, spread over the grid."""
    fa = te.rule_fa.fillna(te.rule_ty).values
    ty = te.rule_ty.values
    sl = te.rule_sl.fillna(te.rule_ty).values
    G = np.empty((len(te), len(GRID)))
    for j, g in enumerate(GRID):
        G[:, j] = np.where(g <= 0.5, fa + (ty - fa) * (g - 0.1) / 0.4, ty + (sl - ty) * (g - 0.5) / 0.4)
    return np.maximum(G, 0)


CALWS = [("b", "lc3", "sc", "wb", "fsb"), ("b", "lc3", "sc", "fsb"), ("b", "lc3", "sc", "wb"),
         ("b", "lc3", "sc"), ("b", "lc3"), ("b",)]
CAL2 = [("b", "lc3", "sc"), ("b", "lc3"), ("b",)]


def gbm_reg(hgb, rounds=200):
    if hgb:
        from sklearn.ensemble import HistGradientBoostingRegressor as H
        return H(loss="absolute_error", learning_rate=0.05, max_iter=rounds, max_leaf_nodes=15,
                 min_samples_leaf=1000, l2_regularization=10.0, random_state=0)
    import lightgbm as lgb
    return lgb.LGBMRegressor(objective="l1", learning_rate=0.05, n_estimators=rounds, num_leaves=15,
                             min_child_samples=1000, reg_lambda=10.0, subsample=0.7, subsample_freq=1,
                             colsample_bytree=0.8, verbose=-1, n_jobs=4)


def two_stage(tr, dur, ev, te, feats, raw, hgb, nbins=20, min_events=40):
    rule = np.ones(len(tr)) if raw else tr.rule_ty.values
    lr = np.log((dur + 0.5) / (rule + 0.5))
    fit_m = (ev == 1) | (dur > 24)
    X = tr[feats].values.astype(float)
    gid = pd.util.hash_array(tr.id.values.astype(str)) % 3
    oof = np.zeros(len(tr))
    for g in range(3):
        a = (gid != g) & fit_m
        oof[gid == g] = gbm_reg(hgb).fit(X[a], lr[a]).predict(X[gid == g])
    m = gbm_reg(hgb).fit(X[fit_m], lr[fit_m])
    edges = np.quantile(oof, np.linspace(0, 1, nbins + 1)[1:-1])
    keys = pd.DataFrame({"b": tr.b.values, "s": np.searchsorted(edges, oof)})
    levels = [("s",)] if raw else [("b", "s"), ("s",)]
    top = 4 * np.max(dur / rule)
    table = km_cells(keys, levels, dur / rule, ev, min_events, top)
    te2 = te.assign(s=np.searchsorted(edges, m.predict(te[feats].values.astype(float))))
    mult = np.ones(len(te)) if raw else te.rule_ty.values
    return km_apply(te2, table, levels, mult, rule_grid(te))


EDGES = np.array([0, .25, .5, 1, 2, 3, 4.5, 6, 8, 10, 12, 15, 18, 21, 24, 30, 36, 42, 48, 60, 72, 96, 120, 168, 240, 336])


def hazard(tr, dur, ev, te, feats, hgb, rounds=300):
    """Discrete-time hazard: P(start in bin k | not started before), boosted trees; censoring-correct."""
    NB = len(EDGES) - 1
    b = np.clip(np.searchsorted(EDGES, dur, side="right") - 1, 0, NB - 1)
    nb = np.where(ev == 1, b + 1, b)
    rep = np.repeat(np.arange(len(dur)), nb)
    kk = np.concatenate([np.arange(n) for n in nb])
    y = np.zeros(len(rep), int)
    last = np.cumsum(nb) - 1
    y[last[(ev == 1) & (nb > 0)]] = 1
    X = tr[feats].values.astype(float)
    Xe = np.column_stack([X[rep], kk, EDGES[kk]])
    # each job counts once per point; big arrays would otherwise dominate
    w = 1.0 / tr.groupby(["T", "user"])["id"].transform("count").values[rep]
    if hgb:
        from sklearn.ensemble import HistGradientBoostingClassifier as C
        m = C(learning_rate=0.05, max_iter=rounds, max_leaf_nodes=31, min_samples_leaf=500, l2_regularization=10.0,
              random_state=0).fit(Xe, y, sample_weight=w)
    else:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(learning_rate=0.05, n_estimators=rounds, num_leaves=31, min_child_samples=500,
                               reg_lambda=10.0, subsample=0.7, subsample_freq=1, colsample_bytree=0.8,
                               verbose=-1, n_jobs=4).fit(Xe, y, sample_weight=w)
    Xt = te[feats].values.astype(float)
    n = len(Xt)
    Xq = np.column_stack([np.repeat(Xt, NB, axis=0), np.tile(np.arange(NB), n), np.tile(EDGES[:-1], n)])
    h = m.predict_proba(Xq)[:, 1].reshape(n, NB)
    S = np.cumprod(1 - h, axis=1)
    Sp = np.column_stack([np.ones(n), S[:, :-1]])
    G = np.zeros((n, len(GRID)))
    for j, g in enumerate(GRID):
        t = 1 - g
        k = (S > t).sum(axis=1)
        kc = np.minimum(k, NB - 1)
        s0, s1 = Sp[np.arange(n), kc], S[np.arange(n), kc]
        with np.errstate(divide="ignore", invalid="ignore"):
            fr = np.nan_to_num(np.clip(np.log(s0 / t) / np.log(s0 / np.maximum(s1, 1e-12)), 0, 1), nan=0.5)
        G[:, j] = np.where(k >= NB, EDGES[-1] * 1.5, EDGES[kc] + (EDGES[kc + 1] - EDGES[kc]) * fr)
    return G


def logmix(*Gs):
    return np.expm1(np.mean([np.log1p(np.maximum(G, 0)) for G in Gs], axis=0))


# ------------------------------------------------------------------ ranges and metrics
def cdf_at(Gi, x):
    """Predicted P(wait <= x) from one row of grid quantiles (linear between grid points)."""
    xs = np.concatenate([[0.0], Gi])
    ps = np.concatenate([[0.0], GRID])
    return np.interp(x, xs, ps, left=0.0, right=GRID[-1])


def ranges(G, mode):
    n = len(G)
    lo, hi = G[:, I10].copy(), G[:, I90].copy()
    if mode == "q10-q90":
        return lo, hi
    med = G[:, I50]
    if mode == "median+-12":
        return np.maximum(lo, med - CAP / 2), np.minimum(hi, med + CAP / 2)
    if mode == "best24":
        out_lo, out_hi = np.empty(n), np.empty(n)
        for i in range(n):
            Gi = np.maximum.accumulate(np.maximum(G[i], 0))
            if Gi[I90] - Gi[I10] <= CAP:
                out_lo[i], out_hi[i] = Gi[I10], Gi[I90]
                continue
            cands = np.unique(np.concatenate([[0.0], Gi, np.maximum(Gi - CAP, 0)]))
            mass = cdf_at(Gi, cands + CAP) - cdf_at(Gi, cands)
            a = cands[np.argmax(mass)]
            out_lo[i], out_hi[i] = a, a + CAP
        return out_lo, out_hi
    raise ValueError(mode)


def wq(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return x[o][min(np.searchsorted(c, q), len(x) - 1)]


def metrics(G, y, w, mode):
    lo, hi = ranges(G, mode)
    med = np.maximum(G[:, I50], 0)
    ae = np.abs(med - y)
    hit = ((lo <= y) & (y <= hi)).astype(float)
    return {"med_err": float(np.median(ae)), "mean_err": float(ae.mean()), "hit": float(hit.mean()),
            "width": float((hi - lo).mean()), "u_med_err": float(wq(ae, w, .5)),
            "u_hit": float((hit * w).sum() / w.sum()),
            "hit_by": {k: float(hit[m].mean()) for k, m in buckets(y).items() if m.any()},
            "err_by": {k: float(np.median(ae[m])) for k, m in buckets(y).items() if m.any()}}


def buckets(y):
    return {"<6h": y < 6, "6-24h": (y >= 6) & (y < 24), "1-3d": (y >= 24) & (y < 72), ">3d": y >= 72}


def fmt(name, m):
    return ("%-16s 中位误差 %5.1fh 平均误差 %5.1fh | 区间命中 %3.0f%% 平均宽 %4.1fh | 用户同权: 误差 %4.1fh 命中 %3.0f%% | 分档命中 %s | 分档误差 %s"
            % (name, m["med_err"], m["mean_err"], 100 * m["hit"], m["width"], m["u_med_err"], 100 * m["u_hit"],
               " ".join("%s %.0f%%" % (k, 100 * v) for k, v in m["hit_by"].items()),
               " ".join("%s %.0fh" % (k, v) for k, v in m["err_by"].items())))


# ------------------------------------------------------------------ main
def predict_all(df, D, fold_days, models, hgb, feats_extra):
    tr = df[(df["T"] < D) & df.sel]
    dur, ev, keep = outcome(tr, D)
    tr, dur, ev = tr[keep], dur[keep], ev[keep]
    m = tr.rule_ty.values > 0.25
    tr, dur, ev = tr[m], dur[m], ev[m]
    te = df[(df["T"] >= D) & (df["T"] < D + fold_days * 86400) & df.sel & df.truth.notna() & (df.rule_ty > 0.25)]
    feats = FEATS + [c for c in feats_extra if c in df.columns]
    out = {}
    base = rule_grid(te)
    top = 4 * np.max(dur / tr.rule_ty.values)
    cache = {}

    def get(name):
        if name in cache:
            return cache[name]
        if name == "rule":
            G = base
        elif name == "cal2":
            G = km_apply(te, km_cells(tr, CAL2, dur / tr.rule_ty.values, ev, 30, top), CAL2, te.rule_ty.values, base)
        elif name == "calws":
            G = km_apply(te, km_cells(tr, CALWS, dur / tr.rule_ty.values, ev, 30, top), CALWS, te.rule_ty.values, base)
        elif name == "two":
            G = two_stage(tr, dur, ev, te, FEATS, False, hgb)
        elif name == "twod":
            G = two_stage(tr, dur, ev, te, FEATS, True, hgb)
        elif name == "learned":          # the app's current model, before widening
            G = logmix(get("calws"), get("two"), get("twod"))
        elif name == "two+":
            G = two_stage(tr, dur, ev, te, feats, False, hgb)
        elif name == "twod+":
            G = two_stage(tr, dur, ev, te, feats, True, hgb)
        elif name == "haz":
            G = hazard(tr, dur, ev, te, FEATS, hgb)
        elif name == "haz+":
            G = hazard(tr, dur, ev, te, feats, hgb)
        elif name == "mix+":
            G = logmix(get("calws"), get("two+"), get("twod+"))
        elif name == "mixh+":
            G = logmix(get("calws"), get("two+"), get("twod+"), get("haz+"))
        else:
            raise ValueError(name)
        cache[name] = G
        return G
    for name in models:
        out[name] = get(name)
    y = ((te.truth - te["T"]) / 3600).values
    return out, y, 1.0 / te.ucnt.values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--folds", default="2,3,4,5,6,7")
    ap.add_argument("--fold-days", type=float, default=7)
    ap.add_argument("--models", default="rule,cal2,learned,haz")
    ap.add_argument("--modes", default="q10-q90,median+-12,best24")
    ap.add_argument("--hgb", action="store_true", help="scikit-learn instead of LightGBM")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    fetched, df = load(a.path)
    print("rows %d, points %d, span %.1f days" % (len(df), df["T"].nunique(), (df["T"].max() - df["T"].min()) / 86400),
          flush=True)
    t_end = df["T"].max()
    models = a.models.split(",")
    acc = {m: ([], [], []) for m in models}
    for k in [int(x) for x in a.folds.split(",")]:
        D = t_end - 7 * 86400 * k
        t0 = time.time()
        out, y, w = predict_all(df, D, a.fold_days, models, a.hgb, EXTRA)
        for name, G in out.items():
            acc[name][0].append(G); acc[name][1].append(y); acc[name][2].append(w)
        print("  fold %d: %d test rows, %ds" % (k, len(y), time.time() - t0), flush=True)
    res = {}
    for name, (Gs, Ys, Ws) in acc.items():
        G, y, w = np.concatenate(Gs), np.concatenate(Ys), np.concatenate(Ws)
        for mode in a.modes.split(","):
            m = metrics(G, y, w, mode)
            res["%s|%s" % (name, mode)] = m
            print(fmt("%s [%s]" % (name, mode), m), flush=True)
    if a.json:
        import json
        with open(a.json, "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
