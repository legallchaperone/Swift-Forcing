"""Compare start-time models on a feature table from build_ds.py, with a strict time split.

usage: python research/evaluate.py <features.pkl> [--fold-days 7] [--folds 2,3,4,5,6,7] [--models cal2,calfs,two]
Needs numpy, pandas, scikit-learn (research only; the app itself stays standard-library).

Fold k tests on points T in [D, D + fold_days), D = last point - 7 days * k, and trains only on
points before D, with outcomes as known at D (jobs still waiting at D are censored).
"""
import argparse, os, pickle, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmon import calib  # noqa

QS = (0.1, 0.5, 0.9)


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
    df["u_eta"] = (df.u_ahead_n + 1) / (df.u_st24 / 24.0 + 0.05)
    df["a_eta"] = (df.a_ahead + 1) / (df.a_st24 / 24.0 + 0.05)
    df["b"] = df.rule_ty.apply(calib.bucket_of)
    df["lc3"] = np.where(df.limit_h <= 4, 0, np.where(df.limit_h <= 24, 1, 2))
    df["sc"] = np.where(df["eq"] <= 1, 0, np.where(df["eq"] <= 4, 1, 2))
    df["wb"] = np.where(df.waited_h < 6, 0, np.where(df.waited_h < 24, 1, np.where(df.waited_h < 72, 2, 3)))
    df["fsb"] = np.where(df.fs < 0.02, 0, np.where(df.fs < 0.1, 1, 2))
    df["heavy"] = (df.u_pend_n > 30).astype(int)
    return d["fetched"], df


def outcome(d, D):
    """(duration h, started 0/1, keep mask) as known at time D; cancelled-before-D rows dropped."""
    started = d.truth.notna() & (d.truth <= D)
    canc = ~started & d.end.notna() & (d.end <= D)
    dur = np.where(started, (d.truth - d["T"]) / 3600.0, (D - d["T"]) / 3600.0)
    return dur, started.values.astype(int), (~canc).values


# ------------------------------------------------------------------ metrics
def wq(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return x[o][min(np.searchsorted(c, q), len(x) - 1)]


def score(P, y, w=None):
    if w is None:
        w = np.ones(len(y))
    P = np.maximum(P, 0)
    ae = np.abs(P[:, 1] - y)
    cov = ((P[:, 0] <= y) & (y <= P[:, 2])).astype(float)
    pb = sum(np.maximum(q * (y - P[:, k]), (q - 1) * (y - P[:, k])) for k, q in enumerate(QS)) / 3
    return {"n": int(len(y)), "mae": float(wq(ae, w, .5)), "cov": float((cov * w).sum() / w.sum()),
            "pin": float((pb * w).sum() / w.sum())}


def bucket(h):
    return np.where(h < 6, "<6h", np.where(h < 24, "6-24h", np.where(h < 72, "1-3d", ">3d")))


def line(name, P, y, w):
    o, u = score(P, y), score(P, y, w)
    br = bucket(y); bp = bucket(np.maximum(P[:, 1], 0))
    parts = ["%s %.2f" % (k, score(P[br == k], y[br == k])["cov"]) for k in ("<6h", "6-24h", "1-3d", ">3d") if (br == k).any()]
    parts2 = ["%s %.2f" % (k, score(P[bp == k], y[bp == k])["cov"]) for k in ("<6h", "6-24h", "1-3d", ">3d") if (bp == k).any()]
    return ("%-12s rows: err %5.1fh cov %.2f pin %5.1f | users equal: err %5.1fh cov %.2f | cov by real wait: %s | by predicted: %s"
            % (name, o["mae"], o["cov"], o["pin"], u["mae"], u["cov"], " ".join(parts), " ".join(parts2)))


# ------------------------------------------------------------------ models
def km_table(keys_df, levels, ratio, ev, min_events):
    table = {}
    for lev in levels:
        for k, ix in keys_df.groupby(list(lev)).indices.items():
            if ev[ix].sum() < min_events:
                continue
            q = calib.km_quantiles(list(zip(ratio[ix], ev[ix])), QS)
            if q and None not in q:
                table[(lev, k if isinstance(k, tuple) else (k,))] = q
    return table


def km_apply(te, table, levels, rule):
    P = np.column_stack([te.rule_fa.fillna(te.rule_ty), te.rule_ty, te.rule_sl.fillna(te.rule_ty)]).astype(float)
    cols = {c: te[c].values for lev in levels for c in lev}
    for i in range(len(te)):
        if rule[i] <= 0.25:
            continue
        for lev in levels:
            q = table.get((lev, tuple(cols[c][i] for c in lev)))
            if q:
                P[i] = [rule[i] * q[0], rule[i] * q[1], rule[i] * q[2]]
                break
    return P


LEVELS = {
    "cal2": [("b", "lc3", "sc"), ("b", "lc3"), ("b",)],
    "calfs": [("b", "lc3", "sc", "fsb", "heavy"), ("b", "lc3", "sc", "fsb"), ("b", "lc3", "sc", "heavy"),
              ("b", "lc3", "sc"), ("b", "lc3"), ("b",)],
    "calws": [("b", "lc3", "sc", "wb", "fsb"), ("b", "lc3", "sc", "fsb"), ("b", "lc3", "sc", "wb"),
              ("b", "lc3", "sc"), ("b", "lc3"), ("b",)],
}

FEATS = ["rule_fa", "rule_ty", "rule_sl", "rank", "ahead", "hist10", "hist50", "hist90", "waited_h", "eq", "nodes",
         "limit_h", "interactive", "prio", "fs", "age0", "pool_busy", "pool_pend_eq", "pool_start24", "u_run_eq",
         "u_run_n", "u_pend_n", "u_pend_eq", "u_ahead_n", "u_ahead_eq", "u_start24", "u_wait_med", "u_wait_n",
         "is_array", "u_st6", "u_st24", "a_ahead", "a_pend", "a_st6", "a_st24", "lh10", "lh50", "lh90", "u_eta", "a_eta"]


def two_stage(tr, dur, ev, te, nbins=12, rounds=200, min_events=40, raw=False):
    """A boosted model ranks jobs by log(real / rule wait); Kaplan-Meier per (score bin, rule bucket)."""
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB

    def mk():
        return HGB(loss="absolute_error", learning_rate=0.05, max_iter=rounds, max_leaf_nodes=15,
                   min_samples_leaf=1000, l2_regularization=10.0, random_state=0)
    rule = np.ones(len(tr)) if raw else tr.rule_ty.values
    lr = np.log((dur + 0.5) / (rule + 0.5))
    fit_m = (ev == 1) | (dur > 24)
    X = tr[FEATS].values.astype(float)
    gid = pd.util.hash_array(tr.id.values.astype(str)) % 3
    oof = np.zeros(len(tr))
    for g in range(3):
        a = (gid != g) & fit_m
        oof[gid == g] = mk().fit(X[a], lr[a]).predict(X[gid == g])
    m = mk().fit(X[fit_m], lr[fit_m])
    edges = np.quantile(oof, np.linspace(0, 1, nbins + 1)[1:-1])
    keys = pd.DataFrame({"b": tr.b.values, "s": np.searchsorted(edges, oof)})
    levels = [("s",)] if raw else [("b", "s"), ("s",)]
    table = km_table(keys, levels, dur / rule, ev, min_events)
    te2 = te.assign(s=np.searchsorted(edges, m.predict(te[FEATS].values.astype(float))))
    return km_apply(te2, table, levels, np.ones(len(te)) if raw else te.rule_ty.values)


def widen(P, f):
    """Stretch the 10% and 90% points away from the median by factor f, in log(hours + 1) space."""
    if f == 1:
        return P
    L = np.log1p(np.maximum(P, 0))
    L[:, 0] = L[:, 1] - f * (L[:, 1] - L[:, 0])
    L[:, 2] = L[:, 1] + f * (L[:, 2] - L[:, 1])
    return np.expm1(np.maximum(L, 0))


def mix(Pa, Pb):
    """Average two quantile predictions in log(hours + 1) space."""
    return np.expm1((np.log1p(np.maximum(Pa, 0)) + np.log1p(np.maximum(Pb, 0))) / 2)


def run(df, folds, fold_days, models, train_sel=True, train_days=None, widen_f=(1.0,)):
    t_end = df["T"].max()
    out = {m: ([], [], []) for m in models}
    for k in folds:
        D = t_end - 7 * 86400 * k
        tr = df[(df["T"] < D) & (df.sel if train_sel else True)]
        if train_days:
            tr = tr[tr["T"] >= D - train_days * 86400]
        dur, ev, keep = outcome(tr, D)
        tr, dur, ev = tr[keep], dur[keep], ev[keep]
        m = tr.rule_ty.values > 0.25
        tr, dur, ev = tr[m], dur[m], ev[m]
        te = df[(df["T"] >= D) & (df["T"] < D + fold_days * 86400) & df.sel & df.truth.notna()]
        y = ((te.truth - te["T"]) / 3600).values
        w = 1.0 / te.ucnt.values
        for name in models:
            t0 = time.time()
            if name == "rule":
                P = np.column_stack([te.rule_fa.fillna(te.rule_ty), te.rule_ty, te.rule_sl.fillna(te.rule_ty)]).astype(float)
            elif name in LEVELS:
                tab = km_table(tr, LEVELS[name], dur / tr.rule_ty.values, ev, 30)
                P = km_apply(te, tab, LEVELS[name], te.rule_ty.values)
            elif name == "two":
                P = two_stage(tr, dur, ev, te)
            elif name == "two20":
                P = two_stage(tr, dur, ev, te, nbins=20)
            elif name == "twod":
                P = two_stage(tr, dur, ev, te, nbins=20, raw=True)
            elif name == "mix":
                tab = km_table(tr, LEVELS["calws"], dur / tr.rule_ty.values, ev, 30)
                P = mix(two_stage(tr, dur, ev, te), km_apply(te, tab, LEVELS["calws"], te.rule_ty.values))
            out[name][0].append(P); out[name][1].append(y); out[name][2].append(w)
        print("  fold %d: train %d rows, test %d rows" % (k, len(tr), len(te)), flush=True)
    res = {}
    for name, (Ps, Ys, Ws) in out.items():
        P, y, w = np.concatenate(Ps), np.concatenate(Ys), np.concatenate(Ws)
        for f in widen_f:
            print(line(name if f == 1 else "%s x%.2g" % (name, f), widen(P, f), y, w), flush=True)
        res[name] = (P, y, w)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--folds", default="2,3,4,5,6,7")
    ap.add_argument("--fold-days", type=float, default=7)
    ap.add_argument("--models", default="rule,cal2,calfs,calws,two")
    ap.add_argument("--train-days", type=float, default=None)
    ap.add_argument("--widen", default="1", help="comma list of interval stretch factors to report")
    a = ap.parse_args()
    fetched, df = load(a.path)
    print("rows %d, points %d, span %.1f days" % (len(df), df["T"].nunique(), (df["T"].max() - df["T"].min()) / 86400))
    run(df, [int(x) for x in a.folds.split(",")], a.fold_days, a.models.split(","), train_days=a.train_days,
        widen_f=[float(x) for x in a.widen.split(",")])
