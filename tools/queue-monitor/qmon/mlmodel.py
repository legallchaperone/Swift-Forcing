"""Apply the learned start-time model (JSON from qmon.learn). Standard library only.

Version 2 (hazard model): a wait distribution per job; the app shows its median and the 24-hour
window with the most probability. Version 1 (the earlier mix of three parts) still loads.
"""
import bisect
import json
import math
import os

from qmon import features


def gbm_predict(g, x):
    s = g["base"]
    for tree in g["trees"]:
        i = 0
        while True:
            f, thr, miss_left, left, right, leaf, val = tree[i]
            if leaf:
                s += val
                break
            v = x[f]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                i = left if miss_left else right
            else:
                i = left if v <= thr else right
    return s


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


GRID = [round(0.05 * i, 2) for i in range(1, 20)]          # 5%, 10%, ..., 95%
I10, I50, I90 = 1, 9, 17


def haz_grid(p, x):
    """Wait quantiles (hours) on GRID from the hazard model; x: feature list without the bin columns."""
    E = p["edges"]
    NB = len(E) - 1
    S, s = [], 1.0
    for k in range(NB):
        s *= 1.0 - sigmoid(gbm_predict(p["gbm"], x + [float(k), float(E[k])]))
        S.append(s)
        if s < 0.05:          # every quantile up to 95% is known
            break
    G = []
    for g in GRID:
        t = 1.0 - g
        k = next((i for i, v in enumerate(S) if v <= t), None)
        if k is None:
            G.append(E[-1] * 1.5 if len(S) == NB else E[len(S)])
            continue
        s0, s1 = (S[k - 1] if k else 1.0), max(S[k], 1e-12)
        try:
            fr = min(max(math.log(s0 / t) / math.log(s0 / s1), 0.0), 1.0)
        except (ValueError, ZeroDivisionError):
            fr = 0.5
        G.append(E[k] + (E[k + 1] - E[k]) * fr)
    return G


def _cdf(xs, ps, x):
    if x <= xs[0]:
        return 0.0
    if x >= xs[-1]:
        return ps[-1]
    i = bisect.bisect_right(xs, x) - 1
    return ps[i] + (ps[i + 1] - ps[i]) * (x - xs[i]) / (xs[i + 1] - xs[i])


def best_window(G, cap):
    """The 10-90% range if it is at most `cap` hours wide, else the `cap`-hour window with the most probability."""
    Gi, m = [], 0.0
    for v in G:
        m = max(m, v, 0.0)
        Gi.append(m)
    if Gi[I90] - Gi[I10] <= cap:
        return Gi[I10], Gi[I90]
    xs, ps = [0.0] + Gi, [0.0] + GRID
    best, a_best = -1.0, 0.0
    for a in sorted(set([0.0] + Gi + [max(v - cap, 0.0) for v in Gi])):
        mass = _cdf(xs, ps, a + cap) - _cdf(xs, ps, a)
        if mass > best + 1e-12:
            best, a_best = mass, a
    return a_best, a_best + cap


def _lookup(table_idx, levels, key):
    for lev in levels:
        q = table_idx.get((tuple(lev), tuple(key[c] for c in lev)))
        if q:
            return q
    return None


class Model(object):
    def __init__(self, d):
        self.d = d
        self.idx = {}
        if d.get("version") == 2:
            return
        for name in ("calws", "two", "twod"):
            self.idx[name] = {(tuple(lev), tuple(k)): q for lev, k, q in d[name]["table"]}

    @classmethod
    def load(cls, path):
        try:
            with open(path) as f:
                d = json.load(f)
            if d.get("version") not in (1, 2):
                return None
            return cls(d)
        except (IOError, OSError, ValueError, KeyError):
            return None

    def _part(self, name, f, key, x):
        p = self.d[name]
        s = gbm_predict(p["gbm"], x)
        sb = sum(1 for e in p["edges"] if s > e)
        k = dict(key, s=sb)
        q = _lookup(self.idx[name], p["levels"], k)
        if not q:
            return None   # caller falls back to the rule range
        mult = 1.0 if p["raw"] else f["rule_ty"]
        return [mult * v for v in q]

    def predict(self, f):
        """f: feature dict from features.compute -> (p10, p50, p90) hours from now, or None."""
        if f["rule_ty"] is None or f["rule_ty"] <= 0.25:
            return None
        if self.d.get("version") == 2:
            x = [f.get(c) for c in self.d["feats"]]
            G = haz_grid(self.d["haz"], x)
            lo, hi = best_window(G, self.d.get("cap_h", 24.0))
            med = min(max(max(G[I50], 0.0), lo), hi)
            return lo, med, hi
        key = features.keys(f)
        x = [f.get(c) for c in self.d["feats"]]
        qs = []
        q = _lookup(self.idx["calws"], self.d["calws"]["levels"], key)
        rf = f.get("rule_fa") if f.get("rule_fa") is not None else f["rule_ty"]
        rs = f.get("rule_sl") if f.get("rule_sl") is not None else f["rule_ty"]
        qs.append([f["rule_ty"] * v for v in q] if q else [rf, f["rule_ty"], rs])
        for name in ("two", "twod"):
            r = self._part(name, f, key, x)
            qs.append(r if r is not None else [rf, f["rule_ty"], rs])
        L = [sum(math.log1p(max(q[i], 0.0)) for q in qs) / len(qs) for i in range(3)]
        lo, up = self.d.get("widen", (1.0, 1.0))
        L = [max(L[1] - lo * (L[1] - L[0]), 0.0), L[1], L[1] + up * (L[2] - L[1])]
        return tuple(math.expm1(v) for v in L)


def default_path(data_dir, cluster):
    return os.path.join(data_dir, "bench", "%s_model.json" % cluster)
