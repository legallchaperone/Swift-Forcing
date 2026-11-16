"""Apply the learned start-time model (JSON from qmon.learn). Standard library only."""
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
        for name in ("calws", "two", "twod"):
            self.idx[name] = {(tuple(lev), tuple(k)): q for lev, k, q in d[name]["table"]}

    @classmethod
    def load(cls, path):
        try:
            with open(path) as f:
                d = json.load(f)
            if d.get("version") != 1:
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
