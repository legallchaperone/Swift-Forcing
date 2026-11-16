"""Per-job features for the learned start-time model.

The same code runs on a live snapshot (the app) and on a queue rebuilt from sacct
history (training), so the model sees the same inputs in both places.
Python 3.9 standard library only.
"""
import time

from qmon import calib, model

# Order matters: the exported trees index features by position.
FEATS = ["rule_fa", "rule_ty", "rule_sl", "rank", "ahead", "hist10", "hist50", "hist90", "waited_h", "eq", "nodes",
         "limit_h", "interactive", "prio", "fs", "age0", "pool_busy", "pool_pend_eq", "pool_start24", "u_run_eq",
         "u_run_n", "u_pend_n", "u_pend_eq", "u_ahead_n", "u_ahead_eq", "u_start24", "u_wait_med", "u_wait_n",
         "is_array", "u_st6", "u_st24", "a_ahead", "a_pend", "a_st6", "a_st24", "lh10", "lh50", "lh90", "u_eta", "a_eta"]


def _lc(lim_s):
    return 0 if lim_s <= 2 * 3600 else 1 if lim_s <= 8 * 3600 else 2 if lim_s <= 24 * 3600 else 3


def hist_jobs_from_snap(snap):
    """Recent jobs of the pools in the shape cand_hist expects, from a live snapshot."""
    out = []
    for h in snap["history"]:
        end = h["start"] + h["elapsed_s"] if (h.get("elapsed_s") and h.get("state") not in ("RUNNING",)) else None
        out.append({"pool": h["pool"], "eq": h["eq"], "limit_s": h.get("limit_s") or 0,
                    "elig": h.get("elig") or h["start"], "start": h["start"], "end": end})
    for p in snap["pending"]:
        if p.get("competing", True):
            out.append({"pool": p["pool"], "eq": p["eq"], "limit_s": p["limit_s"],
                        "elig": p.get("elig") or p.get("submit"), "start": None, "end": None})
    return out


def compute(snap, cfg, allp, hist_jobs, only=None):
    """{job id: feature dict} for every pending job that the rule simulation placed.

    allp: model.predict(..., log_all=True)["all"]; hist_jobs: list of recent jobs
    ({pool, eq, limit_s, elig, start, end}) for the similar-jobs baselines.
    """
    from qmon import bench   # local import: bench imports this module's neighbours
    T = snap["now"]
    W = snap["weights"]
    jbp = {}
    for h in hist_jobs:
        jbp.setdefault(h["pool"], []).append(h)
    for j in snap["pending"]:
        if not j.get("elig"):
            j["elig"] = j.get("submit") or T
    hist = bench.cand_hist(snap, jbp)

    urun_eq, urun_n, pool_busy = {}, {}, {}
    for r in snap["running"]:
        k = (r["user"], r["pool"])
        urun_eq[k] = urun_eq.get(k, 0) + r["eq"]
        urun_n[k] = urun_n.get(k, 0) + 1
        pool_busy[r["pool"]] = pool_busy.get(r["pool"], 0) + r["eq"]
    upend, pool_pend = {}, {}
    for j in snap["pending"]:
        upend.setdefault((j["user"], j["pool"]), []).append(j)
        pool_pend[j["pool"]] = pool_pend.get(j["pool"], 0) + j["eq"] * j.get("count", 1)
    ustart24, uwait, pstart24 = {}, {}, {}
    ust6, ust24, ast6, ast24 = {}, {}, {}, {}
    for h in snap["history"]:
        age = T - h["start"]
        k = (h["user"], h["pool"])
        base = h["id"].split("_")[0]
        if age < 86400:
            ustart24[k] = ustart24.get(k, 0) + h["eq"]
            pstart24[h["pool"]] = pstart24.get(h["pool"], 0) + h["eq"]
            ust24[k] = ust24.get(k, 0) + 1
            ast24[base] = ast24.get(base, 0) + 1
        if age < 6 * 3600:
            ust6[k] = ust6.get(k, 0) + 1
            ast6[base] = ast6.get(base, 0) + 1
        if age < 3 * 86400 and h.get("elig"):
            uwait.setdefault(k, []).append(h["start"] - h["elig"])
    # pending tasks per array; a live squeue line "123_[4-100]" stands for many tasks
    arr = {}
    for j in snap["pending"]:
        if "_" in j["id"] and "[" not in j["id"]:
            b, t = j["id"].split("_", 1)
            try:
                arr.setdefault(b, []).append(int(t))
            except ValueError:
                pass
    for b in arr:
        arr[b].sort()
    lim_obs = {}
    pool0 = cfg["pools"][0] if len(cfg["pools"]) == 1 else None
    for h in jbp.get(pool0, ()):
        if h["elig"] is None or not (T - 3 * 86400 <= h["elig"] < T):
            continue
        key = (model.size_class(h["eq"]), _lc(h["limit_s"]))
        if h["start"] is not None and h["start"] <= T:
            lim_obs.setdefault(key, []).append((h["start"] - h["elig"], 1))
        elif h["start"] is not None or h["end"] is None or h["end"] > T:
            lim_obs.setdefault(key, []).append((T - h["elig"], 0))
    lim_cache = {}
    out = {}
    for j in snap["pending"]:
        if only is not None and j["id"] not in only:
            continue
        a = allp.get(j["id"])
        if not a or a.get("typical") is None:
            continue
        k = (j["user"], j["pool"])
        mine = upend.get(k, [])
        rank_key = (j["prio"], -j["elig"])
        ahead = [x for x in mine if (x["prio"], -x["elig"]) > rank_key]
        hq = hist.get(j["id"])
        uw = sorted(uwait.get(k, []))
        base = j["id"].split("_")[0]
        if "[" in j["id"]:
            a_ahead, a_pend = 0, j.get("count", 1)
        elif "_" in j["id"] and base in arr:
            try:
                a_ahead = sum(1 for t in arr[base] if t < int(j["id"].split("_", 1)[1]))
            except ValueError:
                a_ahead = None
            a_pend = len(arr[base])
        else:
            a_ahead, a_pend = None, None
        w0 = T - j["elig"]
        lkey = (model.size_class(j["eq"]), _lc(j["limit_s"]), int(w0 // (6 * 3600)) if w0 < 48 * 3600 else 8)
        if lkey not in lim_cache:
            cond = [(w - w0, e) for (w, e) in lim_obs.get(lkey[:2], []) if w > w0]
            lim_cache[lkey] = calib.km_quantiles(cond, (0.1, 0.5, 0.9)) if sum(e for _, e in cond) >= 5 else None
        lq = lim_cache[lkey]
        f = {
            "rule_fa": (a["fast"] - T) / 3600.0 if a.get("fast") else None,
            "rule_ty": (a["typical"] - T) / 3600.0,
            "rule_sl": (a["slow"] - T) / 3600.0 if a.get("slow") else None,
            "rank": a["rank"], "ahead": a["ahead"],
            "hist10": (hq[0] - T) / 3600.0 if hq else None,
            "hist50": (hq[1] - T) / 3600.0 if hq else None,
            "hist90": (hq[2] - T) / 3600.0 if hq else None,
            "waited_h": w0 / 3600.0,
            "eq": j["eq"], "nodes": j["nodes"], "limit_h": j["limit_s"] / 3600.0,
            "interactive": int("interactive" in j["partition"]), "preempt": int("preempt" in j["partition"]),
            "prio": j["prio"], "fs": (j["prio"] - j["age0"] - (j.get("part_w") or 0)) / W["fs"], "age0": j["age0"],
            "pool_busy": pool_busy.get(j["pool"], 0), "pool_pend_eq": pool_pend.get(j["pool"], 0),
            "pool_start24": pstart24.get(j["pool"], 0),
            "u_run_eq": urun_eq.get(k, 0), "u_run_n": urun_n.get(k, 0),
            "u_pend_n": len(mine), "u_pend_eq": sum(x["eq"] for x in mine),
            "u_ahead_n": len(ahead), "u_ahead_eq": sum(x["eq"] for x in ahead),
            "u_start24": ustart24.get(k, 0),
            "u_wait_med": (uw[len(uw) // 2] / 3600.0) if uw else None, "u_wait_n": len(uw),
            "is_array": int("_" in j["id"]),
            "u_st6": ust6.get(k, 0), "u_st24": ust24.get(k, 0),
            "a_ahead": a_ahead, "a_pend": a_pend, "a_st6": ast6.get(base, 0), "a_st24": ast24.get(base, 0),
            "lh10": (lq[0] / 3600.0 if lq and lq[0] is not None else None),
            "lh50": (lq[1] / 3600.0 if lq and lq[1] is not None else None),
            "lh90": (lq[2] / 3600.0 if lq and lq[2] is not None else None),
        }
        f["u_eta"] = (f["u_ahead_n"] + 1) / (f["u_st24"] / 24.0 + 0.05)
        f["a_eta"] = (f["a_ahead"] + 1) / (f["a_st24"] / 24.0 + 0.05) if f["a_ahead"] is not None else None
        out[j["id"]] = f
    return out


def keys(f):
    """Calibration-cell keys used by the Kaplan-Meier tables."""
    rt, lim, eq, w, fs = f["rule_ty"], f["limit_h"], f["eq"], f["waited_h"], f["fs"]
    return {
        "b": calib.bucket_of(rt),
        "lc3": 0 if lim <= 4 else 1 if lim <= 24 else 2,
        "sc": 0 if eq <= 1 else 1 if eq <= 4 else 2,
        "wb": 0 if w < 6 else 1 if w < 24 else 2 if w < 72 else 3,
        "fsb": 0 if fs < 0.02 else 1 if fs < 0.1 else 2,
    }


def local_hour(T):
    lt = time.localtime(T)
    return lt.tm_hour + lt.tm_min / 60.0
