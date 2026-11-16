"""Backtest benchmark: which start-time prediction algorithm is most accurate?

Two kinds of test points:

  * Reconstructed history. `sacct` keeps every job's submit / eligible / start /
    end time, size and time limit for weeks. From it we rebuild the queue every
    6 hours: which jobs were pending, which were running (with only their time
    limit known, as at that moment), and each user's decayed usage, which gives
    the Fair Tree fair-share factor. Not recoverable: Slurm's own start estimate,
    why a job was pending (held / dependency jobs look like normal pending jobs),
    changes to account shares, and usage on partitions outside the pools.
  * Archived live snapshots (every 2 hours since logging began). These are exact,
    and also hold Slurm's own estimate.

Split by time, never by job: a prediction made at time T may only use data known
at T (calibration is fitted on earlier test points whose outcome was known by T).

Metrics per algorithm, on jobs that actually started (cancelled jobs dropped):
median absolute error (hours), median |log ratio|, share of real starts inside
the 10-90% range, and mean pinball loss over the 10/50/90% quantiles. Test points
are kept only if they are at least `guard_days` older than the data, so almost
every job has an outcome; the share still unresolved is reported.

Run:  python3 -m qmon.bench --cluster DeltaAI [--fetch]
"""
import argparse
import glob
import gzip
import json
import math
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from qmon import calib, collect, model  # noqa: E402

QS = (0.1, 0.5, 0.9)


# ------------------------------------------------------------------ history dump

HIST_FIELDS = "JobID,User,Account,Partition,NNodes,ReqTRES%160,AllocTRES%160,Timelimit,Elapsed,State,Submit,Eligible,Start,End"


def fetch_history(alias, partitions, days, out_path, timeout=900):
    """One read-only sacct call for all jobs in the pools over the last `days` days."""
    script = ("date +%%s; date +%%z; sacct -a -X -P -n -S $(date -d '-%d days' +%%FT%%T) -r %s -o %s"
              % (days, ",".join(partitions), HIST_FIELDS))
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", alias, "bash", "-s"],
                       input=script.encode(), capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError("sacct history fetch failed: %s" % r.stderr.decode()[-300:])
    with gzip.open(out_path, "wb") as f:
        f.write(r.stdout)
    return out_path


def load_history(path, cfg):
    with gzip.open(path, "rt") as f:
        lines = f.read().splitlines()
    fetched_at = float(lines[0])
    tparse = model.make_time_parser(lines[1])
    cpg = cfg.get("cpus_per_gpu", {})
    pools = set(cfg["pools"])
    jobs = []
    for line in lines[2:]:
        p = line.split("|")
        if len(p) < 14:
            continue
        jid, user, acct, part, nn, req, alloc, limit, elapsed, state, sub, elig, start, end = p[:14]
        part = part.split(",")[0]
        pool = model.pool_of(part)
        if pool not in pools:
            continue
        tres = alloc or req
        g = model.gpus_from(tres)
        cpus = model.tres_value(tres, "cpu")
        c = cpg.get(pool, cpg.get("default", 72))
        bill = model.tres_value(tres, "billing")
        jobs.append({
            "id": jid, "user": user, "account": acct, "partition": part, "pool": pool,
            "nodes": int(nn or 1), "gpus": g, "eq": max(float(g), cpus / float(c)), "billing": bill,
            "limit_s": model.parse_duration(limit) or 172800, "elapsed_s": model.parse_duration(elapsed),
            "state": state.split(" ")[0], "submit": tparse(sub), "elig": tparse(elig) or tparse(sub),
            "start": tparse(start), "end": tparse(end),
        })
    return fetched_at, jobs


# ------------------------------------------------------------------ reconstruction

def reconstruct(T, jobs, tree, weights, part_w, me):
    """Queue state at time T, using only what was knowable at T."""
    hl = weights["half_life_s"]
    pending, running, history = [], [], []
    usage = {}
    for j in jobs:
        st, el = j["start"], j["elig"]
        if el is None:
            continue
        if st is not None and st <= T:
            end = j["end"] if (j["end"] and j["end"] <= T) else None
            run_to = end if end else T
            # decayed usage accrued in [st, run_to], seen from T
            if j["billing"] > 0 and run_to > st:
                k = math.log(2) / hl
                u = j["billing"] * (math.exp(-k * (T - run_to)) - math.exp(-k * (T - st))) / k
                key = (j["account"], j["user"])
                usage[key] = usage.get(key, 0.0) + u
            history.append({"id": j["id"], "pool": j["pool"], "user": j["user"], "start": st, "elig": el,
                            "eq": j["eq"], "limit_s": j["limit_s"],
                            "elapsed_s": (end - st) if end else (T - st),
                            "state": j["state"] if end else "RUNNING"})
            if end is None:
                running.append({"id": j["id"], "user": j["user"], "account": j["account"],
                                "partition": j["partition"], "pool": j["pool"], "nodes": j["nodes"],
                                "gpus": j["gpus"], "eq": j["eq"], "billing": j["billing"],
                                "left_s": max(st + j["limit_s"] - T, 0), "limit_s": j["limit_s"],
                                "start": st, "mine": j["user"] == me})
        elif el <= T and (st is None or st > T):
            if st is None and j["end"] is not None and j["end"] <= T:
                continue                     # cancelled before T
            pending.append(j)
    # fair share from reconstructed usage
    for nd in tree.nodes:
        nd["_raw_save"] = nd["raw"]
        nd["raw"] = 0.0
    for (a, u), val in usage.items():
        i = tree.user_index.get((a, u))
        while i is not None:
            tree.nodes[i]["raw"] += val
            i = tree.nodes[i]["parent"]
    f = tree.factors(1.0, {})
    snap_tree_raw = [nd["raw"] for nd in tree.nodes]
    for nd in tree.nodes:
        nd["raw"] = nd.pop("_raw_save")
    pend = []
    for j in pending:
        i = tree.user_index.get((j["account"], j["user"]))
        fs = f.get(i, 0.0) if i is not None else 0.0
        age = min(max(T - j["elig"], 0) / weights["max_age_s"], 1.0) * weights["age"]
        pw = part_w.get(j["partition"], 0.0)
        pend.append({
            "id": j["id"], "partition": j["partition"], "pool": j["pool"], "user": j["user"],
            "account": j["account"], "prio": round(fs * weights["fs"]) + age + pw, "age0": age,
            "nodes": j["nodes"], "gpus": j["gpus"], "eq": j["eq"], "limit_s": j["limit_s"],
            "submit": j["submit"], "elig": j["elig"], "slurm_est": None, "reason": "Priority",
            "count": 1, "competing": True, "mine": j["user"] == me, "part_w": pw,
            "_truth": j["start"], "_end": j["end"], "_state": j["state"],
        })
    return {"now": T, "me": me, "weights": weights, "pending": pend, "running": running,
            "history": history, "tree": _TreeAt(tree, snap_tree_raw), "part_w": part_w}


class _TreeAt(object):
    """A FairTree view whose raw usage is the reconstructed usage at T."""

    def __init__(self, tree, raw):
        self._t, self._raw = tree, raw
        self.user_index = tree.user_index
        self.nodes = tree.nodes

    def factors(self, decay, adds):
        save = [nd["raw"] for nd in self._t.nodes]
        for nd, r in zip(self._t.nodes, self._raw):
            nd["raw"] = r
        try:
            return self._t.factors(decay, adds)
        finally:
            for nd, r in zip(self._t.nodes, save):
                nd["raw"] = r

    def user_count(self):
        return self._t.user_count()


# ------------------------------------------------------------------ candidates

def cand_rule(snap, cfg, static=False):
    pr = model.predict(snap, cfg, log_all=True, static=static, use_blend=False)
    out = {}
    for jid, v in pr["all"].items():
        if v.get("typical") is None:
            continue
        ty = v["typical"]
        fa = min(v.get("fast") or ty, ty)
        sl = max(v.get("slow") or ty, ty)
        out[jid] = (fa, ty, sl)
    return out


def cand_hist(snap, jobs_by_pool, window_d=3):
    """Baseline: waits of similar jobs (same pool, same GPU size) in the last week,
    conditioned on how long this job has already waited (Kaplan-Meier)."""
    T = snap["now"]
    out = {}
    cache = {}
    for j in snap["pending"]:
        size = 1 if j["eq"] <= 1 else 2 if j["eq"] <= 3 else 4 if j["eq"] <= 4 else 8
        key = (j["pool"], size)
        if key not in cache:
            obs = []
            for h in jobs_by_pool.get(j["pool"], ()):
                if h["elig"] is None or not (T - window_d * 86400 <= h["elig"] < T):
                    continue
                hs = 1 if h["eq"] <= 1 else 2 if h["eq"] <= 3 else 4 if h["eq"] <= 4 else 8
                if hs != size:
                    continue
                if h["start"] is not None and h["start"] <= T:
                    obs.append((h["start"] - h["elig"], 1))
                elif h["start"] is not None or h["end"] is None or h["end"] > T:
                    obs.append((T - h["elig"], 0))
            cache[key] = obs
        w0 = T - j["elig"] if j.get("elig") else 0
        cond = [(w - w0, e) for (w, e) in cache[key] if w > w0]
        if sum(e for _, e in cond) < 5:
            continue
        q = calib.km_quantiles(cond, QS)
        if q is None or q[1] is None:
            continue
        q = [x if x is not None else (cond and max(w for w, _ in cond)) for x in q]
        out[j["id"]] = (T + q[0], T + q[1], T + q[2])
    return out


def strata(rem_h, limit_s, eq):
    """Calibration cell: predicted wait bucket x time-limit class x size class."""
    lc = 0 if limit_s <= 4 * 3600 else 1 if limit_s <= 24 * 3600 else 2
    sc = 0 if eq <= 1 else 1 if eq <= 4 else 2
    return (calib.bucket_of(rem_h), lc, sc)


def cal2_from(train, T, min_events=30):
    """Like cal_from, but per (bucket, limit class, size class) cell, falling back to
    (bucket, limit class) and then bucket when a cell has too few started jobs."""
    cells = {}
    for (t0, rem_h, truth, end, lim, eq) in train:
        if t0 >= T:
            continue
        if truth is not None and truth <= T:
            o = ((truth - t0) / 3600.0 / rem_h, 1)
        elif truth is None and end is not None and end <= T:
            continue
        else:
            o = ((T - t0) / 3600.0 / rem_h, 0)
        b, lc, sc = strata(rem_h, lim, eq)
        for key in ((b, lc, sc), (b, lc), (b,)):
            cells.setdefault(key, []).append(o)
    table = {}
    for key, o in cells.items():
        if sum(e for _, e in o) >= min_events:
            q = calib.km_quantiles(o, QS)
            if q and None not in q:
                table[key] = q
    return table


def cal2_lookup(table, rem_h, lim, eq):
    b, lc, sc = strata(rem_h, lim, eq)
    for key in ((b, lc, sc), (b, lc), (b,)):
        if key in table:
            return table[key]
    return None


def blend(p, q, T):
    return model.blend_quantiles(p, q, T)


def _old_blend(p, q, T):
    """Average two quantile predictions in log space (hours from T)."""
    out = []
    for a, b in zip(p, q):
        ha, hb = max((a - T) / 3600.0, 0) + 0.5, max((b - T) / 3600.0, 0) + 0.5
        out.append(T + (math.sqrt(ha * hb) - 0.5) * 3600)
    return tuple(sorted(out))


def cal_from(train, T):
    """KM ratio quantiles per bucket from earlier rule predictions, as known at T."""
    table = {}
    obs = {}
    for (t0, rem_h, truth, end, _lim, _eq) in train:
        if t0 >= T:
            continue
        if truth is not None and truth <= T:
            obs.setdefault(calib.bucket_of(rem_h), []).append((((truth - t0) / 3600.0) / rem_h, 1))
        else:
            if truth is None and end is not None and end <= T:
                continue  # cancelled
            obs.setdefault(calib.bucket_of(rem_h), []).append((((T - t0) / 3600.0) / rem_h, 0))
    for b, o in obs.items():
        if sum(e for _, e in o) >= calib.MIN_EVENTS:
            q = calib.km_quantiles(o, QS)
            table[b] = q if q and None not in q else None
    return table


# ------------------------------------------------------------------ metrics

def pinball(y, q, tau):
    d = y - q
    return max(tau * d, (tau - 1) * d)


def score(rows):
    """rows: [(pred (p10,p50,p90) in hours from T, real hours)]"""
    if not rows:
        return None
    ae = sorted(abs(p[1] - y) for p, y in rows)
    le = sorted(abs(math.log((p[1] + 0.5) / (y + 0.5))) for p, y in rows)
    cov = sum(1 for p, y in rows if p[0] <= y <= p[2]) / float(len(rows))
    pb = sum(sum(pinball(y, p[k], QS[k]) for k in range(3)) / 3.0 for p, y in rows) / len(rows)
    return {"n": len(rows), "median_abs_err_h": ae[len(ae) // 2], "median_abs_log_err": le[len(le) // 2],
            "coverage_10_90": cov, "pinball_h": pb}


def bucket_name(h):
    return "<6h" if h < 6 else "6-24h" if h < 24 else "1-3d" if h < 72 else ">3d"


# ------------------------------------------------------------------ main loop

def run(cfg_cluster, data_dir, snap_now, days=21, step_h=6, lookback_d=5, guard_d=7, max_jobs_per_point=400,
        fetch=False, log=print):
    """Backtest one cluster. snap_now: a current parsed snapshot (for tree, weights, partition weights)."""
    name = cfg_cluster["name"]
    bdir = os.path.join(data_dir, "bench")
    os.makedirs(bdir, exist_ok=True)
    hpath = os.path.join(bdir, "%s_sacct.txt.gz" % name)
    if fetch or not os.path.exists(hpath) or time.time() - os.path.getmtime(hpath) > 6 * 86400:
        log("fetching %d days of sacct history for %s ..." % (days, name))
        fetch_history(cfg_cluster["alias"], cfg_cluster["partitions"], days, hpath)
    fetched_at, jobs = load_history(hpath, cfg_cluster)
    t_first = min(j["elig"] for j in jobs if j["elig"])
    jobs_by_pool = {}
    for j in jobs:
        jobs_by_pool.setdefault(j["pool"], []).append(j)
    tree = snap_now["tree"]
    weights = snap_now["weights"]
    part_w = dict(snap_now.get("part_w") or {})
    me = snap_now["me"]

    t0 = t_first + lookback_d * 86400
    t1 = fetched_at - guard_d * 86400
    points = []
    T = t0
    while T <= t1:
        points.append(T)
        T += step_h * 3600
    log("%s: %d jobs, %d test points" % (name, len(jobs), len(points)))

    rnd = random.Random(0)
    algos = ["rule", "rule+cal", "rule+cal2", "blend", "rule-static", "hist-median"]  # rule+cal2 is what the app uses
    rows = {a: [] for a in algos}
    rows_b = {}
    unresolved = total = 0
    train = []   # (t0, rem_h, truth, end) from rule predictions
    fs_check = None
    for T in points:
        snap = reconstruct(T, jobs, tree, weights, part_w, me)
        if not snap["pending"]:
            continue
        rule = cand_rule(snap, cfg_cluster)
        static = cand_rule(snap, cfg_cluster, static=True)
        hist = cand_hist(snap, jobs_by_pool)
        table = cal_from(train, T)
        table2 = cal2_from(train, T)
        pend = snap["pending"]
        if len(pend) > max_jobs_per_point:
            pend = rnd.sample(pend, max_jobs_per_point)
        for j in pend:
            total += 1
            truth = j["_truth"]
            if truth is None:
                if j["_end"] is None or j["_end"] > fetched_at - 60:
                    unresolved += 1      # still pending when the data was fetched
                continue                 # cancelled: not a prediction target
            y = (truth - T) / 3600.0
            preds = {}
            if j["id"] in rule:
                p = rule[j["id"]]
                preds["rule"] = p
                rem = (p[1] - T) / 3600.0
                q = table.get(calib.bucket_of(rem)) if rem > 0.25 else None
                preds["rule+cal"] = (T + rem * q[0] * 3600, T + rem * q[1] * 3600, T + rem * q[2] * 3600) if q else p
                q2 = cal2_lookup(table2, rem, j["limit_s"], j["eq"]) if rem > 0.25 else None
                preds["rule+cal2"] = (T + rem * q2[0] * 3600, T + rem * q2[1] * 3600, T + rem * q2[2] * 3600) if q2 else preds["rule+cal"]
                if rem > 0.25:
                    train.append((T, rem, truth, j["_end"], j["limit_s"], j["eq"]))
            if j["id"] in static:
                preds["rule-static"] = static[j["id"]]
            if j["id"] in hist:
                preds["hist-median"] = hist[j["id"]]
                if "rule+cal2" in preds:
                    preds["blend"] = blend(preds["rule+cal2"], hist[j["id"]], T)
            if len(preds) < len(algos):
                continue                 # compare all algorithms on the same jobs
            for a, p in preds.items():
                ph = tuple(max((x - T) / 3600.0, 0.0) for x in p)
                rows[a].append((ph, y))
                rows_b.setdefault((a, bucket_name(y)), []).append((ph, y))
                if j["mine"]:
                    rows_b.setdefault((a, "mine"), []).append((ph, y))
    res = {
        "cluster": name, "generated": time.time(), "source": "reconstructed sacct history",
        "window": [t0, t1], "points": len(points), "step_h": step_h, "jobs_total": total,
        "unresolved_share": unresolved / float(total) if total else None,
        "overall": {a: score(rows[a]) for a in algos},
        "by_bucket": {"%s|%s" % k: score(v) for k, v in rows_b.items()},
        "algos": algos,
        "notes": "Slurm's own estimate is not in sacct, so it is only compared on archived live snapshots.",
    }
    res["live"] = run_live(cfg_cluster, data_dir, jobs, fetched_at, guard_d)
    with open(os.path.join(bdir, "%s_results.json" % name), "w") as f:
        json.dump(res, f)
    return res


def run_live(cfg_cluster, data_dir, jobs, fetched_at, guard_d):
    """Score archived live snapshots (exact inputs, includes Slurm's own estimate)."""
    from qmon import archive
    truth = {j["id"]: (j["start"], j["end"]) for j in jobs}
    algos = ["slurm", "rule", "rule-static"]
    rows = {a: [] for a in algos}
    n_snaps = 0
    for path in archive.list_snapshots(data_dir, cfg_cluster["name"]):
        sec = archive.load(path)
        snap = model.parse_snapshot(sec, cfg_cluster)
        T = snap["now"]
        if T > fetched_at - guard_d * 86400:
            continue
        n_snaps += 1
        rule = cand_rule(snap, cfg_cluster)
        static = cand_rule(snap, cfg_cluster, static=True)
        for j in snap["pending"]:
            if not j["competing"] or j["id"] not in truth or truth[j["id"]][0] is None:
                continue
            st = truth[j["id"]][0]
            y = (st - T) / 3600.0
            if y < 0 or j["id"] not in rule or j["id"] not in static or not j.get("slurm_est"):
                continue
            se = max((j["slurm_est"] - T) / 3600.0, 0.0)
            for a, p in (("slurm", (se, se, se)),
                         ("rule", tuple(max((x - T) / 3600.0, 0) for x in rule[j["id"]])),
                         ("rule-static", tuple(max((x - T) / 3600.0, 0) for x in static[j["id"]]))):
                rows[a].append((p, y))
    return {"snapshots": n_snaps, "overall": {a: score(rows[a]) for a in algos}, "algos": algos}


def main():
    from qmon import server
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", default="DeltaAI")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--days", type=int, default=21)
    a = ap.parse_args()
    cl = next(c for c in server.CONFIG["clusters"] if c["name"] == a.cluster)
    sec = collect.fetch(cl["alias"], cl["partitions"])
    snap = model.parse_snapshot(sec, cl)
    t = time.time()
    res = run(cl, server.DATA_DIR, snap, days=a.days, fetch=a.fetch)
    print(json.dumps({k: v for k, v in res.items() if k != "by_bucket"}, indent=1, default=str))
    for k, v in sorted(res["by_bucket"].items()):
        print(k, v)
    print("seconds", round(time.time() - t))


if __name__ == "__main__":
    main()
