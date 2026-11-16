"""Training table for the learned model: the queue rebuilt from sacct history every
few hours, with features (qmon.features) and the real start of every pending job.

usage: python3 -m qmon.trainset --cluster DeltaAI --sacct <sacct.gz> --snap <archived snapshot> --out <table.pkl>
Standard library only (it runs with the app's own Python).
"""
import argparse, json, math, os, pickle, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmon import archive, bench, calib, features, model  # noqa

A = None
cfg = None
HPATH = None


def base_snap():
    """Config, priority weights and account tree, from one archived live snapshot."""
    return model.parse_snapshot(archive.load(A.snap), cfg)


def _setup(args, config):
    global A, cfg, HPATH
    A, cfg, HPATH = args, config, args.sacct


SNAP0 = None
JOBS = None
JBP = None
FETCHED = None


CONC = None   # {("a", account) or ("u", user): (grid t0, hourly running GPU-eq array)}


def concurrency(jobs, step=3600.0):
    """Running GPU-equivalents per account and per user on an hourly grid."""
    t0 = min(j["start"] for j in jobs if j["start"])
    t1 = max((j["end"] or j["start"]) for j in jobs if j["start"])
    n = int((t1 - t0) / step) + 2
    out = {}
    for j in jobs:
        if not j["start"]:
            continue
        a = int((j["start"] - t0) / step) + 1
        b = int(((j["end"] or t1) - t0) / step) + 1
        for key in (("a", j["account"]), ("u", j["user"])):
            arr = out.get(key)
            if arr is None:
                arr = out[key] = [0.0] * n
            for i in range(a, min(b, n)):
                arr[i] += j["eq"]
    return t0, step, out


def conc_feats(key, T):
    t0, step, out = CONC
    arr = out.get(key)
    if arr is None:
        return 0.0, 0.0
    i = int((T - t0) / step)
    lo = max(0, i - int(7 * 86400 / step))
    if i < 0 or lo >= len(arr):
        return 0.0, 0.0
    now = arr[min(i, len(arr) - 1)]
    return now, max(arr[lo:i + 1] or [0.0])


def init():
    global SNAP0, JOBS, JBP, FETCHED, CONC
    SNAP0 = SNAP0 or base_snap()
    FETCHED, JOBS = bench.load_history(HPATH, cfg)
    CONC = concurrency(JOBS)
    JBP = {}
    for j in JOBS:
        JBP.setdefault(j["pool"], []).append(j)


def feats_at(T):
    tree, weights = SNAP0["tree"], SNAP0["weights"]
    part_w = dict(SNAP0.get("part_w") or {})
    snap = bench.reconstruct(T, JOBS, tree, weights, part_w, SNAP0["me"])
    if not snap["pending"]:
        return []
    pr = model.predict(snap, cfg, log_all=True)
    fts = features.compute(snap, cfg, pr["all"], JOBS)
    rows = []
    lt = time.localtime(T)
    for j in snap["pending"]:
        f = fts.get(j["id"])
        if f is None:
            continue
        r = dict(f)
        r.update({"T": T, "id": j["id"], "user": j["user"], "account": j["account"], "pool": j["pool"],
                  "partition": j["partition"], "truth": j["_truth"], "end": j["_end"], "state": j["_state"],
                  "since_submit_h": (T - j["submit"]) / 3600.0 if j["submit"] else None,
                  "hour": lt.tm_hour + lt.tm_min / 60.0, "wday": lt.tm_wday,
                  "acct_now": conc_feats(("a", j["account"]), T)[0], "acct_max7": conc_feats(("a", j["account"]), T)[1],
                  "user_now": conc_feats(("u", j["user"]), T)[0], "user_max7": conc_feats(("u", j["user"]), T)[1]})
        rows.append(r)
    return rows


def work(T):
    try:
        return feats_at(T)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return []


def build(args, config, procs=3, log=print):
    _setup(args, config)
    init()
    t_first = min(j["end"] for j in JOBS if j["end"])   # sacct -S cut-off: jobs that ended before it are missing
    t0 = t_first + 4 * 86400
    t1 = FETCHED - 0.5 * 86400
    pts = []
    T = t0
    while T <= t1:
        pts.append(T)
        T += args.step * 3600
    log("%s: %d jobs, %d points" % (config["name"], len(JOBS), len(pts)))
    st = time.time()
    out = []
    with Pool(procs, initializer=_worker_init, initargs=(args, config)) as p:
        for i, r in enumerate(p.imap(work, pts)):
            out.extend(r)
            if i % 50 == 0:
                log("%d/%d points, %d rows, %ds" % (i, len(pts), len(out), time.time() - st))
    tmp = args.out + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"fetched": FETCHED, "rows": out, "points": pts}, f)
    os.replace(tmp, args.out)
    return len(out)


def _worker_init(args, config):
    _setup(args, config)
    init()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--config", required=True, help="the app's config.json")
    ap.add_argument("--sacct", required=True)
    ap.add_argument("--snap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=float, default=3.0)
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    with open(a.config) as f:
        conf = next(c for c in json.load(f)["clusters"] if c["name"] == a.cluster)
    print("rows", build(a, conf, a.procs), flush=True)
