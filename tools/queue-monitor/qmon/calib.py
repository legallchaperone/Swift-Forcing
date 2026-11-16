"""Prediction log, real start times, empirical calibration and accuracy metrics.

Every refresh logs the model's prediction for every pending job on the cluster
(the user's jobs every refresh, other jobs at most once an hour). Real start
times come from squeue (running jobs) and sacct (jobs started in the last
3 days). From the log we learn how far off the model is:

    ratio = real remaining wait / predicted remaining wait

Jobs that have not started yet are censored ("at least this long"), so the ratio
distribution is a Kaplan-Meier estimate, per bucket of predicted remaining wait.
The 10/50/90% quantiles of that distribution turn one prediction into a
calibrated range. Accuracy is measured on held-out jobs (half of the job ids).
"""
import json
import math
import os
import sqlite3
import threading
import time
import zlib

BUCKETS = [(0, 6), (6, 24), (24, 72), (72, 1e9)]   # predicted remaining wait, hours
OTHER_EVERY_S = 3600                                # log other users' jobs at most hourly
KEEP_DAYS = 30
MIN_EVENTS = 40                                     # started-job samples needed per bucket
_LOCK = threading.Lock()


def bucket_of(h):
    for i, (a, b) in enumerate(BUCKETS):
        if a <= h < b:
            return i
    return len(BUCKETS) - 1


def holdout(job):
    """Deterministic 50/50 split of jobs: True = evaluation set."""
    return zlib.crc32(job.encode()) % 2 == 1


class Store(object):
    def __init__(self, path):
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS preds (
            cluster TEXT, t REAL, job TEXT, mine INTEGER, partition TEXT, pool TEXT,
            gpus REAL, eq REAL, limit_s REAL, prio REAL, fs REAL, age REAL, part_w REAL,
            rank INTEGER, ahead REAL, rate REAL, queue_n INTEGER, waited_s REAL,
            pred REAL, slurm_est REAL);
        CREATE INDEX IF NOT EXISTS preds_job ON preds(cluster, job);
        CREATE INDEX IF NOT EXISTS preds_t ON preds(cluster, t);
        CREATE TABLE IF NOT EXISTS starts (cluster TEXT, job TEXT, start REAL, PRIMARY KEY (cluster, job));
        CREATE TABLE IF NOT EXISTS seen (cluster TEXT, job TEXT, last REAL, gone INTEGER DEFAULT 0,
                                         PRIMARY KEY (cluster, job));
        """)
        self.db.commit()
        self._last_other = {}
        self._cache = {}

    # ------------------------------------------------------------ writing
    def record(self, cluster, snap, preds):
        now = snap["now"]
        rates = preds["pools"]
        log_others = now - self._last_other.get(cluster, 0) >= OTHER_EVERY_S
        rows = []
        pending_ids = []
        queue_n = len([j for j in snap["pending"] if j["competing"]])
        for j in snap["pending"]:
            if not j["competing"] or "[" in j["id"]:
                continue
            pending_ids.append(j["id"])
            if not (j["mine"] or log_others):
                continue
            a = preds["all"].get(j["id"])
            if not a or a["start"] is None:
                continue
            sp = j.get("part_w")
            rows.append((cluster, now, j["id"], int(j["mine"]), j["partition"], j["pool"], j["gpus"], j["eq"],
                         j["limit_s"], j["prio"], None, j["age0"], sp, a["rank"], a["ahead"],
                         rates.get(j["pool"], {}).get("typical"), queue_n,
                         now - j["submit"] if j.get("submit") else None, a["start"], j.get("slurm_est")))
        if log_others:
            self._last_other[cluster] = now
        starts = {}
        for r in snap["running"]:
            if r.get("start"):
                starts[r["id"]] = r["start"]
        for h in snap["history"]:
            if h.get("id") and h.get("start"):
                starts[h["id"]] = h["start"]
        with _LOCK:
            c = self.db.cursor()
            c.executemany("INSERT INTO preds VALUES (%s)" % ",".join("?" * 20), rows)
            c.executemany("INSERT OR IGNORE INTO starts VALUES (?,?,?)",
                          [(cluster, k, v) for k, v in starts.items()])
            c.executemany("INSERT INTO seen(cluster, job, last, gone) VALUES (?,?,?,0) "
                          "ON CONFLICT(cluster, job) DO UPDATE SET last=excluded.last, gone=0",
                          [(cluster, k, now) for k in pending_ids])
            # jobs no longer pending and never seen starting: cancelled or held -> censored at last seen
            c.execute("UPDATE seen SET gone=1 WHERE cluster=? AND last<?", (cluster, now - 1))
            c.execute("DELETE FROM preds WHERE t<?", (now - KEEP_DAYS * 86400,))
            self.db.commit()
        self._cache.pop(cluster, None)

    # ------------------------------------------------------------ reading
    def backtest_cells(self, cluster):
        p = os.path.join(os.path.dirname(self.path), "bench", "%s_cal.json" % cluster)
        try:
            with open(p) as f:
                d = json.load(f)
            return {tuple(k): tuple(v) for k, v in d["cells"]}
        except (IOError, OSError, ValueError, KeyError):
            return {}

    def samples(self, cluster, now):
        """[(bucket, ratio, event, job, pred_rem_h, real_rem_h or None, slurm_err_h or None, t)]"""
        with _LOCK:
            rows = self.db.execute("""
                SELECT p.job, p.t, p.pred, p.slurm_est, s.start, n.last, n.gone, p.mine, p.limit_s, p.eq
                FROM preds p
                LEFT JOIN starts s ON s.cluster=p.cluster AND s.job=p.job
                LEFT JOIN seen n ON n.cluster=p.cluster AND n.job=p.job
                WHERE p.cluster=?""", (cluster,)).fetchall()
        out = []
        for job, t, pred, sest, start, last, gone, mine, lim, eq in rows:
            rem = (pred - t) / 3600.0
            if rem <= 0.25:
                continue
            if start is not None and start >= t:
                real = (start - t) / 3600.0
                serr = abs(sest - start) / 3600.0 if sest else None
                out.append((bucket_of(rem), real / rem, 1, job, rem, real, serr, t, mine, lim or 0, eq or 1))
            elif start is None:
                upto = (last if gone else now) or now
                cens = max(upto - t, 0) / 3600.0
                out.append((bucket_of(rem), cens / rem, 0, job, rem, None, None, t, mine, lim or 0, eq or 1))
        return out

    def calibrator(self, cluster, now):
        """Returns (fn, report). fn(pred_rem_h) -> (r10, r50, r90) or None."""
        key = (cluster, int(now // 600))
        if key in self._cache:
            return self._cache[key]
        smp = self.samples(cluster, now)
        table = dict(self.backtest_cells(cluster))
        table.update(fit_cells(smp))          # live cells win where they exist

        def fn(rem_h, limit_s=None, eq=None):
            return lookup(table, rem_h, limit_s, eq)

        report = accuracy(smp)
        report["buckets"] = [{"range_h": list(BUCKETS[b]), "events": sum(1 for s in smp if s[0] == b and s[2]),
                              "censored": sum(1 for s in smp if s[0] == b and not s[2]),
                              "q": table.get((b,))} for b in range(len(BUCKETS))]
        report["n_jobs_started"] = len({s[3] for s in smp if s[2]})
        report["n_preds"] = len(smp)
        self._cache[key] = (fn, report)
        return fn, report


def cells_of(rem_h, limit_s, eq):
    """Calibration cells, most specific first: predicted-wait bucket x time-limit
    class x size class, then bucket x time-limit class, then bucket."""
    b = bucket_of(rem_h)
    if limit_s is None or eq is None:
        return [(b,)]
    lc = 0 if limit_s <= 4 * 3600 else 1 if limit_s <= 24 * 3600 else 2
    sc = 0 if eq <= 1 else 1 if eq <= 4 else 2
    return [(b, lc, sc), (b, lc), (b,)]


def fit_cells(smp, min_events=MIN_EVENTS):
    cells = {}
    for s in smp:
        for key in cells_of(s[4], s[9], s[10]):
            cells.setdefault(key, []).append((s[1], s[2]))
    table = {}
    for key, o in cells.items():
        if sum(e for _, e in o) >= min_events:
            q = km_quantiles(o, (0.1, 0.5, 0.9))
            if q and None not in q:
                table[key] = q
    return table


def lookup(table, rem_h, limit_s=None, eq=None):
    for key in cells_of(rem_h, limit_s, eq):
        if key in table:
            return table[key]
    return None


def km_quantiles(obs, qs):
    """Kaplan-Meier quantiles of a right-censored sample [(value, event)]."""
    if not obs:
        return None
    obs = sorted(obs)
    n = len(obs)
    surv = 1.0
    res = [None] * len(qs)
    i = 0
    while i < n:
        v = obs[i][0]
        d = c = 0
        while i < n and obs[i][0] == v:
            d += obs[i][1]
            c += 1 - obs[i][1]
            i += 1
        at_risk = n - (i - d - c)
        if d:
            surv *= 1.0 - d / float(at_risk)
        for k, q in enumerate(qs):
            if res[k] is None and 1.0 - surv >= q - 1e-12:
                res[k] = v
    return tuple(res)


def accuracy(smp, days=7):
    """Held-out accuracy of the raw model, the calibrated range and Slurm's own estimate."""
    if not smp:
        return {"n_eval": 0}
    tmax = max(s[7] for s in smp)
    train = [s for s in smp if not holdout(s[3])]
    test = [s for s in smp if holdout(s[3]) and s[7] >= tmax - days * 86400]
    table = fit_cells(train, MIN_EVENTS // 2)
    raw_err, cal_err, slurm_err, hit, decided = [], [], [], 0, 0
    for b, ratio, ev, job, rem, real, serr, t, mine, lim, eq in test:
        q = lookup(table, rem, lim, eq)
        if ev:
            raw_err.append(abs(real - rem))
            if serr is not None:
                slurm_err.append(serr)
            if q:
                cal_err.append(abs(real - rem * q[1]))
                decided += 1
                hit += int(rem * q[0] <= real <= rem * q[2])
        elif q and ratio > q[2]:
            decided += 1        # still waiting and already past the 90% end: a miss
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
    return {
        "n_eval": len([s for s in test if s[2]]),
        "raw_median_err_h": med(raw_err), "cal_median_err_h": med(cal_err),
        "slurm_median_err_h": med(slurm_err),
        "coverage": hit / float(decided) if decided else None, "coverage_n": decided,
    }
