"""Parse Slurm output and predict when pending jobs start.

Model, in short:
  priority(t) = priority_now + W_fs * (fs_user(t) - fs_user(now)) + (age(t) - age(now))

  * fs_user(t) comes from a Fair Tree model: usage decays with the cluster's
    half-life, running jobs (and jobs the simulation starts) add usage, and every
    user is ranked by LevelFS down the account tree, as Slurm does.
  * age grows linearly to W_age over PriorityMaxAge.

  The queue is a pool of GPUs. GPUs are handed to the queue at the rate the pool
  actually started jobs over the last 3 days (slow / typical / fast = 25th / 50th /
  75th percentile of 6-hour windows). A share of that rate goes to jobs that start
  almost at once (new high-priority jobs and backfill); the rest goes to the
  pending jobs in priority order. A job starts when the GPUs handed out reach it.
"""
import math
import re
import time
from datetime import datetime, timedelta, timezone

INF = float("inf")


# ---------------------------------------------------------------- parsing helpers

def parse_duration(s):
    """Slurm duration -> seconds. Returns None for UNLIMITED / N/A."""
    s = (s or "").strip()
    if not s or s in ("UNLIMITED", "N/A", "INVALID", "NOT_SET", "Partition_Limit"):
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(float(p)) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + sec


def make_time_parser(tz_str):
    """Cluster prints naive local times; tz_str is `date +%z` (e.g. -0600)."""
    sign = -1 if tz_str.startswith("-") else 1
    off = timedelta(hours=int(tz_str[1:3]), minutes=int(tz_str[3:5])) * sign
    tz = timezone(off)

    def parse(s):
        s = (s or "").strip()
        if not s or s in ("N/A", "Unknown", "None", "(null)"):
            return None
        try:
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=tz).timestamp()
        except ValueError:
            return None
    return parse


GPU_RE = re.compile(r"gpu(?::[A-Za-z0-9_]+)?[:=](\d+)")


def gpus_from(s):
    vals = [int(x) for x in GPU_RE.findall(s or "")]
    return max(vals) if vals else 0


def tres_value(tres, key):
    m = re.search(r"(?:^|,)" + re.escape(key) + r"=([0-9.]+)", tres or "")
    return float(m.group(1)) if m else 0.0


def array_count(jobid):
    """'123_[1-50%4]' -> 50. Plain ids -> 1."""
    m = re.search(r"\[([^\]]+)\]", jobid)
    if not m:
        return 1
    n = 0
    for part in m.group(1).split("%")[0].split(","):
        if "-" in part:
            a, b = part.split("-")[:2]
            try:
                n += int(b.split(":")[0]) - int(a) + 1
            except ValueError:
                n += 1
        else:
            n += 1
    return max(n, 1)


def pool_of(partition):
    return re.sub(r"-(interactive|preempt)$", "", partition)


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# ---------------------------------------------------------------- snapshot parsing

def parse_snapshot(sec, cfg):
    lines = sec.get("date", "").split()
    now = float(lines[0])
    tparse = make_time_parser(lines[1] if len(lines) > 1 else "+0000")
    me = sec.get("user", "").strip()
    cpg = cfg.get("cpus_per_gpu", {})

    def gpu_eq(part, gpus, cpus):
        c = cpg.get(pool_of(part), cpg.get("default", 72))
        return max(float(gpus), (cpus or 0) / float(c))

    conf = {}
    for line in sec.get("config", "").splitlines():
        if "=" in line:
            k, v = [x.strip() for x in line.split("=", 1)]
            conf[k] = v
    weights = {
        "fs": float(conf.get("PriorityWeightFairShare", 10000) or 0),
        "age": float(conf.get("PriorityWeightAge", 1000) or 0),
        "max_age_s": parse_duration(conf.get("PriorityMaxAge", "7-00:00:00")) or 7 * 86400,
        "half_life_s": parse_duration(conf.get("PriorityDecayHalfLife", "7-00:00:00")) or 7 * 86400,
    }

    # balance
    balances = []
    for line in sec.get("accounts", "").splitlines():
        p = line.split()
        if len(p) >= 3 and p[1].lstrip("-").isdigit() and p[2].lstrip("-").isdigit():
            balances.append({"account": p[0], "balance": int(p[1]), "deposited": int(p[2])})

    # sprio decomposition
    sprio = {}
    for line in sec.get("sprio", "").splitlines():
        p = line.split("|")
        if len(p) >= 7:
            try:
                sprio[p[0].strip()] = {"prio": float(p[3]), "age": float(p[4]), "fs": float(p[5]), "part": float(p[6])}
            except ValueError:
                pass

    pools = set(cfg["pools"])
    pending = []
    for line in sec.get("pending", "").splitlines():
        p = line.split("|")
        if len(p) < 13:
            continue
        jid, part, user, acct, prio, nodes, cpus, gres, limit, submit, est, reason, name = p[:13]
        pool = pool_of(part.split(",")[0])
        if pool not in pools:
            continue
        nodes = int(nodes or 1)
        g = gpus_from(gres) * nodes
        base = jid.split("_")[0]
        sp = sprio.get(jid) or sprio.get(base)
        submit_t = tparse(submit)
        age0 = sp["age"] if sp else min(max(now - (submit_t or now), 0) / weights["max_age_s"], 1.0) * weights["age"]
        pending.append({
            "id": jid, "partition": part, "pool": pool, "user": user, "account": acct,
            "prio": float(prio or 0), "age0": age0, "nodes": nodes, "gpus": g,
            "eq": gpu_eq(pool, g, int(cpus or 0)), "limit_s": parse_duration(limit) or 172800,
            "submit": submit_t, "slurm_est": tparse(est), "reason": reason, "name": name,
            "count": array_count(jid) if "[" in jid else 1,
            "competing": reason.split(",")[0] in ("Priority", "Resources"),
            "mine": user == me, "part_w": sp["part"] if sp else None,
        })
    part_w = {}
    for j in pending:
        if j["part_w"] is not None:
            part_w.setdefault(j["partition"], []).append(j["part_w"])
    part_w = {k: quantile(v, 0.5) for k, v in part_w.items()}

    running = []
    for line in sec.get("running", "").splitlines():
        p = line.split()
        if len(p) < 10 or not p[0][0].isdigit():
            continue
        jid, user, acct, part, nodes, cpus, tres, limit, left, start = p[:10]
        g = gpus_from(tres)
        running.append({
            "id": jid, "user": user, "account": acct, "partition": part, "pool": pool_of(part),
            "nodes": int(nodes), "gpus": g, "eq": gpu_eq(part, g, int(cpus)),
            "billing": tres_value(tres, "billing"), "left_s": parse_duration(left) or 0,
            "limit_s": parse_duration(limit) or 0, "start": tparse(start), "mine": user == me,
        })

    history = []
    for line in sec.get("sacct_pool", "").splitlines():
        p = line.split("|")
        if len(p) < 10:
            continue
        jid, user, part, tres, limit, elapsed, state, elig, start, end = p[:10]
        st = tparse(start)
        if st is None or pool_of(part) not in pools:
            continue
        g = gpus_from(tres)
        history.append({
            "pool": pool_of(part), "user": user, "start": st, "elig": tparse(elig), "eq": gpu_eq(part, g, int(tres_value(tres, "cpu"))),
            "limit_s": parse_duration(limit), "elapsed_s": parse_duration(elapsed), "state": state,
        })

    mine_hist = []
    for line in sec.get("sacct_me", "").splitlines():
        p = line.split("|")
        if len(p) < 11:
            continue
        jid, name, part, tres, limit, elapsed, state, submit, elig, start, end = p[:11]
        mine_hist.append({
            "id": jid, "name": name, "partition": part, "gpus": gpus_from(tres), "limit": limit,
            "elapsed": elapsed, "state": state, "submit": tparse(submit), "eligible": tparse(elig),
            "start": tparse(start), "end": tparse(end),
        })

    mine = []
    for line in sec.get("mine", "").splitlines():
        p = line.split("|")
        if len(p) < 15:
            continue
        jid, part, name, state, prio, nodes, cpus, gres, limit, used, left, submit, start, reason, acct = p[:15]
        mine.append({
            "id": jid, "partition": part, "name": name, "state": state, "prio": float(prio or 0),
            "nodes": int(nodes or 1), "gpus": gpus_from(gres) * int(nodes or 1), "limit": limit,
            "limit_s": parse_duration(limit), "used_s": parse_duration(used), "left_s": parse_duration(left),
            "submit": tparse(submit), "start": tparse(start), "reason": reason, "account": acct,
            "sprio": sprio.get(jid),
        })

    logs = parse_logs(sec.get("logs", ""))
    tree = FairTree.parse(sec.get("sshare", ""))
    return {
        "now": now, "me": me, "weights": weights, "balances": balances, "pending": pending,
        "running": running, "history": history, "mine": mine, "mine_hist": mine_hist,
        "logs": logs, "tree": tree, "part_w": part_w,
    }


LOSS_RE = re.compile(r"loss[^0-9\-]{0,12}(-?[0-9]+\.?[0-9]*(?:[eE][-+]?\d+)?)", re.I)
STEP_RE = re.compile(r"(?:step|iter(?:ation)?)[^0-9]{0,6}(\d+)", re.I)


def parse_logs(text):
    out, cur, mode = {}, None, None
    for line in text.splitlines():
        if line.startswith("@@job "):
            cur = line.split()[1]
            out[cur] = {"path": "", "metrics": [], "tail": []}
            mode = None
        elif cur is None:
            continue
        elif line.startswith("@@path "):
            out[cur]["path"] = line[7:].strip()
        elif line == "@@metrics":
            mode = "metrics"
        elif line == "@@tail":
            mode = "tail"
        elif mode == "metrics":
            lm, sm = LOSS_RE.search(line), STEP_RE.search(line)
            if lm:
                out[cur]["metrics"].append({"step": int(sm.group(1)) if sm else None, "loss": float(lm.group(1))})
            elif sm:
                out[cur]["metrics"].append({"step": int(sm.group(1)), "loss": None})
        elif mode == "tail":
            out[cur]["tail"].append(line)
    for v in out.values():
        steps = [m["step"] for m in v["metrics"] if m["step"] is not None]
        v["last_step"] = max(steps) if steps else None
        losses = [m for m in v["metrics"] if m["loss"] is not None]
        v["loss_series"] = losses[-200:]
        v["last_loss"] = losses[-1]["loss"] if losses else None
        del v["metrics"]
    return out


# ---------------------------------------------------------------- Fair Tree

class FairTree(object):
    """Account/user tree from `sshare -l -a -P`, ranked like Slurm's Fair Tree."""

    def __init__(self):
        self.nodes = []       # dicts: name, user, depth, shares, raw, parent, children
        self.user_index = {}  # (account, user) -> node index

    @classmethod
    def parse(cls, text):
        t = cls()
        lines = text.splitlines()
        if not lines:
            return t
        header = lines[0].split("|")
        try:
            ia, iu, ins, iru, ifs = (header.index(k) for k in ("Account", "User", "NormShares", "RawUsage", "FairShare"))
        except ValueError:
            return t
        stack = []
        for line in lines[1:]:
            p = line.split("|")
            if len(p) <= max(ia, iu, ins, iru, ifs):
                continue
            raw_acct = p[ia]
            depth = len(raw_acct) - len(raw_acct.lstrip(" "))
            node = {
                "name": raw_acct.strip(), "user": p[iu].strip(), "depth": depth,
                "shares": float(p[ins] or 0), "raw": float(p[iru] or 0), "children": [],
                "parent": None, "fs_obs": float(p[ifs]) if p[ifs] else None,
            }
            while stack and t.nodes[stack[-1]]["depth"] >= depth:
                stack.pop()
            idx = len(t.nodes)
            if stack:
                node["parent"] = stack[-1]
                t.nodes[stack[-1]]["children"].append(idx)
            t.nodes.append(node)
            if node["user"]:
                t.user_index[(node["name"], node["user"])] = idx
            else:
                stack.append(idx)
        return t

    def user_count(self):
        return len(self.user_index)

    def factors(self, decay, adds):
        """adds: {node index of a user: extra usage}. Returns {user node idx: factor}."""
        n = len(self.nodes)
        if n == 0:
            return {}
        usage = [nd["raw"] * decay for nd in self.nodes]
        for idx, a in adds.items():
            i = idx
            while i is not None:
                usage[i] += a
                i = self.nodes[i]["parent"]
        out = {}
        total = self.user_count()
        rank = [total]

        def level_fs(i):
            nd = self.nodes[i]
            par = nd["parent"]
            pu = usage[par] if par is not None else usage[i]
            if usage[i] <= 0 or pu <= 0:
                return INF
            return nd["shares"] / (usage[i] / pu)

        def visit(i):
            kids = sorted(self.nodes[i]["children"], key=level_fs, reverse=True)
            for k in kids:
                if self.nodes[k]["user"]:
                    out[k] = rank[0] / float(total)
                    rank[0] -= 1
                else:
                    visit(k)

        roots = [i for i, nd in enumerate(self.nodes) if nd["parent"] is None]
        for r in roots:
            if self.nodes[r]["user"]:
                out[r] = rank[0] / float(total)
                rank[0] -= 1
            else:
                visit(r)
        return out

    def self_check(self):
        """Mean |model - sshare| fair-share factor at t=0, and the check for one user."""
        f = self.factors(1.0, {})
        errs = [abs(f[i] - self.nodes[i]["fs_obs"]) for i in f if self.nodes[i]["fs_obs"] is not None]
        return sum(errs) / len(errs) if errs else None


# ---------------------------------------------------------------- simulation

def runtime_ratios(history, pool):
    """Median (elapsed / time limit) of finished jobs, overall and per user."""
    per = {}
    allr = []
    for h in history:
        if h["pool"] != pool or not h["limit_s"] or h["elapsed_s"] is None or h["state"] in ("RUNNING", "PENDING"):
            continue
        r = min(h["elapsed_s"] / float(h["limit_s"]), 1.0)
        per.setdefault(h["user"], []).append(r)
        allr.append(r)
    overall = quantile(allr, 0.5) or 0.6
    users = {u: quantile(v, 0.5) for u, v in per.items() if len(v) >= 5}
    return overall, users


def pool_rates(history, pool, now, busy_now=None, window_h=6.0, span_h=72.0):
    """Work started per hour, in GPU-hours per hour, over 6 h windows.

    A job's work is GPU-equivalents x run time (actual run time when finished,
    time limit x the user's usual run/limit ratio when still running). In steady
    state, work started per hour equals the number of busy GPUs.
    """
    overall, users = runtime_ratios(history, pool)
    rows = [h for h in history if h["pool"] == pool and now - span_h * 3600 <= h["start"] <= now]
    nwin = int(span_h / window_h)
    wins = [0.0] * nwin
    quick, total = 0.0, 0.0
    for h in rows:
        if h["state"] == "RUNNING" or h["elapsed_s"] is None:
            run = max(h["elapsed_s"] or 0, (h["limit_s"] or 0) * users.get(h["user"], overall))
        else:
            run = h["elapsed_s"]
        work = h["eq"] * run / 3600.0
        k = int((now - h["start"]) / (window_h * 3600))
        if 0 <= k < nwin:
            wins[k] += work
        total += work
        if h["elig"] is not None and h["start"] - h["elig"] < 6 * 3600:
            quick += work
    rates = [w / window_h for w in wins]
    p25, p50, p75 = quantile(rates, 0.25) or 0.0, quantile(rates, 0.5) or 0.0, quantile(rates, 0.75) or 0.0
    # The GPUs busy right now are the best level estimate of the drain rate; the
    # history only sets how much it swings (slow / fast relative to typical).
    if busy_now and p50 > 0:
        typical, slow, fast = busy_now, busy_now * p25 / p50, busy_now * p75 / p50
    else:
        typical, slow, fast = p50, p25, p75
    return {
        "slow": slow, "typical": typical, "fast": fast, "hist_typical": p50, "busy_now": busy_now,
        "quick_share": quick / total if total else 0.0,
        "runtime_ratio": overall, "user_ratio": users, "n_starts": len(rows),
    }


def billing_per_eq(running, pool, fallback=1000.0):
    vals = [r["billing"] / r["eq"] for r in running if r["pool"] == pool and r["eq"] > 0 and r["billing"] > 0]
    return quantile(vals, 0.5) or fallback


def simulate(snap, pool, rate, extra_jobs=(), dt_h=0.25, horizon_h=14 * 24, fs_every_h=2.0, trace=False):
    """Returns {job id: start epoch or None} for the user's pending jobs (and extra jobs)."""
    W = snap["weights"]
    tree = snap["tree"]
    now = snap["now"]
    hl = W["half_life_s"] / 3600.0
    age_rate = W["age"] / (W["max_age_s"] / 3600.0)  # priority points per hour
    eff_rate = rate["rate"] * (1.0 - rate["quick_share"])
    run_ratio = rate["runtime_ratio"]
    user_ratio = rate.get("user_ratio", {})

    def work_h(j):
        return j["eq"] * j["limit_s"] / 3600.0 * user_ratio.get(j["user"], run_ratio)
    bpe = billing_per_eq(snap["running"], pool)

    def uidx(acct, user):
        return tree.user_index.get((acct, user))

    # usage streams: (user idx, billing/s, t_end_h); usage added = billing * seconds
    streams = []
    for r in snap["running"]:
        i = uidx(r["account"], r["user"])
        if i is not None and r["billing"] > 0:
            streams.append((i, r["billing"], r["left_s"] / 3600.0, 0.0))

    queue = []
    for j in list(snap["pending"]) + list(extra_jobs):
        if j["pool"] != pool or not (j["competing"] or j.get("hypothetical")):
            continue
        for _ in range(min(j["count"], 50)):
            q = dict(j)
            q["work"] = work_h(q)
            queue.append(q)
    if not queue:
        return {}, None

    f0 = tree.factors(1.0, {})
    targets = {j["id"] for j in queue if j["mine"] or j.get("hypothetical")}
    result = {jid: None for jid in targets}
    tr = {"t": [], "mine": {jid: [] for jid in targets}, "cut": [], "ahead": {jid: [] for jid in targets}}

    t = 0.0
    budget = 0.0
    cap = max(q["work"] for q in queue) + eff_rate * 24
    fs_now = f0
    next_fs = 0.0
    while t <= horizon_h and targets - {k for k, v in result.items() if v is not None}:
        if t >= next_fs:
            adds = {}
            for (i, b, t_end, t_start) in streams:
                # usage accrued in [t_start, min(t, t_end)], decayed to time t
                a, e = t_start, min(t, t_end)
                if e > a:
                    # integral of b * 2^-((t-s)/hl) ds, in seconds
                    k = math.log(2) / hl
                    adds[i] = adds.get(i, 0.0) + b * 3600.0 * (math.exp(-k * (t - e)) - math.exp(-k * (t - a))) / k
            fs_now = tree.factors(2.0 ** (-t / hl), adds)
            next_fs = t + fs_every_h
        for j in queue:
            i = uidx(j["account"], j["user"])
            dfs = (fs_now.get(i, 0.0) - f0.get(i, 0.0)) * W["fs"] if i is not None else 0.0
            dage = min(j["age0"] + age_rate * t, W["age"]) - j["age0"]
            j["p"] = j["prio"] + dfs + dage
        queue.sort(key=lambda j: -j["p"])
        if trace and abs(t - round(t)) < 1e-9:
            tr["t"].append(t)
            for jid in targets:
                mine = [q for q in queue if q["id"] == jid]
                if mine:
                    tr["mine"][jid].append(round(mine[0]["p"], 1))
                    tr["ahead"][jid].append(round(sum(q["work"] for q in queue if q["p"] > mine[0]["p"]), 1))
                else:
                    tr["mine"][jid].append(None)
                    tr["ahead"][jid].append(0)
        budget = min(budget + eff_rate * dt_h, cap)
        while queue and budget >= queue[0]["work"]:
            j = queue.pop(0)
            budget -= j["work"]
            if trace:
                tr["cut"].append([round(t, 2), round(j["p"], 1)])
            if j["id"] in result and result[j["id"]] is None:
                result[j["id"]] = now + t * 3600
            i = uidx(j["account"], j["user"])
            if i is not None:
                streams.append((i, j["eq"] * bpe, t + j["limit_s"] / 3600.0 * user_ratio.get(j["user"], run_ratio), t))
        t += dt_h
    return result, (tr if trace else None)


def predict(snap, cfg, calibration=None, extra_jobs=()):
    """Start ranges for the user's pending jobs (and hypothetical extra jobs)."""
    out = {"pools": {}, "jobs": {}}
    cal = calibration or 1.0
    wanted = {j["pool"] for j in snap["pending"] if j["mine"]} | {j["pool"] for j in extra_jobs}
    for pool in cfg["pools"]:
        busy = sum(x["eq"] for x in snap["running"] if x["pool"] == pool)
        r = pool_rates(snap["history"], pool, snap["now"], busy_now=busy)
        out["pools"][pool] = {k: v for k, v in r.items() if k != "user_ratio"}
        if pool not in wanted:
            continue
        res = {}
        trace = None
        for scen in ("fast", "typical", "slow"):
            rr = dict(r, rate=r[scen])
            starts, tr = simulate(snap, pool, rr, extra_jobs=extra_jobs, trace=(scen == "typical"))
            if tr:
                trace = tr
            for jid, s in starts.items():
                res.setdefault(jid, {})[scen] = s
        for jid, v in res.items():
            adj = {}
            for scen, s in v.items():
                adj[scen] = None if s is None else snap["now"] + (s - snap["now"]) * cal
            out["jobs"][jid] = {
                "pool": pool, "start": adj, "raw": v,
                "trace": {"t": trace["t"], "mine": trace["mine"].get(jid), "ahead": trace["ahead"].get(jid),
                          "cut": trace["cut"]} if trace else None,
            }
    return out


def hypothetical_job(snap, partition, gpus, hours, nodes=1, cpus=None, account=None):
    """A job the user might submit now, for the what-if box."""
    me = snap["me"]
    tree = snap["tree"]
    acct = account or next((a for (a, u) in tree.user_index if u == me and a != "noalloc"), "")
    f0 = tree.factors(1.0, {})
    i = tree.user_index.get((acct, me))
    fs_w = f0.get(i, 0.0) * snap["weights"]["fs"] if i is not None else 0.0
    part_w = snap["part_w"].get(partition, 0.0)
    return {
        "id": "what-if", "partition": partition, "pool": pool_of(partition), "user": me, "account": acct,
        "prio": fs_w + part_w, "age0": 0.0, "nodes": nodes, "gpus": gpus, "eq": float(gpus),
        "limit_s": hours * 3600, "count": 1, "competing": True, "mine": True, "hypothetical": True,
    }
