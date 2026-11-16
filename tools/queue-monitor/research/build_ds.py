"""Build a (test point, pending job) feature table from reconstructed sacct history.

usage: python3 research/build_ds.py <cluster> <sacct.gz> <step_h> <out.pkl>   (run from tools/queue-monitor)
Needs <raw>/deltaai.txt (one raw snapshot) and <raw>/sshare_deltaai.txt; <raw> is $QP_RAW (default ../qp_raw).
Python 3.9 standard library only.
"""
import math, os, pickle, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.environ.get('QP_RAW', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'qp_raw'))
from qmon import model, bench, calib  # noqa
from qmon.collect import split_sections  # noqa

CL = sys.argv[1]
HPATH = sys.argv[2]
STEP = float(sys.argv[3])
OUT = sys.argv[4]

CFGS = {
    "DeltaAI": {"name": "DeltaAI", "alias": "x", "pools": ["ghx4"], "partitions": ["ghx4", "ghx4-interactive"],
                "cpus_per_gpu": {"ghx4": 72, "default": 72}},
    "Delta": {"name": "Delta", "alias": "x", "pools": ["gpuA40x4", "gpuA100x4", "gpuA100x8", "gpuH200x8"],
              "partitions": [], "cpus_per_gpu": {"gpuA40x4": 16, "gpuA100x4": 16, "gpuA100x8": 16, "gpuH200x8": 12,
                                                 "default": 16}},
}
cfg = CFGS[CL]


def base_snap():
    """Current config, weights and account tree, from one raw snapshot of the cluster."""
    raw=open(os.path.join(RAW, 'deltaai.txt')).read()
    s=split_sections(raw)
    d=s['date'].split()
    sec={}
    sec['date']=d[1]+"\n-0600"
    sec['user']='wchen38'
    sec['config']=s['config']; sec['accounts']=s['accounts']; sec['sshare']=s['sshare_all']
    sec['pending']="\n".join(l+"|x" for l in s['pending_all'].splitlines())
    run=[]
    for l in s['running_all'].splitlines():
        p=l.split('|')
        if len(p)<11: continue
        jid,part,user,nodes,cpus,gres,lim,used,left,start,nl=p[:11]
        g=model.gpus_from(gres)*int(nodes)
        run.append(" ".join([jid,user,"acct_"+user,part,nodes,cpus,"cpu=%s,gres/gpu=%d,billing=%d"%(cpus,g,max(g*1000,int(cpus)*1000//72)),lim,left,start]))
    sec['running']="\n".join(run)
    sec['sprio']=s['sprio_all']
    sec['mine']=s['myjobs']
    sp=[]
    for l in s['sacct_part'].splitlines()[1:]:
        p=l.split('|')
        jid,user,acct,part,nn,tres,lim,el,st,sub,elig,start,end=p[:13]
        sp.append("|".join([jid,user,part,tres,lim,el,st,elig,start,end]))
    sec['sacct_pool']="\n".join(sp)
    sec['sacct_me']="\n".join("|".join([q[0],q[1],q[2],q[4],q[7],q[6],q[8],q[9],q[10],q[11],q[12]]) for q in (l.split('|') for l in s['sacct_me'].splitlines()[1:]))
    sec['logs']=''
    sec['sshare'] = open(os.path.join(RAW, 'sshare_deltaai.txt')).read()
    return model.parse_snapshot(sec, cfg)


SNAP0 = None
JOBS = None
JBP = None
FETCHED = None


def init():
    global SNAP0, JOBS, JBP, FETCHED
    SNAP0 = SNAP0 or base_snap()
    FETCHED, JOBS = bench.load_history(HPATH, cfg)
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
    allp = pr["all"]
    hist = bench.cand_hist(snap, JBP)
    # per-user / pool state
    urun_eq, urun_n = {}, {}
    pool_busy = {}
    for r in snap["running"]:
        urun_eq[(r["user"], r["pool"])] = urun_eq.get((r["user"], r["pool"]), 0) + r["eq"]
        urun_n[(r["user"], r["pool"])] = urun_n.get((r["user"], r["pool"]), 0) + 1
        pool_busy[r["pool"]] = pool_busy.get(r["pool"], 0) + r["eq"]
    upend = {}
    pool_pend = {}
    for j in snap["pending"]:
        upend.setdefault((j["user"], j["pool"]), []).append(j)
        pool_pend[j["pool"]] = pool_pend.get(j["pool"], 0) + j["eq"]
    # recent starts of each user (last 24h) and recent waits per user (3 d)
    ustart24, uwait = {}, {}
    pstart24 = {}
    for h in snap["history"]:
        if T - h["start"] < 86400:
            k = (h["user"], h["pool"])
            ustart24[k] = ustart24.get(k, 0) + h["eq"]
            pstart24[h["pool"]] = pstart24.get(h["pool"], 0) + h["eq"]
        if T - h["start"] < 3 * 86400 and h["elig"]:
            uwait.setdefault((h["user"], h["pool"]), []).append(h["start"] - h["elig"])
    # max concurrent eq per user over last 7 days (proxy for a per-user limit)
    # starts in the last 6 h / 24 h per user and per array (job id before "_")
    ust6, ust24, ast24, ast6 = {}, {}, {}, {}
    for h in snap["history"]:
        age = T - h["start"]
        k = (h["user"], h["pool"])
        base = h["id"].split("_")[0]
        if age < 6 * 3600:
            ust6[k] = ust6.get(k, 0) + 1
            ast6[base] = ast6.get(base, 0) + 1
        if age < 86400:
            ust24[k] = ust24.get(k, 0) + 1
            ast24[base] = ast24.get(base, 0) + 1
    # pending tasks per array, sorted by task index
    arr = {}
    for j in snap["pending"]:
        if "_" in j["id"]:
            b, t = j["id"].split("_", 1)
            try:
                arr.setdefault(b, []).append(int(t))
            except ValueError:
                pass
    for b in arr:
        arr[b].sort()
    # waits of recent jobs by (size class, limit class), KM conditioned on time already waited
    def lc(lim_s):
        return 0 if lim_s <= 2 * 3600 else 1 if lim_s <= 8 * 3600 else 2 if lim_s <= 24 * 3600 else 3
    lim_obs = {}
    for h in JBP.get(cfg["pools"][0], ()) if len(cfg["pools"]) == 1 else []:
        if h["elig"] is None or not (T - 3 * 86400 <= h["elig"] < T):
            continue
        key = (model.size_class(h["eq"]), lc(h["limit_s"]))
        if h["start"] is not None and h["start"] <= T:
            lim_obs.setdefault(key, []).append((h["start"] - h["elig"], 1))
        elif h["start"] is not None or h["end"] is None or h["end"] > T:
            lim_obs.setdefault(key, []).append((T - h["elig"], 0))
    _c = calib
    lim_cache = {}
    rows = []
    lt = time.localtime(T)
    for j in snap["pending"]:
        a = allp.get(j["id"])
        if not a or a.get("typical") is None:
            continue
        k = (j["user"], j["pool"])
        mine_pend = upend.get(k, [])
        ahead_same_user = sum(1 for x in mine_pend if (x["prio"], -x["elig"]) > (j["prio"], -j["elig"]))
        ahead_same_user_eq = sum(x["eq"] for x in mine_pend if (x["prio"], -x["elig"]) > (j["prio"], -j["elig"]))
        hq = hist.get(j["id"])
        uw = sorted(uwait.get(k, []))
        base = j["id"].split("_")[0]
        a_ahead = None
        if "_" in j["id"] and base in arr:
            try:
                a_ahead = sum(1 for t in arr[base] if t < int(j["id"].split("_", 1)[1]))
            except ValueError:
                a_ahead = None
        w0 = T - j["elig"]
        lkey = (model.size_class(j["eq"]), lc(j["limit_s"]), int(w0 // (6 * 3600)) if w0 < 48 * 3600 else 8)
        if lkey not in lim_cache:
            cond = [(w - w0, e) for (w, e) in lim_obs.get(lkey[:2], []) if w > w0]
            q = _c.km_quantiles(cond, (0.1, 0.5, 0.9)) if sum(e for _, e in cond) >= 5 else None
            lim_cache[lkey] = q
        lq = lim_cache[lkey]
        rows.append({
            "u_st6": ust6.get(k, 0), "u_st24": ust24.get(k, 0),
            "a_ahead": a_ahead, "a_pend": len(arr.get(base, [])) if "_" in j["id"] else None,
            "a_st6": ast6.get(base, 0), "a_st24": ast24.get(base, 0),
            "lh10": (lq[0] / 3600.0 if lq and lq[0] is not None else None),
            "lh50": (lq[1] / 3600.0 if lq and lq[1] is not None else None),
            "lh90": (lq[2] / 3600.0 if lq and lq[2] is not None else None),
            "T": T, "id": j["id"], "user": j["user"], "account": j["account"], "pool": j["pool"],
            "partition": j["partition"],
            "truth": j["_truth"], "end": j["_end"], "state": j["_state"],
            "rule_fa": (a["fast"] - T) / 3600.0 if a.get("fast") else None,
            "rule_ty": (a["typical"] - T) / 3600.0,
            "rule_sl": (a["slow"] - T) / 3600.0 if a.get("slow") else None,
            "rank": a["rank"], "ahead": a["ahead"],
            "hist10": (hq[0] - T) / 3600.0 if hq else None,
            "hist50": (hq[1] - T) / 3600.0 if hq else None,
            "hist90": (hq[2] - T) / 3600.0 if hq else None,
            "waited_h": (T - j["elig"]) / 3600.0, "since_submit_h": (T - j["submit"]) / 3600.0 if j["submit"] else None,
            "eq": j["eq"], "nodes": j["nodes"], "limit_h": j["limit_s"] / 3600.0,
            "interactive": int("interactive" in j["partition"]), "preempt": int("preempt" in j["partition"]),
            "prio": j["prio"], "fs": (j["prio"] - j["age0"] - j["part_w"]) / weights["fs"], "age0": j["age0"],
            "pool_busy": pool_busy.get(j["pool"], 0), "pool_pend_eq": pool_pend.get(j["pool"], 0),
            "pool_start24": pstart24.get(j["pool"], 0),
            "u_run_eq": urun_eq.get(k, 0), "u_run_n": urun_n.get(k, 0),
            "u_pend_n": len(mine_pend), "u_pend_eq": sum(x["eq"] for x in mine_pend),
            "u_ahead_n": ahead_same_user, "u_ahead_eq": ahead_same_user_eq,
            "u_start24": ustart24.get(k, 0),
            "u_wait_med": (uw[len(uw) // 2] / 3600.0) if uw else None, "u_wait_n": len(uw),
            "is_array": int("_" in j["id"]),
            "hour": lt.tm_hour + lt.tm_min / 60.0, "wday": lt.tm_wday,
        })
    return rows


def work(T):
    try:
        return feats_at(T)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    init()
    t_first = min(j["end"] for j in JOBS if j["end"])   # sacct -S cut-off (jobs that ended before it are missing)
    t0 = t_first + 4 * 86400
    t1 = FETCHED - 0.5 * 86400
    pts = []
    T = t0
    while T <= t1:
        pts.append(T)
        T += STEP * 3600
    print(CL, len(JOBS), "jobs", len(pts), "points", flush=True)
    st = time.time()
    with Pool(4, initializer=init) as p:
        out = []
        for i, r in enumerate(p.imap(work, pts)):
            out.extend(r)
            if i % 20 == 0:
                print(i, len(out), round(time.time() - st), flush=True)
    with open(OUT, "wb") as f:
        pickle.dump({"fetched": FETCHED, "rows": out, "points": pts}, f)
    print("done", len(out), round(time.time() - st))
