"""Build a (test point, pending job) feature table from reconstructed sacct history.

usage: python3 research/build_ds.py <cluster> <sacct.gz> <step_h> <out.pkl>   (run from tools/queue-monitor)
Needs <raw>/deltaai.txt (one raw snapshot) and <raw>/sshare_deltaai.txt; <raw> is $QP_RAW (default ../qp_raw).
Python 3.9 standard library only.
"""
import math, os, pickle, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.environ.get('QP_RAW', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'qp_raw'))
from qmon import model, bench, calib, features  # noqa
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


PREV_H = 6.0     # the cut line looks at jobs that started in the last 6 hours
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
    T_prev = T - PREV_H * 3600
    sp = bench.reconstruct(T_prev, JOBS, tree, weights, part_w, SNAP0["me"])
    prev = (T_prev, {x["id"]: x["prio"] for x in sp["pending"]})
    fts = features.compute(snap, cfg, pr["all"], JOBS, prev=prev)
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
