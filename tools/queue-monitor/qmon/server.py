"""Queue Monitor: a local web server that the Mac app window shows.

Run:  python3 -m qmon.server   (from tools/queue-monitor)
Serves http://127.0.0.1:8765 . Python 3.9 standard library only.
"""
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from qmon import archive, bench, calib, collect, model  # noqa: E402

DATA_DIR = os.path.expanduser("~/Library/Application Support/QueueMonitor")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")

DEFAULT_CONFIG = {
    "port": 8765,
    "refresh_minutes": 10,
    "clusters": [
        {
            "name": "DeltaAI", "alias": "deltaai",
            "pools": ["ghx4"],
            "partitions": ["ghx4", "ghx4-interactive"],
            "cpus_per_gpu": {"ghx4": 72, "default": 72},
            "whatif_partitions": ["ghx4", "ghx4-interactive"],
        },
        {
            "name": "Delta", "alias": "delta",
            "pools": ["gpuA40x4", "gpuA100x4", "gpuA100x8", "gpuH200x8"],
            "partitions": ["gpuA40x4", "gpuA40x4-interactive", "gpuA40x4-preempt",
                           "gpuA100x4", "gpuA100x4-interactive", "gpuA100x4-preempt",
                           "gpuA100x8", "gpuA100x8-interactive", "gpuH200x8", "gpuH200x8-interactive"],
            "cpus_per_gpu": {"gpuA40x4": 16, "gpuA100x4": 16, "gpuA100x8": 16, "gpuH200x8": 12, "default": 16},
            "whatif_partitions": ["gpuA40x4", "gpuA100x4", "gpuH200x8", "gpuA40x4-preempt"],
        },
    ],
}


def load_config():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    for k, v in DEFAULT_CONFIG.items():
        cfg.setdefault(k, v)
    return cfg


CONFIG = load_config()
STORE = calib.Store(os.path.join(DATA_DIR, "history.sqlite"))
STATE = {}        # cluster name -> dict for the UI
SNAPS = {}        # cluster name -> parsed snapshot (for what-if)
LOCK = threading.Lock()
LAST_SEC = {}     # cluster name -> raw sections of the last refresh
BENCH = {}        # cluster name -> latest backtest results
BENCH_STATUS = {}
BUSY = set()


# ---------------------------------------------------------------- refresh

def refresh(cl):
    name = cl["name"]
    with LOCK:
        if name in BUSY:
            return
        BUSY.add(name)
    st = dict(STATE.get(name, {}), name=name, alias=cl["alias"], refreshing=True)
    STATE[name] = st
    try:
        alive = collect.socket_alive(cl["alias"])
        st["socket"] = alive
        if not alive:
            raise RuntimeError("SSH 主连接已断开，需要重新登录（Duo）。")
        sec = collect.fetch(cl["alias"], cl["partitions"])
        snap = model.parse_snapshot(sec, cl)
        fn, report = STORE.calibrator(name, snap["now"])
        preds = model.predict(snap, cl, calibrator=fn, log_all=True)
        STORE.record(name, snap, preds)
        SNAPS[name] = snap
        LAST_SEC[name] = sec
        try:
            archive.maybe_save(DATA_DIR, name, sec, snap["now"])
        except Exception:
            traceback.print_exc()
        st.update(build_view(cl, snap, preds, report))
        st["error"] = None
        st["updated"] = time.time()
    except Exception as e:  # keep the last good data on screen
        st["error"] = str(e)
        traceback.print_exc()
    finally:
        st["refreshing"] = False
        st["last_try"] = time.time()
        STATE[name] = st
        with LOCK:
            BUSY.discard(name)


def build_view(cl, snap, preds, report):
    tree = snap["tree"]
    me = snap["me"]
    my_fs = []
    f0 = tree.factors(1.0, {})
    for (acct, user), idx in tree.user_index.items():
        if user == me and acct != "noalloc":
            my_fs.append({"account": acct, "observed": tree.nodes[idx]["fs_obs"], "model": f0.get(idx),
                          "users": tree.user_count()})
    jobs = []
    for j in snap["mine"]:
        p = preds["jobs"].get(j["id"])
        log = snap["logs"].get(j["id"])
        jobs.append({
            "id": j["id"], "name": j["name"], "partition": j["partition"], "state": j["state"],
            "gpus": j["gpus"], "nodes": j["nodes"], "limit": j["limit"], "limit_s": j["limit_s"],
            "used_s": j["used_s"], "left_s": j["left_s"], "submit": j["submit"], "start": j["start"],
            "reason": j["reason"], "prio": j["prio"], "sprio": j["sprio"],
            "pred": p["start"] if p else None, "trace": p["trace"] if p else None,
            "calibrated": p["calibrated"] if p else False,
            "log": log,
        })
    hist = sorted(snap["mine_hist"], key=lambda r: r["submit"] or 0, reverse=True)[:40]
    competing = [x for x in snap["pending"] if x["competing"]]
    return {
        "now": snap["now"], "me": me, "balances": snap["balances"], "fairshare": my_fs,
        "model_check": tree.self_check(), "pools": preds["pools"], "jobs": jobs, "history": hist,
        "queue": {"pending": len(snap["pending"]), "competing": len(competing),
                  "running": len(snap["running"])},
        "accuracy": report,
        "weights": snap["weights"], "whatif_partitions": cl.get("whatif_partitions", []),
    }


def whatif(name, partition, gpus, hours):
    if name not in SNAPS:
        return {"error": "还没有这台集群的数据，请先刷新。"}
    snap = SNAPS[name]
    cl = next(c for c in CONFIG["clusters"] if c["name"] == name)
    nodes = max(1, -(-gpus // 4)) if gpus > 4 else 1
    job = model.hypothetical_job(snap, partition, gpus, hours, nodes=nodes)
    if job["pool"] not in cl["pools"]:
        return {"error": "这个分区不在预测范围内：%s" % partition}
    fn, _ = STORE.calibrator(name, snap["now"])
    preds = model.predict(snap, cl, calibrator=fn, extra_jobs=[job])
    p = preds["jobs"].get("what-if")
    return {"partition": partition, "gpus": gpus, "hours": hours, "now": snap["now"],
            "start": p["start"] if p else None, "prio": job["prio"]}


def loop():
    while True:
        for cl in CONFIG["clusters"]:
            refresh(cl)
        time.sleep(max(60, CONFIG["refresh_minutes"] * 60))


def load_bench():
    for cl in CONFIG["clusters"]:
        p = os.path.join(DATA_DIR, "bench", "%s_results.json" % cl["name"])
        if os.path.exists(p):
            try:
                with open(p) as f:
                    BENCH[cl["name"]] = json.load(f)
            except ValueError:
                pass


def bench_loop():
    """Re-run the backtest once a day (first run a few minutes after start)."""
    load_bench()
    time.sleep(300)
    while True:
        for cl in CONFIG["clusters"]:
            name = cl["name"]
            last = BENCH.get(name, {}).get("generated", 0)
            if time.time() - last < 20 * 3600 or name not in LAST_SEC:
                continue
            BENCH_STATUS[name] = "running"
            try:
                snap = model.parse_snapshot(LAST_SEC[name], cl)   # private copy of the tree
                BENCH[name] = bench.run(cl, DATA_DIR, snap, log=lambda *a: None)
                BENCH_STATUS[name] = "ok"
            except Exception as e:
                BENCH_STATUS[name] = "error: %s" % e
                traceback.print_exc()
        time.sleep(1800)


def open_login(alias):
    cmd = "ssh %s true && echo '登录成功，可以关掉这个窗口。'" % alias
    script = 'tell application "Terminal"\nactivate\ndo script "%s"\nend tell' % cmd.replace('"', '\\"')
    subprocess.Popen(["osascript", "-e", script])


# ---------------------------------------------------------------- http

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, default=str).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "ui", "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if self.path.startswith("/api/state"):
            return self._send(200, {"clusters": [dict(STATE.get(c["name"], {"name": c["name"], "alias": c["alias"]}),
                                                      bench=BENCH.get(c["name"]), bench_status=BENCH_STATUS.get(c["name"]))
                                                 for c in CONFIG["clusters"]],
                                    "refresh_minutes": CONFIG["refresh_minutes"], "server_time": time.time()})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        # Only accept requests from our own page (simple CSRF guard).
        origin = self.headers.get("Origin", "")
        if origin and not origin.startswith("http://127.0.0.1"):
            return self._send(403, {"error": "forbidden"})
        body = self._json_body()
        name = body.get("cluster")
        cl = next((c for c in CONFIG["clusters"] if c["name"] == name), None)
        if self.path == "/api/refresh":
            for c in ([cl] if cl else CONFIG["clusters"]):
                threading.Thread(target=refresh, args=(c,), daemon=True).start()
            return self._send(200, {"ok": True})
        if self.path == "/api/login" and cl:
            open_login(cl["alias"])
            return self._send(200, {"ok": True})
        if self.path == "/api/whatif" and cl:
            try:
                return self._send(200, whatif(name, body["partition"], int(body["gpus"]), float(body["hours"])))
            except Exception as e:
                return self._send(200, {"error": str(e)})
        return self._send(404, {"error": "not found"})


def main():
    port = int(os.environ.get("QMON_PORT", CONFIG["port"]))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=loop, daemon=True).start()
    threading.Thread(target=bench_loop, daemon=True).start()
    print("Queue Monitor on http://127.0.0.1:%d" % port, flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
