"""Retrain the learned start-time model for one cluster.

Steps: (1) one read-only sacct call for 60 days of history, at most once every 6 days;
(2) rebuild the queue every 3 hours and compute features (qmon.trainset, standard library);
(3) fit the model with the learning environment's Python (qmon.learn: numpy, pandas,
scikit-learn) and write bench/<cluster>_model.json.

The learning environment is <data dir>/venv, made by setup_learning.sh. Without it the
app keeps using the calibrated rule model.

usage: python3 -m qmon.retrain <cluster> [--no-fetch]
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from qmon import archive, bench, mlmodel  # noqa: E402

DATA_DIR = os.path.expanduser("~/Library/Application Support/QueueMonitor")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
VENV_PY = os.path.join(DATA_DIR, "venv", "bin", "python")
HISTORY_DAYS = 60
MAX_AGE_S = 7 * 86400


def available():
    return os.path.exists(VENV_PY)


def model_age(name):
    p = mlmodel.default_path(DATA_DIR, name)
    return time.time() - os.path.getmtime(p) if os.path.exists(p) else None


def run(cl, fetch=True, log=print):
    name = cl["name"]
    if not available():
        raise RuntimeError("learning environment missing: run setup_learning.sh")
    bdir = os.path.join(DATA_DIR, "bench")
    os.makedirs(bdir, exist_ok=True)
    hp = os.path.join(bdir, "%s_sacct%d.txt.gz" % (name, HISTORY_DAYS))
    if fetch and (not os.path.exists(hp) or time.time() - os.path.getmtime(hp) > 6 * 86400):
        log("fetching %d days of sacct history" % HISTORY_DAYS)
        bench.fetch_history(cl["alias"], cl["partitions"], HISTORY_DAYS, hp, timeout=1800)
    if not os.path.exists(hp):
        raise RuntimeError("no sacct history at %s" % hp)
    snaps = archive.list_snapshots(DATA_DIR, name)
    if not snaps:
        raise RuntimeError("no archived snapshot yet")
    table = os.path.join(bdir, "%s_train.pkl" % name)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "qmon.trainset", "--cluster", name, "--config", CONFIG_PATH,
                        "--sacct", hp, "--snap", snaps[-1], "--out", table, "--procs", "3"],
                       cwd=ROOT, capture_output=True, timeout=4 * 3600)
    if r.returncode != 0:
        raise RuntimeError("trainset failed: %s" % r.stderr.decode()[-500:])
    log("training table built in %ds" % (time.time() - t0))
    out = mlmodel.default_path(DATA_DIR, name)
    r = subprocess.run([VENV_PY, "-m", "qmon.learn", table, out + ".tmp"], cwd=ROOT, capture_output=True,
                       timeout=3600, env=dict(os.environ, PYTHONPATH=ROOT))
    if r.returncode != 0:
        raise RuntimeError("learn failed: %s" % r.stderr.decode()[-500:])
    os.replace(out + ".tmp", out)
    log(r.stdout.decode().strip())
    # time-split study: weeks 2-7 before the fetch (the last week has unfinished waits), each week predicted by models trained only on earlier weeks
    study = os.path.join(bdir, "%s_study.json" % name)
    r = subprocess.run([VENV_PY, "-m", "qmon.study", table, "--folds", "2,3,4,5,6,7", "--fold-days", "7",
                        "--models", "rule,cal2,mix3", "--json", study + ".tmp"], cwd=ROOT, capture_output=True,
                       timeout=3600, env=dict(os.environ, PYTHONPATH=ROOT))
    if r.returncode == 0:
        os.replace(study + ".tmp", study)
    else:
        log("study failed: %s" % r.stderr.decode()[-300:])
    return out


if __name__ == "__main__":
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    cl = next(c for c in cfg["clusters"] if c["name"] == sys.argv[1])
    print(run(cl, fetch="--no-fetch" not in sys.argv))
