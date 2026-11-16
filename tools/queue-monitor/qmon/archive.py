"""Archive of raw live snapshots, for replaying new algorithms later.

Every 2 hours the raw sections of a refresh are saved gzip-compressed. The big
`sshare` section is saved at most every 6 hours; other snapshots point to the
latest saved one. Snapshots older than KEEP_DAYS are deleted.
"""
import glob
import gzip
import json
import os
import time

EVERY_S = 2 * 3600
SSHARE_EVERY_S = 6 * 3600
KEEP_DAYS = 30
SKIP = ("logs", "sacct_me")


def _dir(data_dir, cluster):
    d = os.path.join(data_dir, "snapshots", cluster)
    os.makedirs(d, exist_ok=True)
    return d


def list_snapshots(data_dir, cluster):
    return sorted(glob.glob(os.path.join(_dir(data_dir, cluster), "snap-*.json.gz")))


def maybe_save(data_dir, cluster, sec, now):
    d = _dir(data_dir, cluster)
    snaps = list_snapshots(data_dir, cluster)
    last_t = int(os.path.basename(snaps[-1])[5:-8]) if snaps else 0
    if now - last_t < EVERY_S:
        return None
    shares = sorted(glob.glob(os.path.join(d, "sshare-*.txt.gz")))
    last_share_t = int(os.path.basename(shares[-1])[7:-7]) if shares else 0
    if now - last_share_t >= SSHARE_EVERY_S:
        last_share_t = int(now)
        with gzip.open(os.path.join(d, "sshare-%d.txt.gz" % last_share_t), "wt") as f:
            f.write(sec.get("sshare", ""))
    body = {k: v for k, v in sec.items() if k not in SKIP and k != "sshare"}
    body["_sshare_ref"] = "sshare-%d.txt.gz" % last_share_t
    path = os.path.join(d, "snap-%d.json.gz" % int(now))
    with gzip.open(path, "wt") as f:
        json.dump(body, f)
    cutoff = now - KEEP_DAYS * 86400
    for p in glob.glob(os.path.join(d, "*.gz")):
        try:
            t = int(os.path.basename(p).split("-")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        if t < cutoff:
            os.remove(p)
    return path


def load(path):
    with gzip.open(path, "rt") as f:
        sec = json.load(f)
    ref = os.path.join(os.path.dirname(path), sec.pop("_sshare_ref", ""))
    if os.path.exists(ref):
        with gzip.open(ref, "rt") as f:
            sec["sshare"] = f.read()
    else:
        sec["sshare"] = ""
    return sec
