"""Read-only data collection over SSH.

One ssh call per refresh. It reuses the ControlMaster socket from ~/.ssh/config,
so Duo is only needed when that socket has died. Nothing here submits, cancels
or changes anything on the cluster.
"""
import subprocess

REMOTE_SCRIPT = r"""
set +e
sec(){ echo "=====$1====="; }
sec date; date +%s; date +%z
sec user; echo $USER
sec config; scontrol show config | grep -E '^(PriorityWeight|PriorityMaxAge|PriorityDecayHalfLife|PriorityType|PriorityFlags)'
sec accounts; accounts 2>&1 | head -40
sec sshare; sshare -l -a -P 2>&1
sec pending; squeue -h -t PD -o '%i|%P|%u|%a|%Q|%D|%C|%b|%l|%V|%S|%r|%j' 2>&1
sec running; squeue -h -t R -O 'JobID:30,UserName:30,Account:40,Partition:30,NumNodes:8,NumCPUs:10,tres-alloc:200,TimeLimit:16,TimeLeft:16,StartTime:24' 2>&1
sec sprio; sprio -h -o '%i|%u|%r|%Y|%A|%F|%P' 2>&1
sec mine; squeue -u $USER -h -o '%i|%P|%j|%T|%Q|%D|%C|%b|%l|%M|%L|%V|%S|%r|%a' 2>&1
sec sacct_pool; sacct -a -X -P -n -S $(date -d '-3 days' +%FT%T) -r __PARTS__ -o JobID,User,Partition,AllocTRES%120,Timelimit,Elapsed,State,Eligible,Start,End 2>&1
sec sacct_me; sacct -u $USER -X -P -n -S $(date -d '-14 days' +%F) -o JobID,JobName,Partition,AllocTRES%120,Timelimit,Elapsed,State,Submit,Eligible,Start,End 2>&1
sec logs
for j in $(squeue -u $USER -h -t R -o %i); do
  echo "@@job $j"
  out=$(scontrol show job $j | grep -oP 'StdOut=\K\S+')
  echo "@@path $out"
  if [ -f "$out" ]; then
    echo "@@metrics"
    tail -n 5000 "$out" | grep -aiE 'loss|step|iter|epoch' | tail -n 400
    echo "@@tail"
    tail -n 25 "$out"
  fi
done
sec end
"""


def socket_alive(alias):
    """True when the ControlMaster connection for this alias is up."""
    try:
        r = subprocess.run(["ssh", "-O", "check", alias], capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def fetch(alias, partitions, timeout=180):
    """Run the read-only script on the cluster. Returns {section: text}."""
    script = REMOTE_SCRIPT.replace("__PARTS__", ",".join(partitions))
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", alias, "bash", "-s"],
        input=script.encode(), capture_output=True, timeout=timeout,
    )
    text = r.stdout.decode("utf-8", "replace")
    if "=====end=====" not in text:
        err = r.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError("ssh %s failed (exit %s): %s" % (alias, r.returncode, err[-400:]))
    return split_sections(text)


def split_sections(text):
    out, cur, buf = {}, None, []
    for line in text.splitlines():
        if line.startswith("=====") and line.endswith("=====") and len(line) > 10:
            if cur is not None:
                out[cur] = "\n".join(buf)
            cur, buf = line.strip("="), []
        else:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf)
    return out
