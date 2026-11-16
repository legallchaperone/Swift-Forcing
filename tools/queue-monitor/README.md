# Queue Monitor

A small Mac app that watches Simon's Slurm jobs on DeltaAI and Delta and predicts
when pending jobs will start. It connects with plain `ssh` (the aliases `deltaai`
and `delta` in `~/.ssh/config`) and does not need Claude Code. It is read-only:
it never submits, cancels or changes anything on the clusters.

## Install

```bash
cd tools/queue-monitor
bash build_app.sh          # needs Xcode command line tools (swiftc)
open ~/Applications/"Queue Monitor.app"
```

To keep collecting accuracy data while the window is closed, run the server as a
background login item (the window then reuses it):

```bash
bash install_agent.sh            # undo: bash install_agent.sh --remove
```

After rebuilding the app, run `install_agent.sh` again so the background server
picks up the new code.

Without swiftc you can run the same thing in a browser:
`python3 -m qmon.server` and open http://127.0.0.1:8765 .

## What it shows

- SU balance, fair-share factor, queue size, and the calibration factor.
- For each pending job: earliest / most likely / latest start, Slurm's own
  estimate, the priority breakdown, a chart of the job's priority over time against
  the priority of jobs being started, and the work queued ahead of it.
- For each running job: progress against the time limit, the last step and loss
  found in its log, a loss curve, and the last log lines.
- A "what if I submit now" box: partition, GPUs, hours -> start range.
- The jobs of the last 14 days, with how long each one waited.

It refreshes every 10 minutes while the app is open. Data, settings and the
prediction log live in `~/Library/Application Support/QueueMonitor/`
(`config.json` holds the clusters, partitions and refresh interval).

## Duo

The app relies on the SSH ControlMaster connection (`ControlPersist 7d`). When
that connection has dropped, the app shows a button that opens Terminal with
`ssh <alias> true`. Approve Duo there once, then press Refresh.

## How the prediction works

`priority(t) = priority_now + W_fs * (fairshare(t) - fairshare_now) + (age(t) - age_now)`

- **Fair share** follows Slurm's Fair Tree: every account and user gets
  `LevelFS = shares / (usage / parent usage)`, users are ranked down the tree, and
  the factor is rank / number of users. Usage halves every `PriorityDecayHalfLife`;
  running jobs, and jobs the simulation starts, add usage.
- **Age** grows linearly up to `PriorityWeightAge` over `PriorityMaxAge`.
- **Drain rate**: the pool hands out work (GPU-hours) at the rate it is used now
  (busy GPUs). Slow / fast use the 25th / 75th percentile of the last 3 days in
  6-hour windows. The share of work that starts within 6 hours of becoming
  eligible (new high-priority jobs, backfill) is taken off the top. Each job's
  work is GPUs x time limit x that user's usual run-time / time-limit ratio.
- **Calibration** (`qmon/calib.py`, data in `history.sqlite`): every refresh logs
  the prediction for every pending job on the cluster (your jobs every refresh,
  others hourly), and real start times come from squeue and sacct. Per bucket of
  predicted remaining wait (0-6 h, 6-24 h, 1-3 d, >3 d), the ratio real / predicted
  wait is estimated with Kaplan-Meier (jobs still waiting count as "at least this
  long"). Its 10/50/90% quantiles turn the typical simulation into the shown range.
  A bucket needs 40 started jobs before it is used; until then the range comes
  from the slow / fast simulations.
- **Accuracy**: half of the job ids are held out. On predictions from the last
  7 days the app reports the median error of the raw model, of the calibrated
  prediction and of Slurm's own estimate, and how often the real start fell inside
  the range (target 80%).

Known limits: node fragmentation (a 4-GPU job needs a whole free node) and
reservations are not modelled; jobs held or waiting on dependencies are ignored
until they become eligible.
