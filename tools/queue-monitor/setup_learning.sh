#!/bin/bash
# One-time setup for the learned start-time model: a private Python environment with
# numpy, pandas and scikit-learn in the app's data folder (nothing is installed system-wide).
# The app retrains the model once a week with it. Undo: delete the venv folder below.
set -euo pipefail
DATA="$HOME/Library/Application Support/QueueMonitor"
mkdir -p "$DATA"
/usr/bin/python3 -m venv "$DATA/venv"
"$DATA/venv/bin/pip" install -q --upgrade pip
"$DATA/venv/bin/pip" install -q numpy pandas scikit-learn
"$DATA/venv/bin/python" -c "import sklearn; print('learning environment ready, scikit-learn', sklearn.__version__)"
