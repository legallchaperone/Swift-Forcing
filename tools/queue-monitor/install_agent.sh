#!/bin/bash
# Keep the Queue Monitor server running in the background (login item via launchd),
# so it keeps logging predictions and real start times while the window is closed.
# The app window reuses this server. Undo with:  bash install_agent.sh --remove
set -euo pipefail
LABEL=local.simon.queuemonitor
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
RES="$HOME/Applications/Queue Monitor.app/Contents/Resources"
LOG="$HOME/Library/Application Support/QueueMonitor/server.log"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
if [ "${1:-}" = "--remove" ]; then
  rm -f "$PLIST"; echo "Removed $LABEL"; exit 0
fi
[ -d "$RES/qmon" ] || { echo "Build the app first: bash build_app.sh"; exit 1; }
mkdir -p "$HOME/Library/LaunchAgents" "$(dirname "$LOG")"
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>/usr/bin/python3</string><string>-m</string><string>qmon.server</string>
  </array>
  <key>WorkingDirectory</key><string>$RES</string>
  <key>EnvironmentVariables</key><dict>
    <key>PYTHONPATH</key><string>$RES</string>
    <key>PYTHONDONTWRITEBYTECODE</key><string>1</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ProcessType</key><string>Background</string>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict></plist>
PLIST
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Installed $LABEL (server on http://127.0.0.1:8765)"
