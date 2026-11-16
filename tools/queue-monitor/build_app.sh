#!/bin/bash
# Build "Queue Monitor.app" and install it to ~/Applications.
# Needs the Xcode command line tools (swiftc). Run on the Mac:  bash build_app.sh
set -euo pipefail
cd "$(dirname "$0")"
APP="build/Queue Monitor.app"
rm -rf build && mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
swiftc -O app/main.swift -o "$APP/Contents/MacOS/QueueMonitor" -framework Cocoa -framework WebKit
rsync -a --exclude '__pycache__' qmon "$APP/Contents/Resources/"
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Queue Monitor</string>
  <key>CFBundleDisplayName</key><string>Queue Monitor</string>
  <key>CFBundleIdentifier</key><string>local.simon.queuemonitor</string>
  <key>CFBundleExecutable</key><string>QueueMonitor</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
</dict></plist>
PLIST
codesign --force --deep -s - "$APP" >/dev/null 2>&1 || true
mkdir -p "$HOME/Applications"
rm -rf "$HOME/Applications/Queue Monitor.app"
cp -R "$APP" "$HOME/Applications/"
echo "Installed: $HOME/Applications/Queue Monitor.app"
