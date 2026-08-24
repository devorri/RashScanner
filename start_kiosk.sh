#!/bin/bash
# Rashilience Kiosk Launcher for Raspberry Pi
# Starts Flask server + Chromium in fullscreen kiosk mode

cd "$(dirname "$0")"

# Activate virtualenv if present
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

# Kill any existing instances
pkill -f "python3 app.py" 2>/dev/null
pkill -f "chromium" 2>/dev/null
sleep 1

# Start Flask server in background
python3 app.py --port 5000 &
FLASK_PID=$!
echo "Rashilience server started (PID: $FLASK_PID)"

# Wait until Flask server is actually responding
for i in {1..15}; do
  if curl -s http://localhost:5000/ > /dev/null 2>&1 || nc -z localhost 5000 2>/dev/null; then
    echo "Rashilience server is UP and ready!"
    break
  fi
  sleep 1
done

# Ensure display environment variable exists
export DISPLAY="${DISPLAY:-:0}"

# Detect Chromium binary name
CHROME=$(which chromium-browser 2>/dev/null || which chromium 2>/dev/null)
if [ -z "$CHROME" ]; then
  echo "Chromium not found. Install with: sudo apt install chromium"
  exit 1
fi

# Launch Chromium in strict 100% offline kiosk mode (no cloud sync, no telemetry, no networking)
$CHROME \
  --kiosk \
  --touch-events=enabled \
  --enable-virtual-keyboard \
  --noerrdialogs \
  --disable-infobars \
  --disable-session-crashed-bubble \
  --disable-restore-session-state \
  --disable-sync \
  --disable-background-networking \
  --disable-component-update \
  --disable-domain-reliability \
  --disable-features=TranslateUI,OptimizationHints,MediaRouter \
  --no-default-browser-check \
  --no-first-run \
  --disable-gpu-vsync \
  --log-level=3 \
  --incognito \
  http://localhost:5000 2>/dev/null &

echo "100% Offline Kiosk mode launched. Press Ctrl+C to stop."

# Wait for Flask process
wait $FLASK_PID
