#!/bin/bash
# Rashilience Kiosk Launcher for Raspberry Pi
# Starts Flask server + Chromium in fullscreen kiosk mode

cd ~/Rashilience
source venv/bin/activate

# Kill any existing instances
pkill -f "python3 app.py" 2>/dev/null
sleep 1

# Start Flask server in background
python3 app.py --port 5000 &
FLASK_PID=$!
echo "Rashilience server started (PID: $FLASK_PID)"

# Wait for server to be ready
sleep 3

# Set display for Pi's local screen (needed when running via SSH)
export DISPLAY=:0

# Detect Chromium binary name
CHROME=$(which chromium-browser 2>/dev/null || which chromium 2>/dev/null)
if [ -z "$CHROME" ]; then
  echo "Chromium not found. Install with: sudo apt install chromium"
  exit 1
fi

# Launch Chromium in kiosk mode (fullscreen, no toolbar)
$CHROME \
  --kiosk \
  --noerrdialogs \
  --disable-infobars \
  --disable-session-crashed-bubble \
  --disable-restore-session-state \
  --incognito \
  http://localhost:5000 &

echo "Kiosk mode launched. Press Ctrl+C to stop."

# Wait for Flask process
wait $FLASK_PID
