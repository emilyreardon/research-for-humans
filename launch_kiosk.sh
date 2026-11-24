#!/bin/bash
# Launch the Research for Humans kiosk on the second monitor (800x480 at -800,0).
# Run this once; it starts the server if needed, then opens Chrome in app mode.

PROJ="$(cd "$(dirname "$0")" && pwd)"   # this folder
VENV="${VENV:-$HOME/Desktop/isls_env}"     # your Python environment; set VENV to use another

# Start server if nothing is listening on 5050
if ! lsof -i :5050 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Starting server..."
    source "$VENV/bin/activate"
    cd "$PROJ"
    python app.py &
    # Wait until the server accepts connections (max 10 s)
    for i in $(seq 1 20); do
        sleep 0.5
        lsof -i :5050 -sTCP:LISTEN -t >/dev/null 2>&1 && break
    done
    echo "Server ready."
fi

# Open Chrome in app mode pinned to the second monitor (left display, 800x480)
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
    --app=http://localhost:5050 \
    --window-position=-800,0 \
    --window-size=800,480 \
    --no-first-run \
    --disable-infobars \
    --disable-session-crashed-bubble \
    --disable-translate \
    --noerrdialogs \
    --overscroll-history-navigation=0 \
    --disable-features=TouchpadOverscrollHistoryNavigation \
    2>/dev/null &

echo "Kiosk open on second monitor."
