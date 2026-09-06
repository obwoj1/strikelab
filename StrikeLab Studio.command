#!/usr/bin/env bash
# Double-click this file in Finder to start StrikeLab Studio.
# It starts the server, opens your browser, and prints the address to use
# from a phone on the same network.

cd "$(dirname "$0")" || exit 1

PORT="${STRIKELAB_PORT:-7878}"
PYTHON=".venv/bin/python"

if [ ! -x "$PYTHON" ]; then
  echo "No virtual environment found in $(pwd)/.venv"
  echo
  echo "Set one up first:"
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/pip install -e '.[dev,video,web,yolo]'"
  echo
  read -r -p "Press return to close."
  exit 1
fi

if ! "$PYTHON" -c "import fastapi" >/dev/null 2>&1; then
  echo "The web extra is not installed. Installing it now..."
  "$PYTHON" -m pip install -q -e '.[web]' || {
    echo "Install failed. Run: .venv/bin/pip install -e '.[web]'"
    read -r -p "Press return to close."
    exit 1
  }
fi

# Reuse an already-running server rather than failing on a port clash.
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "StrikeLab Studio is already running on port $PORT."
  open "http://localhost:$PORT"
  echo
  read -r -p "Press return to close this window (the server keeps running)."
  exit 0
fi

LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "")"

echo "──────────────────────────────────────────────"
echo "  StrikeLab Studio"
echo "──────────────────────────────────────────────"
echo "  On this Mac:  http://localhost:$PORT"
[ -n "$LAN_IP" ] && echo "  On your phone: http://$LAN_IP:$PORT"
echo
echo "  There is no password on this. Anyone on your"
echo "  network can open it while it is running."
echo
echo "  Press Control-C to stop."
echo "──────────────────────────────────────────────"
echo

( sleep 2; open "http://localhost:$PORT" ) &

exec "$PYTHON" -m strikelab serve --host 0.0.0.0 --port "$PORT"
