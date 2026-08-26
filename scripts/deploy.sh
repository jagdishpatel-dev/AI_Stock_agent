#!/bin/bash
# Deploy latest main to the Kali VM app directory and restart services.
# Requires passwordless sudo for systemctl (see docs or /etc/sudoers.d/trading-deploy).

set -euo pipefail

APP_DIR="/home/kali/AIportfolio/AI_Stock_agent"
USER_NAME="kali"

cd "$APP_DIR"

echo "==> Fetch latest code"
git fetch origin main
git reset --hard origin/main

echo "==> Python dependencies"
# shellcheck disable=SC1091
source "$APP_DIR/.venv/bin/activate"
pip install -q -r requirements.txt

echo "==> Frontend build"
cd "$APP_DIR/frontend"
npm ci
npm run build
cd "$APP_DIR"

echo "==> Restart dashboard"
sudo systemctl restart "trading-dashboard@${USER_NAME}.service"

# Skip agent restart during market session (8:30 AM–3:00 PM CST, swing hours)
TZ=America/Chicago
HOUR=$(date +%H)
MIN=$(date +%M)
NOW_MINS=$((10#$HOUR * 60 + 10#$MIN))
START=$((8 * 60 + 30))
END=$((15 * 60 + 0))

if [ "$NOW_MINS" -ge "$START" ] && [ "$NOW_MINS" -le "$END" ]; then
  echo "==> Market session active — skipping agent restart"
else
  echo "==> Restart trading agent"
  sudo systemctl restart "trading-agent@${USER_NAME}.service"
fi

echo "==> Service status"
# systemctl restart returns immediately; the service can sit in "activating"
# for a moment before systemd reports "active", which was failing this check
# (and the whole deploy) on services that were actually starting up fine.
wait_for_active() {
  local svc="$1"
  for _ in $(seq 1 10); do
    if systemctl is-active --quiet "$svc"; then
      echo "$svc: active"
      return 0
    fi
    sleep 1
  done
  echo "$svc: $(systemctl is-active "$svc" || true)"
  return 1
}

wait_for_active "trading-dashboard@${USER_NAME}.service"
wait_for_active "trading-agent@${USER_NAME}.service"

echo "==> Deploy complete"
