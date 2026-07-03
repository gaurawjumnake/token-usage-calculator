#!/usr/bin/env bash
# Nightly restart of the app container — run via cron on the EC2 host.
# Usage: scripts/restart_docker.sh (run from the app root, e.g. /home/ubuntu/app)
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[$(date -Is)] restarting app container"
sudo docker compose down
sudo docker compose up -d
echo "[$(date -Is)] restart complete"
sudo docker compose ps