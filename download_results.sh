#!/usr/bin/env bash
# Download results from Vast.ai instance
#
# Usage: bash download_results.sh HOST PORT
#   HOST: Vast.ai instance IP
#   PORT: SSH port

HOST="${1:?Usage: bash download_results.sh HOST PORT}"
PORT="${2:?Usage: bash download_results.sh HOST PORT}"

echo "Downloading results from $HOST:$PORT..."

rsync -avz -e "ssh -p $PORT" \
    "root@$HOST:/workspace/nko-brain-scanner/results/" \
    ./results/

echo ""
echo "Results downloaded to ./results/"
ls -la results/
echo ""
ls -la results/figures/ 2>/dev/null || echo "No figures yet."
ls -la results/heatmaps/ 2>/dev/null || echo "No heatmaps yet."
ls -la results/activation_profiles/ 2>/dev/null || echo "No activation profiles yet."
