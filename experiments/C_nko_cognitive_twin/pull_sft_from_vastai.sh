#!/usr/bin/env bash
# pull_sft_from_vastai.sh — Pull cognitive twin SFT data from Vast.ai instance 33195812
#
# Once the twin training completes on the Vast.ai GPU instance, this script:
#   1. Checks what training artifacts exist on the instance
#   2. Pulls SFT JSONL data to local staging directory
#   3. Pulls any trained adapter/checkpoint files
#   4. Verifies the downloaded data
#
# Usage:
#   ./pull_sft_from_vastai.sh              # Pull everything
#   ./pull_sft_from_vastai.sh --check-only # Just check what's there
#   ./pull_sft_from_vastai.sh --adapters   # Also pull adapter weights

set -euo pipefail

# ── Configuration ──────────────────────────────────────────
VAST_KEY="$HOME/.ssh/id_vastai"
VAST_PORT=35812
VAST_HOST="root@ssh3.vast.ai"
VAST_SSH="ssh -i $VAST_KEY -p $VAST_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15"
VAST_SCP="scp -i $VAST_KEY -P $VAST_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15"

LOCAL_STAGING="/tmp/twin_sft_data"
EXPERIMENT_DIR="$(cd "$(dirname "$0")" && pwd)"
FINAL_DATA_DIR="$EXPERIMENT_DIR/data"

# ── Parse args ─────────────────────────────────────────────
CHECK_ONLY=false
PULL_ADAPTERS=false
for arg in "$@"; do
    case "$arg" in
        --check-only) CHECK_ONLY=true ;;
        --adapters)   PULL_ADAPTERS=true ;;
        --help|-h)
            echo "Usage: $0 [--check-only] [--adapters]"
            echo ""
            echo "  --check-only   List files on Vast.ai without downloading"
            echo "  --adapters     Also pull trained adapter weights"
            echo ""
            echo "Vast.ai instance: 33195812"
            echo "SSH: ssh -i ~/.ssh/id_vastai -p $VAST_PORT $VAST_HOST"
            exit 0
            ;;
    esac
done

# ── Verify SSH key exists ──────────────────────────────────
if [ ! -f "$VAST_KEY" ]; then
    echo "ERROR: Vast.ai SSH key not found at $VAST_KEY"
    echo "Generate or retrieve the key for instance 33195812"
    exit 1
fi

echo "============================================"
echo "Vast.ai SFT Data Pull — Instance 33195812"
echo "============================================"
echo ""

# ── Step 1: Check connectivity ────────────────────────────
echo "[1/4] Checking connectivity to Vast.ai instance..."
if ! $VAST_SSH $VAST_HOST 'echo "CONNECTION_OK"' 2>/dev/null; then
    echo ""
    echo "ERROR: Cannot connect to Vast.ai instance."
    echo "Possible causes:"
    echo "  - Instance 33195812 is not running (check vastai CLI: vastai show instances)"
    echo "  - SSH port $VAST_PORT has changed (check Vast.ai dashboard)"
    echo "  - Instance IP/host has changed from ssh3.vast.ai"
    echo ""
    echo "To find current connection details:"
    echo "  vastai show instances | grep 33195812"
    echo "  vastai ssh-url 33195812"
    exit 1
fi
echo "  Connected."

# ── Step 2: Discover training artifacts ────────────────────
echo ""
echo "[2/4] Discovering training artifacts on instance..."
echo ""
echo "--- JSONL files ---"
$VAST_SSH $VAST_HOST 'find /workspace -name "*.jsonl" -type f 2>/dev/null | head -30' || true
echo ""
echo "--- SFT-related files ---"
$VAST_SSH $VAST_HOST 'find /workspace -name "*sft*" -o -name "*train*" -o -name "*twin*" 2>/dev/null | head -30' || true
echo ""
echo "--- Checkpoint/adapter files ---"
$VAST_SSH $VAST_HOST 'find /workspace -name "*.safetensors" -o -name "*.bin" -o -name "adapter_config.json" 2>/dev/null | head -20' || true
echo ""
echo "--- Disk usage ---"
$VAST_SSH $VAST_HOST 'du -sh /workspace/*/ 2>/dev/null | head -10' || true

if $CHECK_ONLY; then
    echo ""
    echo "CHECK ONLY mode. No files downloaded."
    echo "Re-run without --check-only to pull data."
    exit 0
fi

# ── Step 3: Pull data ─────────────────────────────────────
echo ""
echo "[3/4] Pulling SFT data to $LOCAL_STAGING..."
mkdir -p "$LOCAL_STAGING"

# Pull all JSONL files from /workspace/data/ (primary location)
echo "  Pulling /workspace/data/*.jsonl..."
$VAST_SCP "$VAST_HOST:/workspace/data/*.jsonl" "$LOCAL_STAGING/" 2>/dev/null || {
    echo "  No files at /workspace/data/*.jsonl, trying alternative paths..."
    # Try common alternative locations
    $VAST_SCP "$VAST_HOST:/workspace/*.jsonl" "$LOCAL_STAGING/" 2>/dev/null || true
    $VAST_SCP "$VAST_HOST:/workspace/output/*.jsonl" "$LOCAL_STAGING/" 2>/dev/null || true
    $VAST_SCP "$VAST_HOST:/workspace/training/*.jsonl" "$LOCAL_STAGING/" 2>/dev/null || true
    $VAST_SCP "$VAST_HOST:/root/*.jsonl" "$LOCAL_STAGING/" 2>/dev/null || true
}

# Pull training logs if they exist
echo "  Pulling training logs..."
$VAST_SCP "$VAST_HOST:/workspace/*.log" "$LOCAL_STAGING/" 2>/dev/null || true
$VAST_SCP "$VAST_HOST:/workspace/training_log.txt" "$LOCAL_STAGING/" 2>/dev/null || true

# Pull adapter weights if requested
if $PULL_ADAPTERS; then
    echo "  Pulling adapter weights..."
    ADAPTER_LOCAL="$LOCAL_STAGING/adapters"
    mkdir -p "$ADAPTER_LOCAL"
    # Try common adapter paths
    $VAST_SCP -r "$VAST_HOST:/workspace/adapters/" "$ADAPTER_LOCAL/" 2>/dev/null || true
    $VAST_SCP -r "$VAST_HOST:/workspace/output/adapter*" "$ADAPTER_LOCAL/" 2>/dev/null || true
    $VAST_SCP -r "$VAST_HOST:/workspace/checkpoint*" "$ADAPTER_LOCAL/" 2>/dev/null || true
fi

# ── Step 4: Verify and stage ──────────────────────────────
echo ""
echo "[4/4] Verifying downloaded data..."

JSONL_COUNT=$(find "$LOCAL_STAGING" -name "*.jsonl" -type f | wc -l | tr -d ' ')
echo "  JSONL files found: $JSONL_COUNT"

if [ "$JSONL_COUNT" -eq 0 ]; then
    echo ""
    echo "WARNING: No JSONL files were pulled."
    echo "The training may not have produced output yet, or files are in an unexpected location."
    echo "Run with --check-only to inspect the instance filesystem."
    exit 1
fi

# Show what we got
echo ""
echo "Downloaded files:"
find "$LOCAL_STAGING" -type f -exec ls -lh {} \;

# Count lines / validate JSON
echo ""
echo "File statistics:"
for f in "$LOCAL_STAGING"/*.jsonl; do
    [ -f "$f" ] || continue
    lines=$(wc -l < "$f" | tr -d ' ')
    valid=$(python3 -c "
import json, sys
count = 0
with open('$f') as fh:
    for line in fh:
        try:
            json.loads(line.strip())
            count += 1
        except: pass
print(count)
" 2>/dev/null || echo "?")
    echo "  $(basename "$f"): $lines lines, $valid valid JSON"
done

# Copy to experiment data directory
echo ""
echo "Staging to $FINAL_DATA_DIR..."
mkdir -p "$FINAL_DATA_DIR"
cp "$LOCAL_STAGING"/*.jsonl "$FINAL_DATA_DIR/" 2>/dev/null || true

echo ""
echo "============================================"
echo "PULL COMPLETE"
echo "============================================"
echo "  Staging:    $LOCAL_STAGING"
echo "  Final:      $FINAL_DATA_DIR"
echo "  JSONL files: $JSONL_COUNT"
echo ""
echo "Next steps:"
echo "  1. Translate to N'Ko:"
echo "     python3 translate_sft_to_nko.py \\"
echo "         --input $FINAL_DATA_DIR/<sft_file>.jsonl \\"
echo "         --output $FINAL_DATA_DIR/nko_sft.jsonl"
echo ""
echo "  2. Train N'Ko adapter:"
echo "     python3 train_nko_adapter.py \\"
echo "         --input $FINAL_DATA_DIR/nko_sft.jsonl \\"
echo "         --base-model mlx-community/gemma-3-1b-it-4bit \\"
echo "         --output-adapter adapters-nko"
echo ""
echo "  3. Compare twins:"
echo "     python3 compare_twins.py \\"
echo "         --model mlx-community/gemma-3-1b-it-4bit \\"
echo "         --english-adapter <path-to-english-adapter> \\"
echo "         --nko-adapter adapters-nko \\"
echo "         --prompts eval_prompts.jsonl"
