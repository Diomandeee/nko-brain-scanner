#!/usr/bin/env bash
#
# Deploy Experiment B (CTC Script Advantage) to Vast.ai
# =====================================================
#
# Uploads training scripts, installs dependencies, and starts both
# training runs on a Vast.ai A100 instance.
#
# Usage:
#   ./deploy_to_vastai.sh <HOST> <PORT>
#
# Example:
#   ./deploy_to_vastai.sh ssh4.vast.ai 12345
#
# Prerequisites:
#   - Vast.ai instance running with SSH access
#   - ~/.ssh/id_vastai key exists
#   - Feature files already extracted (or will be extracted on-GPU)
#
# The script will:
#   1. Upload all experiment scripts
#   2. Install Python dependencies
#   3. Prepare data (download bam-asr-early, extract features, generate pairs)
#   4. Run N'Ko CTC training (50 epochs)
#   5. Run Latin CTC training (50 epochs)
#   6. Run comparison evaluation
#   7. Download results back to local machine

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────

HOST="${1:?Usage: $0 <HOST> <PORT>}"
PORT="${2:?Usage: $0 <HOST> <PORT>}"
SSH_KEY="$HOME/.ssh/id_vastai"
SSH_OPTS="-i $SSH_KEY -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10"
SCP_OPTS="-i $SSH_KEY -P $PORT -o StrictHostKeyChecking=no"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DIR="/workspace/experiment_b"
FEATURES_DIR="/workspace/features"
DATA_DIR="/workspace/data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()  { echo -e "${GREEN}[OK]${NC} $*"; }
warn(){ echo -e "${YELLOW}[WARN]${NC} $*"; }
err() { echo -e "${RED}[ERROR]${NC} $*"; }

ssh_cmd() {
    ssh $SSH_OPTS "root@$HOST" "$@"
}

scp_to() {
    scp $SCP_OPTS "$1" "root@$HOST:$2"
}

scp_from() {
    scp $SCP_OPTS "root@$HOST:$1" "$2"
}

# ── Validate ──────────────────────────────────────────────────────────

log "Experiment B: CTC Script Advantage Deployment"
log "Target: root@$HOST:$PORT"

if [ ! -f "$SSH_KEY" ]; then
    err "SSH key not found: $SSH_KEY"
    exit 1
fi

log "Testing SSH connection..."
if ! ssh_cmd "echo 'Connection OK'" 2>/dev/null; then
    err "Cannot connect to $HOST:$PORT"
    exit 1
fi
ok "SSH connection established"

# Check GPU
log "Checking GPU..."
GPU_INFO=$(ssh_cmd "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>/dev/null || echo "unknown")
ok "GPU: $GPU_INFO"

# ── Step 1: Upload scripts ───────────────────────────────────────────

log "Creating remote directories..."
ssh_cmd "mkdir -p $REMOTE_DIR $FEATURES_DIR $DATA_DIR /workspace/checkpoints/nko /workspace/checkpoints/latin /workspace/results"

log "Uploading experiment scripts..."
scp_to "$SCRIPT_DIR/train_nko_ctc.py" "$REMOTE_DIR/"
scp_to "$SCRIPT_DIR/train_latin_ctc.py" "$REMOTE_DIR/"
scp_to "$SCRIPT_DIR/compare_scripts.py" "$REMOTE_DIR/"
ok "Scripts uploaded"

# ── Step 2: Install dependencies ─────────────────────────────────────

log "Installing Python dependencies..."
ssh_cmd "pip install --quiet torch transformers datasets numpy 2>&1 | tail -5"
ok "Dependencies installed"

# ── Step 3: Create data preparation script ────────────────────────────

log "Creating data preparation script on remote..."
ssh_cmd "cat > $REMOTE_DIR/prepare_data.py << 'PREPARE_EOF'
#!/usr/bin/env python3
\"\"\"
Prepare data for Experiment B:
  1. Download bam-asr-early from HuggingFace
  2. Extract Whisper Large V3 encoder features
  3. Generate N'Ko pairs (via bridge/transliteration)
  4. Generate Latin pairs (from Whisper output)
  5. Generate aligned test set
\"\"\"
import json
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np


def check_existing_features(features_dir):
    \"\"\"Check if features already exist from a prior run.\"\"\"
    feat_dir = Path(features_dir)
    if feat_dir.exists():
        count = len(list(feat_dir.glob(\"*.pt\")))
        if count > 0:
            print(f\"Found {count} existing feature files in {features_dir}\")
            return count
    return 0


def extract_features(features_dir, cache_dir=\"/workspace/hf_data\"):
    \"\"\"Extract Whisper Large V3 encoder features from bam-asr-early.\"\"\"
    import whisper
    from datasets import load_dataset

    device = \"cuda\" if torch.cuda.is_available() else \"cpu\"
    feat_dir = Path(features_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)

    # Check existing
    existing = check_existing_features(features_dir)
    if existing > 1000:
        print(f\"Sufficient features exist ({existing}). Skipping extraction.\")
        return existing

    print(\"Loading bam-asr-early dataset...\")
    ds = load_dataset(\"RobotsMali/bam-asr-early\", cache_dir=cache_dir)
    train_data = ds[\"train\"]
    print(f\"Dataset loaded: {len(train_data)} train samples\")

    print(\"Loading Whisper large-v3 for feature extraction...\")
    model = whisper.load_model(\"large-v3\", device=device)
    model.eval()

    extracted = 0
    t0 = time.time()

    for i in range(len(train_data)):
        feat_path = feat_dir / f\"{i}.pt\"
        if feat_path.exists():
            extracted += 1
            continue

        try:
            audio = train_data[i][\"audio\"]
            audio_array = torch.tensor(audio[\"array\"], dtype=torch.float32)
            sr = audio[\"sampling_rate\"]

            if sr != 16000:
                ratio = 16000 / sr
                audio_array = torch.nn.functional.interpolate(
                    audio_array.unsqueeze(0).unsqueeze(0),
                    size=int(len(audio_array) * ratio),
                    mode=\"linear\", align_corners=False
                ).squeeze()

            audio_padded = whisper.pad_or_trim(audio_array)
            mel = whisper.log_mel_spectrogram(audio_padded, n_mels=128).unsqueeze(0).to(device)

            with torch.no_grad():
                features = model.encoder(mel)

            torch.save(features.squeeze(0).cpu(), feat_path)
            extracted += 1

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(train_data) - i - 1) / rate / 60
                print(f\"  Extracted {i+1}/{len(train_data)} ({rate:.1f}/s, ETA {eta:.0f}m)\")

        except Exception as e:
            print(f\"  [warn] Sample {i}: {e}\")

    print(f\"Feature extraction complete: {extracted} features\")
    del model
    torch.cuda.empty_cache()
    return extracted


def generate_pairs(features_dir, data_dir, cache_dir=\"/workspace/hf_data\"):
    \"\"\"Generate training pairs for both N'Ko and Latin.\"\"\"
    from datasets import load_dataset

    feat_dir = Path(features_dir)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    nko_path = data_dir / \"nko_pairs.jsonl\"
    latin_path = data_dir / \"latin_pairs.jsonl\"
    test_path = data_dir / \"test_pairs_aligned.jsonl\"

    # Check if pairs already exist
    if nko_path.exists() and latin_path.exists() and test_path.exists():
        nko_count = sum(1 for _ in open(nko_path))
        latin_count = sum(1 for _ in open(latin_path))
        test_count = sum(1 for _ in open(test_path))
        if nko_count > 100 and latin_count > 100:
            print(f\"Pairs already exist: {nko_count} N'Ko, {latin_count} Latin, {test_count} test\")
            return

    print(\"Loading dataset for transcription pairing...\")
    ds = load_dataset(\"RobotsMali/bam-asr-early\", cache_dir=cache_dir)
    train_data = ds[\"train\"]

    available = set(int(p.stem) for p in feat_dir.glob(\"*.pt\"))
    print(f\"Available features: {len(available)}\")

    # Build N'Ko transliteration map (Latin Bambara -> N'Ko)
    # This is a simplified version. The full bridge is in ~/Desktop/NKo/nko/
    NKO_MAP = {}
    # N'Ko digits U+07C0-U+07C9
    nko_letters = {
        'a': '\\u07CA', 'ba': '\\u07CB', 'da': '\\u07CC',
        'e': '\\u07CE', 'fa': '\\u07DA', 'ga': '\\u07DB',
        'ha': '\\u07DC', 'i': '\\u07CF', 'ja': '\\u07DD',
        'ka': '\\u07DE', 'la': '\\u07DF', 'ma': '\\u07E0',
        'na': '\\u07E1', 'o': '\\u07D0', 'pa': '\\u07E2',
        'ra': '\\u07E5', 'sa': '\\u07E6', 'ta': '\\u07E7',
        'u': '\\u07D1', 'wa': '\\u07E8', 'ya': '\\u07E9',
        'nya': '\\u07EA', 'gba': '\\u07EB',
    }

    def latin_to_nko_simple(text):
        \"\"\"Simple Latin->N'Ko transliteration for experiment purposes.\"\"\"
        # For the controlled experiment, we use the raw transcription text.
        # The key insight is that bam-asr-early has 'sentence' field in Latin
        # Bambara, and we need N'Ko equivalents.
        # For this experiment, we use the sentence field directly as Latin,
        # and a transliterated version as N'Ko.
        result = []
        i = 0
        t = text.lower()
        while i < len(t):
            matched = False
            # Try 3-char, 2-char, then 1-char
            for length in [3, 2, 1]:
                chunk = t[i:i+length]
                if chunk in nko_letters:
                    result.append(nko_letters[chunk])
                    i += length
                    matched = True
                    break
            if not matched:
                if t[i] == ' ':
                    result.append(' ')
                i += 1
        return ''.join(result)

    nko_pairs = []
    latin_pairs = []
    all_pairs = []

    for i in range(len(train_data)):
        if i not in available:
            continue

        sample = train_data[i]
        latin_text = sample.get(\"sentence\", \"\").strip()
        if not latin_text:
            continue

        # Latin pair: use the sentence as-is
        latin_pair = {
            \"feat_id\": str(i),
            \"latin\": latin_text.lower(),
            \"text\": latin_text.lower(),
        }
        latin_pairs.append(latin_pair)

        # N'Ko pair: transliterate
        nko_text = latin_to_nko_simple(latin_text)
        if nko_text.strip():
            nko_pair = {
                \"feat_id\": str(i),
                \"nko\": nko_text,
            }
            nko_pairs.append(nko_pair)

        # Aligned pair for comparison
        all_pairs.append({
            \"feat_id\": str(i),
            \"nko\": nko_text,
            \"latin\": latin_text.lower(),
        })

    # Split: last 10% as test set for comparison
    import random
    random.seed(42)
    random.shuffle(all_pairs)
    test_split = int(len(all_pairs) * 0.1)
    test_pairs = all_pairs[:test_split]

    # Write pairs
    with open(nko_path, \"w\") as f:
        for p in nko_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + \"\\n\")

    with open(latin_path, \"w\") as f:
        for p in latin_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + \"\\n\")

    with open(test_path, \"w\") as f:
        for p in test_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + \"\\n\")

    print(f\"Generated pairs: {len(nko_pairs)} N'Ko, {len(latin_pairs)} Latin, {len(test_pairs)} test\")


if __name__ == \"__main__\":
    features_dir = sys.argv[1] if len(sys.argv) > 1 else \"/workspace/features\"
    data_dir = sys.argv[2] if len(sys.argv) > 2 else \"/workspace/data\"

    print(\"=\" * 70)
    print(\"Experiment B: Data Preparation\")
    print(\"=\" * 70)

    # Step 1: Extract features
    print(\"\\n--- Feature Extraction ---\")
    extract_features(features_dir)

    # Step 2: Generate pairs
    print(\"\\n--- Pair Generation ---\")
    generate_pairs(features_dir, data_dir)

    print(\"\\n--- Done ---\")
PREPARE_EOF
chmod +x $REMOTE_DIR/prepare_data.py"
ok "Data preparation script created"

# ── Step 4: Create the master run script ──────────────────────────────

log "Creating master run script on remote..."
ssh_cmd "cat > $REMOTE_DIR/run_experiment_b.sh << 'RUN_EOF'
#!/usr/bin/env bash
#
# Master script for Experiment B on Vast.ai
# Runs data prep, both training arms, and comparison.
#
set -euo pipefail

WORK_DIR=\"/workspace/experiment_b\"
FEATURES_DIR=\"/workspace/features\"
DATA_DIR=\"/workspace/data\"
NKO_CKPT_DIR=\"/workspace/checkpoints/nko\"
LATIN_CKPT_DIR=\"/workspace/checkpoints/latin\"
RESULTS_DIR=\"/workspace/results\"
EPOCHS=50
BATCH_SIZE=16

log() { echo \"[$(date '+%Y-%m-%d %H:%M:%S')] $*\"; }

log \"========================================\"
log \"Experiment B: CTC Script Advantage\"
log \"========================================\"
log \"GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)\"
log \"CUDA: $(python3 -c 'import torch; print(torch.version.cuda)')\"

# ── Phase 1: Data Preparation ──
log \"\"
log \"--- Phase 1: Data Preparation ---\"
if [ -f \"$DATA_DIR/nko_pairs.jsonl\" ] && [ -f \"$DATA_DIR/latin_pairs.jsonl\" ]; then
    NKO_COUNT=$(wc -l < \"$DATA_DIR/nko_pairs.jsonl\")
    LATIN_COUNT=$(wc -l < \"$DATA_DIR/latin_pairs.jsonl\")
    log \"Data already exists: $NKO_COUNT N'Ko pairs, $LATIN_COUNT Latin pairs\"
    if [ \"$NKO_COUNT\" -gt 100 ] && [ \"$LATIN_COUNT\" -gt 100 ]; then
        log \"Skipping data preparation.\"
    else
        log \"Insufficient data. Re-running preparation.\"
        python3 \"$WORK_DIR/prepare_data.py\" \"$FEATURES_DIR\" \"$DATA_DIR\" 2>&1 | tee -a /workspace/experiment_b_data.log
    fi
else
    python3 \"$WORK_DIR/prepare_data.py\" \"$FEATURES_DIR\" \"$DATA_DIR\" 2>&1 | tee -a /workspace/experiment_b_data.log
fi

# Verify data exists
if [ ! -f \"$DATA_DIR/nko_pairs.jsonl\" ]; then
    log \"ERROR: N'Ko pairs not generated. Aborting.\"
    exit 1
fi
if [ ! -f \"$DATA_DIR/latin_pairs.jsonl\" ]; then
    log \"ERROR: Latin pairs not generated. Aborting.\"
    exit 1
fi

# ── Phase 2: Train N'Ko CTC ──
log \"\"
log \"--- Phase 2: N'Ko CTC Training ---\"
RESUME_FLAG=\"\"
if [ -f \"$NKO_CKPT_DIR/checkpoint.pt\" ]; then
    log \"Found existing N'Ko checkpoint. Resuming.\"
    RESUME_FLAG=\"--resume\"
fi

python3 \"$WORK_DIR/train_nko_ctc.py\" \
    --features-dir \"$FEATURES_DIR\" \
    --pairs \"$DATA_DIR/nko_pairs.jsonl\" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --checkpoint-dir \"$NKO_CKPT_DIR\" \
    --checkpoint-interval 500 \
    --log /workspace/experiment_b_nko.log \
    $RESUME_FLAG \
    2>&1 | tee -a /workspace/experiment_b_nko.log

log \"N'Ko training complete.\"

# ── Phase 3: Train Latin CTC ──
log \"\"
log \"--- Phase 3: Latin CTC Training ---\"
RESUME_FLAG=\"\"
if [ -f \"$LATIN_CKPT_DIR/checkpoint.pt\" ]; then
    log \"Found existing Latin checkpoint. Resuming.\"
    RESUME_FLAG=\"--resume\"
fi

python3 \"$WORK_DIR/train_latin_ctc.py\" \
    --features-dir \"$FEATURES_DIR\" \
    --pairs \"$DATA_DIR/latin_pairs.jsonl\" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --checkpoint-dir \"$LATIN_CKPT_DIR\" \
    --checkpoint-interval 500 \
    --log /workspace/experiment_b_latin.log \
    $RESUME_FLAG \
    2>&1 | tee -a /workspace/experiment_b_latin.log

log \"Latin training complete.\"

# ── Phase 4: Compare ──
log \"\"
log \"--- Phase 4: Comparison Evaluation ---\"
cd \"$WORK_DIR\"
python3 compare_scripts.py \
    --nko-checkpoint \"$NKO_CKPT_DIR/best.pt\" \
    --latin-checkpoint \"$LATIN_CKPT_DIR/best.pt\" \
    --test-pairs \"$DATA_DIR/test_pairs_aligned.jsonl\" \
    --features-dir \"$FEATURES_DIR\" \
    --output \"$RESULTS_DIR/comparison.json\" \
    2>&1 | tee -a /workspace/experiment_b_compare.log

log \"\"
log \"========================================\"
log \"Experiment B COMPLETE\"
log \"========================================\"
log \"Results: $RESULTS_DIR/comparison.json\"
log \"Logs: /workspace/experiment_b_*.log\"
log \"N'Ko checkpoints: $NKO_CKPT_DIR/\"
log \"Latin checkpoints: $LATIN_CKPT_DIR/\"

# Print summary
if [ -f \"$RESULTS_DIR/comparison.json\" ]; then
    log \"\"
    log \"--- Results Summary ---\"
    python3 -c \"
import json
with open('$RESULTS_DIR/comparison.json') as f:
    r = json.load(f)
print(f'N Ko CER:  {r[\"nko\"][\"cer_mean\"]:.4f} ({r[\"nko\"][\"cer_mean\"]*100:.1f}%)')
print(f'Latin CER: {r[\"latin\"][\"cer_mean\"]:.4f} ({r[\"latin\"][\"cer_mean\"]*100:.1f}%)')
print(f'Advantage: {r[\"advantage_pct\"]:+.1f}%')
print(f'Verdict:   {r[\"verdict\"]}')
\"
fi
RUN_EOF
chmod +x $REMOTE_DIR/run_experiment_b.sh"
ok "Master run script created"

# ── Step 5: Launch training ──────────────────────────────────────────

log ""
log "============================================================"
log "All scripts uploaded and configured."
log "============================================================"
log ""
log "To start the experiment:"
log "  ssh $SSH_OPTS root@$HOST"
log "  nohup bash /workspace/experiment_b/run_experiment_b.sh > /workspace/experiment_b_master.log 2>&1 &"
log "  tail -f /workspace/experiment_b_master.log"
log ""
log "To monitor:"
log "  ssh $SSH_OPTS root@$HOST 'tail -50 /workspace/experiment_b_nko.log'"
log "  ssh $SSH_OPTS root@$HOST 'tail -50 /workspace/experiment_b_latin.log'"
log "  ssh $SSH_OPTS root@$HOST 'nvidia-smi'"
log ""
log "To download results when done:"
log "  scp $SCP_OPTS root@$HOST:/workspace/results/comparison.json ."
log "  scp $SCP_OPTS root@$HOST:/workspace/checkpoints/nko/best.pt checkpoints_nko_best.pt"
log "  scp $SCP_OPTS root@$HOST:/workspace/checkpoints/latin/best.pt checkpoints_latin_best.pt"
log ""

# Ask whether to auto-start
read -p "Start training now in background? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "Starting experiment in background..."
    ssh_cmd "nohup bash $REMOTE_DIR/run_experiment_b.sh > /workspace/experiment_b_master.log 2>&1 &"
    ok "Experiment B launched! Monitor with:"
    log "  ssh $SSH_OPTS root@$HOST 'tail -f /workspace/experiment_b_master.log'"
else
    log "Not starting. SSH in and run manually when ready."
fi
