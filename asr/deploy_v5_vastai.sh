#!/usr/bin/env bash
set -euo pipefail
#
# N'Ko ASR V5 — Vast.ai A100 80GB Deployment
# =============================================
# Deploys and runs the complete V5 training pipeline on a Vast.ai GPU instance.
#
# Prerequisites:
#   - vastai CLI installed and authenticated
#   - SSH key configured in Vast.ai dashboard
#   - An A100 80GB instance running (or this script creates one)
#
# Usage:
#   # Deploy to an existing instance
#   ./asr/deploy_v5_vastai.sh --instance-id 12345
#
#   # Deploy with monitoring
#   ./asr/deploy_v5_vastai.sh --instance-id 12345 --webhook https://hooks.example.com/notify
#
#   # Resume from a crashed run
#   ./asr/deploy_v5_vastai.sh --instance-id 12345 --resume
#
#   # Monitor an in-progress run
#   ./asr/deploy_v5_vastai.sh --instance-id 12345 --monitor-only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ── Configuration ──────────────────────────────────────────────

INSTANCE_ID=""
WEBHOOK_URL=""
RESUME_FLAG=""
MONITOR_ONLY=false
WANDB_PROJECT="nko-asr-v5"
SSH_PORT=22
REMOTE_USER="root"
WORK_DIR="/workspace"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --instance-id) INSTANCE_ID="$2"; shift 2 ;;
        --webhook) WEBHOOK_URL="$2"; shift 2 ;;
        --resume) RESUME_FLAG="--resume"; shift ;;
        --monitor-only) MONITOR_ONLY=true; shift ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        --ssh-port) SSH_PORT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$INSTANCE_ID" ]; then
    echo "ERROR: --instance-id required"
    echo "Usage: $0 --instance-id <VAST_INSTANCE_ID>"
    exit 1
fi

# Get SSH connection info
echo "Getting instance info for ${INSTANCE_ID}..."
SSH_HOST=$(vastai show instance "$INSTANCE_ID" --raw | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('ssh_host', data.get('public_ipaddr', '')))
" 2>/dev/null || echo "")

SSH_PORT_REMOTE=$(vastai show instance "$INSTANCE_ID" --raw | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('ssh_port', 22))
" 2>/dev/null || echo "$SSH_PORT")

if [ -z "$SSH_HOST" ]; then
    echo "ERROR: Could not get SSH host for instance $INSTANCE_ID"
    echo "Is the instance running? Check: vastai show instances"
    exit 1
fi

SSH_CMD="ssh -p $SSH_PORT_REMOTE $REMOTE_USER@$SSH_HOST"
SCP_CMD="scp -P $SSH_PORT_REMOTE"

echo "Instance: $INSTANCE_ID"
echo "SSH: $REMOTE_USER@$SSH_HOST:$SSH_PORT_REMOTE"

# ── Monitor-only mode ─────────────────────────────────────────

if [ "$MONITOR_ONLY" = true ]; then
    echo ""
    echo "=== MONITOR MODE ==="
    echo "Checking training progress every 30 minutes..."
    echo "Press Ctrl+C to stop monitoring."
    echo ""

    while true; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking progress..."
        $SSH_CMD "
            if [ -f $WORK_DIR/v5_model/checkpoint.pt ]; then
                python3 -c '
import torch, json
ckpt = torch.load(\"$WORK_DIR/v5_model/checkpoint.pt\", map_location=\"cpu\", weights_only=False)
print(json.dumps({
    \"epoch\": ckpt.get(\"epoch\", \"?\"),
    \"batch_idx\": ckpt.get(\"batch_idx\", \"?\"),
    \"best_val\": round(ckpt.get(\"best_val\", 0), 4),
    \"val_loss\": round(ckpt.get(\"val_loss\", 0), 4) if ckpt.get(\"val_loss\") else \"N/A\",
    \"timestamp\": ckpt.get(\"timestamp\", \"?\"),
    \"reason\": ckpt.get(\"reason\", \"?\"),
}, indent=2))
'
            else
                echo 'No checkpoint found yet.'
            fi

            echo ''
            echo 'GPU utilization:'
            nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader

            echo ''
            echo 'Training log (last 20 lines):'
            tail -20 $WORK_DIR/v5_training.log 2>/dev/null || echo 'No log file yet.'

            echo ''
            echo 'Disk usage:'
            df -h $WORK_DIR | tail -1
        " 2>/dev/null || echo "  SSH connection failed. Retrying in 30 minutes..."

        echo ""
        echo "Next check in 30 minutes..."
        sleep 1800
    done
fi

# ── Deploy scripts ────────────────────────────────────────────

echo ""
echo "=== DEPLOYING V5 PIPELINE ==="
echo ""

# 1. Copy all required files
echo "Copying scripts to instance..."

# Core ASR scripts
$SCP_CMD \
    "$SCRIPT_DIR/train_v5_fullscale.py" \
    "$SCRIPT_DIR/prepare_v5_data.py" \
    "$SCRIPT_DIR/beam_search_decoder.py" \
    "$SCRIPT_DIR/eval_v5.py" \
    "$SCRIPT_DIR/download_afvoices.py" \
    "$SCRIPT_DIR/extract_afvoices_features.py" \
    "$SCRIPT_DIR/extract_human_features.py" \
    "$SCRIPT_DIR/bridge_to_nko.py" \
    "$SCRIPT_DIR/postprocess.py" \
    "$REMOTE_USER@$SSH_HOST:$WORK_DIR/asr/"

# N'Ko transliteration bridge
if [ -d "$REPO_DIR/nko" ]; then
    echo "Copying nko/ transliteration package..."
    $SCP_CMD -r "$REPO_DIR/nko" "$REMOTE_USER@$SSH_HOST:$WORK_DIR/"
elif [ -d "$HOME/Desktop/NKo/nko" ]; then
    echo "Copying nko/ from ~/Desktop/NKo..."
    $SCP_CMD -r "$HOME/Desktop/NKo/nko" "$REMOTE_USER@$SSH_HOST:$WORK_DIR/"
fi

# Constrained FSM
if [ -d "$REPO_DIR/constrained" ]; then
    echo "Copying constrained/ FSM package..."
    $SCP_CMD -r "$REPO_DIR/constrained" "$REMOTE_USER@$SSH_HOST:$WORK_DIR/"
fi

echo "Scripts deployed."

# 2. Install dependencies
echo ""
echo "Installing dependencies..."

$SSH_CMD "
    pip install --quiet --upgrade pip
    pip install --quiet \
        torch torchvision torchaudio \
        openai-whisper \
        datasets>=3.6.0 \
        soundfile \
        wandb \
        numpy \
        huggingface_hub
    echo 'Dependencies installed.'
    python3 -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"N/A\"}\")'
"

# 3. Setup workspace
echo ""
echo "Setting up workspace..."

$SSH_CMD "
    mkdir -p $WORK_DIR/asr
    mkdir -p $WORK_DIR/v5_data/manifests
    mkdir -p $WORK_DIR/v5_data/features
    mkdir -p $WORK_DIR/v5_model
    mkdir -p $WORK_DIR/hf_data
    echo 'Workspace ready.'
    echo 'Disk space:'
    df -h $WORK_DIR | tail -1
"

# 4. Set environment variables
echo ""
echo "Configuring environment..."

$SSH_CMD "
    # Set W&B key if available
    if [ -n \"\${WANDB_API_KEY:-}\" ]; then
        echo 'W&B API key found in environment.'
    fi

    # Set HF token if available
    if [ -n \"\${HF_TOKEN:-}\" ]; then
        echo 'HuggingFace token found in environment.'
    fi
"

# 5. Run data preparation
echo ""
echo "=== PHASE 1: DATA PREPARATION ==="
echo ""
echo "Running prepare_v5_data.py (download + bridge + feature extraction)..."
echo "This may take several hours for feature extraction."
echo ""

$SSH_CMD "
    cd $WORK_DIR
    nohup python3 -u asr/prepare_v5_data.py \
        --output-dir $WORK_DIR/v5_data \
        --cache-dir $WORK_DIR/hf_data \
        --skip-common-voice \
        --extract-mel \
        > $WORK_DIR/v5_data_prep.log 2>&1 &

    DATA_PID=\$!
    echo \"Data prep PID: \$DATA_PID\"
    echo \$DATA_PID > $WORK_DIR/data_prep.pid

    echo 'Waiting for data preparation to complete...'
    wait \$DATA_PID
    DATA_EXIT=\$?

    if [ \$DATA_EXIT -ne 0 ]; then
        echo 'ERROR: Data preparation failed!'
        tail -50 $WORK_DIR/v5_data_prep.log
        exit 1
    fi

    echo 'Data preparation complete.'
    echo ''
    cat $WORK_DIR/v5_data/manifests/stats.json 2>/dev/null || echo 'No stats file.'
"

# 6. Run training
echo ""
echo "=== PHASE 2: TRAINING ==="
echo ""

WEBHOOK_ARG=""
if [ -n "$WEBHOOK_URL" ]; then
    WEBHOOK_ARG="--webhook $WEBHOOK_URL"
fi

WANDB_ARG=""
if [ -n "$WANDB_PROJECT" ]; then
    WANDB_ARG="--wandb-project $WANDB_PROJECT"
fi

echo "Starting V5 training..."
echo "  LoRA: rank=64, alpha=128, all 32 layers"
echo "  CTC head: h768 L6 nhead=12"
echo "  Batch size: 16, Epochs: 50"
echo "  LR: 5e-6 (LoRA), 1e-4 (head)"
echo ""

$SSH_CMD "
    cd $WORK_DIR
    nohup python3 -u asr/train_v5_fullscale.py \
        --data-dir $WORK_DIR/v5_data \
        --save-dir $WORK_DIR/v5_model \
        --epochs 50 \
        --batch-size 16 \
        --lr-whisper 5e-6 \
        --lr-head 1e-4 \
        --lora-rank 64 \
        --lora-alpha 128 \
        --lora-layers 32 \
        --checkpoint-interval 1000 \
        --warmup-epochs 5 \
        --num-workers 4 \
        $RESUME_FLAG \
        $WEBHOOK_ARG \
        $WANDB_ARG \
        > $WORK_DIR/v5_training.log 2>&1 &

    TRAIN_PID=\$!
    echo \"Training PID: \$TRAIN_PID\"
    echo \$TRAIN_PID > $WORK_DIR/train.pid

    echo ''
    echo 'Training is running in the background.'
    echo 'Monitor with:'
    echo '  tail -f $WORK_DIR/v5_training.log'
    echo ''
    echo 'Or use this script with --monitor-only flag.'
"

# 7. Create monitoring script on the instance
echo ""
echo "Creating monitoring script..."

$SSH_CMD "cat > $WORK_DIR/check_v5.sh << 'MONITOR_EOF'
#!/bin/bash
echo '=== V5 Training Status ==='
echo \"Time: \$(date)\"
echo ''

if [ -f /workspace/v5_model/checkpoint.pt ]; then
    python3 -c '
import torch, json
ckpt = torch.load(\"/workspace/v5_model/checkpoint.pt\", map_location=\"cpu\", weights_only=False)
print(\"Checkpoint:\" , json.dumps({
    \"epoch\": ckpt.get(\"epoch\", \"?\"),
    \"batch_idx\": ckpt.get(\"batch_idx\", \"?\"),
    \"best_val\": round(ckpt.get(\"best_val\", 0), 4),
    \"val_loss\": round(ckpt.get(\"val_loss\", 0), 4) if ckpt.get(\"val_loss\") else \"N/A\",
    \"timestamp\": ckpt.get(\"timestamp\", \"?\"),
    \"reason\": ckpt.get(\"reason\", \"?\"),
}, indent=2))
'
else
    echo 'No checkpoint yet.'
fi

echo ''
echo 'GPU:'
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv

echo ''
echo 'Recent log:'
tail -30 /workspace/v5_training.log 2>/dev/null || echo 'No log.'

echo ''
echo 'Training process:'
if [ -f /workspace/train.pid ]; then
    PID=\$(cat /workspace/train.pid)
    if kill -0 \$PID 2>/dev/null; then
        echo \"Running (PID \$PID)\"
        RUNTIME=\$(ps -p \$PID -o etime= 2>/dev/null || echo 'unknown')
        echo \"Runtime: \$RUNTIME\"
    else
        echo \"Not running (PID \$PID exited)\"
    fi
else
    echo 'No PID file.'
fi

echo ''
echo 'Disk:'
df -h /workspace | tail -1

echo ''
echo 'Model files:'
ls -lh /workspace/v5_model/ 2>/dev/null || echo 'None yet.'
MONITOR_EOF
chmod +x $WORK_DIR/check_v5.sh
echo 'Created /workspace/check_v5.sh'
"

echo ""
echo "=== DEPLOYMENT COMPLETE ==="
echo ""
echo "Training is running on instance $INSTANCE_ID"
echo ""
echo "To monitor progress:"
echo "  ssh -p $SSH_PORT_REMOTE $REMOTE_USER@$SSH_HOST 'bash /workspace/check_v5.sh'"
echo ""
echo "To watch live logs:"
echo "  ssh -p $SSH_PORT_REMOTE $REMOTE_USER@$SSH_HOST 'tail -f /workspace/v5_training.log'"
echo ""
echo "To auto-monitor every 30 min:"
echo "  $0 --instance-id $INSTANCE_ID --monitor-only"
echo ""
echo "To download results when done:"
echo "  scp -P $SSH_PORT_REMOTE $REMOTE_USER@$SSH_HOST:/workspace/v5_model/best_v5.pt ."
echo "  scp -P $SSH_PORT_REMOTE $REMOTE_USER@$SSH_HOST:/workspace/v5_model/lora_weights_v5.pt ."
echo "  scp -P $SSH_PORT_REMOTE $REMOTE_USER@$SSH_HOST:/workspace/v5_model/ctc_head_v5.pt ."
