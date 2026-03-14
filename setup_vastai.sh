#!/usr/bin/env bash
# NKO Brain Scanner — Vast.ai Setup Script
#
# Usage:
#   1. Rent a 2x A100 80GB instance on vast.ai
#   2. SSH in: ssh -p PORT root@HOST
#   3. Upload this project: rsync -avz --exclude='__pycache__' . root@HOST:/workspace/nko-brain-scanner/
#   4. Run: bash setup_vastai.sh
#
# Instance requirements:
#   - 1x A100 80GB (PCIe or SXM) — AWQ 4-bit model fits in ~36GB VRAM
#   - 150GB disk (model weights + checkpoints)
#   - CUDA 12.1+
#   - PyTorch pre-installed (most vast.ai templates include it)

set -e

echo "=== NKO Brain Scanner — Vast.ai Setup ==="
echo ""

# 1. Install Python dependencies
echo "[1/4] Installing dependencies..."
pip install -q transformers>=4.45 accelerate>=0.28 scipy>=1.12 seaborn>=0.13 tqdm

# 2. Verify GPU setup
echo "[2/4] Verifying GPU setup..."
python3 -c "
import torch
n = torch.cuda.device_count()
print(f'GPUs: {n}')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f'  GPU {i}: {name} ({mem:.1f} GB)')
assert n >= 1, 'No GPUs found!'
total_mem = sum(torch.cuda.get_device_properties(i).total_mem for i in range(n)) / 1e9
print(f'Total VRAM: {total_mem:.1f} GB')
if total_mem < 140:
    print('WARNING: May need quantization for Qwen2-72B FP16 (needs ~144GB)')
"

# 3. Download model weights (AWQ 4-bit — ~36GB VRAM, fits on single A100)
echo "[3/4] Pre-downloading Qwen2-72B-Instruct-AWQ weights..."
pip install -q autoawq
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = 'Qwen/Qwen2-72B-Instruct-AWQ'
print(f'Downloading tokenizer for {model_name}...')
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print('Tokenizer ready.')

print(f'Downloading model weights for {model_name}...')
print('This will take 5-10 minutes on first run (AWQ 4-bit is ~40GB).')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    trust_remote_code=True,
)
print(f'Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params')
del model
torch.cuda.empty_cache()
print('Model weights cached successfully.')
"

# 4. Create results directories
echo "[4/4] Creating output directories..."
mkdir -p results/activation_profiles results/heatmaps results/figures

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Run experiments:"
echo "  # Experiment 1: Activation profiling (~30 min)"
echo "  python3 -m scanner.run_experiment --experiment activation"
echo ""
echo "  # Experiment 2: Coarse heatmap sweep (~6 hours)"
echo "  python3 -m scanner.run_experiment --experiment heatmap --mode coarse"
echo ""
echo "  # Both experiments sequentially"
echo "  python3 -m scanner.run_experiment --experiment both --mode coarse"
echo ""
echo "  # Full sweep (only if coarse shows interesting patterns, ~40 hours)"
echo "  python3 -m scanner.run_experiment --experiment heatmap --mode full"
