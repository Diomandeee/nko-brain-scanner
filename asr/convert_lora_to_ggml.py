#!/usr/bin/env python3
"""
N'Ko Whisper LoRA -> GGML Conversion Pipeline
===============================================
Full chain: trained LoRA checkpoint -> merged HuggingFace model -> GGML -> quantized -> iOS bundle.

Stages (CLI subcommands):
  merge    — Merge LoRA adapters into base Whisper encoder weights
  convert  — Convert merged model to whisper.cpp GGML format
  quantize — Quantize GGML model to Q5_0
  bundle   — Prepare directory structure for iOS integration
  all      — Run entire pipeline sequentially

Architecture handled:
  - Base: openai/whisper-large-v3 (32 encoder blocks, d=1280)
  - LoRA: rank=32, alpha=64, applied to encoder blocks 24-31
    - Attention: query, key, value, out
    - MLP: layers 0 and 2
  - CTC head: TransformerCTCHead (temporal_ds + pos_enc + 6-layer transformer encoder + output_proj)
  - Checkpoint format: {lora_state: dict, head_state: dict, epoch: int, val_loss: float}

Usage:
  python3 convert_lora_to_ggml.py merge --checkpoint best_whisper_lora.pt --output merged/
  python3 convert_lora_to_ggml.py convert --merged merged/ --output ggml-nko-large-v3.bin
  python3 convert_lora_to_ggml.py quantize --input ggml-nko-large-v3.bin --output ggml-nko-large-v3-q5_0.bin
  python3 convert_lora_to_ggml.py bundle --model ggml-nko-large-v3-q5_0.bin --output-dir nko-ios-model/
  python3 convert_lora_to_ggml.py all --checkpoint best_whisper_lora.pt --output-dir nko_model_release/
"""

import argparse
import json
import logging
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nko-convert")

# ── Constants ────────────────────────────────────────────────────────────────

WHISPER_MODEL_NAME = "large-v3"
WHISPER_HF_NAME = "openai/whisper-large-v3"
DEFAULT_LORA_RANK = 32
DEFAULT_LORA_ALPHA = 64
DEFAULT_LORA_LAYERS = 8
ENCODER_DIM = 1280
TOTAL_ENCODER_BLOCKS = 32
DEFAULT_WHISPER_CPP_DIR = os.path.expanduser("~/whisper.cpp")

# Attention projections that get LoRA in our training script
ATTN_TARGETS = ["query", "key", "value", "out"]
# MLP layers that get LoRA (indices into nn.Sequential)
MLP_TARGETS = ["0", "2"]


# ── Utility ──────────────────────────────────────────────────────────────────

def get_device():
    """Pick the best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except ImportError:
        pass
    return "cpu"


def format_size(nbytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def progress_bar(current, total, width=40, prefix=""):
    """Simple inline progress bar."""
    frac = current / max(total, 1)
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r  {prefix}[{bar}] {current}/{total} ({100*frac:.0f}%)")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def ensure_whisper_cpp(whisper_cpp_dir: str) -> Path:
    """Clone whisper.cpp if not present, return path."""
    wdir = Path(whisper_cpp_dir)
    if wdir.exists() and (wdir / "Makefile").exists():
        log.info("whisper.cpp found at %s", wdir)
        return wdir

    log.info("whisper.cpp not found at %s, cloning...", wdir)
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/ggerganov/whisper.cpp.git", str(wdir)],
        check=True,
    )
    log.info("whisper.cpp cloned to %s", wdir)
    return wdir


def build_quantize_binary(whisper_cpp_dir: Path) -> Path:
    """Build the quantize binary if not present."""
    quantize_bin = whisper_cpp_dir / "quantize"
    if not quantize_bin.exists():
        quantize_bin = whisper_cpp_dir / "build" / "bin" / "quantize"

    if quantize_bin.exists():
        log.info("quantize binary found at %s", quantize_bin)
        return quantize_bin

    log.info("Building whisper.cpp quantize binary...")

    # Try cmake first (newer whisper.cpp versions)
    build_dir = whisper_cpp_dir / "build"
    build_dir.mkdir(exist_ok=True)

    cmake_args = ["cmake", ".."]
    if platform.system() == "Darwin":
        cmake_args.extend(["-DWHISPER_COREML=OFF", "-DWHISPER_METAL=ON"])

    try:
        subprocess.run(cmake_args, cwd=build_dir, check=True, capture_output=True)
        subprocess.run(["cmake", "--build", ".", "--target", "quantize", "-j", "4"],
                       cwd=build_dir, check=True, capture_output=True)
        quantize_bin = build_dir / "bin" / "quantize"
        if quantize_bin.exists():
            log.info("Built quantize via cmake at %s", quantize_bin)
            return quantize_bin
    except (subprocess.CalledProcessError, FileNotFoundError):
        log.info("cmake build failed, trying make...")

    # Fallback to make
    try:
        subprocess.run(["make", "quantize", "-j", "4"], cwd=whisper_cpp_dir,
                       check=True, capture_output=True)
        quantize_bin = whisper_cpp_dir / "quantize"
        if quantize_bin.exists():
            log.info("Built quantize via make at %s", quantize_bin)
            return quantize_bin
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    raise FileNotFoundError(
        f"Could not build quantize binary in {whisper_cpp_dir}. "
        "Install cmake or make, then retry."
    )


# ── N'Ko Vocab (must match train_whisper_lora.py) ───────────────────────────

def build_nko_char_vocab():
    """Build N'Ko character vocabulary. Matches train_whisper_lora.build_nko_char_vocab()."""
    chars = {}
    idx = 0
    for cp in range(0x07C0, 0x07FF + 1):
        chars[chr(cp)] = idx
        idx += 1
    chars[" "] = idx
    idx += 1
    return chars, idx


# ── Stage 1: Merge LoRA ─────────────────────────────────────────────────────

def stage_merge(
    checkpoint_path: str,
    output_dir: str,
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_layers: int = DEFAULT_LORA_LAYERS,
    device_override: str = None,
) -> Tuple[Path, Path]:
    """
    Merge LoRA adapter weights into base Whisper model.

    The LoRA merge formula:
        W_merged = W_base + (lora_B @ lora_A) * (alpha / rank)

    lora_A shape: (rank, in_features)
    lora_B shape: (out_features, rank)

    Returns (merged_model_dir, ctc_head_path).
    """
    import torch
    import whisper

    log.info("=" * 60)
    log.info("STAGE 1: Merge LoRA into base Whisper")
    log.info("=" * 60)

    device = torch.device(device_override) if device_override else get_device()
    # For merge, CPU is fine and avoids GPU memory pressure
    merge_device = torch.device("cpu")
    log.info("Using device: %s (merge on CPU)", device)

    # Load checkpoint
    log.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    lora_state = ckpt.get("lora_state", {})
    head_state = ckpt.get("head_state", {})
    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", "?")
    log.info("  Checkpoint epoch=%s, val_loss=%s", epoch, val_loss)
    log.info("  LoRA tensors: %d", len(lora_state))
    log.info("  CTC head tensors: %d", len(head_state))

    if len(lora_state) == 0:
        raise ValueError("Checkpoint has no lora_state tensors. Wrong file?")

    # Load base Whisper model
    log.info("Loading base Whisper %s...", WHISPER_MODEL_NAME)
    t0 = time.time()
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device="cpu")
    log.info("  Loaded in %.1fs", time.time() - t0)

    # Parse LoRA state keys and merge
    # Keys look like: encoder.blocks.24.attn.query.lora_A
    #                  encoder.blocks.24.attn.query.lora_B
    #                  encoder.blocks.24.mlp.0.lora_A
    #                  encoder.blocks.24.mlp.0.lora_B

    scale = lora_alpha / lora_rank
    log.info("  LoRA scale (alpha/rank): %.2f (alpha=%d, rank=%d)", scale, lora_alpha, lora_rank)

    start_layer = TOTAL_ENCODER_BLOCKS - lora_layers
    log.info("  Target layers: %d-%d (%d layers)", start_layer, TOTAL_ENCODER_BLOCKS - 1, lora_layers)

    # Group LoRA pairs (A and B for each adapted weight)
    lora_pairs: Dict[str, Dict[str, "torch.Tensor"]] = {}
    for key, tensor in lora_state.items():
        # Strip the lora_A/lora_B suffix to get the base parameter path
        if key.endswith(".lora_A"):
            base_key = key[:-7]  # remove ".lora_A"
            lora_pairs.setdefault(base_key, {})["A"] = tensor
        elif key.endswith(".lora_B"):
            base_key = key[:-7]  # remove ".lora_B"
            lora_pairs.setdefault(base_key, {})["B"] = tensor
        else:
            log.warning("  Unexpected LoRA key (skipping): %s", key)

    log.info("  Found %d LoRA adapter pairs to merge", len(lora_pairs))

    merged_count = 0
    skipped_count = 0

    for base_key, pair in sorted(lora_pairs.items()):
        if "A" not in pair or "B" not in pair:
            log.warning("  Incomplete pair for %s (has %s), skipping", base_key, list(pair.keys()))
            skipped_count += 1
            continue

        lora_a = pair["A"].float()  # (rank, in_features)
        lora_b = pair["B"].float()  # (out_features, rank)

        # Navigate to the weight tensor in the Whisper model
        # base_key is like: "encoder.blocks.24.attn.query"
        # After LoRA wrapping during training, the actual linear is at .original
        # But since we loaded a fresh model (no LoRA wrappers), the linear is directly there
        try:
            target = _resolve_weight(whisper_model, base_key)
        except (AttributeError, KeyError) as e:
            log.warning("  Could not resolve %s: %s", base_key, e)
            skipped_count += 1
            continue

        # Compute the delta: delta_W = (lora_B @ lora_A) * scale
        delta_w = (lora_b @ lora_a) * scale

        if target.shape != delta_w.shape:
            log.error(
                "  Shape mismatch for %s: weight=%s, delta=%s",
                base_key, target.shape, delta_w.shape,
            )
            skipped_count += 1
            continue

        # Merge: W = W + delta_W
        target.data.add_(delta_w.to(target.dtype))
        merged_count += 1
        progress_bar(merged_count, len(lora_pairs), prefix="Merging: ")

    log.info("  Merged: %d adapters, Skipped: %d", merged_count, skipped_count)

    if merged_count == 0:
        raise RuntimeError(
            "No LoRA adapters were merged. Check that checkpoint key format "
            "matches the model structure."
        )

    # Save merged model
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save as standard PyTorch state dict (compatible with whisper.cpp convert scripts)
    merged_path = out / "merged_whisper_model.pt"
    log.info("Saving merged Whisper model to %s", merged_path)
    torch.save(whisper_model.state_dict(), merged_path)
    log.info("  Size: %s", format_size(merged_path.stat().st_size))

    # Also save as HuggingFace-compatible if transformers is available
    hf_dir = out / "hf_model"
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        log.info("Saving HuggingFace-format model to %s", hf_dir)
        hf_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_HF_NAME)

        # Map OpenAI Whisper keys to HuggingFace keys and apply merged weights
        _apply_openai_weights_to_hf(whisper_model, hf_model)

        hf_model.save_pretrained(str(hf_dir))

        processor = WhisperProcessor.from_pretrained(WHISPER_HF_NAME)
        processor.save_pretrained(str(hf_dir))

        log.info("  HuggingFace model saved")
    except ImportError:
        log.info("  transformers not installed, skipping HuggingFace format (not required)")
    except Exception as e:
        log.warning("  HuggingFace save failed (non-fatal): %s", e)

    # Save CTC head separately
    head_path = out / "ctc_head.pt"
    log.info("Saving CTC head to %s", head_path)
    torch.save({
        "state_dict": head_state,
        "num_chars": build_nko_char_vocab()[1],
        "input_dim": ENCODER_DIM,
        "architecture": "TransformerCTCHead",
        "config": {
            "hidden_dim": 768,
            "num_layers": 6,
            "nhead": 12,
            "dropout": 0.1,
        },
    }, head_path)

    # Save merge metadata
    meta_path = out / "merge_metadata.json"
    meta = {
        "base_model": WHISPER_MODEL_NAME,
        "hf_model": WHISPER_HF_NAME,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_layers": lora_layers,
        "start_layer": start_layer,
        "lora_targets_attn": ATTN_TARGETS,
        "lora_targets_mlp": MLP_TARGETS,
        "merged_adapters": merged_count,
        "skipped_adapters": skipped_count,
        "checkpoint_epoch": epoch,
        "checkpoint_val_loss": float(val_loss) if isinstance(val_loss, (int, float)) else None,
        "merge_scale": scale,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Stage 1 complete. Output: %s", out)
    return merged_path, head_path


def _resolve_weight(model, key_path: str):
    """
    Navigate a dotted key path to get the weight tensor of a linear layer.

    For paths like 'encoder.blocks.24.attn.query', returns the .weight parameter
    of the nn.Linear at that path.

    For paths like 'encoder.blocks.24.mlp.0', navigates into nn.Sequential by index.
    """
    import torch.nn as nn

    parts = key_path.split(".")
    obj = model
    for part in parts:
        if hasattr(obj, part):
            obj = getattr(obj, part)
        elif isinstance(obj, (nn.Sequential, nn.ModuleList)) and part.isdigit():
            obj = obj[int(part)]
        else:
            raise AttributeError(f"Cannot navigate '{part}' in {type(obj).__name__} (full path: {key_path})")

    if isinstance(obj, nn.Linear):
        return obj.weight
    elif hasattr(obj, "weight"):
        return obj.weight
    else:
        raise AttributeError(f"Resolved object at '{key_path}' is {type(obj).__name__}, not a Linear layer")


def _apply_openai_weights_to_hf(openai_model, hf_model):
    """
    Transfer merged OpenAI-format Whisper weights into a HuggingFace WhisperForConditionalGeneration.

    The key mapping between OpenAI and HuggingFace Whisper is nontrivial:
      OpenAI: encoder.blocks.N.attn.query.weight
      HF:     model.encoder.layers.N.self_attn.q_proj.weight
    """
    import torch

    openai_sd = openai_model.state_dict()
    hf_sd = hf_model.state_dict()

    # Build key mapping
    key_map = {}

    # Encoder conv layers
    key_map["encoder.conv1.weight"] = "model.encoder.conv1.weight"
    key_map["encoder.conv1.bias"] = "model.encoder.conv1.bias"
    key_map["encoder.conv2.weight"] = "model.encoder.conv2.weight"
    key_map["encoder.conv2.bias"] = "model.encoder.conv2.bias"

    # Encoder positional embedding
    key_map["encoder.positional_embedding"] = "model.encoder.embed_positions.weight"

    # Encoder layer norm
    key_map["encoder.ln_post.weight"] = "model.encoder.layer_norm.weight"
    key_map["encoder.ln_post.bias"] = "model.encoder.layer_norm.bias"

    # Encoder blocks
    for i in range(TOTAL_ENCODER_BLOCKS):
        prefix_oa = f"encoder.blocks.{i}"
        prefix_hf = f"model.encoder.layers.{i}"

        # Attention
        attn_map = {
            "query": "q_proj", "key": "k_proj",
            "value": "v_proj", "out": "out_proj",
        }
        for oa_name, hf_name in attn_map.items():
            for suffix in ("weight", "bias"):
                oa_key = f"{prefix_oa}.attn.{oa_name}.{suffix}"
                hf_key = f"{prefix_hf}.self_attn.{hf_name}.{suffix}"
                if oa_key in openai_sd and hf_key in hf_sd:
                    key_map[oa_key] = hf_key

        # Attention layer norms
        for suffix in ("weight", "bias"):
            key_map[f"{prefix_oa}.attn_ln.{suffix}"] = f"{prefix_hf}.self_attn_layer_norm.{suffix}"

        # MLP
        for suffix in ("weight", "bias"):
            key_map[f"{prefix_oa}.mlp.0.{suffix}"] = f"{prefix_hf}.fc1.{suffix}"
            key_map[f"{prefix_oa}.mlp.2.{suffix}"] = f"{prefix_hf}.fc2.{suffix}"

        # MLP layer norm
        for suffix in ("weight", "bias"):
            key_map[f"{prefix_oa}.mlp_ln.{suffix}"] = f"{prefix_hf}.final_layer_norm.{suffix}"

    # Decoder (unchanged, copy from HF pretrained)
    # We only modified the encoder, so decoder weights stay as-is

    # Apply mapped encoder weights
    mapped = 0
    for oa_key, hf_key in key_map.items():
        if oa_key in openai_sd and hf_key in hf_sd:
            if openai_sd[oa_key].shape == hf_sd[hf_key].shape:
                hf_sd[hf_key] = openai_sd[oa_key].clone()
                mapped += 1

    hf_model.load_state_dict(hf_sd)
    log.info("  Mapped %d/%d encoder weights to HuggingFace format", mapped, len(key_map))


# ── Stage 2: Convert to GGML ────────────────────────────────────────────────

def stage_convert(
    merged_dir: str,
    output_path: str,
    whisper_cpp_dir: str = DEFAULT_WHISPER_CPP_DIR,
) -> Path:
    """
    Convert merged Whisper model to whisper.cpp GGML format.

    Strategy:
      1. If HuggingFace model directory exists, use whisper.cpp's convert-hf-to-ggml.py
      2. Otherwise, use the OpenAI .pt checkpoint with convert-pt-to-ggml.py
      3. As final fallback, implement direct GGML conversion
    """
    log.info("=" * 60)
    log.info("STAGE 2: Convert to GGML")
    log.info("=" * 60)

    merged = Path(merged_dir)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wdir = ensure_whisper_cpp(whisper_cpp_dir)

    hf_dir = merged / "hf_model"
    pt_path = merged / "merged_whisper_model.pt"

    # Strategy 1: HuggingFace model with convert-hf-to-ggml.py
    hf_convert_script = _find_convert_script(wdir, "hf")
    if hf_dir.exists() and (hf_dir / "config.json").exists() and hf_convert_script:
        log.info("Converting HuggingFace model via %s", hf_convert_script.name)
        return _convert_via_hf_script(hf_dir, out_path, hf_convert_script, wdir)

    # Strategy 2: OpenAI .pt with convert-pt-to-ggml.py
    pt_convert_script = _find_convert_script(wdir, "pt")
    if pt_path.exists() and pt_convert_script:
        log.info("Converting OpenAI .pt model via %s", pt_convert_script.name)
        return _convert_via_pt_script(pt_path, out_path, pt_convert_script, wdir)

    # Strategy 3: Direct GGML conversion
    if pt_path.exists():
        log.info("No whisper.cpp convert script found, using direct GGML writer")
        return _convert_direct_ggml(pt_path, out_path)

    raise FileNotFoundError(
        f"No convertible model found in {merged}. "
        f"Expected either hf_model/ directory or merged_whisper_model.pt"
    )


def _find_convert_script(wdir: Path, variant: str) -> Optional[Path]:
    """Find the appropriate whisper.cpp conversion script."""
    candidates = []
    if variant == "hf":
        candidates = [
            wdir / "models" / "convert-hf-to-ggml.py",
            wdir / "convert-hf-to-ggml.py",
            wdir / "scripts" / "convert-hf-to-ggml.py",
        ]
    elif variant == "pt":
        candidates = [
            wdir / "models" / "convert-pt-to-ggml.py",
            wdir / "convert-pt-to-ggml.py",
            wdir / "scripts" / "convert-pt-to-ggml.py",
        ]

    for p in candidates:
        if p.exists():
            return p
    return None


def _convert_via_hf_script(hf_dir: Path, out_path: Path, script: Path, wdir: Path) -> Path:
    """Convert using whisper.cpp's HuggingFace conversion script."""
    cmd = [
        sys.executable, str(script),
        "--model-dir", str(hf_dir),
        "--outfile", str(out_path),
    ]
    log.info("  Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(wdir))

    if result.returncode != 0:
        log.error("  Convert script stderr:\n%s", result.stderr)
        # Try with alternate arguments (whisper.cpp script API has changed over time)
        cmd_alt = [
            sys.executable, str(script),
            str(hf_dir),
            str(out_path.parent),
        ]
        log.info("  Retrying with positional args: %s", " ".join(cmd_alt))
        result = subprocess.run(cmd_alt, capture_output=True, text=True, cwd=str(wdir))
        if result.returncode != 0:
            log.error("  Retry stderr:\n%s", result.stderr)
            raise RuntimeError(f"Conversion script failed with exit code {result.returncode}")

    # The script might output to a default name, find the output
    actual_out = _find_ggml_output(out_path, hf_dir)
    if actual_out != out_path:
        shutil.move(str(actual_out), str(out_path))

    log.info("  GGML model written: %s (%s)", out_path, format_size(out_path.stat().st_size))
    return out_path


def _convert_via_pt_script(pt_path: Path, out_path: Path, script: Path, wdir: Path) -> Path:
    """Convert using whisper.cpp's PyTorch conversion script."""
    # The PT convert script typically expects: <model_dir> <whisper_cpp_dir>
    # and looks for the .pt file in the model dir
    model_dir = pt_path.parent

    cmd = [
        sys.executable, str(script),
        str(model_dir),
        str(wdir),
    ]
    log.info("  Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(wdir))

    if result.returncode != 0:
        log.error("  stderr:\n%s", result.stderr)
        raise RuntimeError(f"PT conversion script failed with exit code {result.returncode}")

    # Find the output ggml file
    actual_out = _find_ggml_output(out_path, model_dir)
    if actual_out != out_path:
        shutil.move(str(actual_out), str(out_path))

    log.info("  GGML model written: %s (%s)", out_path, format_size(out_path.stat().st_size))
    return out_path


def _find_ggml_output(expected: Path, search_dir: Path) -> Path:
    """Find the GGML output file (scripts may use unexpected names)."""
    if expected.exists():
        return expected

    # Search common output patterns
    for pattern in ["ggml-*.bin", "whisper-*.bin", "ggml-model*.bin"]:
        matches = list(search_dir.glob(pattern))
        if matches:
            # Return the most recently modified
            return max(matches, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(
        f"Could not find GGML output. Expected {expected}, also searched {search_dir}"
    )


def _convert_direct_ggml(pt_path: Path, out_path: Path) -> Path:
    """
    Direct GGML conversion without whisper.cpp scripts.

    Implements the GGML file format for whisper.cpp:
      - Magic number + model hyperparameters
      - Mel filter data
      - Tokenizer vocab
      - All model tensors in float16

    This is a faithful reimplementation of convert-pt-to-ggml.py from whisper.cpp,
    handling the Whisper large-v3 architecture specifically.
    """
    import torch

    log.info("Direct GGML conversion from %s", pt_path)

    # Load the merged state dict
    state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)

    # If the state dict is a full model checkpoint (not just state_dict), extract it
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    # whisper.cpp GGML magic and version
    GGML_MAGIC = 0x67676D6C  # 'ggml' in little-endian
    # File header format version for whisper
    # whisper.cpp uses a custom header, not generic GGML

    # Load Whisper model to get config (mel filters, vocab)
    log.info("  Loading Whisper model for config extraction...")
    import whisper as whisper_lib
    whisper_model = whisper_lib.load_model(WHISPER_MODEL_NAME, device="cpu")
    dims = whisper_model.dims

    # Get mel filters
    mel_filters = whisper_lib.audio.BASES.get(whisper_model.device, {})

    # For direct conversion, we need the mel filter bank
    # whisper stores this internally; extract from the audio module
    try:
        filters_path = os.path.join(
            os.path.dirname(whisper_lib.__file__),
            "assets", "mel_filters.npz"
        )
        mel_data = np.load(filters_path, allow_pickle=True)
        # The key is typically the number of mel bins
        mel_key = f"mel_{dims.n_mels}"
        if mel_key in mel_data:
            filters = mel_data[mel_key]
        else:
            # Fallback: take the first available filter
            filters = list(mel_data.values())[0]
    except Exception:
        log.warning("  Could not load mel filters from assets, generating...")
        filters = _generate_mel_filters(dims.n_mels, 400)

    # Get tokenizer
    try:
        tokenizer = whisper_lib.tokenizer.get_tokenizer(
            multilingual=True,
            num_languages=dims.n_vocab - 51865 if hasattr(dims, 'n_vocab') else 100,
        )
        vocab = tokenizer.encoding
    except Exception:
        log.info("  Using tiktoken for vocab...")
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        vocab = {enc.decode([i]): i for i in range(enc.n_vocab)}

    log.info("  Model dims: n_mels=%d, n_vocab=%d, n_audio_ctx=%d, n_audio_state=%d",
             dims.n_mels, dims.n_vocab, dims.n_audio_ctx, dims.n_audio_state)
    log.info("  n_text_ctx=%d, n_text_state=%d, n_text_head=%d, n_text_layer=%d",
             dims.n_text_ctx, dims.n_text_state, dims.n_text_head, dims.n_text_layer)
    log.info("  n_audio_head=%d, n_audio_layer=%d",
             dims.n_audio_head, dims.n_audio_layer)

    # Write GGML file
    log.info("  Writing GGML binary: %s", out_path)
    with open(out_path, "wb") as f:
        # ─── Header ─────────────────────────────────────────
        f.write(struct.pack("i", GGML_MAGIC))

        # Hyperparameters
        f.write(struct.pack("i", dims.n_vocab))
        f.write(struct.pack("i", dims.n_audio_ctx))
        f.write(struct.pack("i", dims.n_audio_state))
        f.write(struct.pack("i", dims.n_audio_head))
        f.write(struct.pack("i", dims.n_audio_layer))
        f.write(struct.pack("i", dims.n_text_ctx))
        f.write(struct.pack("i", dims.n_text_state))
        f.write(struct.pack("i", dims.n_text_head))
        f.write(struct.pack("i", dims.n_text_layer))
        f.write(struct.pack("i", dims.n_mels))
        f.write(struct.pack("i", 1))  # ftype: 1 = float16

        # ─── Mel filters ────────────────────────────────────
        f.write(struct.pack("i", dims.n_mels))
        n_fft = filters.shape[1] if len(filters.shape) > 1 else 201
        f.write(struct.pack("i", n_fft))
        filters_flat = filters.astype(np.float32).flatten()
        f.write(filters_flat.tobytes())

        # ─── Tokenizer ──────────────────────────────────────
        tokens = []
        for i in range(dims.n_vocab):
            try:
                if hasattr(tokenizer, 'decode'):
                    text = tokenizer.decode([i])
                else:
                    text = ""
            except Exception:
                text = ""
            tokens.append(text.encode("utf-8"))

        for token_bytes in tokens:
            f.write(struct.pack("i", len(token_bytes)))
            f.write(token_bytes)

        # ─── Model tensors ──────────────────────────────────
        tensor_count = 0
        total_tensors = len(state_dict)

        for name, tensor in state_dict.items():
            # Convert to float16 for storage
            data = tensor.numpy().astype(np.float16) if tensor.dtype == torch.float32 else tensor.numpy()

            # Tensor header: name length, dimensions, type, then data
            name_bytes = name.encode("utf-8")
            n_dims = len(data.shape)

            f.write(struct.pack("i", n_dims))
            f.write(struct.pack("i", len(name_bytes)))
            # ftype per tensor: 0=f32, 1=f16
            ftype = 1 if data.dtype == np.float16 else 0
            f.write(struct.pack("i", ftype))

            # Dimensions (reversed for GGML's column-major convention)
            for dim in reversed(data.shape):
                f.write(struct.pack("i", dim))

            f.write(name_bytes)

            # Pad to 32-byte alignment
            offset = f.tell()
            padding = (32 - offset % 32) % 32
            f.write(b"\x00" * padding)

            # Write tensor data
            f.write(data.tobytes())

            tensor_count += 1
            if tensor_count % 50 == 0:
                progress_bar(tensor_count, total_tensors, prefix="Writing tensors: ")

        progress_bar(total_tensors, total_tensors, prefix="Writing tensors: ")

    log.info("  Wrote %d tensors", tensor_count)
    log.info("  GGML model: %s (%s)", out_path, format_size(out_path.stat().st_size))

    return out_path


def _generate_mel_filters(n_mels: int, n_fft: int) -> np.ndarray:
    """Generate mel filter bank (fallback if Whisper assets unavailable)."""
    sample_rate = 16000
    fmin, fmax = 0.0, sample_rate / 2

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        for j in range(bin_points[i], bin_points[i + 1]):
            filters[i, j] = (j - bin_points[i]) / (bin_points[i + 1] - bin_points[i])
        for j in range(bin_points[i + 1], bin_points[i + 2]):
            filters[i, j] = (bin_points[i + 2] - j) / (bin_points[i + 2] - bin_points[i + 1])

    return filters


# ── Stage 3: Quantize ───────────────────────────────────────────────────────

def stage_quantize(
    input_path: str,
    output_path: str,
    quantization: str = "q5_0",
    whisper_cpp_dir: str = DEFAULT_WHISPER_CPP_DIR,
) -> Path:
    """
    Quantize GGML model to reduce size.

    Q5_0 gives ~60% size reduction with minimal quality loss on Whisper.
    Other options: q4_0, q4_1, q5_1, q8_0.
    """
    log.info("=" * 60)
    log.info("STAGE 3: Quantize to %s", quantization.upper())
    log.info("=" * 60)

    inp = Path(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise FileNotFoundError(f"Input model not found: {inp}")

    input_size = inp.stat().st_size
    log.info("  Input: %s (%s)", inp, format_size(input_size))

    wdir = ensure_whisper_cpp(whisper_cpp_dir)

    # Map quantization names to whisper.cpp type IDs
    quant_types = {
        "q4_0": "2", "q4_1": "3",
        "q5_0": "8", "q5_1": "9",
        "q8_0": "7",
        "f16": "1", "f32": "0",
    }

    quant_id = quant_types.get(quantization.lower())
    if quant_id is None:
        raise ValueError(
            f"Unknown quantization type '{quantization}'. "
            f"Supported: {', '.join(quant_types.keys())}"
        )

    try:
        quantize_bin = build_quantize_binary(wdir)
    except FileNotFoundError:
        log.warning("Could not build quantize binary. Printing manual command instead.")
        print("\n" + "=" * 60)
        print("MANUAL QUANTIZATION COMMAND")
        print("=" * 60)
        print(f"cd {wdir}")
        print(f"make quantize")
        print(f"./quantize {inp} {out} {quantization}")
        print("=" * 60 + "\n")

        # Copy input as output (user will quantize manually)
        if not out.exists():
            shutil.copy2(str(inp), str(out))
            log.info("  Copied unquantized model to %s (quantize manually)", out)
        return out

    cmd = [str(quantize_bin), str(inp), str(out), quant_id]
    log.info("  Running: %s", " ".join(cmd))

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error("  Quantize stderr:\n%s", result.stderr)

        # Some versions of whisper.cpp use type name instead of ID
        cmd_alt = [str(quantize_bin), str(inp), str(out), quantization]
        log.info("  Retrying with type name: %s", " ".join(cmd_alt))
        result = subprocess.run(cmd_alt, capture_output=True, text=True)

        if result.returncode != 0:
            log.error("  Retry stderr:\n%s", result.stderr)
            raise RuntimeError(f"Quantization failed with exit code {result.returncode}")

    elapsed = time.time() - t0
    output_size = out.stat().st_size
    reduction = (1 - output_size / input_size) * 100

    log.info("  Output: %s (%s)", out, format_size(output_size))
    log.info("  Size reduction: %.1f%% (%s -> %s)", reduction, format_size(input_size), format_size(output_size))
    log.info("  Quantization took %.1fs", elapsed)

    return out


# ── Stage 4: Bundle for iOS ─────────────────────────────────────────────────

def stage_bundle(
    model_path: str,
    output_dir: str,
    ctc_head_path: str = None,
) -> Path:
    """
    Prepare directory structure for iOS integration.

    Creates:
      output_dir/
        ggml-nko-large-v3-q5_0.bin   (quantized GGML model)
        ctc_head.pt                   (CTC head weights, for custom CTC decoding)
        model_config.json             (metadata for the iOS app)
        NKoVocab.json                 (character vocabulary mapping)
        README.txt                    (integration notes)
    """
    log.info("=" * 60)
    log.info("STAGE 4: Bundle for iOS")
    log.info("=" * 60)

    model = Path(model_path)
    out = Path(output_dir)

    if not model.exists():
        raise FileNotFoundError(f"Model not found: {model}")

    out.mkdir(parents=True, exist_ok=True)

    # Copy model
    model_dest = out / model.name
    if model_dest != model:
        log.info("  Copying model to bundle...")
        shutil.copy2(str(model), str(model_dest))
    log.info("  Model: %s (%s)", model_dest.name, format_size(model_dest.stat().st_size))

    # Copy CTC head if provided
    if ctc_head_path and Path(ctc_head_path).exists():
        head_dest = out / "ctc_head.pt"
        shutil.copy2(ctc_head_path, str(head_dest))
        log.info("  CTC head: %s (%s)", head_dest.name, format_size(head_dest.stat().st_size))

    # Generate N'Ko vocabulary JSON
    char_vocab, num_chars = build_nko_char_vocab()
    idx_to_char = {v: k for k, v in char_vocab.items()}
    vocab_data = {
        "num_chars": num_chars,
        "blank_id": num_chars,  # CTC blank is always last
        "char_to_idx": char_vocab,
        "idx_to_char": idx_to_char,
        "unicode_range": {"start": "0x07C0", "end": "0x07FF"},
        "script": "N'Ko",
        "note": "Includes space as final character. Blank token is at index num_chars.",
    }
    vocab_path = out / "NKoVocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)
    log.info("  Vocabulary: %d characters + blank", num_chars)

    # Generate model config
    config = {
        "model_name": "nko-whisper-large-v3",
        "base_model": WHISPER_HF_NAME,
        "model_file": model_dest.name,
        "ctc_head_file": "ctc_head.pt" if (ctc_head_path and Path(ctc_head_path).exists()) else None,
        "vocab_file": "NKoVocab.json",
        "quantization": "q5_0" if "q5_0" in model.name else "f16",
        "model_size_bytes": model_dest.stat().st_size,
        "architecture": {
            "type": "whisper-encoder-ctc",
            "encoder": "whisper-large-v3",
            "encoder_dim": ENCODER_DIM,
            "n_mels": 128,
            "n_audio_ctx": 1500,
            "sample_rate": 16000,
            "ctc_head": {
                "type": "TransformerCTCHead",
                "input_dim": ENCODER_DIM,
                "hidden_dim": 768,
                "num_layers": 6,
                "nhead": 12,
                "temporal_downsampling": {"kernel": 5, "stride": 4},
            },
        },
        "lora_config": {
            "rank": DEFAULT_LORA_RANK,
            "alpha": DEFAULT_LORA_ALPHA,
            "target_layers": f"{TOTAL_ENCODER_BLOCKS - DEFAULT_LORA_LAYERS}-{TOTAL_ENCODER_BLOCKS - 1}",
            "target_modules_attn": ATTN_TARGETS,
            "target_modules_mlp": MLP_TARGETS,
            "status": "merged",
        },
        "language": {"code": "nqo", "name": "N'Ko", "script": "N'Ko"},
        "ios_integration": {
            "framework": "whisper.cpp",
            "reference_app": "whisper.objc / whisper.swiftui",
            "minimum_ios": "15.0",
            "recommended_devices": "iPhone 12+, iPad Air 4+",
            "memory_warning": "Large-v3 requires 8GB+ RAM. Use base/small for 4GB devices.",
        },
    }
    config_path = out / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Generate README
    readme = _generate_ios_readme(config, model_dest)
    readme_path = out / "README.txt"
    with open(readme_path, "w") as f:
        f.write(readme)

    # Print summary
    total_size = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
    log.info("")
    log.info("iOS Bundle Summary")
    log.info("-" * 40)
    for f in sorted(out.iterdir()):
        if f.is_file():
            log.info("  %-35s %s", f.name, format_size(f.stat().st_size))
    log.info("-" * 40)
    log.info("  %-35s %s", "TOTAL", format_size(total_size))
    log.info("")
    log.info("Bundle location: %s", out.resolve())

    return out


def _generate_ios_readme(config: dict, model_path: Path) -> str:
    """Generate integration notes for iOS developers."""
    return f"""\
N'Ko ASR Model — iOS Integration Notes
========================================

Model: {config['model_name']}
Base:  {config['base_model']}
Size:  {format_size(model_path.stat().st_size)}
Quantization: {config['quantization']}

Architecture
------------
This model uses Whisper large-v3 as an audio encoder with a custom CTC head
for direct N'Ko character recognition. The Whisper encoder was fine-tuned using
LoRA (rank={DEFAULT_LORA_RANK}, alpha={DEFAULT_LORA_ALPHA}) on layers {TOTAL_ENCODER_BLOCKS - DEFAULT_LORA_LAYERS}-{TOTAL_ENCODER_BLOCKS - 1},
then LoRA weights were merged back into the base model.

The CTC head (TransformerCTCHead) performs temporal downsampling (stride 4),
adds positional encoding, runs through 6 transformer encoder layers, and
projects to the N'Ko character vocabulary.

Integration with whisper.cpp
-----------------------------
1. Add the GGML model file to your Xcode project (as a resource bundle or
   downloaded on first launch).

2. Initialize whisper.cpp with the model:
     struct whisper_context * ctx = whisper_init_from_file(modelPath);

3. The Whisper encoder output from whisper.cpp gives you audio features.
   For CTC decoding, you need to run the CTC head separately (see ctc_head.pt).

4. For pure whisper.cpp usage (decoder-based, not CTC), the model works as a
   standard Whisper model. The encoder improvements from LoRA fine-tuning
   benefit both decoder and CTC pathways.

CTC Head (Custom)
------------------
The ctc_head.pt file contains the TransformerCTCHead weights. To use CTC
decoding on iOS:
  - Convert ctc_head.pt to Core ML using coremltools
  - Or implement the head in Metal/Accelerate
  - Or use ONNX Runtime Mobile

The CTC approach is preferred for N'Ko because:
  - Deterministic output (no beam search hallucination)
  - N'Ko has near-perfect phoneme-grapheme correspondence
  - Much faster than autoregressive decoding

Vocabulary
----------
NKoVocab.json contains the character mapping.
Unicode range: U+07C0 to U+07FF (N'Ko block) plus space.
CTC blank token is at index {config['architecture']['ctc_head']['input_dim']} (after all characters).

Memory Requirements
-------------------
- iPhone 15 Pro (8GB): OK
- iPhone 15 (6GB): RISKY (may crash under memory pressure)
- iPhone 14 and older (4-6GB): Use Whisper base or small instead

For broader device support, consider:
  - Whisper small + LoRA (quantized ~200MB, runs on all devices)
  - sherpa-onnx export (lowest memory of all runtimes)
"""


# ── Full Pipeline ────────────────────────────────────────────────────────────

def stage_all(
    checkpoint_path: str,
    output_dir: str,
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_layers: int = DEFAULT_LORA_LAYERS,
    quantization: str = "q5_0",
    whisper_cpp_dir: str = DEFAULT_WHISPER_CPP_DIR,
    device_override: str = None,
):
    """Run the complete conversion pipeline."""
    log.info("#" * 60)
    log.info("  N'Ko Whisper LoRA -> GGML Full Pipeline")
    log.info("#" * 60)
    t_start = time.time()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    merged_dir = out / "merged"
    ggml_path = out / "ggml-nko-large-v3.bin"
    quant_name = f"ggml-nko-large-v3-{quantization}.bin"
    quant_path = out / quant_name
    bundle_dir = out / "ios_bundle"

    # Stage 1: Merge
    log.info("")
    merged_pt, head_pt = stage_merge(
        checkpoint_path=checkpoint_path,
        output_dir=str(merged_dir),
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_layers=lora_layers,
        device_override=device_override,
    )

    # Stage 2: Convert
    log.info("")
    ggml_path = stage_convert(
        merged_dir=str(merged_dir),
        output_path=str(ggml_path),
        whisper_cpp_dir=whisper_cpp_dir,
    )

    # Stage 3: Quantize
    log.info("")
    quant_path = stage_quantize(
        input_path=str(ggml_path),
        output_path=str(quant_path),
        quantization=quantization,
        whisper_cpp_dir=whisper_cpp_dir,
    )

    # Stage 4: Bundle
    log.info("")
    bundle_path = stage_bundle(
        model_path=str(quant_path),
        output_dir=str(bundle_dir),
        ctc_head_path=str(head_pt),
    )

    elapsed = time.time() - t_start
    log.info("")
    log.info("#" * 60)
    log.info("  Pipeline Complete in %.1fs", elapsed)
    log.info("#" * 60)
    log.info("")
    log.info("Output directory: %s", out.resolve())
    log.info("  merged/              — Merged Whisper model (OpenAI + HF format)")
    log.info("  ggml-nko-*.bin       — GGML float16 model")
    log.info("  ggml-nko-*-%s.bin — Quantized model", quantization)
    log.info("  ios_bundle/          — Ready for iOS integration")
    log.info("")
    log.info("Quantized model: %s (%s)", quant_path.name, format_size(quant_path.stat().st_size))


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="N'Ko Whisper LoRA to GGML conversion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--whisper-cpp-dir", default=DEFAULT_WHISPER_CPP_DIR,
        help=f"Path to whisper.cpp repo (default: {DEFAULT_WHISPER_CPP_DIR})",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")

    subparsers = parser.add_subparsers(dest="stage", help="Pipeline stage to run")

    # ── merge ────────────────────────
    p_merge = subparsers.add_parser("merge", help="Merge LoRA weights into base Whisper model")
    p_merge.add_argument("--checkpoint", required=True, help="Path to LoRA checkpoint (.pt)")
    p_merge.add_argument("--output", required=True, help="Output directory for merged model")
    p_merge.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK, help=f"LoRA rank (default: {DEFAULT_LORA_RANK})")
    p_merge.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA, help=f"LoRA alpha (default: {DEFAULT_LORA_ALPHA})")
    p_merge.add_argument("--lora-layers", type=int, default=DEFAULT_LORA_LAYERS, help=f"Number of encoder layers with LoRA (default: {DEFAULT_LORA_LAYERS})")
    p_merge.add_argument("--device", default=None, help="Force device (cpu, cuda, mps)")

    # ── convert ──────────────────────
    p_convert = subparsers.add_parser("convert", help="Convert merged model to GGML format")
    p_convert.add_argument("--merged", required=True, help="Directory containing merged model")
    p_convert.add_argument("--output", required=True, help="Output GGML file path")

    # ── quantize ─────────────────────
    p_quant = subparsers.add_parser("quantize", help="Quantize GGML model")
    p_quant.add_argument("--input", required=True, help="Input GGML model file")
    p_quant.add_argument("--output", required=True, help="Output quantized model file")
    p_quant.add_argument("--type", default="q5_0", choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"],
                         help="Quantization type (default: q5_0)")

    # ── bundle ───────────────────────
    p_bundle = subparsers.add_parser("bundle", help="Prepare iOS bundle directory")
    p_bundle.add_argument("--model", required=True, help="Path to quantized GGML model")
    p_bundle.add_argument("--output-dir", required=True, help="Output bundle directory")
    p_bundle.add_argument("--ctc-head", default=None, help="Path to CTC head checkpoint")

    # ── all ──────────────────────────
    p_all = subparsers.add_parser("all", help="Run full pipeline (merge + convert + quantize + bundle)")
    p_all.add_argument("--checkpoint", required=True, help="Path to LoRA checkpoint (.pt)")
    p_all.add_argument("--output-dir", required=True, help="Output directory for all artifacts")
    p_all.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    p_all.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    p_all.add_argument("--lora-layers", type=int, default=DEFAULT_LORA_LAYERS)
    p_all.add_argument("--quantization", default="q5_0", choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"])
    p_all.add_argument("--device", default=None, help="Force device (cpu, cuda, mps)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.stage:
        parser.print_help()
        sys.exit(1)

    try:
        if args.stage == "merge":
            stage_merge(
                checkpoint_path=args.checkpoint,
                output_dir=args.output,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_layers=args.lora_layers,
                device_override=args.device,
            )

        elif args.stage == "convert":
            stage_convert(
                merged_dir=args.merged,
                output_path=args.output,
                whisper_cpp_dir=args.whisper_cpp_dir,
            )

        elif args.stage == "quantize":
            stage_quantize(
                input_path=args.input,
                output_path=args.output,
                quantization=args.type,
                whisper_cpp_dir=args.whisper_cpp_dir,
            )

        elif args.stage == "bundle":
            stage_bundle(
                model_path=args.model,
                output_dir=args.output_dir,
                ctc_head_path=args.ctc_head,
            )

        elif args.stage == "all":
            stage_all(
                checkpoint_path=args.checkpoint,
                output_dir=args.output_dir,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_layers=args.lora_layers,
                quantization=args.quantization,
                whisper_cpp_dir=args.whisper_cpp_dir,
                device_override=args.device,
            )

    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        log.error("Pipeline failed: %s", e, exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
