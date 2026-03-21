#!/usr/bin/env python3
"""
Build the 5-layer derivation chain for EPOCH inscription.

Takes a conversation turn and runs it through all 5 transformation layers,
producing a final inscription payload with full provenance hashes.

Layers:
  0. Raw conversation text
  1. SFT curation (structured messages)
  2. N'Ko translation (via IPA bridge)
  3. Sigil compression (to 1-5 sigils)
  4. EPOCH inscription payload (all hashes + metadata)

Usage:
    # Build chain without inscribing
    python3 derivation_chain.py \
        --input sample_turn.json \
        --output results/derivation_chain.json

    # Build + inscribe on testnet (requires STX wallet)
    python3 derivation_chain.py \
        --input sample_turn.json \
        --output results/derivation_chain.json \
        --inscribe \
        --network testnet
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Add NKo to path
NKO_PATH = os.path.expanduser("~/Desktop/NKo")
if NKO_PATH not in sys.path:
    sys.path.insert(0, NKO_PATH)

try:
    from nko.transliterate import transliterate, NkoTransliterator
    TRANSLITERATOR = NkoTransliterator()
except ImportError:
    print("WARNING: nko.transliterate not available")
    TRANSLITERATOR = None

# Import sigil encoder from experiment D
SIGIL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), "..", "D_sigil_compression")
if SIGIL_ENCODER_PATH not in sys.path:
    sys.path.insert(0, SIGIL_ENCODER_PATH)

try:
    from sigil_encoder import extract_features, assign_sigils, SIGIL_BY_NAME
except ImportError:
    print("WARNING: sigil_encoder not importable. Run from experiments/ directory.")
    extract_features = None
    assign_sigils = None


def sha256(text):
    """Compute SHA-256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_layer_0(text):
    """Layer 0: Raw conversation text."""
    return {
        "layer": 0,
        "name": "raw_conversation",
        "content": text,
        "hash": sha256(text),
        "char_count": len(text),
    }


def build_layer_1(text, role="user"):
    """Layer 1: SFT curation (structured format)."""
    sft = {
        "messages": [
            {"role": role, "content": text},
        ]
    }
    sft_str = json.dumps(sft, ensure_ascii=False, sort_keys=True)
    return {
        "layer": 1,
        "name": "sft_curation",
        "content": sft_str,
        "hash": sha256(sft_str),
        "format": "SFT-JSONL",
    }


def build_layer_2(text):
    """Layer 2: N'Ko translation via IPA bridge."""
    if TRANSLITERATOR is None:
        return {
            "layer": 2,
            "name": "nko_translation",
            "content": "[transliterator not available]",
            "hash": sha256(text),
            "confidence": 0.0,
            "error": "nko.transliterate not importable",
        }

    try:
        result = TRANSLITERATOR.convert(text, target="nko")
        nko_text = result.target_text
        confidence = result.confidence
        ipa = result.ipa
    except Exception as e:
        nko_text = text
        confidence = 0.0
        ipa = ""

    return {
        "layer": 2,
        "name": "nko_translation",
        "content": nko_text,
        "hash": sha256(nko_text),
        "ipa_intermediate": ipa,
        "confidence": confidence,
        "nko_char_count": len(nko_text),
    }


def build_layer_3(text, prev_text=None):
    """Layer 3: Sigil compression."""
    if extract_features is None or assign_sigils is None:
        return {
            "layer": 3,
            "name": "sigil_compression",
            "content": "ߛ",  # default stabilization
            "hash": sha256("ߛ"),
            "sigil_count": 1,
            "error": "sigil_encoder not importable",
        }

    features = extract_features(text, prev_text)
    sigils = assign_sigils(features)
    sigil_str = "".join(sigils)

    return {
        "layer": 3,
        "name": "sigil_compression",
        "content": sigil_str,
        "hash": sha256(sigil_str),
        "sigil_count": len(sigils),
        "compression_ratio": len(text) / len(sigils) if sigils else 0,
    }


def build_layer_4(layers):
    """Layer 4: EPOCH inscription payload.

    Assembles the final on-chain record with all prior hashes.
    """
    # Collect all intermediate hashes
    hash_chain = [layer["hash"] for layer in layers]

    # Provenance hash: SHA-256 of all intermediate hashes concatenated
    provenance_str = "|".join(hash_chain)
    provenance_hash = sha256(provenance_str)

    # Sigil content (from layer 3)
    sigil_content = layers[3]["content"] if len(layers) > 3 else ""

    # Inscription payload
    payload = {
        "version": 1,
        "type": "knowledge-provenance",
        "sigils": sigil_content,
        "provenance_hash": provenance_hash,
        "layer_hashes": hash_chain,
        "confidence": layers[2].get("confidence", 0) if len(layers) > 2 else 0,
        "timestamp": int(time.time()),
    }

    # Compact form for on-chain storage (fits in ~200 bytes)
    compact = f"NKP1|{sigil_content}|{provenance_hash[:16]}|{payload['confidence']:.2f}"

    return {
        "layer": 4,
        "name": "epoch_inscription",
        "content": json.dumps(payload, ensure_ascii=False),
        "hash": sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        "compact_inscription": compact,
        "compact_bytes": len(compact.encode("utf-8")),
        "full_payload": payload,
    }


def build_chain(text, role="user", prev_text=None):
    """Build the complete 5-layer derivation chain."""
    layers = []

    layer_0 = build_layer_0(text)
    layers.append(layer_0)

    layer_1 = build_layer_1(text, role)
    layers.append(layer_1)

    layer_2 = build_layer_2(text)
    layers.append(layer_2)

    layer_3 = build_layer_3(text, prev_text)
    layers.append(layer_3)

    layer_4 = build_layer_4(layers)
    layers.append(layer_4)

    return {
        "layers": layers,
        "summary": {
            "original_chars": len(text),
            "nko_chars": layer_2.get("nko_char_count", 0),
            "sigils": layer_3["sigil_count"],
            "provenance_hash": layer_4["full_payload"]["provenance_hash"],
            "compact_inscription": layer_4["compact_inscription"],
            "compact_bytes": layer_4["compact_bytes"],
            "total_compression": len(text) / layer_3["sigil_count"] if layer_3["sigil_count"] > 0 else 0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build EPOCH derivation chain")
    parser.add_argument("--input", required=True, help="Input JSON with text field")
    parser.add_argument("--output", required=True, help="Output derivation chain JSON")
    parser.add_argument("--text-field", default="text", help="Field containing the text")
    parser.add_argument("--inscribe", action="store_true", help="Actually inscribe on-chain (requires STX)")
    parser.add_argument("--network", choices=["testnet", "mainnet"], default="testnet")
    args = parser.parse_args()

    # Load input
    with open(args.input) as f:
        data = json.load(f)

    if isinstance(data, str):
        text = data
    elif isinstance(data, dict):
        text = data.get(args.text_field, data.get("content", ""))
    else:
        print("ERROR: Input must be a JSON object with a text field or a plain string.")
        sys.exit(1)

    if not text:
        print("ERROR: No text found in input.")
        sys.exit(1)

    print(f"Input: {len(text)} chars")
    print(f"Preview: {text[:100]}...")

    # Build chain
    chain = build_chain(text)

    # Print summary
    s = chain["summary"]
    print(f"\nDerivation Chain Summary:")
    print(f"  Original chars:      {s['original_chars']}")
    print(f"  N'Ko chars:          {s['nko_chars']}")
    print(f"  Sigils:              {s['sigils']}")
    print(f"  Total compression:   {s['total_compression']:.1f}x")
    print(f"  Provenance hash:     {s['provenance_hash'][:32]}...")
    print(f"  Compact inscription: {s['compact_inscription']}")
    print(f"  Compact bytes:       {s['compact_bytes']}")

    if args.inscribe:
        print(f"\n  [INSCRIPTION NOT YET IMPLEMENTED]")
        print(f"  Network: {args.network}")
        print(f"  Would inscribe: {s['compact_inscription']}")
        print(f"  Requires: STX wallet + contract deployment")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(chain, f, indent=2, ensure_ascii=False)
    print(f"\nChain saved to {output_path}")


if __name__ == "__main__":
    main()
