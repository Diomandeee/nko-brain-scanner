#!/usr/bin/env python3
"""
Verify the integrity of EPOCH inscription derivation chains.

Takes one or more inscription payloads and recomputes the SHA-256 hash
at each layer, verifying that the provenance chain is intact.

Reports: VALID or BROKEN (and exactly where it broke).

Usage:
    # Verify a single inscription JSONL
    python3 verify_chain.py --input results/sample_inscriptions.jsonl

    # Verify with verbose layer-by-layer output
    python3 verify_chain.py --input results/sample_inscriptions.jsonl --verbose

    # Verify a single line from stdin (pipe)
    echo '{"layers": [...]}' | python3 verify_chain.py --input -
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(text):
    """Compute SHA-256 hash of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_single_chain(chain, verbose=False):
    """Verify a single derivation chain.

    Checks:
    1. Each layer's hash matches SHA-256(content)
    2. Layer 4's layer_hashes array matches layers 0-3
    3. Provenance hash = SHA-256(layer_hashes joined by |)
    4. Compact inscription format is consistent

    Returns (is_valid, report_lines)
    """
    report = []
    errors = []
    layers = chain.get("layers", [])

    if len(layers) != 5:
        return False, [f"BROKEN: Expected 5 layers, got {len(layers)}"]

    input_text = chain.get("input_text", layers[0].get("content", ""))
    report.append(f"Input: {input_text[:60]}...")

    # ---- Check layers 0-3: content hash integrity ----
    for i in range(4):
        layer = layers[i]
        content = layer.get("content", "")
        expected_hash = layer.get("hash", "")
        recomputed = sha256(content)

        layer_name = layer.get("name", f"layer_{i}")

        if recomputed == expected_hash:
            if verbose:
                report.append(f"  Layer {i} ({layer_name}): VALID")
                report.append(f"    Content: {content[:80]}...")
                report.append(f"    Hash: {recomputed[:32]}...")
        else:
            errors.append(f"Layer {i} ({layer_name})")
            report.append(f"  Layer {i} ({layer_name}): BROKEN")
            report.append(f"    Content: {content[:80]}...")
            report.append(f"    Expected: {expected_hash[:32]}...")
            report.append(f"    Got:      {recomputed[:32]}...")

    # ---- Check layer 4: hash chain linkage ----
    layer4 = layers[4]
    payload = layer4.get("full_payload", {})

    # Verify layer_hashes array matches individual layer hashes
    stored_hashes = payload.get("layer_hashes", [])
    actual_hashes = [layers[i]["hash"] for i in range(4)]

    if stored_hashes == actual_hashes:
        if verbose:
            report.append(f"  Hash chain linkage: VALID (4 hashes match)")
    else:
        errors.append("Hash chain linkage")
        report.append(f"  Hash chain linkage: BROKEN")
        for i in range(min(len(stored_hashes), len(actual_hashes))):
            if stored_hashes[i] != actual_hashes[i]:
                report.append(f"    Layer {i}: stored {stored_hashes[i][:16]}... != actual {actual_hashes[i][:16]}...")

    # ---- Check provenance hash ----
    hash_chain_input = "|".join(actual_hashes)
    expected_provenance = sha256(hash_chain_input)
    stored_provenance = payload.get("provenance_hash", "")

    if expected_provenance == stored_provenance:
        if verbose:
            report.append(f"  Provenance hash: VALID")
            report.append(f"    Hash: {expected_provenance[:32]}...")
    else:
        errors.append("Provenance hash")
        report.append(f"  Provenance hash: BROKEN")
        report.append(f"    Expected: {expected_provenance[:32]}...")
        report.append(f"    Stored:   {stored_provenance[:32]}...")

    # ---- Check hash_chain_input field ----
    stored_chain_input = payload.get("hash_chain_input", "")
    if stored_chain_input == hash_chain_input:
        if verbose:
            report.append(f"  Hash chain input string: VALID")
    else:
        errors.append("Hash chain input string")
        report.append(f"  Hash chain input string: BROKEN")

    # ---- Check compact inscription format ----
    compact = layer4.get("compact_inscription", "")
    if compact.startswith("NKP1|"):
        parts = compact.split("|")
        if len(parts) == 4:
            compact_hash_prefix = parts[2]
            if stored_provenance.startswith(compact_hash_prefix):
                if verbose:
                    report.append(f"  Compact inscription: VALID ({compact})")
            else:
                errors.append("Compact hash prefix")
                report.append(f"  Compact inscription: BROKEN (hash prefix mismatch)")
                report.append(f"    Compact prefix: {compact_hash_prefix}")
                report.append(f"    Provenance:     {stored_provenance[:16]}")
        else:
            errors.append("Compact format")
            report.append(f"  Compact inscription: BROKEN (expected 4 pipe-delimited parts, got {len(parts)})")
    else:
        errors.append("Compact prefix")
        report.append(f"  Compact inscription: BROKEN (missing NKP1| prefix)")

    # ---- Check layer 4 content hash ----
    payload_canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    expected_l4_hash = sha256(payload_canonical)
    stored_l4_hash = layer4.get("hash", "")

    if expected_l4_hash == stored_l4_hash:
        if verbose:
            report.append(f"  Layer 4 payload hash: VALID")
    else:
        errors.append("Layer 4 payload hash")
        report.append(f"  Layer 4 payload hash: BROKEN")
        report.append(f"    Expected: {expected_l4_hash[:32]}...")
        report.append(f"    Stored:   {stored_l4_hash[:32]}...")

    # ---- Verdict ----
    is_valid = len(errors) == 0
    if is_valid:
        report.insert(1, "  Status: VALID (all 5 layers + provenance chain intact)")
    else:
        report.insert(1, f"  Status: BROKEN at: {', '.join(errors)}")

    return is_valid, report


def main():
    parser = argparse.ArgumentParser(
        description="Verify EPOCH inscription derivation chain integrity"
    )
    parser.add_argument("--input", required=True,
                        help="Input JSONL file with inscription chains (or - for stdin)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show layer-by-layer verification details")
    args = parser.parse_args()

    # Load chains
    chains = []
    if args.input == "-":
        for line in sys.stdin:
            line = line.strip()
            if line:
                chains.append(json.loads(line))
    else:
        with open(args.input) as f:
            for line in f:
                line = line.strip()
                if line:
                    chains.append(json.loads(line))

    if not chains:
        print("ERROR: No chains found in input.")
        sys.exit(1)

    print(f"Verifying {len(chains)} inscription chain(s)...")
    print()

    valid_count = 0
    broken_count = 0

    for i, chain in enumerate(chains):
        is_valid, report = verify_single_chain(chain, verbose=args.verbose)

        prefix = "VALID" if is_valid else "BROKEN"
        print(f"Chain {i+1}/{len(chains)}: [{prefix}]")
        for line in report:
            print(line)
        print()

        if is_valid:
            valid_count += 1
        else:
            broken_count += 1

    # Summary
    print(f"{'='*60}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total chains:  {len(chains)}")
    print(f"  Valid:          {valid_count}")
    print(f"  Broken:         {broken_count}")
    print(f"  Integrity:      {'100% INTACT' if broken_count == 0 else f'{valid_count}/{len(chains)} passed'}")

    sys.exit(0 if broken_count == 0 else 1)


if __name__ == "__main__":
    main()
