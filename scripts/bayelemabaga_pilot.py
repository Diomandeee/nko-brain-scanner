#!/usr/bin/env python3
"""Bayelemabaga Pilot: Convert Bambara-French pairs to N'Ko via cross-script bridge.

Downloads 500 Bambara-French pairs from the Bayelemabaga dataset,
converts Bambara (Latin) to N'Ko using the cross-script bridge,
and produces SFT-ready training examples.

Usage:
  python3 scripts/bayelemabaga_pilot.py
"""

import json
import os
import sys
import random

# Add cross-script bridge to path
from core.bridge import Bridge

OUTPUT_DIR = os.path.expanduser("~/Desktop/nko-brain-scanner/data/bayelemabaga")
PILOT_SIZE = 500
REVIEW_SIZE = 100

# Try to fetch from HuggingFace datasets
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("datasets library not available, trying direct download...")


def fetch_from_github():
    """Fetch Bayelemabaga from GitHub tar.gz archive."""
    import urllib.request
    import tarfile
    import io
    import csv

    url = "https://github.com/RobotsMali-AI/datasets/raw/main/bayelemabaga.tar.gz"
    print(f"Downloading from {url}...")

    try:
        response = urllib.request.urlopen(url)
        data = response.read()
        print(f"  Downloaded {len(data)} bytes")

        # Extract tar.gz
        tar = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
        members = tar.getnames()
        print(f"  Archive contains: {members[:10]}...")

        pairs = []
        for member in members:
            if member.endswith(".csv") or member.endswith(".tsv"):
                f = tar.extractfile(member)
                if f is None:
                    continue
                content = f.read().decode("utf-8")
                reader = csv.reader(io.StringIO(content))
                header = next(reader, None)
                print(f"  Reading {member} (header: {header})")
                for row in reader:
                    if len(row) >= 2:
                        bam = row[0].strip()
                        fra = row[1].strip()
                        if bam and fra and len(bam) > 3:
                            pairs.append({"bambara": bam, "french": fra})

        if not pairs:
            # Try looking for txt or json files
            for member in members:
                if "train" in member or "data" in member:
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    content = f.read().decode("utf-8", errors="replace")
                    lines = content.strip().split("\n")
                    for line in lines:
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            bam = parts[0].strip()
                            fra = parts[1].strip()
                            if bam and fra and len(bam) > 3:
                                pairs.append({"bambara": bam, "french": fra})

        tar.close()
        return pairs if pairs else None

    except Exception as e:
        print(f"  GitHub download failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def fetch_from_hf():
    """Fetch from HuggingFace datasets (parquet branch)."""
    if not HAS_DATASETS:
        return None

    try:
        ds = load_dataset(
            "RobotsMaliAI/bayelemabaga",
            split="train",
            revision="refs/convert/parquet",
        )
        pairs = []
        for row in ds:
            t = row.get("translation", {})
            bam = t.get("bam", "")
            fra = t.get("fr", "")
            if bam and fra and len(bam) > 5:
                pairs.append({"bambara": bam, "french": fra})
        print(f"  Loaded {len(pairs)} pairs from HF parquet")
        return pairs
    except Exception as e:
        print(f"  HF download failed: {e}")
        return None


def convert_bambara_to_nko(bridge, text):
    """Convert Latin Bambara text to N'Ko script."""
    try:
        return bridge.convert(text, "latin", "nko")
    except Exception:
        return None


def build_sft_examples(pairs_with_nko):
    """Build SFT training examples from converted pairs."""
    examples = []

    for item in pairs_with_nko:
        bambara = item["bambara"]
        french = item["french"]
        nko = item["nko"]

        if not nko:
            continue

        # Example 1: Translation prompt (French → N'Ko)
        examples.append({
            "messages": [
                {"role": "user", "content": f"Translate to N'Ko: {french}"},
                {"role": "assistant", "content": nko}
            ]
        })

        # Example 2: Script conversion (Latin Bambara → N'Ko)
        examples.append({
            "messages": [
                {"role": "user", "content": f"Write this Bambara sentence in N'Ko script: {bambara}"},
                {"role": "assistant", "content": nko}
            ]
        })

        # Example 3: Trilingual (every 5th pair)
        if len(examples) % 10 == 0:
            examples.append({
                "messages": [
                    {"role": "user", "content": f"What is '{french}' in Bambara (both Latin and N'Ko)?"},
                    {"role": "assistant", "content": f"Latin: {bambara}\nN'Ko: {nko}"}
                ]
            })

    return examples


def main():
    print("Bayelemabaga Pilot: Bambara → N'Ko Conversion")
    print("=" * 60)

    # Fetch data
    pairs = fetch_from_hf()
    if not pairs:
        pairs = fetch_from_github()
    if not pairs:
        print("ERROR: Could not fetch Bayelemabaga data")
        return

    print(f"Fetched {len(pairs)} Bambara-French pairs")

    # Sample pilot set
    random.seed(42)
    pilot = random.sample(pairs, min(PILOT_SIZE, len(pairs)))
    print(f"Sampled {len(pilot)} pairs for pilot")

    # Convert Bambara → N'Ko
    bridge = Bridge()
    converted = 0
    failed = 0

    for item in pilot:
        nko = convert_bambara_to_nko(bridge, item["bambara"])
        item["nko"] = nko
        if nko:
            converted += 1
        else:
            failed += 1

    print(f"\nConversion results: {converted} success, {failed} failed")

    # Build SFT examples
    sft_examples = build_sft_examples(pilot)
    print(f"Generated {len(sft_examples)} SFT training examples")

    # Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save converted pairs
    pairs_file = os.path.join(OUTPUT_DIR, "pilot_pairs.jsonl")
    with open(pairs_file, "w") as f:
        for item in pilot:
            if item.get("nko"):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nSaved pairs to {pairs_file}")

    # Save SFT examples
    sft_file = os.path.join(OUTPUT_DIR, "pilot_sft.jsonl")
    with open(sft_file, "w") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved SFT examples to {sft_file}")

    # Save review sample (first 100 for Mohamed to review)
    review_file = os.path.join(OUTPUT_DIR, "review_sample.jsonl")
    review_pairs = [p for p in pilot if p.get("nko")][:REVIEW_SIZE]
    with open(review_file, "w") as f:
        for item in review_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(review_pairs)} pairs for review to {review_file}")

    # Print sample
    print("\n" + "=" * 60)
    print("SAMPLE CONVERSIONS")
    print("=" * 60)
    for item in review_pairs[:10]:
        print(f"\n  Bambara: {item['bambara']}")
        print(f"  French:  {item['french']}")
        print(f"  N'Ko:    {item['nko']}")


if __name__ == "__main__":
    main()
