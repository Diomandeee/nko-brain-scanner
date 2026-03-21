#!/usr/bin/env python3
"""
Build the 5-layer derivation chain for EPOCH inscription.

Takes raw English text and runs it through all 5 transformation layers,
producing a final inscription payload with full provenance hashes.

Layers:
  0. Raw text input (English or any source language)
  1. Concept extraction (TF-IDF keyword extraction)
  2. N'Ko transliteration (via IPA bridge from ~/Desktop/NKo)
  3. Sigil compression (information-theoretic sigil mapping from Experiment D)
  4. EPOCH inscription payload (JSON with all layers + SHA-256 provenance hash)

Inputs:
  --input   : Plain text, single JSON, or JSONL file
  --output  : Output JSONL file (one inscription payload per line)
  --limit   : Max entries to process from JSONL input
  --text-field : JSON field name containing text (default: "english")

Usage:
    # From parallel corpus (JSONL)
    python3 derivation_chain.py \\
        --input ../../data/parallel_corpus.jsonl \\
        --output results/sample_inscriptions.jsonl \\
        --limit 10

    # From plain text
    python3 derivation_chain.py \\
        --input "Knowledge is power" \\
        --output results/single_inscription.jsonl

    # From single JSON file
    python3 derivation_chain.py \\
        --input sample_turn.json \\
        --output results/derivation_chain.jsonl
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

# ---- Dependency imports ----

# Add NKo to path
NKO_PATH = os.path.expanduser("~/Desktop/NKo")
if NKO_PATH not in sys.path:
    sys.path.insert(0, NKO_PATH)

try:
    from nko.transliterate import transliterate, NkoTransliterator
    TRANSLITERATOR = NkoTransliterator()
except ImportError:
    print("WARNING: nko.transliterate not available. Layer 2 will use fallback.")
    TRANSLITERATOR = None
    transliterate = None

# Import sigil encoder from experiment D
SIGIL_ENCODER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "D_sigil_compression")
if SIGIL_ENCODER_PATH not in sys.path:
    sys.path.insert(0, SIGIL_ENCODER_PATH)

try:
    from sigil_encoder import extract_features, assign_sigils, SIGIL_BY_NAME, SIGILS
except ImportError:
    print("WARNING: sigil_encoder not importable. Layer 3 will use fallback.")
    extract_features = None
    assign_sigils = None
    SIGIL_BY_NAME = None
    SIGILS = None


# ---- Hashing ----

def sha256(text):
    """Compute SHA-256 hash of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---- Layer 0: Raw Text ----

def build_layer_0(text):
    """Layer 0: Accept raw English text, hash it."""
    return {
        "layer": 0,
        "name": "raw_text",
        "content": text,
        "hash": sha256(text),
        "char_count": len(text),
        "word_count": len(text.split()),
    }


# ---- Layer 1: Concept Extraction ----

# Simple stop words for English TF-IDF
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "am", "it", "its",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "they", "them", "their", "this", "that", "these",
    "those", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "and", "but", "or", "nor", "not", "no", "so",
    "if", "then", "than", "too", "very", "just", "about", "up", "out",
    "off", "over", "under", "again", "further", "once", "here", "there",
    "when", "where", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "what", "which", "who", "whom", "don't", "doesn't", "didn't",
    "won't", "wouldn't", "shan't", "shouldn't", "can't", "cannot",
    "couldn't", "mustn't", "let's", "that's", "who's", "what's",
    "here's", "there's", "when's", "where's", "why's", "how's",
}


def extract_concepts(text, top_k=5):
    """Extract top-k keywords using TF-IDF-like scoring.

    Uses term frequency with inverse document frequency approximation
    based on word rarity (shorter, more common words score lower).
    """
    # Tokenize: lowercase, strip punctuation
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    # Remove stop words and very short words
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    if not tokens:
        return [], ""

    # Term frequency
    tf = Counter(tokens)
    total = len(tokens)

    # IDF approximation: longer, rarer words get higher scores
    # log(word_length) serves as a proxy for specificity
    scores = {}
    for word, count in tf.items():
        term_freq = count / total
        # IDF proxy: word length correlates with specificity
        idf_proxy = math.log(1 + len(word))
        scores[word] = term_freq * idf_proxy

    # Sort by score, take top_k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, score in ranked[:top_k]]

    # Canonical concept string (sorted for deterministic hashing)
    concept_str = "|".join(sorted(keywords))

    return keywords, concept_str


def build_layer_1(text, top_k=5):
    """Layer 1: Concept extraction using TF-IDF keyword scoring."""
    keywords, concept_str = extract_concepts(text, top_k=top_k)

    return {
        "layer": 1,
        "name": "concept_extraction",
        "content": concept_str,
        "hash": sha256(concept_str) if concept_str else sha256(""),
        "keywords": keywords,
        "keyword_count": len(keywords),
    }


# ---- Layer 2: N'Ko Transliteration ----

def build_layer_2(text):
    """Layer 2: N'Ko transliteration via IPA bridge."""
    if TRANSLITERATOR is None:
        # Fallback: hash the raw text, mark as unavailable
        return {
            "layer": 2,
            "name": "nko_transliteration",
            "content": "[transliterator not available]",
            "hash": sha256(text),
            "confidence": 0.0,
            "ipa_intermediate": "",
            "nko_char_count": 0,
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
        "name": "nko_transliteration",
        "content": nko_text,
        "hash": sha256(nko_text),
        "ipa_intermediate": ipa,
        "confidence": confidence,
        "nko_char_count": len(nko_text),
    }


# ---- Layer 3: Sigil Compression ----

def build_layer_3(text, prev_text=None):
    """Layer 3: Sigil compression via Experiment D encoder."""
    if extract_features is None or assign_sigils is None:
        # Fallback: single stabilization sigil
        fallback_sigil = "\u07DB"  # N'Ko Sa (stabilization)
        return {
            "layer": 3,
            "name": "sigil_compression",
            "content": fallback_sigil,
            "hash": sha256(fallback_sigil),
            "sigil_count": 1,
            "compression_ratio": len(text),
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
        "compression_ratio": round(len(text) / len(sigils), 2) if sigils else 0,
    }


# ---- Layer 4: EPOCH Inscription Payload ----

def build_layer_4(layers):
    """Layer 4: EPOCH inscription payload with full provenance chain.

    Assembles the final inscription record containing:
    - All 4 intermediate layer hashes
    - Provenance hash (SHA-256 of concatenated layer hashes)
    - Sigil content and confidence score
    - Compact form for on-chain storage
    - Clarity contract call parameters
    """
    # Collect all intermediate hashes (layers 0-3)
    hash_chain = [layer["hash"] for layer in layers]

    # Provenance hash: SHA-256 of pipe-delimited layer hashes
    provenance_str = "|".join(hash_chain)
    provenance_hash = sha256(provenance_str)

    # Extract key data from prior layers
    sigil_content = layers[3]["content"]
    confidence = layers[2].get("confidence", 0.0)
    nko_text = layers[2]["content"]
    keywords = layers[1].get("keywords", [])

    # Timestamp for this inscription
    ts = int(time.time())

    # Full inscription payload
    payload = {
        "version": 1,
        "type": "knowledge-provenance",
        "sigils": sigil_content,
        "provenance_hash": provenance_hash,
        "layer_hashes": hash_chain,
        "hash_chain_input": provenance_str,
        "confidence": round(confidence, 4),
        "timestamp": ts,
        "keywords": keywords,
        "nko_text": nko_text,
    }

    # Deterministic payload string for the layer-4 hash
    # (sort_keys + ensure_ascii for reproducibility)
    payload_canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    # Compact form for on-chain memo/storage
    compact = f"NKP1|{sigil_content}|{provenance_hash[:16]}|{confidence:.2f}"

    # Clarity contract parameters (matching nko-inscription.clar)
    clarity_params = {
        "nko-text": nko_text[:1024],  # max 1024 UTF-8 chars
        "inscription-hash": provenance_hash,
        "claim-type": _sigil_to_claim_type(sigil_content),
        "sigil": sigil_content[:4],   # max 4 UTF-8 chars
        "confidence": min(int(confidence * 10000), 10000),  # scaled uint, max 10000
        "density": _compute_density(layers),
        "basin-id": "knowledge-provenance",
        "depth": len(hash_chain),
    }

    return {
        "layer": 4,
        "name": "epoch_inscription",
        "content": payload_canonical,
        "hash": sha256(payload_canonical),
        "compact_inscription": compact,
        "compact_bytes": len(compact.encode("utf-8")),
        "full_payload": payload,
        "clarity_params": clarity_params,
    }


def _sigil_to_claim_type(sigil_content):
    """Map the first sigil character to its claim type for the Clarity contract."""
    if SIGILS is None:
        return "stabilize"
    sigil_map = {s["char"]: s["name"] for s in SIGILS}
    # Map sigil names to Clarity claim type constants
    name_to_claim = {
        "stabilization": "stabilize",
        "dispersion": "dispersion",
        "transition": "transition",
        "return": "return",
        "dwell": "dwell",
        "oscillation": "oscillation",
        "recovery": "recovery",
        "novelty": "novelty",
        "place_shift": "place-shift",
        "echo": "echo",
    }
    if sigil_content:
        first_char = sigil_content[0]
        name = sigil_map.get(first_char, "stabilization")
        return name_to_claim.get(name, "stabilize")
    return "stabilize"


def _compute_density(layers):
    """Compute information density: bits of state per N'Ko stroke.

    density = (char_entropy_of_original * word_count) / nko_char_count
    Scaled by 1000 for uint storage (e.g., 2500 = 2.5 bits/stroke).
    """
    original_chars = layers[0].get("char_count", 1)
    nko_chars = layers[2].get("nko_char_count", 1) or 1
    sigil_count = layers[3].get("sigil_count", 1) or 1

    # Rough density: original chars compressed through sigils, relative to N'Ko length
    density = (original_chars / nko_chars) * (1 / sigil_count) * 1000
    return min(int(density), 9999)


# ---- Full Chain Builder ----

def build_chain(text, prev_text=None):
    """Build the complete 5-layer derivation chain for a single text input."""
    layers = []

    layer_0 = build_layer_0(text)
    layers.append(layer_0)

    layer_1 = build_layer_1(text)
    layers.append(layer_1)

    layer_2 = build_layer_2(text)
    layers.append(layer_2)

    layer_3 = build_layer_3(text, prev_text)
    layers.append(layer_3)

    layer_4 = build_layer_4(layers)
    layers.append(layer_4)

    return {
        "input_text": text,
        "layers": [
            {k: v for k, v in layer.items()} for layer in layers
        ],
        "summary": {
            "original_chars": layer_0["char_count"],
            "original_words": layer_0["word_count"],
            "keywords": layer_1.get("keywords", []),
            "nko_chars": layer_2.get("nko_char_count", 0),
            "sigil_count": layer_3["sigil_count"],
            "sigils": layer_3["content"],
            "compression_ratio": layer_3.get("compression_ratio", 0),
            "confidence": layer_2.get("confidence", 0),
            "provenance_hash": layer_4["full_payload"]["provenance_hash"],
            "compact_inscription": layer_4["compact_inscription"],
            "compact_bytes": layer_4["compact_bytes"],
            "clarity_claim_type": layer_4["clarity_params"]["claim-type"],
        },
    }


# ---- Input Loading ----

def load_texts(input_path, text_field="english", limit=None):
    """Load texts from various input formats.

    Supports:
    - JSONL files (one JSON object per line)
    - Single JSON files
    - Plain text (treated as a single entry)

    Returns list of text strings.
    """
    input_path = str(input_path)

    # Check if input is a file path or raw text
    if not os.path.exists(input_path):
        # Treat as raw text input
        return [input_path]

    texts = []
    path = Path(input_path)

    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Try multiple field names
                text = None
                if isinstance(entry, str):
                    text = entry
                elif isinstance(entry, dict):
                    # Priority: specified field > english > text > content
                    for field in [text_field, "english", "text", "content"]:
                        if field in entry and entry[field]:
                            text = entry[field]
                            break
                    # SFT message format
                    if text is None and "messages" in entry:
                        for msg in entry["messages"]:
                            if msg.get("content", "").strip():
                                text = msg["content"]
                                break

                if text and text.strip():
                    texts.append(text.strip())

                if limit and len(texts) >= limit:
                    break

    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)

        if isinstance(data, str):
            texts.append(data)
        elif isinstance(data, dict):
            for field in [text_field, "english", "text", "content"]:
                if field in data and data[field]:
                    texts.append(data[field])
                    break
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    for field in [text_field, "english", "text", "content"]:
                        if field in item and item[field]:
                            texts.append(item[field])
                            break
                if limit and len(texts) >= limit:
                    break
    else:
        # Plain text file
        with open(path) as f:
            content = f.read().strip()
        if content:
            texts.append(content)

    return texts


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description="Build EPOCH derivation chain: text -> concepts -> N'Ko -> sigils -> inscription"
    )
    parser.add_argument("--input", required=True,
                        help="Input: plain text, JSON file, or JSONL file")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file for inscription payloads")
    parser.add_argument("--text-field", default="english",
                        help="JSON field containing the text (default: english)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max entries to process from JSONL input")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of keywords to extract per text (default: 5)")
    parser.add_argument("--inscribe", action="store_true",
                        help="[Not implemented] Actually inscribe on-chain")
    parser.add_argument("--network", choices=["testnet", "mainnet"], default="testnet",
                        help="Blockchain network (for --inscribe)")
    args = parser.parse_args()

    # Load texts
    texts = load_texts(args.input, text_field=args.text_field, limit=args.limit)

    if not texts:
        print("ERROR: No texts found in input.")
        sys.exit(1)

    print(f"Loaded {len(texts)} text(s) from {args.input}")
    print(f"Transliterator: {'ACTIVE' if TRANSLITERATOR else 'UNAVAILABLE'}")
    print(f"Sigil encoder:  {'ACTIVE' if extract_features else 'UNAVAILABLE'}")
    print()

    # Build chains
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    prev_text = None
    total_compact_bytes = 0
    total_payload_bytes = 0

    for i, text in enumerate(texts):
        chain = build_chain(text, prev_text=prev_text)
        results.append(chain)
        prev_text = text

        s = chain["summary"]
        total_compact_bytes += s["compact_bytes"]
        payload_bytes = len(chain["layers"][4]["content"].encode("utf-8"))
        total_payload_bytes += payload_bytes

        # Progress output
        print(f"  [{i+1}/{len(texts)}] {text[:50]}...")
        print(f"         Keywords: {', '.join(s['keywords'])}")
        print(f"         Sigils: {s['sigils']} ({s['sigil_count']}) | "
              f"Confidence: {s['confidence']:.2f} | "
              f"Claim: {s['clarity_claim_type']}")
        print(f"         Compact: {s['compact_inscription']} ({s['compact_bytes']}B)")

    # Write output JSONL
    with open(output_path, "w") as f:
        for chain in results:
            f.write(json.dumps(chain, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"DERIVATION CHAIN SUMMARY")
    print(f"{'='*60}")
    print(f"  Texts processed:      {len(results)}")
    print(f"  Total compact bytes:  {total_compact_bytes}")
    print(f"  Total payload bytes:  {total_payload_bytes}")
    print(f"  Avg compact bytes:    {total_compact_bytes / len(results):.1f}")
    print(f"  Avg payload bytes:    {total_payload_bytes / len(results):.1f}")

    # Aggregate stats
    avg_compression = sum(r["summary"]["compression_ratio"] for r in results) / len(results)
    avg_confidence = sum(r["summary"]["confidence"] for r in results) / len(results)
    claim_types = Counter(r["summary"]["clarity_claim_type"] for r in results)

    print(f"  Avg compression:      {avg_compression:.1f}x")
    print(f"  Avg confidence:       {avg_confidence:.2f}")
    print(f"  Claim type breakdown: {dict(claim_types)}")
    print(f"\nOutput: {output_path}")

    if args.inscribe:
        print(f"\n  [INSCRIPTION NOT YET IMPLEMENTED]")
        print(f"  Network: {args.network}")
        print(f"  Would inscribe {len(results)} payloads to nko-inscription.clar")
        print(f"  Estimated gas: ~{len(results) * 0.02:.2f} STX "
              f"(~${len(results) * 0.02:.2f} at $1/STX)")


if __name__ == "__main__":
    main()
