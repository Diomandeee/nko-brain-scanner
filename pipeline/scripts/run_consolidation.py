#!/usr/bin/env python3
"""
Pass 2: Consolidation + Deduplication

Queries all detections from Supabase, deduplicates N'Ko phrases,
and creates vocabulary entries for unique phrases.

Usage:
    python run_consolidation.py                 # Run consolidation
    python run_consolidation.py --dry-run       # Show stats without writing
    python run_consolidation.py --threshold 0.8 # Custom similarity threshold
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
import re
import aiohttp

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from nko_analyzer import load_config

# Output file for Pass 3
VOCABULARY_FILE = Path(__file__).parent.parent / "data" / "vocabulary.json"
CONSOLIDATION_STATS = Path(__file__).parent.parent / "data" / "consolidation_stats.json"


def normalize_nko_text(text: str) -> str:
    """
    Normalize N'Ko text for deduplication.
    
    - Removes extra whitespace
    - Removes punctuation
    - Lowercases (N'Ko doesn't have case, but for consistency)
    """
    if not text:
        return ""
    
    # Remove common punctuation
    text = re.sub(r'[،؟!.,:;""''\[\](){}]', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def compute_similarity(text1: str, text2: str) -> float:
    """
    Compute similarity between two N'Ko texts using character-level Jaccard.
    
    Returns:
        Similarity score 0.0 to 1.0
    """
    if not text1 or not text2:
        return 0.0
    
    # Character-level comparison (better for N'Ko)
    chars1 = set(text1)
    chars2 = set(text2)
    
    intersection = len(chars1 & chars2)
    union = len(chars1 | chars2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def deduplicate_phrases(
    detections: List[Dict[str, Any]],
    similarity_threshold: float = 0.85,
) -> Tuple[Dict[str, Dict], Dict[str, List[str]]]:
    """
    Deduplicate N'Ko phrases across all detections.
    
    Args:
        detections: List of detection records from Supabase
        similarity_threshold: Minimum similarity to consider as duplicate
        
    Returns:
        Tuple of:
        - vocabulary: Dict mapping canonical text to metadata
        - phrase_to_detections: Dict mapping canonical text to detection IDs
    """
    print(f"Deduplicating {len(detections)} detections...")
    
    vocabulary = {}  # canonical_text -> {latin, english, confidence, count, sources}
    phrase_to_detections = defaultdict(list)  # canonical_text -> [detection_ids]
    
    # Group by normalized text first (exact matches)
    normalized_groups = defaultdict(list)
    for det in detections:
        nko_text = det.get("nko_text", "")
        if not nko_text:
            continue
        
        normalized = normalize_nko_text(nko_text)
        if normalized:
            normalized_groups[normalized].append(det)
    
    print(f"  Exact match groups: {len(normalized_groups)}")
    
    # Now find similar groups (fuzzy matching)
    canonical_texts = list(normalized_groups.keys())
    merged = set()  # Texts that have been merged into another
    
    for i, text1 in enumerate(canonical_texts):
        if text1 in merged:
            continue
        
        # Find similar texts
        similar_texts = [text1]
        for j, text2 in enumerate(canonical_texts[i+1:], i+1):
            if text2 in merged:
                continue
            
            if compute_similarity(text1, text2) >= similarity_threshold:
                similar_texts.append(text2)
                merged.add(text2)
        
        # Merge all similar texts into the longest one (canonical)
        canonical = max(similar_texts, key=len)
        
        # Collect all detections for this group
        all_dets = []
        for text in similar_texts:
            all_dets.extend(normalized_groups[text])
        
        # Find best metadata (highest confidence)
        best_det = max(all_dets, key=lambda d: d.get("confidence", 0))
        
        # Build vocabulary entry
        vocabulary[canonical] = {
            "nko_text": canonical,
            "latin_text": best_det.get("latin_text"),
            "english_text": best_det.get("english_text"),
            "confidence": best_det.get("confidence", 0),
            "count": len(all_dets),
            "sources": list(set(d.get("frame_id", "") for d in all_dets)),
            "variants": similar_texts if len(similar_texts) > 1 else [],
        }
        
        # Map to detection IDs
        phrase_to_detections[canonical] = [d.get("id") for d in all_dets if d.get("id")]
    
    print(f"  Unique phrases after dedup: {len(vocabulary)}")
    print(f"  Dedup ratio: {len(detections) / len(vocabulary):.1f}x")
    
    return vocabulary, dict(phrase_to_detections)


async def fetch_all_detections(supabase_url: str, supabase_key: str) -> List[Dict]:
    """Fetch all detections from Supabase."""
    print("Fetching detections from Supabase...")
    
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
    }
    
    all_detections = []
    offset = 0
    limit = 1000
    
    async with aiohttp.ClientSession() as session:
        while True:
            url = f"{supabase_url}/rest/v1/nko_detections?select=*&offset={offset}&limit={limit}"
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise Exception(f"Supabase error: {error}")
                
                detections = await resp.json()
                if not detections:
                    break
                
                all_detections.extend(detections)
                offset += limit
                print(f"  Fetched {len(all_detections)} detections...")
    
    return all_detections


async def run_consolidation(
    config: Dict[str, Any],
    similarity_threshold: float = 0.85,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run Pass 2: Consolidation + Deduplication.
    
    Returns:
        Consolidation statistics
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    
    print(f"\n{'='*60}")
    print(f"PASS 2: CONSOLIDATION + DEDUPLICATION")
    print(f"{'='*60}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"{'='*60}\n")
    
    # Fetch all detections
    detections = await fetch_all_detections(supabase_url, supabase_key)
    
    if not detections:
        print("No detections found! Run Pass 1 first.")
        return {"error": "No detections"}
    
    print(f"\nTotal detections: {len(detections)}")
    
    # Deduplicate
    vocabulary, phrase_to_detections = deduplicate_phrases(
        detections,
        similarity_threshold=similarity_threshold,
    )
    
    # Statistics
    stats = {
        "timestamp": datetime.now().isoformat(),
        "total_detections": len(detections),
        "unique_phrases": len(vocabulary),
        "dedup_ratio": len(detections) / len(vocabulary) if vocabulary else 0,
        "top_phrases": sorted(
            vocabulary.values(),
            key=lambda x: x["count"],
            reverse=True
        )[:20],
        "phrase_to_detections": phrase_to_detections,
    }
    
    print(f"\n{'='*60}")
    print(f"CONSOLIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Unique phrases: {stats['unique_phrases']}")
    print(f"Dedup ratio: {stats['dedup_ratio']:.1f}x")
    print(f"\nTop 10 phrases by frequency:")
    for i, phrase in enumerate(stats["top_phrases"][:10], 1):
        nko = phrase["nko_text"][:30]
        latin = (phrase.get("latin_text") or "")[:20]
        print(f"  {i}. ({phrase['count']}x) {nko}... → {latin}...")
    
    if dry_run:
        print(f"\n[DRY RUN] Would save {len(vocabulary)} vocabulary entries")
        return stats
    
    # Save vocabulary for Pass 3
    VOCABULARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(VOCABULARY_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "vocabulary": list(vocabulary.values()),
            "phrase_to_detections": phrase_to_detections,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nVocabulary saved to: {VOCABULARY_FILE}")
    
    # Save stats
    with open(CONSOLIDATION_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"Stats saved to: {CONSOLIDATION_STATS}")
    
    # Estimate Pass 3 cost
    world_cost = len(vocabulary) * 5 * 0.0001
    print(f"\nEstimated Pass 3 cost: ${world_cost:.2f} ({len(vocabulary)} phrases × 5 worlds)")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Pass 2: Consolidation + Deduplication")
    parser.add_argument("--dry-run", action="store_true", help="Show stats without writing")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold (0.0-1.0)")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()
    
    # Load environment
    from dotenv import load_dotenv
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    
    # Load config
    config = load_config(args.config)
    
    # Run consolidation
    stats = asyncio.run(run_consolidation(
        config=config,
        similarity_threshold=args.threshold,
        dry_run=args.dry_run,
    ))
    
    print(f"\n{'='*60}")
    print(f"PASS 2 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

