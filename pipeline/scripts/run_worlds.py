#!/usr/bin/env python3
"""
Pass 3: World Generation

Generates 5 world variants for each unique phrase from Pass 2 vocabulary.
Stores trajectories in Supabase linked back to original detections.

Usage:
    python run_worlds.py                    # Generate worlds for all phrases
    python run_worlds.py --limit 100        # Limit to first 100 phrases
    python run_worlds.py --resume           # Resume from checkpoint
    python run_worlds.py --batch-size 10    # Process in batches of 10
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import aiohttp

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from nko_analyzer import load_config
from world_generator import WorldGenerator
from retry_utils import retry_with_backoff, RetryError

# Input/output files
VOCABULARY_FILE = Path(__file__).parent.parent / "data" / "vocabulary.json"
CHECKPOINT_FILE = Path(__file__).parent.parent / "data" / "worlds_checkpoint.json"
PROGRESS_FILE = Path(__file__).parent.parent / "data" / "worlds_progress.json"


def load_vocabulary() -> Dict[str, Any]:
    """Load vocabulary from Pass 2."""
    if not VOCABULARY_FILE.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {VOCABULARY_FILE}\nRun Pass 2 first!")
    
    with open(VOCABULARY_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint() -> Dict[str, Any]:
    """Load checkpoint for resumability."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed_phrases": [], "failed_phrases": [], "last_updated": None}


def save_checkpoint(checkpoint: Dict[str, Any]):
    """Save checkpoint."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    checkpoint["last_updated"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def save_progress(progress: Dict[str, Any]):
    """Save progress statistics."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


async def store_trajectory_in_supabase(
    session: aiohttp.ClientSession,
    supabase_url: str,
    supabase_key: str,
    phrase: Dict[str, Any],
    worlds: List[Dict[str, Any]],
    detection_ids: List[str],
) -> Optional[str]:
    """Store world trajectory in Supabase."""
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    
    # Create trajectory
    trajectory_data = {
        "name": f"Worlds for: {phrase['nko_text'][:50]}",
        "description": f"5 world variants for '{phrase.get('latin_text', '')}'",
        "trajectory_type": "world_exploration",
        "total_nodes": len(worlds),
        "dominant_phase": "exploration",
        "is_complete": True,
        "is_successful": True,
        "metadata": {
            "source_phrase": phrase["nko_text"],
            "latin": phrase.get("latin_text"),
            "english": phrase.get("english_text"),
            "detection_count": phrase.get("count", 0),
            "detection_ids": detection_ids[:10],  # Store first 10 for reference
        },
    }
    
    async with session.post(
        f"{supabase_url}/rest/v1/nko_trajectories",
        json=trajectory_data,
        headers=headers,
    ) as resp:
        if resp.status not in (200, 201):
            error = await resp.text()
            print(f"    Warning: Failed to create trajectory: {error[:100]}")
            return None
        
        result = await resp.json()
        trajectory_id = result[0]["id"] if result else None
    
    if not trajectory_id:
        return None
    
    # Create trajectory nodes for each world
    for i, world in enumerate(worlds):
        node_data = {
            "trajectory_id": trajectory_id,
            "node_index": i,
            "trajectory_depth": 1,
            "trajectory_phase": "exploration",
            "salience_score": 0.5,
            "content_preview": world.get("world_name", "")[:255],
            "content_type": f"world_{world.get('world_name', 'unknown')}",
            "outcome": "success" if world.get("variants") else "empty",
            "timestamp_ms": int(datetime.now().timestamp() * 1000),
            "metadata": world,
        }
        
        async with session.post(
            f"{supabase_url}/rest/v1/nko_trajectory_nodes",
            json=node_data,
            headers=headers,
        ) as resp:
            if resp.status not in (200, 201):
                error = await resp.text()
                print(f"    Warning: Failed to create node {i}: {error[:50]}")
    
    return trajectory_id


async def run_world_generation(
    config: Dict[str, Any],
    limit: Optional[int] = None,
    resume: bool = False,
    batch_size: int = 10,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run Pass 3: World Generation for all unique phrases.
    
    Returns:
        Progress statistics
    """
    # Load vocabulary from Pass 2
    vocab_data = load_vocabulary()
    vocabulary = vocab_data.get("vocabulary", [])
    phrase_to_detections = vocab_data.get("phrase_to_detections", {})
    
    if not vocabulary:
        raise ValueError("Empty vocabulary! Run Pass 2 first.")
    
    # Apply limit
    if limit:
        vocabulary = vocabulary[:limit]
    
    # Resume from checkpoint
    checkpoint = load_checkpoint() if resume else {"completed_phrases": [], "failed_phrases": []}
    completed_set = set(checkpoint["completed_phrases"])
    
    if resume:
        vocabulary = [p for p in vocabulary if p["nko_text"] not in completed_set]
        print(f"Resuming: {len(completed_set)} completed, {len(vocabulary)} remaining")
    
    if dry_run:
        print(f"\n=== DRY RUN ===")
        print(f"Phrases to process: {len(vocabulary)}")
        print(f"Estimated cost: ${len(vocabulary) * 5 * 0.0001:.2f}")
        return {"dry_run": True, "total_phrases": len(vocabulary)}
    
    # Supabase config
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    
    # Progress tracking
    progress = {
        "start_time": datetime.now().isoformat(),
        "total_phrases": len(vocabulary),
        "completed": 0,
        "failed": 0,
        "total_worlds": 0,
        "total_variants": 0,
        "estimated_cost": 0.0,
    }
    
    print(f"\n{'='*60}")
    print(f"PASS 3: WORLD GENERATION")
    print(f"{'='*60}")
    print(f"Phrases to process: {len(vocabulary)}")
    print(f"Worlds per phrase: 5")
    print(f"Batch size: {batch_size}")
    print(f"Estimated cost: ${len(vocabulary) * 5 * 0.0001:.2f}")
    print(f"{'='*60}\n")
    
    # Initialize world generator
    api_key = os.getenv("GEMINI_API_KEY")
    generator = WorldGenerator(api_key=api_key)
    
    # Process in batches
    async with aiohttp.ClientSession() as session:
        for batch_start in range(0, len(vocabulary), batch_size):
            batch = vocabulary[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(vocabulary) + batch_size - 1) // batch_size
            
            print(f"\n--- Batch {batch_num}/{total_batches} ({len(batch)} phrases) ---")
            
            for phrase in batch:
                nko_text = phrase["nko_text"]
                latin_text = phrase.get("latin_text", "")
                english_text = phrase.get("english_text", "")
                
                try:
                    # Generate worlds
                    result = await generator.generate_worlds(
                        nko_text=nko_text,
                        latin_text=latin_text,
                        translation=english_text,
                    )
                    
                    if result.worlds:
                        # Store in Supabase
                        detection_ids = phrase_to_detections.get(nko_text, [])
                        
                        trajectory_id = await store_trajectory_in_supabase(
                            session=session,
                            supabase_url=supabase_url,
                            supabase_key=supabase_key,
                            phrase=phrase,
                            worlds=[w.__dict__ for w in result.worlds],
                            detection_ids=detection_ids,
                        )
                        
                        checkpoint["completed_phrases"].append(nko_text)
                        progress["completed"] += 1
                        progress["total_worlds"] += len(result.worlds)
                        progress["total_variants"] += sum(len(w.variants) for w in result.worlds)
                        progress["estimated_cost"] += len(result.worlds) * 0.0001
                        
                        print(f"  ✓ {nko_text[:30]}... → {len(result.worlds)} worlds, {sum(len(w.variants) for w in result.worlds)} variants")
                    else:
                        checkpoint["failed_phrases"].append(nko_text)
                        progress["failed"] += 1
                        print(f"  ✗ {nko_text[:30]}... → No worlds generated")
                        
                except Exception as e:
                    checkpoint["failed_phrases"].append(nko_text)
                    progress["failed"] += 1
                    print(f"  ✗ {nko_text[:30]}... → Error: {e}")
                
                # Rate limiting
                await asyncio.sleep(0.2)
            
            # Save checkpoint after each batch
            save_checkpoint(checkpoint)
            save_progress(progress)
            
            # Batch rate limiting
            await asyncio.sleep(1)
    
    progress["end_time"] = datetime.now().isoformat()
    save_progress(progress)
    
    print(f"\n{'='*60}")
    print(f"PASS 3 COMPLETE")
    print(f"{'='*60}")
    print(f"Completed: {progress['completed']}/{progress['total_phrases']}")
    print(f"Failed: {progress['failed']}")
    print(f"Total worlds: {progress['total_worlds']}")
    print(f"Total variants: {progress['total_variants']}")
    print(f"Estimated cost: ${progress['estimated_cost']:.2f}")
    print(f"{'='*60}\n")
    
    return progress


def main():
    parser = argparse.ArgumentParser(description="Pass 3: World Generation")
    parser.add_argument("--limit", type=int, help="Limit number of phrases to process")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--dry-run", action="store_true", help="Show stats without processing")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()
    
    # Load environment
    from dotenv import load_dotenv
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    
    # Load config
    config = load_config(args.config)
    
    # Run world generation
    progress = asyncio.run(run_world_generation(
        config=config,
        limit=args.limit,
        resume=args.resume,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    ))
    
    print(f"\nProgress saved to: {PROGRESS_FILE}")


if __name__ == "__main__":
    main()

