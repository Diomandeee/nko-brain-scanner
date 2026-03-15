#!/usr/bin/env python3
"""
N'Ko Training Pipeline Orchestrator

Runs the complete 3-pass pipeline:
  Pass 1: Extraction + OCR
  Pass 2: Consolidation + Deduplication  
  Pass 3: World Generation

Usage:
    python run_pipeline.py                      # Run full pipeline
    python run_pipeline.py --test               # Test on 5 videos
    python run_pipeline.py --pass 1             # Run only Pass 1
    python run_pipeline.py --pass 2             # Run only Pass 2
    python run_pipeline.py --pass 3             # Run only Pass 3
    python run_pipeline.py --resume             # Resume from checkpoint
    python run_pipeline.py --estimate           # Show cost estimate only
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from nko_analyzer import load_config

# Import pass scripts
import run_extraction
import run_consolidation
import run_worlds


def estimate_costs(num_videos: int, config: Dict[str, Any]) -> Dict[str, float]:
    """
    Estimate costs for the full pipeline.
    
    Returns:
        Dict with cost breakdown
    """
    target_frames = config.get("extraction", {}).get("target_frames", 100)
    
    # Assumptions
    frames_after_dedup = target_frames * 0.55  # ~45% reduction
    detection_rate = 0.80  # 80% of frames have N'Ko
    dedup_ratio = 3.0  # 3x deduplication in Pass 2
    
    # Pass 1: OCR
    total_ocr_calls = num_videos * frames_after_dedup
    pass1_cost = total_ocr_calls * 0.002
    
    # Pass 2: Just processing, no API calls
    pass2_cost = 0.0
    
    # Pass 3: World generation
    total_detections = total_ocr_calls * detection_rate
    unique_phrases = total_detections / dedup_ratio
    worlds_per_phrase = 5
    pass3_cost = unique_phrases * worlds_per_phrase * 0.0001
    
    return {
        "num_videos": num_videos,
        "total_frames": int(total_ocr_calls),
        "estimated_detections": int(total_detections),
        "unique_phrases": int(unique_phrases),
        "pass1_ocr": pass1_cost,
        "pass2_consolidation": pass2_cost,
        "pass3_worlds": pass3_cost,
        "total": pass1_cost + pass2_cost + pass3_cost,
    }


async def run_full_pipeline(
    config: Dict[str, Any],
    test_mode: bool = False,
    single_pass: Optional[int] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """
    Run the complete 3-pass pipeline.
    
    Args:
        config: Pipeline configuration
        test_mode: If True, only process 5 videos
        single_pass: If set, only run specified pass (1, 2, or 3)
        resume: Resume from checkpoint
        
    Returns:
        Combined results from all passes
    """
    results = {
        "start_time": datetime.now().isoformat(),
        "test_mode": test_mode,
        "passes": {},
    }
    
    limit = 5 if test_mode else None
    
    print(f"\n{'='*60}")
    print(f"N'KO TRAINING PIPELINE")
    print(f"{'='*60}")
    print(f"Mode: {'TEST (5 videos)' if test_mode else 'FULL'}")
    print(f"Resume: {resume}")
    if single_pass:
        print(f"Running: Pass {single_pass} only")
    print(f"{'='*60}\n")
    
    # Get video count for estimation
    channel_url = config.get("channel", {}).get("url", "https://www.youtube.com/@babamamadidiane")
    videos = run_extraction.get_channel_videos(channel_url, limit=limit)
    num_videos = len(videos)
    
    # Cost estimation
    costs = estimate_costs(num_videos, config)
    print(f"\n--- Cost Estimate ---")
    print(f"Videos: {costs['num_videos']}")
    print(f"Estimated frames: {costs['total_frames']}")
    print(f"Pass 1 (OCR): ${costs['pass1_ocr']:.2f}")
    print(f"Pass 2 (Consolidation): ${costs['pass2_consolidation']:.2f}")
    print(f"Pass 3 (Worlds): ${costs['pass3_worlds']:.2f}")
    print(f"Total: ${costs['total']:.2f}")
    print(f"---------------------\n")
    
    results["cost_estimate"] = costs
    
    # === PASS 1: Extraction + OCR ===
    if single_pass is None or single_pass == 1:
        print(f"\n{'='*60}")
        print(f"STARTING PASS 1: EXTRACTION + OCR")
        print(f"{'='*60}\n")
        
        pass1_result = await run_extraction.run_extraction(
            videos=videos,
            config=config,
            resume=resume,
        )
        results["passes"]["pass1"] = pass1_result
        
        if pass1_result.get("failed", 0) == pass1_result.get("total_videos", 0):
            print("Pass 1 failed completely! Stopping.")
            return results
    
    # === PASS 2: Consolidation + Deduplication ===
    if single_pass is None or single_pass == 2:
        print(f"\n{'='*60}")
        print(f"STARTING PASS 2: CONSOLIDATION + DEDUPLICATION")
        print(f"{'='*60}\n")
        
        try:
            pass2_result = await run_consolidation.run_consolidation(
                config=config,
                similarity_threshold=0.85,
            )
            results["passes"]["pass2"] = pass2_result
        except Exception as e:
            print(f"Pass 2 error: {e}")
            results["passes"]["pass2"] = {"error": str(e)}
            if single_pass == 2:
                return results
    
    # === PASS 3: World Generation ===
    if single_pass is None or single_pass == 3:
        print(f"\n{'='*60}")
        print(f"STARTING PASS 3: WORLD GENERATION")
        print(f"{'='*60}\n")
        
        try:
            pass3_result = await run_worlds.run_world_generation(
                config=config,
                resume=resume,
                batch_size=10,
            )
            results["passes"]["pass3"] = pass3_result
        except Exception as e:
            print(f"Pass 3 error: {e}")
            results["passes"]["pass3"] = {"error": str(e)}
    
    results["end_time"] = datetime.now().isoformat()
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    
    if "pass1" in results["passes"]:
        p1 = results["passes"]["pass1"]
        print(f"Pass 1: {p1.get('completed', 0)} videos, {p1.get('total_detections', 0)} detections")
    
    if "pass2" in results["passes"]:
        p2 = results["passes"]["pass2"]
        print(f"Pass 2: {p2.get('unique_phrases', 0)} unique phrases (from {p2.get('total_detections', 0)} detections)")
    
    if "pass3" in results["passes"]:
        p3 = results["passes"]["pass3"]
        print(f"Pass 3: {p3.get('total_worlds', 0)} worlds, {p3.get('total_variants', 0)} variants")
    
    # Calculate actual cost
    actual_cost = 0
    if "pass1" in results["passes"]:
        actual_cost += results["passes"]["pass1"].get("estimated_cost", 0)
    if "pass3" in results["passes"]:
        actual_cost += results["passes"]["pass3"].get("estimated_cost", 0)
    
    print(f"\nActual cost: ${actual_cost:.2f}")
    print(f"{'='*60}\n")
    
    # Save final results
    results_file = Path(__file__).parent.parent / "data" / "pipeline_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="N'Ko Training Pipeline Orchestrator")
    parser.add_argument("--test", action="store_true", help="Test mode (5 videos only)")
    parser.add_argument("--pass", type=int, dest="single_pass", choices=[1, 2, 3], help="Run only specified pass")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--estimate", action="store_true", help="Show cost estimate only")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()
    
    # Load environment
    from dotenv import load_dotenv
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    
    # Load config
    config = load_config(args.config)
    
    if args.estimate:
        # Just show cost estimate
        channel_url = config.get("channel", {}).get("url")
        videos = run_extraction.get_channel_videos(channel_url)
        costs = estimate_costs(len(videos), config)
        
        print(f"\n{'='*60}")
        print(f"COST ESTIMATE")
        print(f"{'='*60}")
        print(f"Videos: {costs['num_videos']}")
        print(f"Estimated frames: {costs['total_frames']}")
        print(f"Estimated detections: {costs['estimated_detections']}")
        print(f"Unique phrases (after dedup): {costs['unique_phrases']}")
        print(f"\nCost breakdown:")
        print(f"  Pass 1 (OCR): ${costs['pass1_ocr']:.2f}")
        print(f"  Pass 2 (Consolidation): ${costs['pass2_consolidation']:.2f}")
        print(f"  Pass 3 (Worlds): ${costs['pass3_worlds']:.2f}")
        print(f"  Total: ${costs['total']:.2f}")
        print(f"{'='*60}\n")
        return
    
    # Run pipeline
    asyncio.run(run_full_pipeline(
        config=config,
        test_mode=args.test,
        single_pass=args.single_pass,
        resume=args.resume,
    ))


if __name__ == "__main__":
    main()

