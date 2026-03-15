#!/usr/bin/env python3
"""
Sync local training manifest files to Supabase.

Reads JSON manifests from training/data/videos/*/manifest.json
and inserts data into nko_sources, nko_frames, nko_detections tables.

Usage:
    python sync_local_to_supabase.py

Environment variables required:
    SUPABASE_URL - Training database URL (zceeunlfhcherokveyek.supabase.co)
    SUPABASE_SERVICE_KEY - Service role key for database access
"""

import json
import os
import sys
from pathlib import Path
from typing import Any
from datetime import datetime

try:
    from supabase import create_client, Client
except ImportError:
    print("Error: supabase-py not installed. Run: pip install supabase")
    sys.exit(1)

# Constants
VIDEOS_DIR = Path(__file__).parent.parent / "data" / "videos"


def get_supabase_client() -> Client:
    """Create Supabase client from environment variables."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")

    if not url or not key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        print("  export SUPABASE_URL=https://zceeunlfhcherokveyek.supabase.co")
        print("  export SUPABASE_SERVICE_KEY=<your-service-key>")
        sys.exit(1)

    return create_client(url, key)


def sync_manifest(manifest_path: Path, client: Client) -> dict[str, Any]:
    """Sync a single manifest file to Supabase."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    video_id = manifest.get("video_id")
    if not video_id:
        return {"path": str(manifest_path), "status": "error", "error": "No video_id in manifest"}

    # Check if source already exists
    existing = client.table("nko_sources").select("id").eq("external_id", video_id).execute()
    if existing.data:
        return {"video_id": video_id, "status": "skipped", "reason": "already exists", "source_id": existing.data[0]["id"]}

    # Calculate stats from manifest
    scenes = manifest.get("scenes", [])
    stats = manifest.get("stats", {})
    total_frames = stats.get("total_frames", len(scenes))
    nko_frames = stats.get("nko_frames", sum(1 for s in scenes if s.get("has_nko")))
    total_detections = sum(1 for s in scenes if s.get("has_nko") and s.get("nko_text"))

    # 1. Insert nko_sources entry
    source_data = {
        "source_type": "youtube",
        "external_id": video_id,
        "url": manifest.get("youtube_url"),
        "title": manifest.get("title"),
        "channel_name": manifest.get("channel_name"),
        "duration_seconds": (manifest.get("duration_ms", 0) or 0) / 1000,
        "status": "completed",
        "frame_count": total_frames,
        "nko_frame_count": nko_frames,
        "total_detections": total_detections,
        "metadata": stats,
    }

    source_result = client.table("nko_sources").insert(source_data).execute()
    if not source_result.data:
        return {"video_id": video_id, "status": "error", "error": "Failed to insert source"}

    source_id = source_result.data[0]["id"]

    # 2. Insert nko_frames and nko_detections for each scene
    frames_inserted = 0
    detections_inserted = 0

    for scene in scenes:
        # Insert frame
        frame_data = {
            "source_id": source_id,
            "frame_index": scene.get("index", 0),
            "timestamp_ms": scene.get("start_ms", 0),
            "has_nko": scene.get("has_nko", False),
            "detection_count": 1 if scene.get("has_nko") and scene.get("nko_text") else 0,
            "confidence": scene.get("confidence", 0.0),
            "storage_path": scene.get("frame_path"),
            "width": 1920,  # Default video dimensions
            "height": 1080,
        }

        frame_result = client.table("nko_frames").insert(frame_data).execute()
        if not frame_result.data:
            continue

        frames_inserted += 1
        frame_id = frame_result.data[0]["id"]

        # Insert detection if N'Ko text exists
        if scene.get("has_nko") and scene.get("nko_text"):
            detection_data = {
                "frame_id": frame_id,
                "nko_text": scene.get("nko_text"),
                "latin_text": scene.get("latin_text"),
                "english_text": scene.get("english_text"),
                "confidence": scene.get("confidence", 0.0),
                "status": "validated",
            }

            detection_result = client.table("nko_detections").insert(detection_data).execute()
            if detection_result.data:
                detections_inserted += 1

    return {
        "video_id": video_id,
        "source_id": source_id,
        "status": "synced",
        "frames": frames_inserted,
        "detections": detections_inserted,
    }


def main():
    """Sync all local manifests to Supabase."""
    print("=" * 60)
    print("N'Ko Training Data Sync - Local Manifests to Supabase")
    print("=" * 60)

    # Get Supabase client
    client = get_supabase_client()
    print(f"\nConnected to Supabase")

    # Find all manifest files
    manifests = list(VIDEOS_DIR.glob("*/manifest.json"))
    print(f"Found {len(manifests)} manifest files in {VIDEOS_DIR}\n")

    if not manifests:
        print("No manifest files found. Ensure training data exists in training/data/videos/*/")
        return

    results = []
    for manifest_path in manifests:
        try:
            result = sync_manifest(manifest_path, client)
            results.append(result)

            status = result.get("status")
            video_id = result.get("video_id", "unknown")

            if status == "synced":
                print(f"  [SYNCED] {video_id}: {result.get('frames', 0)} frames, {result.get('detections', 0)} detections")
            elif status == "skipped":
                print(f"  [SKIP]   {video_id}: {result.get('reason')}")
            else:
                print(f"  [ERROR]  {video_id}: {result.get('error')}")

        except Exception as e:
            print(f"  [ERROR]  {manifest_path.parent.name}: {e}")
            results.append({"path": str(manifest_path), "status": "error", "error": str(e)})

    # Summary
    synced = sum(1 for r in results if r.get("status") == "synced")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    errors = sum(1 for r in results if r.get("status") == "error")
    total_frames = sum(r.get("frames", 0) for r in results if r.get("status") == "synced")
    total_detections = sum(r.get("detections", 0) for r in results if r.get("status") == "synced")

    print("\n" + "=" * 60)
    print("SYNC COMPLETE")
    print("=" * 60)
    print(f"  Videos synced:  {synced}")
    print(f"  Videos skipped: {skipped}")
    print(f"  Errors:         {errors}")
    print(f"  Total frames:   {total_frames}")
    print(f"  Total detections: {total_detections}")
    print("=" * 60)

    # Verify counts in database
    print("\nVerifying database counts...")
    sources_count = client.table("nko_sources").select("id", count="exact").execute()
    frames_count = client.table("nko_frames").select("id", count="exact").execute()
    detections_count = client.table("nko_detections").select("id", count="exact").execute()

    print(f"  nko_sources:    {sources_count.count} records")
    print(f"  nko_frames:     {frames_count.count} records")
    print(f"  nko_detections: {detections_count.count} records")


if __name__ == "__main__":
    main()
