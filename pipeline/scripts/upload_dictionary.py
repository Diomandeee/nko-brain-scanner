#!/usr/bin/env python3
"""
Upload Dictionary Data to Supabase

Uploads the scraped Ankataa dictionary JSON to Supabase.
Requires the migrations to be applied first.

Usage:
    python upload_dictionary.py                    # Upload all entries
    python upload_dictionary.py --file path.json   # Upload from specific file
    python upload_dictionary.py --batch-size 100   # Adjust batch size
"""

import asyncio
import aiohttp
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Load environment
def load_env():
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ.setdefault(key, value.strip('"\''))

load_env()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")


async def upload_batch(
    session: aiohttp.ClientSession,
    entries: List[Dict[str, Any]],
    batch_num: int,
    total_batches: int,
) -> int:
    """Upload a batch of entries to Supabase."""
    url = f"{SUPABASE_URL}/rest/v1/dictionary_entries"
    
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",  # Upsert
    }
    
    try:
        async with session.post(url, headers=headers, json=entries) as resp:
            if resp.status in (200, 201):
                print(f"  Batch {batch_num}/{total_batches}: {len(entries)} entries uploaded")
                return len(entries)
            else:
                error = await resp.text()
                print(f"  Batch {batch_num}/{total_batches}: Error {resp.status} - {error[:100]}")
                return 0
    except Exception as e:
        print(f"  Batch {batch_num}/{total_batches}: Exception - {e}")
        return 0


async def upload_dictionary(
    entries: List[Dict[str, Any]],
    batch_size: int = 50,
) -> int:
    """Upload all dictionary entries to Supabase."""
    total = len(entries)
    total_batches = (total + batch_size - 1) // batch_size
    
    print(f"Uploading {total} entries in {total_batches} batches...")
    print(f"Target: {SUPABASE_URL}")
    print()
    
    uploaded = 0
    
    async with aiohttp.ClientSession() as session:
        for i in range(0, total, batch_size):
            batch = entries[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            uploaded += await upload_batch(session, batch, batch_num, total_batches)
            await asyncio.sleep(0.1)  # Small delay between batches
    
    return uploaded


def main():
    parser = argparse.ArgumentParser(description="Upload dictionary to Supabase")
    parser.add_argument("--file", type=str, help="Path to dictionary JSON file")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for uploads")
    args = parser.parse_args()
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_KEY required in environment")
        sys.exit(1)
    
    # Find dictionary file
    if args.file:
        json_file = Path(args.file)
    else:
        # Find most recent dictionary file
        data_dir = Path(__file__).parent.parent / "data" / "dictionary"
        files = sorted(data_dir.glob("ankataa_dictionary_*.json"), reverse=True)
        if not files:
            print("No dictionary files found. Run ankataa_scraper.py first.")
            sys.exit(1)
        json_file = files[0]
    
    print(f"Loading: {json_file}")
    
    with open(json_file) as f:
        entries = json.load(f)
    
    print(f"Loaded {len(entries)} entries")
    
    # Upload
    uploaded = asyncio.run(upload_dictionary(entries, args.batch_size))
    
    print()
    print(f"✓ Uploaded {uploaded}/{len(entries)} entries")


if __name__ == "__main__":
    main()

