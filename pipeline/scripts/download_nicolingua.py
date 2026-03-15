#!/usr/bin/env python3
"""
Download and Integrate nicolingua-0005 Corpus
==============================================

Downloads the nicolingua-0005 corpus from GitHub and integrates it into
the LearnN'Ko training data pipeline.

nicolingua-0005 contains:
- 130,850 parallel segments (N'Ko ↔ English/French)
- 3.96 million monolingual N'Ko words
- 25,848 trilingual entries (N'Ko + English + French)

Source: https://github.com/mdoumbouya/nicolingua-0005-nqo-nmt-resources
Paper: "Machine Translation for Nko" (WMT 2023, Doumbouya et al.)

Usage:
    python download_nicolingua.py
    python download_nicolingua.py --upload-supabase
"""

import os
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import csv

# Configuration
REPO_URL = "https://github.com/mdoumbouya/nicolingua-0005-nqo-nmt-resources.git"
DATA_DIR = Path(__file__).parent.parent / "data" / "nicolingua"
EXPORTS_DIR = Path(__file__).parent.parent / "data" / "exports"

# Supabase config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")


def clone_repository() -> bool:
    """Clone the nicolingua repository."""
    print("=" * 60)
    print("📥 Downloading nicolingua-0005 corpus")
    print("=" * 60)
    
    if DATA_DIR.exists():
        print(f"   Repository already exists at {DATA_DIR}")
        print("   Pulling latest changes...")
        try:
            subprocess.run(
                ["git", "-C", str(DATA_DIR), "pull"],
                check=True,
                capture_output=True
            )
            print("   ✅ Updated to latest")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️ Pull failed: {e}")
            return True  # Continue with existing data
    
    print(f"   Cloning from {REPO_URL}...")
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(DATA_DIR)],
            check=True,
        )
        print("   ✅ Repository cloned successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Clone failed: {e}")
        return False


def find_data_files() -> Dict[str, List[Path]]:
    """Find all data files in the repository."""
    files = {
        "parallel_sets": [],  # Groups of aligned files
        "monolingual": [],
        "all_files": [],
    }
    
    data_subdir = DATA_DIR / "data"
    if not data_subdir.exists():
        data_subdir = DATA_DIR
    
    if not data_subdir.exists():
        return files
    
    # nicolingua uses line-aligned files with extensions like:
    # .nqo_Nkoo (N'Ko), .eng_Latn (English), .fra_Latn (French)
    
    # Group files by base name
    file_groups = {}
    
    for f in data_subdir.iterdir():
        if f.is_file() and not f.name.startswith('.'):
            files["all_files"].append(f)
            
            # Extract base name (before language code)
            name = f.name
            if ".nqo_Nkoo" in name:
                base = name.replace(".nqo_Nkoo", "")
                if base not in file_groups:
                    file_groups[base] = {}
                file_groups[base]["nko"] = f
            elif ".eng_Latn" in name:
                base = name.replace(".eng_Latn", "")
                if base not in file_groups:
                    file_groups[base] = {}
                file_groups[base]["english"] = f
            elif ".fra_Latn" in name:
                base = name.replace(".fra_Latn", "")
                if base not in file_groups:
                    file_groups[base] = {}
                file_groups[base]["french"] = f
    
    # Separate parallel from monolingual
    for base, group in file_groups.items():
        if "nko" in group and (len(group) > 1):
            # Has N'Ko plus at least one other language
            files["parallel_sets"].append(group)
        elif "nko" in group:
            files["monolingual"].append(group["nko"])
    
    return files


def parse_parallel_file(file_path: Path) -> List[Dict[str, str]]:
    """Parse a parallel text file."""
    pairs = []
    
    suffix = file_path.suffix.lower()
    
    try:
        if suffix == ".tsv":
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                header = next(reader, None)
                
                for row in reader:
                    if len(row) >= 2:
                        # Determine which columns are which
                        pair = {}
                        for i, val in enumerate(row):
                            if header and i < len(header):
                                col = header[i].lower()
                                if 'nko' in col or 'nqo' in col:
                                    pair['nko'] = val
                                elif 'en' in col:
                                    pair['english'] = val
                                elif 'fr' in col:
                                    pair['french'] = val
                                else:
                                    pair[f'col_{i}'] = val
                            else:
                                pair[f'col_{i}'] = val
                        
                        if pair:
                            pairs.append(pair)
        
        elif suffix == ".json":
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    pairs = data
                elif isinstance(data, dict):
                    pairs = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else [data]
        
        elif suffix == ".jsonl":
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        pairs.append(json.loads(line))
        
        elif suffix == ".txt":
            # Assume tab-separated or parallel lines
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        pairs.append({"source": parts[0], "target": parts[1]})
    
    except Exception as e:
        print(f"   ⚠️ Error parsing {file_path.name}: {e}")
    
    return pairs


def analyze_corpus() -> Dict[str, Any]:
    """Analyze the downloaded corpus."""
    print("\n📊 Analyzing corpus structure...")
    
    files = find_data_files()
    stats = {
        "parallel_sets": len(files["parallel_sets"]),
        "monolingual_files": len(files["monolingual"]),
        "total_files": len(files["all_files"]),
        "total_pairs": 0,
        "sample_pairs": [],
    }
    
    print(f"\n   Found:")
    print(f"   • Parallel file sets: {stats['parallel_sets']}")
    print(f"   • Monolingual files: {stats['monolingual_files']}")
    print(f"   • Total files: {stats['total_files']}")
    
    # List parallel sets
    print(f"\n   Parallel sets:")
    for i, pset in enumerate(files["parallel_sets"], 1):
        langs = ", ".join(pset.keys())
        nko_file = pset.get("nko")
        if nko_file:
            lines = sum(1 for _ in open(nko_file, 'r', encoding='utf-8'))
            stats["total_pairs"] += lines
            print(f"   {i}. {nko_file.stem} [{langs}] - {lines:,} lines")
    
    # Show sample
    if files["parallel_sets"]:
        first_set = files["parallel_sets"][0]
        nko_file = first_set.get("nko")
        eng_file = first_set.get("english")
        
        if nko_file and eng_file:
            with open(nko_file, 'r', encoding='utf-8') as f_nko, \
                 open(eng_file, 'r', encoding='utf-8') as f_eng:
                for i, (nko, eng) in enumerate(zip(f_nko, f_eng)):
                    if i >= 3:
                        break
                    stats["sample_pairs"].append({
                        "nko": nko.strip()[:60],
                        "english": eng.strip()[:60]
                    })
            
            print(f"\n   Sample pairs:")
            for i, pair in enumerate(stats["sample_pairs"], 1):
                print(f"   {i}. N'Ko: {pair['nko']}...")
                print(f"      Eng: {pair['english']}...")
    
    print(f"\n   Total parallel pairs: {stats['total_pairs']:,}")
    
    return stats


def convert_to_training_format(output_dir: Optional[Path] = None) -> Dict[str, int]:
    """Convert nicolingua data to LearnN'Ko training formats."""
    print("\n🔄 Converting to training formats...")
    
    output_dir = output_dir or EXPORTS_DIR / "nicolingua"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = find_data_files()
    counts = {
        "huggingface": 0,
        "openai": 0,
        "vocabulary": 0,
        "monolingual": 0,
    }
    
    if not files["parallel_sets"]:
        print("   ⚠️ No parallel data found")
        return counts
    
    # Read all parallel sets and combine
    all_pairs = []
    
    for pset in files["parallel_sets"]:
        nko_file = pset.get("nko")
        eng_file = pset.get("english")
        fra_file = pset.get("french")
        
        if not nko_file:
            continue
        
        # Read all lines
        with open(nko_file, 'r', encoding='utf-8') as f:
            nko_lines = [line.strip() for line in f]
        
        eng_lines = []
        if eng_file and eng_file.exists():
            with open(eng_file, 'r', encoding='utf-8') as f:
                eng_lines = [line.strip() for line in f]
        
        fra_lines = []
        if fra_file and fra_file.exists():
            with open(fra_file, 'r', encoding='utf-8') as f:
                fra_lines = [line.strip() for line in f]
        
        # Combine aligned lines
        for i, nko in enumerate(nko_lines):
            eng = eng_lines[i] if i < len(eng_lines) else ""
            fra = fra_lines[i] if i < len(fra_lines) else ""
            
            if nko and (eng or fra):
                all_pairs.append({
                    "nko": nko,
                    "english": eng,
                    "french": fra,
                    "source_file": nko_file.stem,
                })
    
    print(f"   Loaded {len(all_pairs):,} parallel pairs")
    
    # HuggingFace JSONL format
    hf_file = output_dir / "nicolingua_translations.jsonl"
    with open(hf_file, 'w', encoding='utf-8') as f:
        for pair in all_pairs:
            entry = {
                "nko_text": pair["nko"],
                "english": pair["english"],
                "french": pair["french"],
                "source": "nicolingua-0005",
                "source_file": pair["source_file"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            counts["huggingface"] += 1
    
    print(f"   ✅ HuggingFace: {hf_file.name} ({counts['huggingface']:,} entries)")
    
    # OpenAI fine-tuning format (N'Ko → English)
    openai_file = output_dir / "nicolingua_finetune.jsonl"
    with open(openai_file, 'w', encoding='utf-8') as f:
        for pair in all_pairs:
            if pair["nko"] and pair["english"]:
                entry = {
                    "messages": [
                        {"role": "user", "content": f"Translate to English: {pair['nko']}"},
                        {"role": "assistant", "content": pair["english"]}
                    ]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                counts["openai"] += 1
    
    print(f"   ✅ OpenAI: {openai_file.name} ({counts['openai']:,} entries)")
    
    # Vocabulary extraction
    vocab_file = output_dir / "nicolingua_vocabulary.jsonl"
    seen_words = set()
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for pair in all_pairs:
            nko = pair["nko"]
            # N'Ko word extraction (split on spaces and punctuation)
            if nko:
                for word in nko.split():
                    # Strip common punctuation
                    word = word.strip('.,!?;:،؟!؛')
                    if word and word not in seen_words and len(word) > 1:
                        seen_words.add(word)
                        entry = {
                            "word": word,
                            "source": "nicolingua-0005",
                            "context": nko[:150],
                        }
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        counts["vocabulary"] += 1
    
    print(f"   ✅ Vocabulary: {vocab_file.name} ({counts['vocabulary']:,} unique words)")
    
    # Process monolingual data
    mono_file = output_dir / "nicolingua_monolingual.txt"
    with open(mono_file, 'w', encoding='utf-8') as f_out:
        for mono_path in files["monolingual"]:
            with open(mono_path, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    line = line.strip()
                    if line:
                        f_out.write(line + '\n')
                        counts["monolingual"] += 1
    
    if counts["monolingual"] > 0:
        print(f"   ✅ Monolingual: {mono_file.name} ({counts['monolingual']:,} lines)")
    
    return counts


async def upload_to_supabase(limit: Optional[int] = None):
    """Upload nicolingua data to Supabase."""
    print("\n☁️ Uploading to Supabase...")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("   ❌ SUPABASE_URL and SUPABASE_KEY required")
        return
    
    try:
        import aiohttp
    except ImportError:
        print("   ❌ aiohttp required: pip install aiohttp")
        return
    
    exports_file = EXPORTS_DIR / "nicolingua" / "nicolingua_translations.jsonl"
    
    if not exports_file.exists():
        print(f"   ❌ Export file not found: {exports_file}")
        print("   Run conversion first: python download_nicolingua.py")
        return
    
    # Read entries
    entries = []
    with open(exports_file, 'r', encoding='utf-8') as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    
    if limit:
        entries = entries[:limit]
    
    print(f"   Uploading {len(entries)} entries...")
    
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    
    async with aiohttp.ClientSession() as session:
        # Upload in batches
        batch_size = 100
        uploaded = 0
        
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i+batch_size]
            
            # Transform to vocabulary format
            vocab_entries = []
            for entry in batch:
                vocab_entries.append({
                    "word": entry.get("nko_text", "")[:100],
                    "latin": "",  # Would need transliteration
                    "meaning_primary": entry.get("english", "")[:500],
                    "meaning_french": entry.get("french", "")[:500],
                    "source": "nicolingua-0005",
                    "confidence": 0.9,  # High confidence - academic corpus
                    "is_dictionary_verified": True,
                })
            
            url = f"{SUPABASE_URL}/rest/v1/nko_vocabulary"
            
            async with session.post(url, headers=headers, json=vocab_entries) as resp:
                if resp.status in (200, 201):
                    uploaded += len(batch)
                    print(f"   Uploaded {uploaded}/{len(entries)}", end='\r')
                else:
                    error = await resp.text()
                    print(f"\n   ⚠️ Batch failed: {error[:100]}")
        
        print(f"\n   ✅ Uploaded {uploaded} entries to Supabase")


def print_integration_guide():
    """Print guide for integrating nicolingua with existing pipeline."""
    print("\n" + "=" * 60)
    print("📖 Integration Guide")
    print("=" * 60)
    
    print("""
The nicolingua-0005 corpus provides a strong foundation for your
N'Ko translation models. Here's how to integrate it:

1. COMBINE WITH YOUR DATA
   The corpus is now exported to:
   - training/data/exports/nicolingua/nicolingua_translations.jsonl
   
   Merge with your video-extracted translations:
   
   cat training/data/exports/nicolingua/*.jsonl \\
       training/data/exports/huggingface/*.jsonl \\
       > training/data/combined_corpus.jsonl

2. USE FOR NLLB FINE-TUNING
   Update nko/src/training/nllb_finetune.py to load combined data:
   
   data_dir = "training/data/combined_corpus.jsonl"

3. EXPAND WITH 5-WORLD GENERATION
   Queue nicolingua vocabulary for world expansion:
   
   python training/scripts/scheduled_exploration.py --source nicolingua

4. BENCHMARK COMPARISON
   nicolingua baseline: 30.83 chrF++ (English → N'Ko)
   Your target: >35 chrF++ with expanded corpus

Paper citation:
   Doumbouya et al. "Machine Translation for Nko: Tools, Corpora,
   and Baseline Results." WMT 2023.
   https://aclanthology.org/2023.wmt-1.34/
""")


def main():
    parser = argparse.ArgumentParser(
        description="Download and integrate nicolingua-0005 corpus"
    )
    parser.add_argument(
        "--upload-supabase",
        action="store_true",
        help="Upload to Supabase after conversion"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit entries for Supabase upload"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, only convert existing data"
    )
    
    args = parser.parse_args()
    
    # Download
    if not args.skip_download:
        if not clone_repository():
            print("❌ Failed to download corpus")
            return
    
    # Analyze
    stats = analyze_corpus()
    
    # Convert
    counts = convert_to_training_format()
    
    # Upload if requested
    if args.upload_supabase:
        import asyncio
        asyncio.run(upload_to_supabase(limit=args.limit))
    
    # Print guide
    print_integration_guide()
    
    print("\n✅ nicolingua-0005 integration complete!")
    print(f"   Total parallel pairs available: ~130,850")
    print(f"   Monolingual N'Ko words: ~3.96 million")


if __name__ == "__main__":
    main()

