#!/usr/bin/env python3
"""
World Generation Pipeline for N'Ko Phrases

Reads 1,755 unique N'Ko phrases from dynamic_pairs.jsonl,
generates 5 contextual world variants per phrase using Gemini 2.5 Flash.
Saves results to results/world_generations.jsonl with resume support.

Worlds: everyday, formal, storytelling, proverbs, educational
"""

import asyncio
import aiohttp
import json
import os
import sys
import time
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "results" / "dynamic_ocr" / "dynamic_pairs.jsonl"
OUTPUT_FILE = BASE_DIR / "results" / "world_generations.jsonl"

# ─── Gemini config ────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"  # stable, fast, handles N'Ko well
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_URL = f"{GEMINI_BASE}/{GEMINI_MODEL}:generateContent"
# 2.5-flash is in preview and returns 500/503 too frequently — use 2.0-flash directly

WORLD_TYPES = [
    ("world_everyday",      "casual everyday conversation"),
    ("world_formal",        "formal or official writing"),
    ("world_storytelling",  "griot oral storytelling tradition"),
    ("world_proverbs",      "Mande proverb or wisdom saying"),
    ("world_educational",   "educational or teaching context"),
]

# ─── Rate-limit & concurrency ─────────────────────────────────────────────────
MAX_CONCURRENT = 20         # 20 phrases at a time = 100 requests in-flight max
REQUESTS_PER_SECOND = 50.0  # gemini-2.0-flash handles 100 concurrent at 1.5s no 429s
MAX_RETRIES = 3

# Global rate-limiter state
_rate_lock = None
_last_call_time = 0.0
_min_interval = 1.0 / REQUESTS_PER_SECOND


def build_prompt(nko_text: str, world_desc: str) -> str:
    return (
        f"You are an expert in N'Ko language and Mande cultures.\n\n"
        f"Given this N'Ko text: {nko_text}\n\n"
        f"Generate a {world_desc} sentence or phrase that uses or extends this text "
        f"in its natural context. The output MUST be written entirely in N'Ko script "
        f"(Unicode range ߀-߿). Do NOT include Latin transliteration, English, or "
        f"any explanation.\n\n"
        f"Return ONLY the N'Ko script sentence, nothing else."
    )


async def rate_limit():
    global _last_call_time
    async with _rate_lock:
        now = asyncio.get_event_loop().time()
        elapsed = now - _last_call_time
        if elapsed < _min_interval:
            await asyncio.sleep(_min_interval - elapsed)
        _last_call_time = asyncio.get_event_loop().time()


async def _post_gemini(
    session: aiohttp.ClientSession, url: str, payload: dict
) -> tuple[int, dict | str]:
    """POST to Gemini, return (status, json_data_or_error_text)."""
    raw_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    async with session.post(
        f"{url}?key={GOOGLE_API_KEY}",
        data=raw_body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        timeout=aiohttp.ClientTimeout(total=60),
    ) as resp:
        if resp.status == 200:
            return 200, await resp.json()
        else:
            return resp.status, await resp.text()


async def call_gemini(session: aiohttp.ClientSession, prompt: str) -> str:
    """Call Gemini 2.0 Flash API with retry/backoff. Returns text or empty string."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.8, "maxOutputTokens": 256},
    }

    for attempt in range(MAX_RETRIES):
        try:
            await rate_limit()
            status, data = await _post_gemini(session, GEMINI_URL, payload)

            if status == 200:
                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
                except (KeyError, IndexError, TypeError):
                    return ""
            elif status == 429:
                wait = 2 ** (attempt + 1)
                print(f"    [429] rate-limited, sleeping {wait}s …", flush=True)
                await asyncio.sleep(wait)
            else:
                err_text = data if isinstance(data, str) else str(data)
                print(f"    [ERR {status}] {err_text[:80]}", flush=True)
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)

        except asyncio.TimeoutError:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                print(f"    [EXC] {e}", flush=True)

    return ""


async def generate_all_worlds(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    nko_text: str,
    idx: int,
    total: int,
) -> dict:
    """Generate 5 world variants for one N'Ko phrase sequentially (same semaphore slot)."""
    async with semaphore:
        results = {"nko_original": nko_text}
        for world_key, world_desc in WORLD_TYPES:
            prompt = build_prompt(nko_text, world_desc)
            generated = await call_gemini(session, prompt)
            results[world_key] = generated

        if idx % 50 == 0 or idx <= 5:
            print(f"  [{idx}/{total}] done: {nko_text[:30]!r}", flush=True)

        return results


async def main():
    global _rate_lock

    if not GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # ── Load input phrases ──────────────────────────────────────────────────
    print(f"Loading phrases from {INPUT_FILE} …")
    phrases = []
    seen = set()
    with open(INPUT_FILE) as f:
        for line in f:
            data = json.loads(line)
            nko = data.get("nko_text", "").strip()
            if nko and nko not in seen:
                seen.add(nko)
                phrases.append(nko)

    print(f"Unique N'Ko phrases: {len(phrases)}")

    # ── Resume: skip already-generated ─────────────────────────────────────
    already_done = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    already_done.add(entry.get("nko_original", ""))
                except json.JSONDecodeError:
                    pass
        print(f"Resuming: {len(already_done)} already generated, "
              f"{len(phrases) - len(already_done)} remaining")

    todo = [p for p in phrases if p not in already_done]
    if not todo:
        print("All phrases already generated. Done.")
        return

    # ── Run generation ──────────────────────────────────────────────────────
    _rate_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    total = len(phrases)
    done_so_far = len(already_done)

    print(f"Generating worlds for {len(todo)} phrases "
          f"(model={GEMINI_MODEL}, concurrency={MAX_CONCURRENT}) …")
    t0 = time.time()

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT * 5 + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            generate_all_worlds(session, semaphore, phrase, done_so_far + i + 1, total)
            for i, phrase in enumerate(todo)
        ]

        # Process in batches and write incrementally to avoid memory buildup
        BATCH = 50
        success = 0
        with open(OUTPUT_FILE, "a") as out_f:
            for i in range(0, len(tasks), BATCH):
                batch = tasks[i : i + BATCH]
                results = await asyncio.gather(*batch, return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        print(f"  [BATCH ERR] {r}", flush=True)
                    else:
                        out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        success += 1
                out_f.flush()
                elapsed = time.time() - t0
                pct = (done_so_far + success) / total * 100
                print(
                    f"  Progress: {done_so_far + success}/{total} "
                    f"({pct:.1f}%) — {elapsed:.0f}s elapsed",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"World generation complete!")
    print(f"  Generated: {success} new entries")
    print(f"  Total in output: {done_so_far + success}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if success > 0:
        print(f"  Avg per phrase: {elapsed/success:.2f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
