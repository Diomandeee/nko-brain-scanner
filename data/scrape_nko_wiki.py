#!/usr/bin/env python3
"""
Scrape all N'Ko Wikipedia articles for a pre-training corpus.

Uses the MediaWiki API at https://nqo.wikipedia.org/w/api.php to:
1. Enumerate all non-redirect article titles via action=query&list=allpages
2. Fetch wikitext content in batches of 50 via revisions API
3. Strip wikitext markup to produce plain text
4. Fallback to action=parse (HTML→plain text) for stubborn articles
5. Save a plain-text corpus (.txt) and structured JSONL (.jsonl)
"""

import html as html_mod
import json
import os
import re
import sys
import time
import urllib.request
import urllib.parse
import urllib.error

API_URL = "https://nqo.wikipedia.org/w/api.php"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TXT_PATH = os.path.join(DATA_DIR, "nko_wikipedia_corpus.txt")
JSONL_PATH = os.path.join(DATA_DIR, "nko_wikipedia_corpus.jsonl")
BATCH_SIZE = 50
MAX_RETRIES = 3
BACKOFF_BASE = 2.0

# N'Ko Unicode block: U+07C0 to U+07FF
NKO_RANGE = re.compile(r"[\u07C0-\u07FF]")


def api_request(params: dict, retries: int = MAX_RETRIES) -> dict:
    """Make a GET request to the Wikipedia API with retry + exponential backoff."""
    params["format"] = "json"
    url = API_URL + "?" + urllib.parse.urlencode(params)

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "NKoBrainScanner/1.0 (corpus builder; Python/3)"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError) as e:
            wait = BACKOFF_BASE ** attempt
            print(f"  [retry {attempt + 1}/{retries}] {e} -- waiting {wait:.1f}s", file=sys.stderr)
            time.sleep(wait)

    raise RuntimeError(f"API request failed after {retries} retries: {url[:200]}")


def count_nko_chars(text: str) -> int:
    """Count characters in the N'Ko Unicode block."""
    return len(NKO_RANGE.findall(text))


def strip_wikitext(wikitext: str) -> str:
    """Convert wikitext to plain text by stripping markup."""
    text = wikitext

    # Skip redirects
    if text.strip().upper().startswith("#REDIRECT"):
        return ""

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Remove <ref> tags and their content
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/]*/\s*>", "", text)

    # Remove other HTML tags but keep content
    text = re.sub(r"<br\s*/?\s*>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)

    # Remove {{...}} templates (handle nesting up to 3 levels deep)
    for _ in range(4):
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)

    # Remove {| ... |} tables
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)

    # Remove remaining table markup lines
    text = re.sub(r"^\s*[\|!].*$", "", text, flags=re.MULTILINE)

    # Remove categories [[Category:...]] / [[ߦߌߟߡߊ:...]]
    text = re.sub(r"\[\[[^\]]*?:[^\]]*?\]\]", "", text)

    # Convert [[link|display]] to display, [[link]] to link
    text = re.sub(r"\[\[[^\]]*?\|([^\]]*?)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]*?)\]\]", r"\1", text)

    # Remove external links [http://... display] -> display
    text = re.sub(r"\[https?://[^\s\]]*\s+([^\]]*)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\]]*\]", "", text)

    # Remove bold/italic markup
    text = re.sub(r"'{2,5}", "", text)

    # Remove magic words / behavior switches
    text = re.sub(r"__[A-Z]+__", "", text)

    # Remove section headers markup but keep text
    text = re.sub(r"^(=+)\s*(.*?)\s*\1\s*$", r"\2", text, flags=re.MULTILINE)

    # Remove list markers at start of lines
    text = re.sub(r"^[*#:;]+\s*", "", text, flags=re.MULTILINE)

    # Unescape HTML entities
    text = html_mod.unescape(text)

    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def html_to_plain(html_text: str) -> str:
    """Convert rendered HTML to plain text."""
    text = re.sub(r"<br\s*/?\s*>", "\n", html_text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_mod.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def enumerate_all_titles() -> list:
    """Fetch every non-redirect article title from the N'Ko Wikipedia."""
    titles = []
    params = {
        "action": "query",
        "list": "allpages",
        "aplimit": "500",
        "apnamespace": "0",
        "apfilterredir": "nonredirects",
    }

    while True:
        data = api_request(params)
        pages = data.get("query", {}).get("allpages", [])
        for p in pages:
            titles.append(p["title"])

        cont = data.get("continue")
        if cont and "apcontinue" in cont:
            params["apcontinue"] = cont["apcontinue"]
            params["continue"] = cont.get("continue", "")
        else:
            break

    return titles


def fetch_wikitext_batch(titles: list) -> dict:
    """Fetch raw wikitext for a batch of titles (max 50) via revisions API."""
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "titles": "|".join(titles),
    }
    data = api_request(params)
    pages = data.get("query", {}).get("pages", {})
    result = {}
    for page_id, page in pages.items():
        title = page.get("title", "")
        revisions = page.get("revisions", [])
        if revisions:
            wikitext = revisions[0].get("slots", {}).get("main", {}).get("*", "")
            result[title] = wikitext
    return result


def fetch_parsed_text(title: str) -> str:
    """Fallback: fetch rendered HTML via action=parse and convert to plain text."""
    try:
        params = {
            "action": "parse",
            "page": title,
            "prop": "text",
            "disablelimitreport": "true",
        }
        data = api_request(params)
        html_text = data.get("parse", {}).get("text", {}).get("*", "")
        return html_to_plain(html_text)
    except Exception:
        return ""


def main():
    print("=== N'Ko Wikipedia Corpus Builder ===")
    print(f"API: {API_URL}")
    print(f"Output TXT:  {TXT_PATH}")
    print(f"Output JSONL: {JSONL_PATH}")
    print()

    # Step 1: Enumerate all titles (non-redirects only)
    print("Step 1: Enumerating all non-redirect article titles...")
    t0 = time.time()
    titles = enumerate_all_titles()
    elapsed = time.time() - t0
    print(f"  Found {len(titles)} article titles in {elapsed:.1f}s")
    print()

    # Step 2: Fetch content in batches via revisions API
    print(f"Step 2: Fetching article texts in batches of {BATCH_SIZE}...")
    t0 = time.time()

    total_articles = 0
    total_chars = 0
    total_nko = 0
    total_lines = 0
    skipped = 0
    fallback_count = 0

    with open(TXT_PATH, "w", encoding="utf-8") as txt_f, \
         open(JSONL_PATH, "w", encoding="utf-8") as jsonl_f:

        for batch_start in range(0, len(titles), BATCH_SIZE):
            batch = titles[batch_start : batch_start + BATCH_SIZE]

            try:
                wikitext_map = fetch_wikitext_batch(batch)
            except RuntimeError as e:
                print(f"  [ERROR] Batch at {batch_start} failed: {e}", file=sys.stderr)
                skipped += len(batch)
                continue

            for title in batch:
                wikitext = wikitext_map.get(title, "")
                if not wikitext:
                    skipped += 1
                    continue

                # Strip wikitext to plain text
                text = strip_wikitext(wikitext)

                # If stripping yielded very little, try the parse fallback
                nko_in_text = count_nko_chars(text)
                nko_in_source = count_nko_chars(wikitext)
                if nko_in_text < 10 and nko_in_source >= 10:
                    fallback_text = fetch_parsed_text(title)
                    if len(fallback_text) > len(text):
                        text = fallback_text
                        fallback_count += 1

                # Skip if no meaningful content
                if len(text) < 5:
                    skipped += 1
                    continue

                nko_chars = count_nko_chars(text)
                line_count = text.count("\n") + 1

                # TXT: one article block, blank line separator
                txt_f.write(text + "\n\n")

                # JSONL
                record = {"title": title, "text": text, "nko_chars": nko_chars}
                jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                total_articles += 1
                total_chars += len(text)
                total_nko += nko_chars
                total_lines += line_count

            processed = batch_start + len(batch)
            if processed % 100 < BATCH_SIZE or processed >= len(titles):
                elapsed = time.time() - t0
                print(f"  [{processed}/{len(titles)}] articles processed "
                      f"({total_articles} saved, {skipped} skipped) -- {elapsed:.1f}s")

    elapsed_total = time.time() - t0

    # Step 3: Final stats
    txt_size = os.path.getsize(TXT_PATH)
    jsonl_size = os.path.getsize(JSONL_PATH)

    print()
    print("=" * 50)
    print("FINAL STATS")
    print("=" * 50)
    print(f"Total articles saved:    {total_articles:,}")
    print(f"Skipped (empty/error):   {skipped:,}")
    print(f"Parse fallbacks used:    {fallback_count:,}")
    print(f"Total characters:        {total_chars:,}")
    print(f"Total N'Ko characters:   {total_nko:,}")
    print(f"Total lines:             {total_lines:,}")
    print(f"TXT file size:           {txt_size:,} bytes ({txt_size / 1024 / 1024:.2f} MB)")
    print(f"JSONL file size:         {jsonl_size:,} bytes ({jsonl_size / 1024 / 1024:.2f} MB)")
    print(f"Fetch time:              {elapsed_total:.1f}s")
    print(f"Files written:")
    print(f"  {TXT_PATH}")
    print(f"  {JSONL_PATH}")


if __name__ == "__main__":
    main()
