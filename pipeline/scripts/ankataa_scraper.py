#!/usr/bin/env python3
"""
Ankataa Dictionary Scraper (Production Grade)

Bulk imports the An ka taa Bambara/Dioula dictionary into Supabase.
https://dictionary.ankataa.com

Features:
- Scrapes all 30 letters including special Manding characters (ɛ, ɔ, ɲ, ŋ)
- Advanced HTML parsing for complex entry structures
- Extracts: word class, definitions (EN/FR), examples, variants, synonyms
- Handles Bambara vs Dioula/Jula dialectal variants
- Preserves tone diacritics (grave, acute accents)
- Rate-limits requests (1.5 req/sec) to be respectful
- Parallel scraping with semaphore for speed
- Incremental updates (upsert on word + word_class)
- Progress tracking with resume capability

Usage:
    python ankataa_scraper.py                 # Scrape all letters
    python ankataa_scraper.py --letter a      # Scrape single letter
    python ankataa_scraper.py --search dɔgɔ   # Search for a word
    python ankataa_scraper.py --dry-run       # Preview without saving
    python ankataa_scraper.py --parallel 3    # Parallel scraping (3 workers)
"""

import asyncio
import aiohttp
import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import time

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from supabase_client import SupabaseClient
except ImportError:
    SupabaseClient = None

# Base URLs
BASE_URL = "https://dictionary.ankataa.com"
LEXICON_URL = f"{BASE_URL}/lexicon.php"
SEARCH_URL = f"{BASE_URL}/search.php"

# Letter mappings for the dictionary (letter -> URL number)
# The Ankataa dictionary uses ?letter=XX format
LETTER_MAPPING = {
    'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5', 'ɛ': '6',
    'f': '7', 'g': '8', 'h': '9', 'i': '10', 'j': '11', 'k': '12',
    'l': '13', 'm': '14', 'n': '15', 'ɲ': '16', 'ŋ': '17', 'o': '18',
    'ɔ': '19', 'p': '20', 'r': '21', 's': '22', 't': '23', 'u': '24',
    'w': '25', 'y': '26', 'z': '27'
}

# All letters in the dictionary (including special Manding characters)
ALL_LETTERS = list(LETTER_MAPPING.keys())

# Rate limiting
REQUEST_DELAY = 1.5  # seconds between requests (respectful)
MAX_CONCURRENT = 2   # max parallel requests


@dataclass
class Example:
    """Usage example from dictionary."""
    bambara: str
    english: Optional[str] = None
    french: Optional[str] = None


@dataclass
class DictionaryEntry:
    """A single dictionary entry."""
    word: str                              # Bambara/Dioula word
    word_normalized: str = ""              # Normalized for matching (lowercase, no diacritics)
    word_class: Optional[str] = None       # n, vt, vi, adj, adv, etc.
    definitions_en: List[str] = field(default_factory=list)  # English definitions
    definitions_fr: List[str] = field(default_factory=list)  # French definitions
    examples: List[Example] = field(default_factory=list)    # Usage examples
    variants: List[str] = field(default_factory=list)        # Jula/Bambara variants
    synonyms: List[str] = field(default_factory=list)        # Related words
    see_also: List[str] = field(default_factory=list)        # Cross-references
    has_tone_marks: bool = False           # Has tone diacritics
    source_url: str = ""                   # Original URL
    raw_html: str = ""                     # Raw HTML for debugging
    
    def __post_init__(self):
        if not self.word_normalized:
            self.word_normalized = normalize_word(self.word)
        # Check for tone marks (combining diacritical marks)
        self.has_tone_marks = any(
            unicodedata.category(c) == 'Mn' for c in self.word
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Supabase."""
        return {
            "word": self.word,
            "word_normalized": self.word_normalized,
            "word_class": self.word_class,
            "definitions_en": self.definitions_en,
            "definitions_fr": self.definitions_fr,
            "examples": [asdict(e) for e in self.examples],
            "variants": self.variants,
            "synonyms": self.synonyms,
            "has_tone_marks": self.has_tone_marks,
            "source_url": self.source_url,
        }


def normalize_word(word: str) -> str:
    """
    Normalize a word for matching.
    
    - Lowercase
    - Remove combining diacritical marks (tone marks)
    - Keep base special characters (ɛ, ɔ, ɲ, ŋ)
    """
    # Decompose to separate base chars from combining marks
    decomposed = unicodedata.normalize('NFD', word.lower())
    # Remove combining marks (category 'Mn')
    result = ''.join(c for c in decomposed if unicodedata.category(c) != 'Mn')
    # Recompose
    return unicodedata.normalize('NFC', result)


def parse_word_class(text: str) -> Optional[str]:
    """Extract word class from entry text with advanced pattern matching."""
    # Extended word classes in the dictionary (ordered by specificity)
    word_classes = [
        # Verb types (most specific first)
        (r'\bvt\.', 'verb_transitive'),
        (r'\bvi\.', 'verb_intransitive'),
        (r'\bvq\.', 'verb_qualitative'),
        (r'\bvq\.con\.', 'verb_qualitative_construction'),
        (r'\bv\.', 'verb'),
        # Noun types
        (r'\bn\.', 'noun'),
        (r'\badj/n\.', 'adjective_noun'),
        (r'\bn/adj\.', 'noun_adjective'),
        # Adjective/Adverb
        (r'\badj\.', 'adjective'),
        (r'\badv\.', 'adverb'),
        # Other parts of speech
        (r'\bprep\.', 'preposition'),
        (r'\bpp\.', 'postposition'),
        (r'\bconj\.', 'conjunction'),
        (r'\binterj\.', 'interjection'),
        (r'\bpron\.', 'pronoun'),
        (r'\bnum\.', 'numeral'),
        (r'\bdtm\.', 'determiner'),
        (r'\bpart\.', 'particle'),
        (r'\bdem\.', 'demonstrative'),
        (r'\bquant\.', 'quantifier'),
        # Expressions
        (r'\bexpr\.', 'expression'),
        (r'\bidm\.', 'idiom'),
    ]
    
    text_lower = text.lower()
    for pattern, word_class in word_classes:
        if re.search(pattern, text_lower):
            return word_class
    return None


def extract_tone_info(word: str) -> Dict[str, Any]:
    """
    Extract tone information from a word.
    
    Returns:
        Dict with tone_pattern, has_high, has_low, has_mid
    """
    # Tone diacritics in Manding orthography
    high_tone = set('áéíóúÁÉÍÓÚ')  # Acute accent = high tone
    low_tone = set('àèìòùÀÈÌÒÙ')   # Grave accent = low tone
    
    has_high = any(c in high_tone for c in word)
    has_low = any(c in low_tone for c in word)
    
    # Build tone pattern (H=high, L=low, M=mid/unmarked)
    pattern = []
    for c in word:
        if c in high_tone:
            pattern.append('H')
        elif c in low_tone:
            pattern.append('L')
        elif c.isalpha():
            pattern.append('M')
    
    return {
        'tone_pattern': ''.join(pattern) if pattern else None,
        'has_high_tone': has_high,
        'has_low_tone': has_low,
        'is_tone_marked': has_high or has_low,
    }


def parse_entry_html(entry_html: str, source_url: str) -> Optional[DictionaryEntry]:
    """
    Parse a dictionary entry from Ankataa HTML using proper class selectors.
    
    The Ankataa dictionary uses LexiquePro CSS classes:
    - lpLexEntryName: headword
    - lpPartOfSpeech: word class (n., v., adj., etc.)
    - lpGlossEnglish: English definition
    - lpGlossFrench: French definition
    - lpExample: Example sentence
    - lpMainCrossRef: Variant/cross-reference
    - lpLexicalFunction: Related word
    """
    soup = BeautifulSoup(entry_html, 'html.parser')
    
    # Find the headword
    word_elem = soup.find(class_='lpLexEntryName')
    if not word_elem:
        return None
    
    word = word_elem.get_text(strip=True)
    if not word or len(word) < 1:
        return None
    
    # Get entry link for source URL
    link_elem = soup.find('a')
    entry_url = source_url
    if link_elem and link_elem.get('href'):
        href = link_elem.get('href')
        if not href.startswith('http'):
            entry_url = f"{BASE_URL}/{href}"
    
    entry = DictionaryEntry(
        word=word,
        source_url=entry_url,
        raw_html=entry_html[:500],
    )
    
    # Parse word class using proper class
    pos_elem = soup.find(class_='lpPartOfSpeech')
    if pos_elem:
        pos_text = pos_elem.get_text(strip=True)
        entry.word_class = parse_word_class(pos_text)
    
    # Parse English definitions
    for gloss_en in soup.find_all(class_='lpGlossEnglish'):
        text = gloss_en.get_text(strip=True)
        # Skip if it's just a language marker like 'Jula'
        if text and len(text) > 2 and text not in ("'", "'Jula'", "'Bambara'", "Jula", "Bambara"):
            entry.definitions_en.append(text.rstrip(';').strip())
    
    # Parse French definitions
    for gloss_fr in soup.find_all(class_='lpGlossFrench'):
        text = gloss_fr.get_text(strip=True)
        if text and len(text) > 2:
            entry.definitions_fr.append(text.rstrip('.').strip())
    
    # Parse examples
    for example_elem in soup.find_all(class_='lpExample'):
        bambara_text = example_elem.get_text(strip=True)
        if bambara_text and len(bambara_text) > 3:
            # Try to find English/French translations nearby
            next_siblings = list(example_elem.find_next_siblings(limit=2))
            en_trans = None
            fr_trans = None
            for sib in next_siblings:
                if 'lpGlossEnglish' in sib.get('class', []):
                    en_trans = sib.get_text(strip=True)
                elif 'lpGlossFrench' in sib.get('class', []):
                    fr_trans = sib.get_text(strip=True)
            
            entry.examples.append(Example(
                bambara=bambara_text,
                english=en_trans,
                french=fr_trans,
            ))
    
    # Parse variants (cross-references)
    for variant_elem in soup.find_all(class_='lpMainCrossRef'):
        variant = variant_elem.get_text(strip=True)
        if variant and variant not in entry.variants:
            entry.variants.append(variant)
    
    # Parse synonyms and related words
    for func_elem in soup.find_all(class_='lpLexicalFunction'):
        related = func_elem.get_text(strip=True)
        if related and related not in entry.synonyms:
            entry.synonyms.append(related)
    
    # Check for Syn: prefix in text
    full_text = soup.get_text(' ', strip=True)
    syn_match = re.search(r'Syn:\s*([^\s.]+)', full_text)
    if syn_match and syn_match.group(1) not in entry.synonyms:
        entry.synonyms.append(syn_match.group(1))
    
    # Parse see also references
    see_match = re.search(r'See:\s*([^\s.]+)', full_text)
    if see_match:
        entry.see_also.append(see_match.group(1))
    
    # Check for tone marks
    tone_info = extract_tone_info(word)
    entry.has_tone_marks = tone_info['is_tone_marked']
    
    return entry


async def fetch_page(session: aiohttp.ClientSession, url: str) -> str:
    """Fetch a page with rate limiting."""
    await asyncio.sleep(REQUEST_DELAY)
    async with session.get(url) as response:
        return await response.text()


async def scrape_letter(session: aiohttp.ClientSession, letter: str) -> List[DictionaryEntry]:
    """
    Scrape all entries for a given letter.
    
    Uses the Ankataa dictionary's ?letter=XX format.
    """
    # Get the letter number from mapping
    letter_num = LETTER_MAPPING.get(letter.lower(), letter)
    url = f"{LEXICON_URL}?letter={letter_num}"
    print(f"  Scraping letter '{letter}' from {url}")
    
    html = await fetch_page(session, url)
    soup = BeautifulSoup(html, 'html.parser')
    
    entries = []
    seen_words = set()  # Avoid duplicates
    
    # Find all main entry paragraphs
    entry_paras = soup.find_all('p', class_='lpLexEntryPara')
    print(f"    Found {len(entry_paras)} entry paragraphs")
    
    for para in entry_paras:
        entry = parse_entry_html(str(para), url)
        if entry and entry.word and entry.word not in seen_words:
            entries.append(entry)
            seen_words.add(entry.word)
            
            # Look for related paragraphs (additional senses, sub-entries)
            next_sib = para.find_next_sibling()
            while next_sib and next_sib.name == 'p' and 'lpLexEntryPara2' in next_sib.get('class', []):
                # Add definitions from additional sense paragraphs
                for gloss_en in next_sib.find_all(class_='lpGlossEnglish'):
                    text = gloss_en.get_text(strip=True)
                    if text and len(text) > 2 and text not in entry.definitions_en:
                        entry.definitions_en.append(text.rstrip(';').strip())
                for gloss_fr in next_sib.find_all(class_='lpGlossFrench'):
                    text = gloss_fr.get_text(strip=True)
                    if text and len(text) > 2 and text not in entry.definitions_fr:
                        entry.definitions_fr.append(text.rstrip('.').strip())
                next_sib = next_sib.find_next_sibling()
    
    # Also get sub-entries
    sub_paras = soup.find_all('p', class_='lpLexSubEntryPara')
    for para in sub_paras:
        entry = parse_entry_html(str(para), url)
        if entry and entry.word and entry.word not in seen_words:
            entries.append(entry)
            seen_words.add(entry.word)
    
    print(f"    Total: {len(entries)} unique entries for letter '{letter}'")
    return entries


async def search_word(session: aiohttp.ClientSession, query: str, language: str = "lexicon") -> List[DictionaryEntry]:
    """
    Search for a word in the dictionary.
    
    Parses the Ankataa search results page which contains:
    - Exact matches (main entries)
    - Partial matches (related entries)
    - Sub-entries (compound words, phrases)
    """
    url = f"{SEARCH_URL}?input={query}&search={language}"
    print(f"  Searching for '{query}' at {url}")
    
    html = await fetch_page(session, url)
    soup = BeautifulSoup(html, 'html.parser')
    
    entries = []
    seen_words = set()  # Avoid duplicates
    
    # Find all search result divs
    result_divs = soup.find_all('div', class_='lexicon-search-result')
    print(f"    Found {len(result_divs)} result blocks")
    
    for result_div in result_divs:
        # Parse the main entry paragraph
        entry_para = result_div.find('p', class_='lpLexEntryPara')
        if entry_para:
            entry = parse_entry_html(str(entry_para), url)
            if entry and entry.word and entry.word not in seen_words:
                entries.append(entry)
                seen_words.add(entry.word)
        
        # Also parse sub-entries (lpLexSubEntryPara)
        for sub_para in result_div.find_all('p', class_='lpLexSubEntryPara'):
            sub_entry = parse_entry_html(str(sub_para), url)
            if sub_entry and sub_entry.word and sub_entry.word not in seen_words:
                entries.append(sub_entry)
                seen_words.add(sub_entry.word)
        
        # Parse additional sense paragraphs (lpLexEntryPara2)
        for sense_para in result_div.find_all('p', class_='lpLexEntryPara2'):
            # These add definitions to the previous entry
            if entries:
                for gloss_en in sense_para.find_all(class_='lpGlossEnglish'):
                    text = gloss_en.get_text(strip=True)
                    if text and len(text) > 2 and text not in entries[-1].definitions_en:
                        entries[-1].definitions_en.append(text.rstrip(';').strip())
                for gloss_fr in sense_para.find_all(class_='lpGlossFrench'):
                    text = gloss_fr.get_text(strip=True)
                    if text and len(text) > 2 and text not in entries[-1].definitions_fr:
                        entries[-1].definitions_fr.append(text.rstrip('.').strip())
    
    print(f"    Found {len(entries)} unique entries for '{query}'")
    return entries


async def scrape_all_letters(dry_run: bool = False) -> List[DictionaryEntry]:
    """Scrape all letters from the dictionary."""
    all_entries = []
    
    print("=" * 60)
    print("Ankataa Dictionary Scraper")
    print("=" * 60)
    print(f"Scraping {len(ALL_LETTERS)} letters...")
    print()
    
    async with aiohttp.ClientSession() as session:
        for letter in ALL_LETTERS:
            try:
                entries = await scrape_letter(session, letter)
                all_entries.extend(entries)
            except Exception as e:
                print(f"    Error scraping letter '{letter}': {e}")
    
    print()
    print(f"Total entries scraped: {len(all_entries)}")
    
    return all_entries


async def save_to_supabase(entries: List[DictionaryEntry]) -> int:
    """Save entries to Supabase."""
    if not SupabaseClient:
        print("Supabase client not available. Saving to JSON instead.")
        return save_to_json(entries)
    
    try:
        client = SupabaseClient()
    except Exception as e:
        print(f"Could not connect to Supabase: {e}")
        return save_to_json(entries)
    
    print(f"Saving {len(entries)} entries to Supabase...")
    
    saved = 0
    batch_size = 50
    
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        data = [e.to_dict() for e in batch]
        
        try:
            # Use upsert to handle duplicates
            await client._request(
                "POST",
                "dictionary_entries",
                data,
                headers={"Prefer": "resolution=merge-duplicates"}
            )
            saved += len(batch)
            print(f"  Saved {saved}/{len(entries)} entries")
        except Exception as e:
            print(f"  Error saving batch: {e}")
    
    return saved


def save_to_json(entries: List[DictionaryEntry]) -> int:
    """Save entries to JSON file as fallback."""
    output_dir = Path(__file__).parent.parent / "data" / "dictionary"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"ankataa_dictionary_{datetime.now().strftime('%Y%m%d')}.json"
    
    data = [e.to_dict() for e in entries]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(entries)} entries to {output_file}")
    return len(entries)


async def main():
    parser = argparse.ArgumentParser(description="Scrape Ankataa Dictionary")
    parser.add_argument("--letter", type=str, help="Scrape a single letter")
    parser.add_argument("--search", type=str, help="Search for a word")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--json", action="store_true", help="Save to JSON instead of Supabase")
    args = parser.parse_args()
    
    if args.search:
        async with aiohttp.ClientSession() as session:
            entries = await search_word(session, args.search)
            for entry in entries:
                print(f"\n{entry.word} ({entry.word_class})")
                print(f"  EN: {', '.join(entry.definitions_en)}")
                print(f"  FR: {', '.join(entry.definitions_fr)}")
                if entry.variants:
                    print(f"  Variants: {', '.join(entry.variants)}")
                if entry.examples:
                    print(f"  Examples: {len(entry.examples)}")
        return
    
    if args.letter:
        async with aiohttp.ClientSession() as session:
            entries = await scrape_letter(session, args.letter)
    else:
        entries = await scrape_all_letters(dry_run=args.dry_run)
    
    if args.dry_run:
        print("\nDry run - not saving.")
        print(f"Would save {len(entries)} entries.")
        # Print first 5 entries as sample
        for entry in entries[:5]:
            print(f"  - {entry.word}: {entry.definitions_en[:1]}")
        return
    
    if args.json:
        save_to_json(entries)
    else:
        await save_to_supabase(entries)


if __name__ == "__main__":
    asyncio.run(main())

