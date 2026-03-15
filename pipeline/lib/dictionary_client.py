"""
Dictionary Client with Caching and Real-Time Lookup

Provides access to the Ankataa Bambara/Dioula dictionary with:
- Local cache lookup first (Supabase)
- Real-time API lookup for cache misses
- Fuzzy matching for partial matches
- Batch enrichment for vocabulary

Usage:
    client = DictionaryClient()
    
    # Single lookup
    entry = await client.lookup("dɔgɔ")
    
    # Batch enrichment
    enriched = await client.enrich_vocabulary(detections)
"""

import asyncio
import aiohttp
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

# Supabase config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_KEY")

# Ankataa URLs
ANKATAA_SEARCH_URL = "https://dictionary.ankataa.com/search.php"

# Rate limiting for API calls
REQUEST_DELAY = 1.0


@dataclass
class DictionaryLookupResult:
    """Result from a dictionary lookup."""
    word: str
    word_normalized: str
    word_class: Optional[str] = None
    definitions_en: List[str] = field(default_factory=list)
    definitions_fr: List[str] = field(default_factory=list)
    examples: List[Dict] = field(default_factory=list)
    variants: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    has_tone_marks: bool = False
    source: str = "cache"  # "cache" or "api"
    match_score: float = 1.0  # Similarity score for fuzzy matches
    
    @property
    def primary_english(self) -> Optional[str]:
        """Get primary English definition."""
        return self.definitions_en[0] if self.definitions_en else None
    
    @property
    def primary_french(self) -> Optional[str]:
        """Get primary French definition."""
        return self.definitions_fr[0] if self.definitions_fr else None


def normalize_word(word: str) -> str:
    """
    Normalize a word for matching.
    
    - Lowercase
    - Remove combining diacritical marks (tone marks)
    - Keep base special characters (ɛ, ɔ, ɲ, ŋ)
    """
    decomposed = unicodedata.normalize('NFD', word.lower())
    result = ''.join(c for c in decomposed if unicodedata.category(c) != 'Mn')
    return unicodedata.normalize('NFC', result)


class DictionaryClient:
    """
    Client for looking up words in the Ankataa dictionary.
    
    Uses a hybrid approach:
    1. Check local Supabase cache first
    2. If not found, query the Ankataa API
    3. Cache new results for future lookups
    """
    
    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        enable_api_fallback: bool = True,
        cache_api_results: bool = True,
    ):
        self.supabase_url = supabase_url or SUPABASE_URL
        self.supabase_key = supabase_key or SUPABASE_KEY
        self.enable_api_fallback = enable_api_fallback
        self.cache_api_results = cache_api_results
        self._session: Optional[aiohttp.ClientSession] = None
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY required")
    
    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
    
    def _get_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def lookup(
        self,
        word: str,
        fuzzy: bool = True,
        limit: int = 5,
    ) -> Optional[DictionaryLookupResult]:
        """
        Look up a word in the dictionary.
        
        Args:
            word: Word to look up (Bambara/Dioula or Latin transliteration)
            fuzzy: Allow fuzzy matching for typos/variations
            limit: Max results for fuzzy search
            
        Returns:
            DictionaryLookupResult if found, None otherwise
        """
        normalized = normalize_word(word)
        
        # 1. Check local cache first
        cached = await self._lookup_cache(normalized, fuzzy, limit)
        if cached:
            return cached[0] if len(cached) == 1 else max(cached, key=lambda x: x.match_score)
        
        # 2. Fall back to API if enabled
        if self.enable_api_fallback:
            api_results = await self._lookup_api(word)
            if api_results:
                # Cache the results
                if self.cache_api_results:
                    await self._cache_results(api_results)
                return api_results[0]
        
        return None
    
    async def lookup_many(
        self,
        words: List[str],
        fuzzy: bool = True,
    ) -> Dict[str, Optional[DictionaryLookupResult]]:
        """
        Look up multiple words in parallel.
        
        Args:
            words: List of words to look up
            fuzzy: Allow fuzzy matching
            
        Returns:
            Dict mapping word -> result (None if not found)
        """
        tasks = [self.lookup(word, fuzzy) for word in words]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            word: result if not isinstance(result, Exception) else None
            for word, result in zip(words, results)
        }
    
    async def _lookup_cache(
        self,
        normalized_word: str,
        fuzzy: bool,
        limit: int,
    ) -> List[DictionaryLookupResult]:
        """Look up word in Supabase cache."""
        session = self._get_session()
        
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
        }
        
        if fuzzy:
            # Use the search_dictionary function for fuzzy matching
            url = f"{self.supabase_url}/rest/v1/rpc/search_dictionary"
            payload = {
                "search_term": normalized_word,
                "limit_count": limit,
            }
            
            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return [self._row_to_result(row, "cache") for row in data]
            except Exception:
                pass  # Fall through to exact match
        
        # Exact match fallback
        url = f"{self.supabase_url}/rest/v1/dictionary_entries"
        params = {"word_normalized": f"eq.{normalized_word}"}
        
        try:
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [self._row_to_result(row, "cache") for row in data]
        except Exception:
            pass
        
        return []
    
    async def _lookup_api(self, word: str) -> List[DictionaryLookupResult]:
        """Look up word via Ankataa API (scraping)."""
        session = self._get_session()
        
        url = f"{ANKATAA_SEARCH_URL}?input={word}&search=lexicon"
        
        try:
            await asyncio.sleep(REQUEST_DELAY)  # Rate limiting
            
            async with session.get(url) as resp:
                if resp.status != 200:
                    return []
                
                html = await resp.text()
                return self._parse_search_results(html, url)
                
        except Exception:
            return []
    
    def _parse_search_results(self, html: str, source_url: str) -> List[DictionaryLookupResult]:
        """Parse search results from Ankataa HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        
        # Find entries (paragraphs with word definitions)
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            
            # Pattern: "word   word_class. definition; définition"
            match = re.match(
                r'^([a-zɛɔɲŋàáâãäåèéêëìíîïòóôõöùúûü]+\d?)\s+'
                r'(n\.|vt?\.|vi?\.|adj\.|adv\.)?\s*'
                r'(?:\d+\s*[•·])?\s*([^;]+?)(?:;\s*(.+))?$',
                text,
                re.IGNORECASE
            )
            
            if match:
                word = match.group(1)
                word_class = match.group(2).rstrip('.') if match.group(2) else None
                en_def = match.group(3).strip() if match.group(3) else None
                fr_def = match.group(4).strip() if match.group(4) else None
                
                # Map word class abbreviations
                class_map = {
                    'n': 'noun',
                    'v': 'verb',
                    'vt': 'verb_transitive',
                    'vi': 'verb_intransitive',
                    'adj': 'adjective',
                    'adv': 'adverb',
                }
                
                result = DictionaryLookupResult(
                    word=word,
                    word_normalized=normalize_word(word),
                    word_class=class_map.get(word_class, word_class),
                    definitions_en=[en_def] if en_def else [],
                    definitions_fr=[fr_def] if fr_def else [],
                    source="api",
                )
                results.append(result)
        
        return results
    
    async def _cache_results(self, results: List[DictionaryLookupResult]):
        """Cache API results to Supabase."""
        if not results:
            return
        
        session = self._get_session()
        
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        }
        
        url = f"{self.supabase_url}/rest/v1/dictionary_entries"
        
        for result in results:
            data = {
                "word": result.word,
                "word_normalized": result.word_normalized,
                "word_class": result.word_class,
                "definitions_en": result.definitions_en,
                "definitions_fr": result.definitions_fr,
                "source_url": ANKATAA_SEARCH_URL,
            }
            
            try:
                async with session.post(url, headers=headers, json=data) as resp:
                    pass  # Ignore errors for caching
            except Exception:
                pass
    
    def _row_to_result(self, row: Dict, source: str) -> DictionaryLookupResult:
        """Convert a database row to a lookup result."""
        return DictionaryLookupResult(
            word=row.get("word", ""),
            word_normalized=row.get("word_normalized", ""),
            word_class=row.get("word_class"),
            definitions_en=row.get("definitions_en", []) or [],
            definitions_fr=row.get("definitions_fr", []) or [],
            examples=row.get("examples", []) or [],
            variants=row.get("variants", []) or [],
            synonyms=row.get("synonyms", []) or [],
            has_tone_marks=row.get("has_tone_marks", False),
            source=source,
            match_score=row.get("similarity", 1.0),
        )
    
    async def enrich_detection(
        self,
        latin_text: str,
        current_english: Optional[str] = None,
        current_french: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Enrich a detection with dictionary data.
        
        Args:
            latin_text: Latin transliteration of N'Ko text
            current_english: Current English translation (from OCR)
            current_french: Current French translation (from OCR)
            
        Returns:
            Dict with enriched data or empty if not found
        """
        result = await self.lookup(latin_text, fuzzy=True)
        
        if not result:
            return {}
        
        return {
            "dictionary_entry_id": None,  # Would need UUID from cache
            "word_class": result.word_class,
            "verified_english": result.primary_english or current_english,
            "verified_french": result.primary_french or current_french,
            "variants": result.variants,
            "synonyms": result.synonyms,
            "is_dictionary_verified": True,
            "dictionary_match_score": result.match_score,
        }
    
    async def enrich_vocabulary_batch(
        self,
        vocabulary: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Enrich a batch of vocabulary entries with dictionary data.
        
        Args:
            vocabulary: List of vocabulary dicts with 'latin_text' key
            
        Returns:
            List of enriched vocabulary dicts
        """
        latin_texts = [v.get("latin_text", "") for v in vocabulary if v.get("latin_text")]
        
        # Batch lookup
        results = await self.lookup_many(latin_texts, fuzzy=True)
        
        # Merge results
        enriched = []
        for vocab in vocabulary:
            latin = vocab.get("latin_text", "")
            result = results.get(latin)
            
            if result:
                vocab.update({
                    "word_class": result.word_class,
                    "verified_english": result.primary_english or vocab.get("english_text"),
                    "verified_french": result.primary_french,
                    "variants": result.variants,
                    "is_dictionary_verified": True,
                })
            
            enriched.append(vocab)
        
        return enriched

