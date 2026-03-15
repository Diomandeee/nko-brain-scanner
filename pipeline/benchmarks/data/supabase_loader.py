"""
Supabase Data Loader for N'Ko Benchmark.

Pulls verified vocabulary and translations from Supabase nko_vocabulary
and nko_translations tables for gold-standard testing.
"""

import os
import json
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from ..config import get_supabase_config


@dataclass
class VerifiedVocabulary:
    """A verified vocabulary entry from Supabase."""
    id: str
    word: str
    word_normalized: str
    latin: Optional[str] = None
    meaning_primary: Optional[str] = None
    meanings: List[str] = field(default_factory=list)
    definition: Optional[str] = None
    pos: Optional[str] = None  # Part of speech
    frequency: int = 0
    is_dictionary_verified: bool = False
    verified_english: Optional[str] = None
    verified_french: Optional[str] = None
    cefr_level: Optional[str] = None
    status: str = "unverified"


@dataclass
class VerifiedTranslation:
    """A verified translation pair from Supabase."""
    id: str
    nko_text: str
    latin_text: Optional[str] = None
    english_text: Optional[str] = None
    french_text: Optional[str] = None
    text_type: Optional[str] = None  # word, phrase, sentence, etc.
    confidence: float = 0.0
    validated: bool = False
    quality_tier: Optional[str] = None  # gold, silver, bronze, raw


class SupabaseLoader:
    """
    Loads verified N'Ko data from Supabase for gold-standard benchmark tests.
    
    Uses the nko_vocabulary and nko_translations tables which contain
    dictionary-verified entries with high confidence.
    """
    
    def __init__(self):
        config = get_supabase_config()
        self.url = config.get("url")
        self.key = config.get("key")
        self._session: Optional[aiohttp.ClientSession] = None
    
    @property
    def is_configured(self) -> bool:
        """Check if Supabase is properly configured."""
        return bool(self.url and self.key)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required: pip install aiohttp")
        
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "apikey": self.key,
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                }
            )
        return self._session
    
    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def fetch_verified_vocabulary(
        self,
        limit: int = 500,
        verified_only: bool = False,  # Changed default: include all vocab
        min_confidence: float = 0.0,  # Changed: no confidence filter by default
    ) -> List[VerifiedVocabulary]:
        """
        Fetch vocabulary entries from Supabase.
        
        Args:
            limit: Maximum entries to fetch
            verified_only: Only fetch dictionary-verified entries (default False to get all)
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of VerifiedVocabulary objects
        """
        if not self.is_configured:
            print("  Warning: Supabase not configured, skipping vocabulary fetch")
            return []
        
        session = await self._get_session()
        
        # Build query - get ALL vocabulary by default
        query_params = [
            f"limit={limit}",
            "order=id.asc",  # Ordered by ID for consistent results
        ]
        
        # Only filter by verified if explicitly requested
        if verified_only:
            query_params.append("is_dictionary_verified=eq.true")
        
        query_string = "&".join(query_params)
        url = f"{self.url}/rest/v1/nko_vocabulary?{query_string}"
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    entries = []
                    for row in data:
                        entries.append(VerifiedVocabulary(
                            id=row.get("id", ""),
                            word=row.get("word", ""),
                            word_normalized=row.get("word_normalized", ""),
                            latin=row.get("latin"),
                            meaning_primary=row.get("meaning_primary"),
                            meanings=row.get("meanings", []) or [],
                            definition=row.get("definition"),
                            pos=row.get("pos"),
                            frequency=row.get("frequency", 0),
                            is_dictionary_verified=row.get("is_dictionary_verified", False),
                            verified_english=row.get("verified_english"),
                            verified_french=row.get("verified_french"),
                            cefr_level=row.get("cefr_level"),
                            status=row.get("status", "unverified"),
                        ))
                    
                    return entries
                else:
                    error = await response.text()
                    print(f"  Warning: Supabase vocabulary fetch failed: {error[:100]}")
                    return []
                    
        except Exception as e:
            print(f"  Warning: Supabase connection error: {e}")
            return []
    
    async def fetch_verified_translations(
        self,
        limit: int = 1000,
        quality_tier: Optional[str] = None,  # Changed: no filter by default
        validated_only: bool = False,  # Changed: include all translations
    ) -> List[VerifiedTranslation]:
        """
        Fetch translation pairs from Supabase.
        
        Args:
            limit: Maximum entries to fetch
            quality_tier: Filter by quality tier (gold, silver, bronze) - None for all
            validated_only: Only fetch validated translations (default False to get all)
            
        Returns:
            List of VerifiedTranslation objects
        """
        if not self.is_configured:
            print("  Warning: Supabase not configured, skipping translation fetch")
            return []
        
        session = await self._get_session()
        
        # Build query - get ALL translations by default
        query_params = [
            f"limit={limit}",
            "order=id.asc",  # Consistent ordering
        ]
        
        # Only apply filters if explicitly requested
        if validated_only:
            query_params.append("validated=eq.true")
        
        if quality_tier:
            query_params.append(f"quality_tier=eq.{quality_tier}")
        
        query_string = "&".join(query_params)
        url = f"{self.url}/rest/v1/nko_translations?{query_string}"
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    entries = []
                    for row in data:
                        entries.append(VerifiedTranslation(
                            id=row.get("id", ""),
                            nko_text=row.get("nko_text", ""),
                            latin_text=row.get("latin_text"),
                            english_text=row.get("english_text"),
                            french_text=row.get("french_text"),
                            text_type=row.get("text_type"),
                            confidence=row.get("confidence", 0.0),
                            validated=row.get("validated", False),
                            quality_tier=row.get("quality_tier"),
                        ))
                    
                    return entries
                else:
                    error = await response.text()
                    print(f"  Warning: Supabase translation fetch failed: {error[:100]}")
                    return []
                    
        except Exception as e:
            print(f"  Warning: Supabase connection error: {e}")
            return []
    
    async def fetch_grammar_rules(
        self,
        limit: int = 50,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch grammar rules from Supabase for grammar testing.
        
        Args:
            limit: Maximum rules to fetch
            category: Filter by rule category
            
        Returns:
            List of grammar rule dictionaries
        """
        if not self.is_configured:
            return []
        
        session = await self._get_session()
        
        query_params = [f"limit={limit}"]
        if category:
            query_params.append(f"rule_category=eq.{category}")
        
        query_string = "&".join(query_params)
        url = f"{self.url}/rest/v1/nko_grammar_rules?{query_string}"
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                return []
        except Exception:
            return []
    
    async def fetch_examples(
        self,
        vocabulary_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Fetch example sentences from Supabase.
        
        Args:
            vocabulary_id: Optional filter by vocabulary entry
            limit: Maximum examples to fetch
            
        Returns:
            List of example dictionaries
        """
        if not self.is_configured:
            return []
        
        session = await self._get_session()
        
        query_params = [f"limit={limit}", "is_verified=eq.true"]
        if vocabulary_id:
            query_params.append(f"vocabulary_id=eq.{vocabulary_id}")
        
        query_string = "&".join(query_params)
        url = f"{self.url}/rest/v1/nko_examples?{query_string}"
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                return []
        except Exception:
            return []


async def load_supabase_data(
    vocabulary_limit: int = 1000,
    translation_limit: int = 1000,
) -> Dict[str, List]:
    """
    Convenience function to load all Supabase data.
    
    Returns:
        Dict with 'vocabulary' and 'translations' lists
    """
    async with SupabaseLoader() as loader:
        if not loader.is_configured:
            print("  Supabase not configured, using local data only")
            return {"vocabulary": [], "translations": []}
        
        print("  Loading ALL data from Supabase (no filters)...")
        
        # Get ALL vocab entries (not just verified)
        vocabulary = await loader.fetch_verified_vocabulary(
            limit=vocabulary_limit,
            verified_only=False,  # Get everything
        )
        print(f"    Vocabulary entries: {len(vocabulary)}")
        
        # Get ALL translations (not just gold-tier)
        translations = await loader.fetch_verified_translations(
            limit=translation_limit,
            validated_only=False,
            quality_tier=None,
        )
        print(f"    Translation pairs: {len(translations)}")
        
        return {
            "vocabulary": vocabulary,
            "translations": translations,
        }


def load_supabase_data_sync(
    vocabulary_limit: int = 500,
    translation_limit: int = 200,
) -> Dict[str, List]:
    """Synchronous wrapper for load_supabase_data."""
    return asyncio.run(load_supabase_data(vocabulary_limit, translation_limit))

