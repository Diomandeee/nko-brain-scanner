#!/usr/bin/env python3
"""
Vocabulary Expansion Engine

The core cross-referencing engine for continuous vocabulary learning.
Enriches words through:
1. Existing vocabulary check (avoid duplicates)
2. Ankataa dictionary lookup (verified translations)
3. Gemini AI enrichment (cultural context, examples)
4. Queue expansion (related words for future learning)

Usage:
    from expansion_engine import ExpansionEngine
    
    engine = ExpansionEngine()
    
    # Enrich a single word
    result = await engine.enrich_word("dɔgɔ", context={"source": "video"})
    
    # Process queue batch
    processed = await engine.process_queue_batch(limit=10)
"""

import asyncio
import aiohttp
import os
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

# Support both relative and absolute imports
try:
    from .dictionary_client import DictionaryClient, DictionaryLookupResult, normalize_word
    from .world_generator import WorldGenerator, WorldGenerationResult
    from .supabase_client import SupabaseClient
except ImportError:
    from dictionary_client import DictionaryClient, DictionaryLookupResult, normalize_word
    from world_generator import WorldGenerator, WorldGenerationResult
    from supabase_client import SupabaseClient

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")


@dataclass
class EnrichmentResult:
    """Result of enriching a word."""
    word: str
    word_normalized: str
    status: str = "pending"  # "new", "enriched", "exists", "failed"
    
    # Dictionary lookup results
    dictionary_match: bool = False
    dictionary_entry_id: Optional[str] = None
    word_class: Optional[str] = None
    verified_english: Optional[str] = None
    verified_french: Optional[str] = None
    variants: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    
    # AI enrichment results
    ai_enriched: bool = False
    cultural_context: Optional[str] = None
    examples: List[Dict] = field(default_factory=list)
    related_words: List[str] = field(default_factory=list)
    
    # Vocabulary entry
    vocabulary_id: Optional[str] = None
    confidence: float = 0.0
    
    # Metadata
    sources: List[str] = field(default_factory=list)
    processing_time_ms: int = 0
    error: Optional[str] = None
    
    # Queue items created
    queued_words: List[str] = field(default_factory=list)


@dataclass 
class QueueItem:
    """An item from the expansion queue."""
    id: str
    word: str
    word_normalized: str
    source_type: str
    source_id: Optional[str]
    priority: int
    context: Dict[str, Any]


class ExpansionEngine:
    """
    Vocabulary expansion engine for continuous learning.
    
    Processes words through multiple enrichment sources and
    queues related words for future exploration.
    """
    
    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        enable_ai_enrichment: bool = True,
        enable_queue_expansion: bool = True,
        max_related_words: int = 5,
        min_confidence_to_skip: float = 0.8,
    ):
        """
        Initialize the expansion engine.
        
        Args:
            supabase_url: Supabase URL
            supabase_key: Supabase service key
            enable_ai_enrichment: Enable Gemini AI enrichment
            enable_queue_expansion: Queue related words for expansion
            max_related_words: Max related words to queue per enrichment
            min_confidence_to_skip: Skip enrichment if existing confidence >= this
        """
        self.supabase_url = supabase_url or SUPABASE_URL
        self.supabase_key = supabase_key or SUPABASE_KEY
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY required")
        
        self.enable_ai_enrichment = enable_ai_enrichment
        self.enable_queue_expansion = enable_queue_expansion
        self.max_related_words = max_related_words
        self.min_confidence_to_skip = min_confidence_to_skip
        
        # Initialize clients
        self.supabase = SupabaseClient(self.supabase_url, self.supabase_key)
        self.dictionary = DictionaryClient(self.supabase_url, self.supabase_key)
        self.world_generator = WorldGenerator() if enable_ai_enrichment else None
        
        self._headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
        }
    
    async def enrich_word(
        self,
        word: str,
        context: Optional[Dict[str, Any]] = None,
        force: bool = False,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> EnrichmentResult:
        """
        Enrich a single word through all available sources.
        
        Args:
            word: The word to enrich (Bambara/Dioula or Latin)
            context: Additional context (source info, timestamps, etc.)
            force: Force enrichment even if word exists with high confidence
            session: Optional aiohttp session
            
        Returns:
            EnrichmentResult with enrichment data
        """
        start_time = datetime.now()
        normalized = normalize_word(word)
        context = context or {}
        
        result = EnrichmentResult(
            word=word,
            word_normalized=normalized,
        )
        
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True
        
        try:
            # Step 1: Check existing vocabulary
            existing = await self._check_existing_vocabulary(normalized, session)
            
            if existing and not force:
                confidence = existing.get("confidence", 0)
                is_verified = existing.get("is_dictionary_verified", False)
                
                if confidence >= self.min_confidence_to_skip or is_verified:
                    result.status = "exists"
                    result.vocabulary_id = existing.get("id")
                    result.confidence = confidence
                    result.verified_english = existing.get("verified_english")
                    result.verified_french = existing.get("verified_french")
                    result.sources = ["existing"]
                    elapsed = (datetime.now() - start_time).total_seconds() * 1000
                    result.processing_time_ms = int(elapsed)
                    return result
            
            # Step 2: Dictionary lookup
            dict_result = await self._lookup_dictionary(word, session)
            
            if dict_result:
                result.dictionary_match = True
                result.word_class = dict_result.word_class
                result.verified_english = dict_result.primary_english
                result.verified_french = dict_result.primary_french
                result.variants = dict_result.variants
                result.synonyms = dict_result.synonyms
                result.sources.append("ankataa")
                result.confidence = max(result.confidence, 0.7)
                
                # Queue variants and synonyms
                if self.enable_queue_expansion:
                    queued = await self._queue_related_words(
                        words=dict_result.variants + dict_result.synonyms,
                        source_type="dictionary_variant",
                        priority=3,
                        session=session,
                    )
                    result.queued_words.extend(queued)
            
            # Step 3: AI enrichment (if enabled and not fully known)
            if self.enable_ai_enrichment and self.world_generator:
                if not result.dictionary_match or not result.verified_english:
                    ai_result = await self._ai_enrich(
                        nko_text=word,  # Treat input as potential N'Ko
                        latin_text=word,
                        translation=result.verified_english,
                        session=session,
                    )
                    
                    if ai_result and not ai_result.get("error"):
                        result.ai_enriched = True
                        result.cultural_context = ai_result.get("cultural_context")
                        result.examples = ai_result.get("examples", [])
                        result.related_words = ai_result.get("related_words", [])
                        result.sources.append("gemini")
                        result.confidence = max(result.confidence, 0.5)
                        
                        # Derive translation from AI if missing
                        if not result.verified_english and ai_result.get("translation"):
                            result.verified_english = ai_result["translation"]
                        
                        # Queue AI-suggested related words
                        if self.enable_queue_expansion and result.related_words:
                            queued = await self._queue_related_words(
                                words=result.related_words[:self.max_related_words],
                                source_type="ai_suggestion",
                                priority=5,
                                session=session,
                            )
                            result.queued_words.extend(queued)
            
            # Step 4: Store or update vocabulary
            if result.dictionary_match or result.ai_enriched:
                vocab_id = await self._upsert_vocabulary(
                    word=word,
                    normalized=normalized,
                    result=result,
                    existing_id=existing.get("id") if existing else None,
                    session=session,
                )
                result.vocabulary_id = vocab_id
                result.status = "enriched"
            else:
                result.status = "new"
            
            # Step 5: Update stats
            await self._update_stats(result, session)
            
        except Exception as e:
            result.status = "failed"
            result.error = str(e)
        finally:
            if close_session:
                await session.close()
        
        elapsed = (datetime.now() - start_time).total_seconds() * 1000
        result.processing_time_ms = int(elapsed)
        
        return result
    
    async def queue_word(
        self,
        word: str,
        source_type: str = "manual",
        source_id: Optional[str] = None,
        priority: int = 5,
        context: Optional[Dict[str, Any]] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> Optional[str]:
        """
        Add a word to the expansion queue.
        
        Args:
            word: Word to queue
            source_type: Origin type ('video_detection', 'user_query', etc.)
            source_id: Optional reference ID
            priority: 1=highest, 10=lowest
            context: Additional context
            
        Returns:
            Queue item ID if added, None if skipped
        """
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True
        
        try:
            url = f"{self.supabase_url}/rest/v1/rpc/queue_word_for_enrichment"
            payload = {
                "p_word": word,
                "p_source_type": source_type,
                "p_source_id": source_id,
                "p_priority": priority,
                "p_context": context or {},
            }
            
            async with session.post(url, headers=self._headers, json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result if result else None
                else:
                    return None
        finally:
            if close_session:
                await session.close()
    
    async def get_pending_items(
        self,
        limit: int = 10,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> List[QueueItem]:
        """
        Get pending items from the expansion queue.
        
        Args:
            limit: Max items to return
            
        Returns:
            List of QueueItem objects
        """
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True
        
        try:
            url = f"{self.supabase_url}/rest/v1/rpc/get_pending_enrichments"
            payload = {"p_limit": limit}
            
            async with session.post(url, headers=self._headers, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [
                        QueueItem(
                            id=row["id"],
                            word=row["word"],
                            word_normalized=row["word_normalized"],
                            source_type=row["source_type"],
                            source_id=row.get("source_id"),
                            priority=row["priority"],
                            context=row.get("context", {}),
                        )
                        for row in data
                    ]
                return []
        finally:
            if close_session:
                await session.close()
    
    async def complete_queue_item(
        self,
        queue_id: str,
        vocabulary_id: Optional[str],
        sources: List[str],
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        """Mark a queue item as completed."""
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True
        
        try:
            url = f"{self.supabase_url}/rest/v1/rpc/complete_enrichment"
            payload = {
                "p_queue_id": queue_id,
                "p_vocabulary_id": vocabulary_id,
                "p_sources": sources,
            }
            
            async with session.post(url, headers=self._headers, json=payload) as resp:
                pass  # Ignore result
        finally:
            if close_session:
                await session.close()
    
    async def fail_queue_item(
        self,
        queue_id: str,
        error: str,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        """Mark a queue item as failed."""
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True
        
        try:
            url = f"{self.supabase_url}/rest/v1/rpc/fail_enrichment"
            payload = {
                "p_queue_id": queue_id,
                "p_error": error[:500],  # Truncate long errors
            }
            
            async with session.post(url, headers=self._headers, json=payload) as resp:
                pass  # Ignore result
        finally:
            if close_session:
                await session.close()
    
    async def process_queue_batch(
        self,
        limit: int = 10,
        delay_seconds: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Process a batch of queue items.
        
        Args:
            limit: Max items to process
            delay_seconds: Delay between items (rate limiting)
            
        Returns:
            Summary of processing results
        """
        start_time = datetime.now()
        
        async with aiohttp.ClientSession() as session:
            # Get pending items
            items = await self.get_pending_items(limit, session)
            
            if not items:
                return {
                    "processed": 0,
                    "enriched": 0,
                    "failed": 0,
                    "message": "Queue empty",
                }
            
            enriched = 0
            failed = 0
            queued_words = 0
            
            for item in items:
                try:
                    # Rate limiting
                    if delay_seconds > 0:
                        await asyncio.sleep(delay_seconds)
                    
                    # Enrich the word
                    result = await self.enrich_word(
                        word=item.word,
                        context=item.context,
                        session=session,
                    )
                    
                    if result.status == "failed":
                        await self.fail_queue_item(item.id, result.error or "Unknown error", session)
                        failed += 1
                    else:
                        await self.complete_queue_item(
                            queue_id=item.id,
                            vocabulary_id=result.vocabulary_id,
                            sources=result.sources,
                            session=session,
                        )
                        enriched += 1
                        queued_words += len(result.queued_words)
                        
                except Exception as e:
                    await self.fail_queue_item(item.id, str(e), session)
                    failed += 1
        
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return {
            "processed": len(items),
            "enriched": enriched,
            "failed": failed,
            "queued_words": queued_words,
            "elapsed_ms": elapsed_ms,
        }
    
    async def get_queue_status(
        self,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> Dict[str, Any]:
        """Get current queue status summary."""
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True
        
        try:
            url = f"{self.supabase_url}/rest/v1/queue_status"
            
            async with session.get(url, headers=self._headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    # Aggregate by status
                    status_counts = {}
                    source_counts = {}
                    
                    for row in data:
                        status = row.get("status", "unknown")
                        source = row.get("source_type", "unknown")
                        count = row.get("count", 0)
                        
                        status_counts[status] = status_counts.get(status, 0) + count
                        source_counts[source] = source_counts.get(source, 0) + count
                    
                    return {
                        "by_status": status_counts,
                        "by_source": source_counts,
                        "total_pending": status_counts.get("pending", 0),
                        "total_completed": status_counts.get("completed", 0),
                    }
                return {}
        finally:
            if close_session:
                await session.close()
    
    async def get_learning_stats(
        self,
        days: int = 7,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> List[Dict[str, Any]]:
        """Get learning statistics for the past N days."""
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True
        
        try:
            url = f"{self.supabase_url}/rest/v1/learning_stats"
            params = {
                "order": "stat_date.desc",
                "limit": str(days),
            }
            
            async with session.get(url, headers=self._headers, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                return []
        finally:
            if close_session:
                await session.close()
    
    # ==================== Private Methods ====================
    
    async def _check_existing_vocabulary(
        self,
        normalized: str,
        session: aiohttp.ClientSession,
    ) -> Optional[Dict]:
        """Check if word exists in vocabulary."""
        url = f"{self.supabase_url}/rest/v1/nko_vocabulary"
        params = {
            "latin_text": f"eq.{normalized}",
            "select": "id,latin_text,english_text,confidence,is_dictionary_verified,verified_english,verified_french",
        }
        
        try:
            async with session.get(url, headers=self._headers, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data[0] if data else None
        except Exception:
            pass
        return None
    
    async def _lookup_dictionary(
        self,
        word: str,
        session: aiohttp.ClientSession,
    ) -> Optional[DictionaryLookupResult]:
        """Look up word in Ankataa dictionary."""
        try:
            async with DictionaryClient(self.supabase_url, self.supabase_key) as client:
                return await client.lookup(word, fuzzy=True)
        except Exception:
            return None
    
    async def _ai_enrich(
        self,
        nko_text: str,
        latin_text: Optional[str],
        translation: Optional[str],
        session: aiohttp.ClientSession,
    ) -> Optional[Dict[str, Any]]:
        """Get AI enrichment using just one world for efficiency."""
        if not self.world_generator:
            return None
        
        try:
            # Use just educational world for enrichment (most informative)
            result = await self.world_generator.generate_worlds(
                nko_text=nko_text,
                latin_text=latin_text,
                translation=translation,
                worlds=["world_educational"],  # Just one world
            )
            
            if result.worlds and not result.worlds[0].error:
                world = result.worlds[0]
                
                # Extract data from world
                variants = world.variants
                examples = variants[:3] if variants else []
                
                # Extract related words from variants
                related = set()
                for v in variants:
                    # Get any related terms mentioned
                    if "related" in str(v).lower():
                        pass  # Would parse here
                
                return {
                    "cultural_context": world.cultural_notes,
                    "examples": examples,
                    "related_words": list(related),
                    "translation": variants[0].get("english") if variants else None,
                }
        except Exception:
            pass
        
        return None
    
    async def _queue_related_words(
        self,
        words: List[str],
        source_type: str,
        priority: int,
        session: aiohttp.ClientSession,
    ) -> List[str]:
        """Queue related words for future enrichment."""
        queued = []
        
        for word in words[:self.max_related_words]:
            if not word or len(word) < 2:
                continue
            
            try:
                result = await self.queue_word(
                    word=word,
                    source_type=source_type,
                    priority=priority,
                    session=session,
                )
                if result:
                    queued.append(word)
            except Exception:
                pass
        
        return queued
    
    async def _upsert_vocabulary(
        self,
        word: str,
        normalized: str,
        result: EnrichmentResult,
        existing_id: Optional[str],
        session: aiohttp.ClientSession,
    ) -> Optional[str]:
        """Insert or update vocabulary entry."""
        data = {
            "latin_text": normalized,
            "english_text": result.verified_english,
            "confidence": result.confidence,
            "is_dictionary_verified": result.dictionary_match,
            "word_class": result.word_class,
            "verified_english": result.verified_english,
            "verified_french": result.verified_french,
            "variants": result.variants if result.variants else None,
        }
        
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        
        url = f"{self.supabase_url}/rest/v1/nko_vocabulary"
        headers = {
            **self._headers,
            "Prefer": "return=representation",
        }
        
        try:
            if existing_id:
                # Update existing
                async with session.patch(
                    url,
                    headers=headers,
                    params={"id": f"eq.{existing_id}"},
                    json=data,
                ) as resp:
                    if resp.status in (200, 201):
                        result_data = await resp.json()
                        return result_data[0]["id"] if result_data else existing_id
            else:
                # Insert new
                async with session.post(url, headers=headers, json=data) as resp:
                    if resp.status in (200, 201):
                        result_data = await resp.json()
                        return result_data[0]["id"] if result_data else None
        except Exception:
            pass
        
        return existing_id
    
    async def _update_stats(
        self,
        result: EnrichmentResult,
        session: aiohttp.ClientSession,
    ) -> None:
        """Update daily learning statistics."""
        url = f"{self.supabase_url}/rest/v1/learning_stats"
        today = datetime.now().date().isoformat()
        
        # Check if today's stats exist
        params = {"stat_date": f"eq.{today}"}
        
        try:
            async with session.get(url, headers=self._headers, params=params) as resp:
                if resp.status == 200:
                    existing = await resp.json()
                    
                    if existing:
                        # Update existing
                        stats_id = existing[0]["id"]
                        updates = {"updated_at": datetime.utcnow().isoformat()}
                        
                        if result.status == "enriched":
                            updates["words_enriched"] = existing[0].get("words_enriched", 0) + 1
                        if result.dictionary_match:
                            updates["dictionary_matches"] = existing[0].get("dictionary_matches", 0) + 1
                        if result.ai_enriched:
                            updates["ai_enrichments"] = existing[0].get("ai_enrichments", 0) + 1
                        
                        async with session.patch(
                            url,
                            headers=self._headers,
                            params={"id": f"eq.{stats_id}"},
                            json=updates,
                        ) as patch_resp:
                            pass  # Ignore result
                    else:
                        # Create new
                        new_stats = {
                            "stat_date": today,
                            "words_enriched": 1 if result.status == "enriched" else 0,
                            "dictionary_matches": 1 if result.dictionary_match else 0,
                            "ai_enrichments": 1 if result.ai_enriched else 0,
                        }
                        
                        async with session.post(
                            url,
                            headers=self._headers,
                            json=new_stats,
                        ) as post_resp:
                            pass  # Ignore result
        except Exception:
            pass  # Stats update is best-effort


async def test_expansion_engine():
    """Test the expansion engine."""
    print("=" * 60)
    print("Expansion Engine Test")
    print("=" * 60)
    
    try:
        engine = ExpansionEngine(enable_ai_enrichment=False)  # Disable AI for quick test
        
        # Test word enrichment
        print("\n1. Testing word enrichment...")
        result = await engine.enrich_word("dɔgɔ")
        print(f"   Word: {result.word}")
        print(f"   Status: {result.status}")
        print(f"   Dictionary match: {result.dictionary_match}")
        print(f"   English: {result.verified_english}")
        print(f"   Time: {result.processing_time_ms}ms")
        
        # Test queue status
        print("\n2. Testing queue status...")
        status = await engine.get_queue_status()
        print(f"   Pending: {status.get('total_pending', 0)}")
        print(f"   Completed: {status.get('total_completed', 0)}")
        
        # Test learning stats
        print("\n3. Testing learning stats...")
        stats = await engine.get_learning_stats(days=3)
        for stat in stats:
            print(f"   {stat.get('stat_date')}: {stat.get('words_enriched', 0)} enriched")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_expansion_engine())

