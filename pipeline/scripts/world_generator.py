#!/usr/bin/env python3
"""
World Generator for N'Ko Learning Pipeline

Generates 5 "world" variants for each N'Ko detection using Gemini text-only API.
This is ~10x cheaper than multimodal prompts (~$0.0001/call vs ~$0.002/call).

Worlds:
1. Everyday - Casual conversation usage
2. Formal - Official/formal writing
3. Storytelling - Griot/oral tradition
4. Proverbs - Wisdom sayings
5. Educational - Teaching content

Usage:
    from world_generator import WorldGenerator
    
    generator = WorldGenerator()
    worlds = await generator.generate_worlds(
        nko_text="ߒߞߏ",
        latin_text="N'Ko",
        translation="I declare"
    )
"""

import asyncio
import aiohttp
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from prompts.loader import PromptLoader
except ImportError:
    # Fallback if prompts module not in path
    PromptLoader = None

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# World definitions
WORLDS = [
    "world_everyday",
    "world_formal", 
    "world_storytelling",
    "world_proverbs",
    "world_educational",
]


@dataclass
class WorldVariant:
    """A generated world variant for N'Ko text."""
    world_name: str
    variants: List[Dict[str, Any]] = field(default_factory=list)
    cultural_notes: Optional[str] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None
    generation_time_ms: int = 0


@dataclass
class WorldGenerationResult:
    """Complete world generation result for a detection."""
    nko_text: str
    latin_text: Optional[str]
    translation: Optional[str]
    worlds: List[WorldVariant] = field(default_factory=list)
    total_variants: int = 0
    success_count: int = 0
    error_count: int = 0
    total_time_ms: int = 0


class WorldGenerator:
    """
    Generate world variants for N'Ko text using Gemini text-only API.
    
    Features:
    - Loads prompts from YAML or Supabase via PromptLoader
    - Rate limiting (configurable requests/second)
    - Retry logic with exponential backoff
    - Concurrent generation with semaphore
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_concurrent: int = 5,
        requests_per_second: float = 10.0,
        max_retries: int = 3,
    ):
        """
        Initialize the WorldGenerator.
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            max_concurrent: Max concurrent API calls
            requests_per_second: Rate limit
            max_retries: Max retry attempts per request
        """
        self.api_key = api_key or GEMINI_API_KEY
        self.max_concurrent = max_concurrent
        self.min_interval = 1.0 / requests_per_second
        self.max_retries = max_retries
        
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()
        
        # Load prompts
        self._prompts: Dict[str, str] = {}
        self._load_prompts()
    
    def _load_prompts(self):
        """Load world prompts from YAML or fallback to embedded."""
        # Try PromptLoader first
        if PromptLoader:
            try:
                loader = PromptLoader().from_env()
                for world_id in WORLDS:
                    template = loader.get_template(world_id)
                    if template:
                        self._prompts[world_id] = template
                if self._prompts:
                    print(f"Loaded {len(self._prompts)} prompts via PromptLoader")
                    return
            except Exception as e:
                print(f"PromptLoader failed: {e}, falling back to YAML")
        
        # Try direct YAML loading
        yaml_path = Path(__file__).parent.parent.parent / "prompts" / "nko" / "world_exploration.yaml"
        if yaml_path.exists():
            try:
                import yaml
                with open(yaml_path) as f:
                    data = yaml.safe_load(f)
                for prompt_id, prompt_data in data.get("prompts", {}).items():
                    self._prompts[prompt_id] = prompt_data.get("template", "")
                print(f"Loaded {len(self._prompts)} prompts from YAML")
                return
            except Exception as e:
                print(f"YAML loading failed: {e}")
        
        # Fallback to embedded minimal prompts
        self._prompts = self._get_fallback_prompts()
        print(f"Using {len(self._prompts)} fallback prompts")
    
    def _get_fallback_prompts(self) -> Dict[str, str]:
        """Return minimal fallback prompts if YAML fails."""
        base_prompt = """Given the following N'Ko text and translation:
N'Ko: {nko_text}
Latin: {latin_text}
English: {translation}

Generate {world_desc} variants of this text.

Respond in JSON format:
{{
    "world": "{world_name}",
    "variants": [
        {{
            "nko_text": "N'Ko script text",
            "latin_text": "romanized version",
            "english": "English translation",
            "context": "usage context"
        }}
    ],
    "cultural_note": "any cultural context"
}}

Generate 2-3 variants."""

        return {
            "world_everyday": base_prompt.format(
                nko_text="{nko_text}", latin_text="{latin_text}", translation="{translation}",
                world_desc="casual everyday conversation", world_name="everyday"
            ),
            "world_formal": base_prompt.format(
                nko_text="{nko_text}", latin_text="{latin_text}", translation="{translation}",
                world_desc="formal/official writing", world_name="formal"
            ),
            "world_storytelling": base_prompt.format(
                nko_text="{nko_text}", latin_text="{latin_text}", translation="{translation}",
                world_desc="griot storytelling tradition", world_name="storytelling"
            ),
            "world_proverbs": base_prompt.format(
                nko_text="{nko_text}", latin_text="{latin_text}", translation="{translation}",
                world_desc="Mande proverb and wisdom", world_name="proverbs"
            ),
            "world_educational": base_prompt.format(
                nko_text="{nko_text}", latin_text="{latin_text}", translation="{translation}",
                world_desc="educational/teaching", world_name="educational"
            ),
        }
    
    async def _rate_limit(self):
        """Enforce rate limiting."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_request_time
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last_request_time = asyncio.get_event_loop().time()
    
    async def _call_gemini(
        self,
        prompt: str,
        session: aiohttp.ClientSession,
    ) -> Dict[str, Any]:
        """
        Call Gemini text-only API with retry logic.
        
        Args:
            prompt: The prompt to send
            session: aiohttp session
            
        Returns:
            Parsed JSON response or error dict
        """
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,  # More creative for world generation
                "maxOutputTokens": 2048,
            }
        }
        
        for attempt in range(self.max_retries):
            try:
                await self._rate_limit()
                
                async with self._semaphore:
                    async with session.post(
                        f"{GEMINI_API_URL}?key={self.api_key}",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            try:
                                text = data["candidates"][0]["content"]["parts"][0]["text"]
                                # Parse JSON from response
                                if "```json" in text:
                                    text = text.split("```json")[1].split("```")[0]
                                elif "```" in text:
                                    text = text.split("```")[1].split("```")[0]
                                return json.loads(text.strip())
                            except (KeyError, IndexError, json.JSONDecodeError) as e:
                                return {"error": f"Parse error: {e}", "raw": str(data)}
                        
                        elif response.status == 429:
                            # Rate limited - exponential backoff
                            wait = (2 ** attempt) + 1
                            print(f"Rate limited, waiting {wait}s...")
                            await asyncio.sleep(wait)
                            continue
                        
                        else:
                            error_text = await response.text()
                            return {"error": f"API error {response.status}: {error_text[:200]}"}
                            
            except asyncio.TimeoutError:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {"error": "Request timed out"}
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {"error": str(e)}
        
        return {"error": "Max retries exceeded"}
    
    async def generate_single_world(
        self,
        world_id: str,
        nko_text: str,
        latin_text: Optional[str],
        translation: Optional[str],
        session: aiohttp.ClientSession,
    ) -> WorldVariant:
        """
        Generate a single world variant.
        
        Args:
            world_id: World prompt ID (e.g., "world_everyday")
            nko_text: The N'Ko text to expand
            latin_text: Latin transliteration
            translation: English translation
            session: aiohttp session
            
        Returns:
            WorldVariant with results or error
        """
        start_time = datetime.now()
        world_name = world_id.replace("world_", "")
        
        # Get prompt template
        template = self._prompts.get(world_id)
        if not template:
            return WorldVariant(
                world_name=world_name,
                error=f"No prompt found for {world_id}"
            )
        
        # Fill template
        prompt = template.replace("{nko_text}", nko_text or "")
        prompt = prompt.replace("{latin_text}", latin_text or "N/A")
        prompt = prompt.replace("{translation}", translation or "N/A")
        
        # Call API
        result = await self._call_gemini(prompt, session)
        
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        if "error" in result:
            return WorldVariant(
                world_name=world_name,
                error=result["error"],
                raw_response=result.get("raw"),
                generation_time_ms=elapsed_ms,
            )
        
        # Extract variants
        variants = result.get("variants", [])
        if not variants and "related_proverbs" in result:
            variants = result["related_proverbs"]  # Proverbs world uses different key
        
        # Extract cultural notes
        cultural_note = (
            result.get("cultural_note") or 
            result.get("cultural_significance") or
            result.get("cultural_context", {}).get("origin")
        )
        
        return WorldVariant(
            world_name=world_name,
            variants=variants,
            cultural_notes=cultural_note,
            raw_response=json.dumps(result),
            generation_time_ms=elapsed_ms,
        )
    
    async def generate_worlds(
        self,
        nko_text: str,
        latin_text: Optional[str] = None,
        translation: Optional[str] = None,
        worlds: Optional[List[str]] = None,
    ) -> WorldGenerationResult:
        """
        Generate all world variants for N'Ko text.
        
        Args:
            nko_text: The N'Ko text to expand
            latin_text: Latin transliteration
            translation: English translation
            worlds: List of world IDs to generate (defaults to all 5)
            
        Returns:
            WorldGenerationResult with all variants
        """
        if not self.api_key:
            return WorldGenerationResult(
                nko_text=nko_text,
                latin_text=latin_text,
                translation=translation,
                worlds=[WorldVariant(world_name="error", error="No API key")],
                error_count=1,
            )
        
        worlds_to_generate = worlds or WORLDS
        start_time = datetime.now()
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.generate_single_world(
                    world_id=world_id,
                    nko_text=nko_text,
                    latin_text=latin_text,
                    translation=translation,
                    session=session,
                )
                for world_id in worlds_to_generate
            ]
            
            results = await asyncio.gather(*tasks)
        
        # Compile results
        total_variants = sum(len(r.variants) for r in results)
        success_count = sum(1 for r in results if not r.error)
        error_count = sum(1 for r in results if r.error)
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return WorldGenerationResult(
            nko_text=nko_text,
            latin_text=latin_text,
            translation=translation,
            worlds=list(results),
            total_variants=total_variants,
            success_count=success_count,
            error_count=error_count,
            total_time_ms=elapsed_ms,
        )


async def test_world_generator():
    """Test the world generator with sample N'Ko text."""
    print("=" * 60)
    print("World Generator Test")
    print("=" * 60)
    
    generator = WorldGenerator()
    
    # Sample N'Ko text
    result = await generator.generate_worlds(
        nko_text="ߒ ߓߊ߬ ߓߏ߲߬ ߞߊ߲߫",
        latin_text="N ba bon kan",
        translation="I want to learn",
    )
    
    print(f"\nInput: {result.nko_text}")
    print(f"Latin: {result.latin_text}")
    print(f"Translation: {result.translation}")
    print(f"\nResults: {result.success_count}/{len(result.worlds)} worlds succeeded")
    print(f"Total variants: {result.total_variants}")
    print(f"Time: {result.total_time_ms}ms")
    
    for world in result.worlds:
        print(f"\n--- {world.world_name.upper()} ---")
        if world.error:
            print(f"  Error: {world.error}")
        else:
            print(f"  Variants: {len(world.variants)}")
            for i, v in enumerate(world.variants[:2], 1):
                nko = v.get("nko_text", "N/A")[:50]
                eng = v.get("english", v.get("literal_meaning", "N/A"))[:50]
                print(f"    {i}. {nko}... → {eng}...")
            if world.cultural_notes:
                print(f"  Cultural: {world.cultural_notes[:100]}...")


if __name__ == "__main__":
    asyncio.run(test_world_generator())

