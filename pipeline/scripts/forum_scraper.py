#!/usr/bin/env python3
"""
Ankataa Forum Scraper

Extracts linguistic knowledge from the Ankataa Discourse forum:
- Word questions and clarifications
- New vocabulary discussions
- Expert answers and corrections
- Usage examples

Forum: https://ankataa.discourse.group

Usage:
    python forum_scraper.py --categories word-questions word-not-in-dictionary
    python forum_scraper.py --save-json forum_knowledge.json
    python forum_scraper.py --save-supabase
    python forum_scraper.py --dry-run --limit 10
"""

import asyncio
import aiohttp
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

# Add parent lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

try:
    from supabase_client import SupabaseClient
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False

# Forum configuration
FORUM_BASE_URL = "https://ankataa.discourse.group"

# Categories to scrape (slug names)
DEFAULT_CATEGORIES = [
    "word-questions",
    "word-not-in-dictionary", 
    "media-and-references",
    "user-guide",
]

# Rate limiting
REQUEST_DELAY = 1.0  # seconds between requests


@dataclass
class ForumAnswer:
    """An answer/reply in a forum topic."""
    author: str
    content: str
    is_solution: bool = False
    likes: int = 0
    created_at: Optional[str] = None


@dataclass
class ForumTopic:
    """A forum topic with its content and answers."""
    topic_id: str
    category: str
    title: str
    content: str
    author: str
    answers: List[ForumAnswer] = field(default_factory=list)
    related_words: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    url: str = ""
    views: int = 0
    reply_count: int = 0
    created_at: Optional[str] = None
    last_posted_at: Optional[str] = None


class ForumScraper:
    """
    Scrapes the Ankataa Discourse forum for linguistic knowledge.
    
    Uses the Discourse API where available, falls back to HTML scraping.
    """
    
    def __init__(
        self,
        base_url: str = FORUM_BASE_URL,
        categories: Optional[List[str]] = None,
        request_delay: float = REQUEST_DELAY,
    ):
        self.base_url = base_url.rstrip("/")
        self.categories = categories or DEFAULT_CATEGORIES
        self.request_delay = request_delay
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            headers={"User-Agent": "LearnNKo Forum Scraper/1.0"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
    
    async def _get(self, url: str) -> Optional[str]:
        """Make a GET request with rate limiting."""
        if not self._session:
            raise RuntimeError("Use async context manager")
        
        await asyncio.sleep(self.request_delay)
        
        try:
            async with self._session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    print(f"Error {response.status} for {url}")
                    return None
        except Exception as e:
            print(f"Request failed: {e}")
            return None
    
    async def _get_json(self, url: str) -> Optional[Dict]:
        """Make a GET request and parse JSON."""
        if not self._session:
            raise RuntimeError("Use async context manager")
        
        await asyncio.sleep(self.request_delay)
        
        try:
            async with self._session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Error {response.status} for {url}")
                    return None
        except Exception as e:
            print(f"Request failed: {e}")
            return None
    
    async def get_category_topics(
        self,
        category_slug: str,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """
        Get all topics in a category using the Discourse API.
        
        Args:
            category_slug: Category slug (e.g., "word-questions")
            limit: Max topics to fetch
            
        Returns:
            List of topic metadata dicts
        """
        topics = []
        page = 0
        
        while True:
            url = f"{self.base_url}/c/{category_slug}.json?page={page}"
            data = await self._get_json(url)
            
            if not data:
                break
            
            topic_list = data.get("topic_list", {}).get("topics", [])
            if not topic_list:
                break
            
            for topic in topic_list:
                topics.append({
                    "id": str(topic.get("id")),
                    "title": topic.get("title", ""),
                    "slug": topic.get("slug", ""),
                    "views": topic.get("views", 0),
                    "reply_count": topic.get("reply_count", 0),
                    "created_at": topic.get("created_at"),
                    "last_posted_at": topic.get("last_posted_at"),
                    "tags": topic.get("tags", []),
                })
                
                if limit and len(topics) >= limit:
                    return topics[:limit]
            
            # Check for more pages
            if len(topic_list) < 30:  # Default page size
                break
            page += 1
        
        return topics
    
    async def get_topic_details(
        self,
        topic_id: str,
        category: str,
    ) -> Optional[ForumTopic]:
        """
        Get full topic details including posts/answers.
        
        Args:
            topic_id: Topic ID
            category: Category slug
            
        Returns:
            ForumTopic with all data
        """
        url = f"{self.base_url}/t/{topic_id}.json"
        data = await self._get_json(url)
        
        if not data:
            return None
        
        # Parse posts
        posts = data.get("post_stream", {}).get("posts", [])
        
        # First post is the topic content
        first_post = posts[0] if posts else {}
        content = first_post.get("cooked", "")  # HTML content
        
        # Clean HTML to plain text
        if content:
            soup = BeautifulSoup(content, "html.parser")
            content = soup.get_text(separator="\n").strip()
        
        # Extract related words (N'Ko or Bambara words mentioned)
        related_words = self._extract_words(content)
        
        # Parse answers (remaining posts)
        answers = []
        for post in posts[1:]:
            post_content = post.get("cooked", "")
            if post_content:
                soup = BeautifulSoup(post_content, "html.parser")
                post_content = soup.get_text(separator="\n").strip()
            
            answers.append(ForumAnswer(
                author=post.get("username", "unknown"),
                content=post_content,
                is_solution=post.get("accepted_answer", False),
                likes=post.get("like_count", 0),
                created_at=post.get("created_at"),
            ))
        
        return ForumTopic(
            topic_id=topic_id,
            category=category,
            title=data.get("title", ""),
            content=content,
            author=first_post.get("username", "unknown"),
            answers=answers,
            related_words=related_words,
            tags=data.get("tags", []),
            url=f"{self.base_url}/t/{topic_id}",
            views=data.get("views", 0),
            reply_count=data.get("reply_count", 0),
            created_at=data.get("created_at"),
            last_posted_at=data.get("last_posted_at"),
        )
    
    def _extract_words(self, text: str) -> List[str]:
        """
        Extract potential Bambara/N'Ko words from text.
        
        Looks for:
        - Words with special Manding characters (ɛ, ɔ, ɲ, ŋ)
        - Words in quotes that look like vocabulary
        - N'Ko script (U+07C0-U+07FF)
        """
        words = set()
        
        # N'Ko script
        nko_pattern = r'[\u07C0-\u07FF]+'
        for match in re.finditer(nko_pattern, text):
            words.add(match.group())
        
        # Bambara words with special characters
        bambara_pattern = r'\b[a-zA-ZɛɔɲŋÈÉÊËèéêëÒÓÔÕòóôõ]{2,}\b'
        for match in re.finditer(bambara_pattern, text):
            word = match.group().lower()
            # Filter to words with special chars or likely vocabulary
            if any(c in word for c in 'ɛɔɲŋ'):
                words.add(word)
        
        # Words in quotes (often vocabulary being discussed)
        quoted_pattern = r'["\']([a-zA-ZɛɔɲŋÈÉÊËèéêëÒÓÔÕòóôõ\s]{2,})["\']'
        for match in re.finditer(quoted_pattern, text):
            word = match.group(1).strip().lower()
            if len(word) <= 30:  # Reasonable word length
                words.add(word)
        
        return list(words)
    
    async def scrape_all(
        self,
        categories: Optional[List[str]] = None,
        limit_per_category: Optional[int] = None,
    ) -> List[ForumTopic]:
        """
        Scrape all topics from specified categories.
        
        Args:
            categories: Categories to scrape (defaults to self.categories)
            limit_per_category: Max topics per category
            
        Returns:
            List of ForumTopic objects
        """
        categories = categories or self.categories
        all_topics = []
        
        for category in categories:
            print(f"\n--- Category: {category} ---")
            
            topic_list = await self.get_category_topics(category, limit_per_category)
            print(f"Found {len(topic_list)} topics")
            
            for i, topic_meta in enumerate(topic_list):
                topic_id = topic_meta["id"]
                print(f"  [{i+1}/{len(topic_list)}] {topic_meta['title'][:50]}...")
                
                topic = await self.get_topic_details(topic_id, category)
                if topic:
                    all_topics.append(topic)
                    print(f"    Answers: {len(topic.answers)}, Words: {len(topic.related_words)}")
        
        return all_topics
    
    def to_supabase_format(self, topic: ForumTopic) -> Dict[str, Any]:
        """Convert ForumTopic to Supabase forum_knowledge table format."""
        return {
            "topic_id": topic.topic_id,
            "category": topic.category,
            "title": topic.title,
            "content": topic.content,
            "answers": [asdict(a) for a in topic.answers],
            "related_words": topic.related_words,
            "tags": topic.tags,
            "url": topic.url,
        }


async def main():
    parser = argparse.ArgumentParser(
        description="Scrape Ankataa Forum for linguistic knowledge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES,
                        help="Categories to scrape")
    parser.add_argument("--limit", type=int, help="Limit topics per category")
    parser.add_argument("--dry-run", action="store_true",
                        help="List topics without full scraping")
    parser.add_argument("--save-json", type=str, metavar="FILE",
                        help="Save to JSON file")
    parser.add_argument("--save-supabase", action="store_true",
                        help="Save to Supabase forum_knowledge table")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Ankataa Forum Scraper")
    print("=" * 60)
    print(f"Categories: {', '.join(args.categories)}")
    print(f"Limit per category: {args.limit or 'unlimited'}")
    
    async with ForumScraper(categories=args.categories) as scraper:
        if args.dry_run:
            # Just list topics
            for category in args.categories:
                print(f"\n--- {category} ---")
                topics = await scraper.get_category_topics(category, args.limit)
                for t in topics:
                    print(f"  [{t['id']}] {t['title'][:60]}")
                    print(f"       Views: {t['views']}, Replies: {t['reply_count']}")
            return
        
        # Full scrape
        topics = await scraper.scrape_all(
            categories=args.categories,
            limit_per_category=args.limit,
        )
        
        print(f"\n\n=== Scrape Complete ===")
        print(f"Total topics: {len(topics)}")
        print(f"Total answers: {sum(len(t.answers) for t in topics)}")
        print(f"Unique words found: {len(set(w for t in topics for w in t.related_words))}")
        
        # Save to JSON
        if args.save_json:
            output = [asdict(t) for t in topics]
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"\nSaved to: {args.save_json}")
        
        # Save to Supabase
        if args.save_supabase:
            if not HAS_SUPABASE:
                print("Error: Supabase client not available")
                return
            
            print("\nUploading to Supabase...")
            
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
            
            if not supabase_url or not supabase_key:
                print("Error: SUPABASE_URL and SUPABASE_SERVICE_KEY required")
                return
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "apikey": supabase_key,
                    "Authorization": f"Bearer {supabase_key}",
                    "Content-Type": "application/json",
                    "Prefer": "resolution=merge-duplicates",
                }
                
                url = f"{supabase_url}/rest/v1/forum_knowledge"
                
                success = 0
                for topic in topics:
                    data = scraper.to_supabase_format(topic)
                    
                    try:
                        async with session.post(url, headers=headers, json=data) as resp:
                            if resp.status in (200, 201):
                                success += 1
                            elif resp.status == 409:
                                # Already exists, that's fine
                                success += 1
                            else:
                                text = await resp.text()
                                print(f"Error {resp.status}: {text[:100]}")
                    except Exception as e:
                        print(f"Upload failed: {e}")
                
                print(f"Uploaded: {success}/{len(topics)} topics")


if __name__ == "__main__":
    asyncio.run(main())
