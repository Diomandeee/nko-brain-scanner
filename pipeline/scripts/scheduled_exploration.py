#!/usr/bin/env python3
"""
Scheduled Vocabulary Exploration

Background process that continuously enriches vocabulary through:
1. Processing the expansion queue (words waiting to be enriched)
2. Rate-limited API calls to avoid abuse
3. Organic vocabulary growth through cross-referencing

Run modes:
- Once: Process a batch and exit
- Continuous: Run indefinitely, processing batches on interval
- Daemon: Run as a background service

Usage:
    # Process one batch (10 words) and exit
    python scheduled_exploration.py --once
    
    # Run continuously (default: every 5 minutes)
    python scheduled_exploration.py --interval 300
    
    # Dry run - show what would be processed
    python scheduled_exploration.py --dry-run
    
    # Process with AI enrichment enabled
    python scheduled_exploration.py --enable-ai
"""

import asyncio
import argparse
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from expansion_engine import ExpansionEngine, EnrichmentResult

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ExplorationScheduler:
    """
    Scheduled vocabulary exploration with rate limiting and graceful shutdown.
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        delay_between_words: float = 2.0,
        interval_seconds: int = 300,
        enable_ai: bool = False,
        dry_run: bool = False,
    ):
        """
        Initialize the scheduler.
        
        Args:
            batch_size: Words to process per batch
            delay_between_words: Seconds between word enrichments
            interval_seconds: Seconds between batches (for continuous mode)
            enable_ai: Enable Gemini AI enrichment (costs ~$0.0001/word)
            dry_run: Don't actually process, just show what would happen
        """
        self.batch_size = batch_size
        self.delay_between_words = delay_between_words
        self.interval_seconds = interval_seconds
        self.enable_ai = enable_ai
        self.dry_run = dry_run
        
        self._running = True
        self._engine: Optional[ExpansionEngine] = None
        
        # Stats
        self.total_processed = 0
        self.total_enriched = 0
        self.total_failed = 0
        self.start_time = datetime.now()
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown on signals."""
        def handle_shutdown(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self._running = False
        
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)
    
    async def _init_engine(self):
        """Initialize the expansion engine."""
        if self._engine is None:
            self._engine = ExpansionEngine(
                enable_ai_enrichment=self.enable_ai,
                enable_queue_expansion=True,
            )
        return self._engine
    
    async def run_once(self) -> dict:
        """
        Run a single batch and return results.
        
        Returns:
            Summary of processed items
        """
        engine = await self._init_engine()
        
        logger.info(f"Processing batch of {self.batch_size} words...")
        
        if self.dry_run:
            # Just show queue status
            items = await engine.get_pending_items(self.batch_size)
            logger.info(f"Would process {len(items)} items:")
            for item in items:
                logger.info(f"  - {item.word} (source: {item.source_type}, priority: {item.priority})")
            return {
                "mode": "dry_run",
                "would_process": len(items),
                "items": [{"word": i.word, "source": i.source_type} for i in items],
            }
        
        # Process the batch
        result = await engine.process_queue_batch(
            limit=self.batch_size,
            delay_seconds=self.delay_between_words,
        )
        
        self.total_processed += result.get("processed", 0)
        self.total_enriched += result.get("enriched", 0)
        self.total_failed += result.get("failed", 0)
        
        logger.info(
            f"Batch complete: {result['processed']} processed, "
            f"{result['enriched']} enriched, {result['failed']} failed"
        )
        
        if result.get("queued_words", 0) > 0:
            logger.info(f"  Queued {result['queued_words']} related words for future exploration")
        
        return result
    
    async def run_continuous(self):
        """
        Run continuously, processing batches on interval.
        """
        self._setup_signal_handlers()
        
        logger.info("=" * 60)
        logger.info("Vocabulary Exploration Scheduler")
        logger.info("=" * 60)
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Interval: {self.interval_seconds}s")
        logger.info(f"  AI enrichment: {'Enabled' if self.enable_ai else 'Disabled'}")
        logger.info(f"  Delay between words: {self.delay_between_words}s")
        logger.info("=" * 60)
        
        # Show initial queue status
        engine = await self._init_engine()
        status = await engine.get_queue_status()
        logger.info(f"Queue status: {status.get('total_pending', 0)} pending")
        
        while self._running:
            try:
                # Check if there's work to do
                items = await engine.get_pending_items(1)
                
                if items:
                    await self.run_once()
                else:
                    logger.debug("Queue empty, waiting...")
                
                # Wait for next interval (check _running flag periodically)
                for _ in range(self.interval_seconds):
                    if not self._running:
                        break
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                await asyncio.sleep(60)  # Back off on error
        
        # Final summary
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info("\n" + "=" * 60)
        logger.info("Scheduler Summary")
        logger.info("=" * 60)
        logger.info(f"  Runtime: {int(elapsed)}s ({int(elapsed/60)}m)")
        logger.info(f"  Total processed: {self.total_processed}")
        logger.info(f"  Total enriched: {self.total_enriched}")
        logger.info(f"  Total failed: {self.total_failed}")
    
    async def show_status(self):
        """Show current queue status without processing."""
        engine = await self._init_engine()
        
        status = await engine.get_queue_status()
        stats = await engine.get_learning_stats(days=7)
        
        print("\n" + "=" * 60)
        print("Vocabulary Expansion Queue Status")
        print("=" * 60)
        
        print("\n📊 Queue:")
        print(f"  Pending:    {status.get('total_pending', 0)}")
        print(f"  Completed:  {status.get('total_completed', 0)}")
        
        by_source = status.get('by_source', {})
        if by_source:
            print("\n  By source:")
            for source, count in sorted(by_source.items()):
                print(f"    {source}: {count}")
        
        print("\n📈 Last 7 days:")
        for stat in stats[:7]:
            date = stat.get('stat_date', 'Unknown')
            enriched = stat.get('words_enriched', 0)
            matches = stat.get('dictionary_matches', 0)
            print(f"  {date}: {enriched} enriched, {matches} dictionary matches")
        
        print()


async def main():
    parser = argparse.ArgumentParser(
        description="Scheduled Vocabulary Exploration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--once", action="store_true",
                        help="Process one batch and exit")
    parser.add_argument("--status", action="store_true",
                        help="Show queue status and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without doing it")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Words to process per batch (default: 10)")
    parser.add_argument("--interval", type=int, default=300,
                        help="Seconds between batches (default: 300)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between words (default: 2.0)")
    parser.add_argument("--enable-ai", action="store_true",
                        help="Enable AI enrichment (costs ~$0.0001/word)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check environment
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_KEY"):
        logger.error("SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables required")
        sys.exit(1)
    
    scheduler = ExplorationScheduler(
        batch_size=args.batch_size,
        delay_between_words=args.delay,
        interval_seconds=args.interval,
        enable_ai=args.enable_ai,
        dry_run=args.dry_run,
    )
    
    if args.status:
        await scheduler.show_status()
    elif args.once:
        result = await scheduler.run_once()
        if not args.dry_run:
            print(f"\n✓ Processed {result.get('processed', 0)} words")
    else:
        await scheduler.run_continuous()


if __name__ == "__main__":
    asyncio.run(main())

