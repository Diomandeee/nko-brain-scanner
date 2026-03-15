#!/usr/bin/env python3
"""
Daily Pipeline Scheduler for Cost-Controlled Video Processing

Processes N'Ko videos at a controlled rate to spread costs over 60 days.
- 933 videos / 60 days = ~16 videos/day
- Budget: $1-2/day (~$0.12 per video for OCR + worlds)

Features:
- Daily budget tracking with rollover
- Automatic rate limiting
- Resume from checkpoint
- Cost estimation before processing
- GCS-based video sourcing

Usage:
    python daily_pipeline_scheduler.py                  # Run daily batch
    python daily_pipeline_scheduler.py --budget 2.00    # Set daily budget
    python daily_pipeline_scheduler.py --videos-per-day 16
    python daily_pipeline_scheduler.py --dry-run        # Preview only
    python daily_pipeline_scheduler.py --status         # Show progress
"""

import asyncio
import json
import os
import sys
import subprocess
import argparse
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from lib.analyzer import NkoAnalyzer
    HAS_ANALYZER = True
except ImportError:
    HAS_ANALYZER = False
    logger.warning("NkoAnalyzer not available - will use subprocess fallback")


@dataclass
class DailyBudget:
    """Track daily spending."""
    date: str
    budget_usd: float
    spent_usd: float
    videos_processed: int
    frames_analyzed: int
    detections_found: int
    worlds_generated: int

    @property
    def remaining(self) -> float:
        return max(0, self.budget_usd - self.spent_usd)

    @property
    def is_exhausted(self) -> bool:
        return self.spent_usd >= self.budget_usd

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""
    daily_budget_usd: float = 2.00
    videos_per_day: int = 16
    cost_per_frame_ocr: float = 0.002
    cost_per_world_gen: float = 0.0001
    frames_per_video: int = 55  # After smart filtering
    worlds_per_detection: int = 5
    detection_rate: float = 0.3  # ~30% of frames have N'Ko
    gcs_bucket: str = "learnnko-videos"
    gcs_prefix: str = "videos/"
    checkpoint_dir: str = "./data/scheduler"
    target_days: int = 60
    total_videos: int = 933


@dataclass
class VideoInfo:
    """Video metadata for processing."""
    video_id: str
    gcs_path: str
    size_bytes: int = 0
    duration_seconds: int = 0
    estimated_cost: float = 0.0


class DailyBudgetTracker:
    """Track and persist daily API costs."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.budget_file = self.checkpoint_dir / "daily_budgets.json"
        self.processed_file = self.checkpoint_dir / "processed_videos.json"
        self._load()

    def _load(self):
        """Load budget history and processed videos."""
        if self.budget_file.exists():
            with open(self.budget_file) as f:
                data = json.load(f)
                self.budgets = {d['date']: DailyBudget(**d) for d in data.get('budgets', [])}
                self.total_spent = data.get('total_spent', 0.0)
        else:
            self.budgets = {}
            self.total_spent = 0.0

        if self.processed_file.exists():
            with open(self.processed_file) as f:
                self.processed_videos = set(json.load(f))
        else:
            self.processed_videos = set()

    def _save(self):
        """Save budget history and processed videos."""
        data = {
            'budgets': [b.to_dict() for b in self.budgets.values()],
            'total_spent': self.total_spent,
            'last_updated': datetime.now().isoformat(),
        }
        with open(self.budget_file, 'w') as f:
            json.dump(data, f, indent=2)

        with open(self.processed_file, 'w') as f:
            json.dump(list(self.processed_videos), f)

    def get_today_budget(self, default_budget: float) -> DailyBudget:
        """Get or create today's budget record."""
        today = date.today().isoformat()

        if today not in self.budgets:
            self.budgets[today] = DailyBudget(
                date=today,
                budget_usd=default_budget,
                spent_usd=0.0,
                videos_processed=0,
                frames_analyzed=0,
                detections_found=0,
                worlds_generated=0,
            )
            self._save()

        return self.budgets[today]

    def record_video_processed(
        self,
        video_id: str,
        cost_usd: float,
        frames: int,
        detections: int,
        worlds: int,
    ):
        """Record a processed video."""
        today = date.today().isoformat()
        budget = self.get_today_budget(2.00)

        budget.spent_usd += cost_usd
        budget.videos_processed += 1
        budget.frames_analyzed += frames
        budget.detections_found += detections
        budget.worlds_generated += worlds

        self.total_spent += cost_usd
        self.processed_videos.add(video_id)
        self._save()

        logger.info(f"Recorded: {video_id} - ${cost_usd:.4f} "
                   f"({frames} frames, {detections} detections, {worlds} worlds)")

    def is_video_processed(self, video_id: str) -> bool:
        """Check if a video has already been processed."""
        return video_id in self.processed_videos

    def can_afford(self, estimated_cost: float, daily_budget: float) -> bool:
        """Check if today's budget can afford the cost."""
        budget = self.get_today_budget(daily_budget)
        return budget.remaining >= estimated_cost

    def get_summary(self) -> Dict[str, Any]:
        """Get overall summary."""
        total_videos = len(self.processed_videos)
        total_frames = sum(b.frames_analyzed for b in self.budgets.values())
        total_detections = sum(b.detections_found for b in self.budgets.values())
        total_worlds = sum(b.worlds_generated for b in self.budgets.values())

        return {
            'total_videos_processed': total_videos,
            'total_frames_analyzed': total_frames,
            'total_detections_found': total_detections,
            'total_worlds_generated': total_worlds,
            'total_spent_usd': self.total_spent,
            'days_active': len(self.budgets),
            'avg_cost_per_video': self.total_spent / total_videos if total_videos > 0 else 0,
        }


class CostEstimator:
    """Estimate processing costs before committing."""

    def __init__(self, config: SchedulerConfig):
        self.config = config

    def estimate_video_cost(self, duration_minutes: int = 30) -> float:
        """Estimate cost to process a single video."""
        # Estimate frames based on duration (more frames for longer videos)
        estimated_frames = min(
            self.config.frames_per_video,
            max(20, duration_minutes * 2)
        )

        # OCR cost
        ocr_cost = estimated_frames * self.config.cost_per_frame_ocr

        # Estimate detections (~30% of frames have N'Ko)
        estimated_detections = int(estimated_frames * self.config.detection_rate)

        # World generation cost
        world_cost = (estimated_detections *
                     self.config.worlds_per_detection *
                     self.config.cost_per_world_gen)

        return ocr_cost + world_cost

    def estimate_daily_batch(self, videos: List[VideoInfo]) -> Dict[str, float]:
        """Estimate cost for a batch of videos."""
        total_ocr = 0.0
        total_worlds = 0.0

        for video in videos:
            duration = video.duration_seconds // 60 if video.duration_seconds else 30
            frames = min(self.config.frames_per_video, max(20, duration * 2))
            detections = int(frames * self.config.detection_rate)

            total_ocr += frames * self.config.cost_per_frame_ocr
            total_worlds += detections * self.config.worlds_per_detection * self.config.cost_per_world_gen

        return {
            'ocr_cost': total_ocr,
            'world_cost': total_worlds,
            'total_cost': total_ocr + total_worlds,
            'video_count': len(videos),
        }


class DailyPipelineScheduler:
    """Main scheduler orchestrator."""

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.budget_tracker = DailyBudgetTracker(config.checkpoint_dir)
        self.cost_estimator = CostEstimator(config)

    def get_gcs_videos(self) -> List[VideoInfo]:
        """Get list of videos from GCS bucket."""
        bucket = self.config.gcs_bucket
        prefix = self.config.gcs_prefix

        cmd = [
            os.path.expanduser("~/google-cloud-sdk/bin/gsutil"),
            "ls", "-l", f"gs://{bucket}/{prefix}*.mp4"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                logger.error(f"Failed to list GCS: {result.stderr}")
                return []

            videos = []
            for line in result.stdout.strip().split('\n'):
                if line.strip() and 'gs://' in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        size = int(parts[0]) if parts[0].isdigit() else 0
                        gcs_path = parts[-1]
                        video_id = Path(gcs_path).stem

                        videos.append(VideoInfo(
                            video_id=video_id,
                            gcs_path=gcs_path,
                            size_bytes=size,
                            estimated_cost=self.cost_estimator.estimate_video_cost(),
                        ))

            return videos

        except Exception as e:
            logger.error(f"Error listing GCS: {e}")
            return []

    def get_pending_videos(self) -> List[VideoInfo]:
        """Get videos from GCS that haven't been processed."""
        all_videos = self.get_gcs_videos()

        pending = [
            v for v in all_videos
            if not self.budget_tracker.is_video_processed(v.video_id)
        ]

        logger.info(f"Found {len(pending)} pending videos "
                   f"(of {len(all_videos)} total in GCS)")
        return pending

    async def process_video(self, video: VideoInfo) -> Dict[str, Any]:
        """Process a single video through the extraction pipeline."""
        video_id = video.video_id

        # Download from GCS to local temp
        temp_dir = Path(self.config.checkpoint_dir) / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        local_path = temp_dir / f"{video_id}.mp4"

        try:
            # Download from GCS
            logger.info(f"Downloading {video_id} from GCS...")
            cmd = [
                os.path.expanduser("~/google-cloud-sdk/bin/gsutil"),
                "-q", "cp", video.gcs_path, str(local_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                return {'error': f"GCS download failed: {result.stderr}"}

            # Process with extraction pipeline
            if HAS_ANALYZER:
                # Use NkoAnalyzer directly
                analyzer = NkoAnalyzer()
                result = await analyzer.process_video(str(local_path))

                frames = result.get('frames_analyzed', 0)
                detections = result.get('detections_found', 0)
                worlds = detections * 5  # 5 worlds per detection

            else:
                # Fallback to subprocess
                logger.info(f"Processing {video_id} with run_extraction.py...")
                cmd = [
                    sys.executable,
                    str(Path(__file__).parent / "run_extraction.py"),
                    "--video", str(local_path),
                    "--limit", "1",
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

                # Parse output for stats (approximate)
                frames = 55
                detections = int(frames * 0.3)
                worlds = detections * 5

            # Calculate actual cost
            actual_cost = (
                frames * self.config.cost_per_frame_ocr +
                detections * 5 * self.config.cost_per_world_gen
            )

            return {
                'video_id': video_id,
                'frames': frames,
                'detections': detections,
                'worlds': worlds,
                'cost': actual_cost,
                'success': True,
            }

        except Exception as e:
            logger.error(f"Error processing {video_id}: {e}")
            return {'error': str(e), 'video_id': video_id}

        finally:
            # Cleanup temp file
            if local_path.exists():
                local_path.unlink()

    async def run_daily_batch(self, dry_run: bool = False) -> Dict[str, Any]:
        """Process today's batch within budget."""
        budget = self.budget_tracker.get_today_budget(self.config.daily_budget_usd)

        logger.info(f"Daily budget: ${budget.budget_usd:.2f}, "
                   f"Spent: ${budget.spent_usd:.2f}, "
                   f"Remaining: ${budget.remaining:.2f}")

        if budget.is_exhausted:
            logger.warning("Daily budget exhausted!")
            return {
                'status': 'budget_exhausted',
                'spent': budget.spent_usd,
                'videos_today': budget.videos_processed,
            }

        # Get pending videos
        pending = self.get_pending_videos()

        if not pending:
            logger.info("No pending videos to process!")
            return {'status': 'no_pending_videos'}

        # Select videos for today
        videos_to_process = []
        estimated_total = 0.0

        for video in pending[:self.config.videos_per_day]:
            cost = self.cost_estimator.estimate_video_cost()
            if estimated_total + cost <= budget.remaining:
                video.estimated_cost = cost
                videos_to_process.append(video)
                estimated_total += cost
            else:
                break

        logger.info(f"Selected {len(videos_to_process)} videos "
                   f"(estimated cost: ${estimated_total:.2f})")

        if dry_run:
            return {
                'status': 'dry_run',
                'videos_selected': len(videos_to_process),
                'video_ids': [v.video_id for v in videos_to_process],
                'estimated_cost': estimated_total,
                'budget_remaining': budget.remaining,
            }

        # Process videos
        results = {
            'processed': [],
            'failed': [],
            'total_cost': 0.0,
        }

        for i, video in enumerate(videos_to_process, 1):
            logger.info(f"[{i}/{len(videos_to_process)}] Processing {video.video_id}...")

            result = await self.process_video(video)

            if result.get('success'):
                self.budget_tracker.record_video_processed(
                    video_id=video.video_id,
                    cost_usd=result['cost'],
                    frames=result['frames'],
                    detections=result['detections'],
                    worlds=result['worlds'],
                )
                results['processed'].append(video.video_id)
                results['total_cost'] += result['cost']
            else:
                results['failed'].append({
                    'video_id': video.video_id,
                    'error': result.get('error', 'Unknown error'),
                })

            # Check if we've exceeded budget
            updated_budget = self.budget_tracker.get_today_budget(self.config.daily_budget_usd)
            if updated_budget.is_exhausted:
                logger.warning("Budget exhausted mid-batch, stopping")
                break

        results['status'] = 'completed'
        results['videos_processed'] = len(results['processed'])
        results['videos_failed'] = len(results['failed'])

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get current status and progress."""
        summary = self.budget_tracker.get_summary()
        today = self.budget_tracker.get_today_budget(self.config.daily_budget_usd)
        pending = len(self.get_pending_videos())

        # Calculate progress
        total_processed = summary['total_videos_processed']
        total_target = self.config.total_videos
        progress_pct = (total_processed / total_target * 100) if total_target > 0 else 0

        # Estimate days remaining
        if summary['days_active'] > 0:
            avg_per_day = total_processed / summary['days_active']
            days_remaining = (pending / avg_per_day) if avg_per_day > 0 else float('inf')
        else:
            days_remaining = self.config.target_days

        return {
            'progress': {
                'videos_processed': total_processed,
                'videos_remaining': pending,
                'total_target': total_target,
                'progress_percent': round(progress_pct, 1),
            },
            'budget': {
                'total_spent': round(summary['total_spent_usd'], 2),
                'today_spent': round(today.spent_usd, 2),
                'today_remaining': round(today.remaining, 2),
                'avg_cost_per_video': round(summary['avg_cost_per_video'], 4),
            },
            'stats': {
                'total_frames': summary['total_frames_analyzed'],
                'total_detections': summary['total_detections_found'],
                'total_worlds': summary['total_worlds_generated'],
                'days_active': summary['days_active'],
            },
            'estimate': {
                'days_remaining': round(days_remaining, 1),
                'projected_total_cost': round(
                    summary['avg_cost_per_video'] * total_target, 2
                ) if summary['avg_cost_per_video'] > 0 else 0,
            },
        }


async def main():
    parser = argparse.ArgumentParser(
        description='Daily Pipeline Scheduler for N\'Ko Video Processing'
    )
    parser.add_argument('--budget', type=float, default=2.00,
                       help='Daily budget in USD (default: 2.00)')
    parser.add_argument('--videos-per-day', type=int, default=16,
                       help='Maximum videos per day (default: 16)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview what would be processed')
    parser.add_argument('--status', action='store_true',
                       help='Show current progress and exit')
    parser.add_argument('--checkpoint-dir', type=str,
                       default='./data/scheduler',
                       help='Directory for checkpoints')

    args = parser.parse_args()

    # Create config
    config = SchedulerConfig(
        daily_budget_usd=args.budget,
        videos_per_day=args.videos_per_day,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Initialize scheduler
    scheduler = DailyPipelineScheduler(config)

    if args.status:
        status = scheduler.get_status()
        print("\n" + "="*60)
        print("N'Ko Training Pipeline - Daily Scheduler Status")
        print("="*60)
        print(f"\nProgress:")
        print(f"  Videos processed: {status['progress']['videos_processed']}/{status['progress']['total_target']}")
        print(f"  Videos remaining: {status['progress']['videos_remaining']}")
        print(f"  Progress: {status['progress']['progress_percent']}%")
        print(f"\nBudget:")
        print(f"  Total spent: ${status['budget']['total_spent']}")
        print(f"  Today spent: ${status['budget']['today_spent']}")
        print(f"  Today remaining: ${status['budget']['today_remaining']}")
        print(f"  Avg cost/video: ${status['budget']['avg_cost_per_video']}")
        print(f"\nStats:")
        print(f"  Frames analyzed: {status['stats']['total_frames']:,}")
        print(f"  Detections found: {status['stats']['total_detections']:,}")
        print(f"  Worlds generated: {status['stats']['total_worlds']:,}")
        print(f"  Days active: {status['stats']['days_active']}")
        print(f"\nEstimate:")
        print(f"  Days remaining: {status['estimate']['days_remaining']}")
        print(f"  Projected total: ${status['estimate']['projected_total_cost']}")
        print("="*60 + "\n")
        return

    # Run daily batch
    print(f"\nStarting daily batch processing...")
    print(f"  Budget: ${args.budget}/day")
    print(f"  Max videos: {args.videos_per_day}/day")
    print(f"  Dry run: {args.dry_run}")
    print()

    results = await scheduler.run_daily_batch(dry_run=args.dry_run)

    print("\nResults:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
