# N'Ko Training Pipeline

A comprehensive pipeline for extracting, analyzing, and generating training data from N'Ko educational videos.

## Overview

This pipeline processes YouTube videos from the [@babamamadidiane](https://www.youtube.com/@babamamadidiane) channel to create structured training data for N'Ko language learning.

### Key Features

- **Smart Frame Extraction**: Scene detection and perceptual hashing to capture unique slides
- **N'Ko OCR**: Gemini multimodal API for accurate N'Ko text extraction
- **5-World Generation**: Diverse context variants for each detected phrase
- **Audio Segmentation**: Scene-aligned audio for future ASR transcription
- **Resume/Checkpoint**: Graceful shutdown and resumable processing

## Quick Start

```bash
# Activate environment
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)

# Check status
python scripts/run_extraction.py --status

# Run pipeline (resumes from checkpoint)
python scripts/run_extraction.py --resume

# Retry failed videos
python scripts/run_extraction.py --retry-failed
```

## Pipeline Passes

| Pass | Script | Description | Cost |
|------|--------|-------------|------|
| 1 | `run_extraction.py` | Download + OCR + Audio | ~$60 |
| 2 | `run_consolidation.py` | Deduplicate phrases | Free |
| 3 | `run_worlds.py` | Generate 5 worlds | ~$5 |
| 4 | `run_transcription.py` | ASR (optional) | ~$190 |

## Folder Structure

```
training/
├── scripts/              # Pipeline entry points
│   ├── run_extraction.py     # Pass 1: Download + OCR + Audio
│   ├── run_consolidation.py  # Pass 2: Deduplicate
│   ├── run_worlds.py         # Pass 3: World generation
│   └── run_transcription.py  # Pass 4: ASR (optional)
│
├── lib/                  # Core modules
│   ├── analyzer.py           # NkoAnalyzer class
│   ├── audio_extractor.py    # Audio extraction
│   ├── frame_filter.py       # Smart frame sampling
│   ├── world_generator.py    # 5-world generation
│   ├── supabase_client.py    # Database operations
│   └── retry_utils.py        # Retry logic
│
├── config/
│   └── production.yaml       # Pipeline configuration
│
├── data/                 # All data (gitignored)
│   ├── videos/               # Processed videos
│   │   └── {video_id}/
│   │       ├── frames/       # Extracted frames
│   │       ├── audio_segments/   # Scene audio
│   │       └── manifest.json # Timestamps + metadata
│   ├── temp/                 # Temporary downloads
│   ├── checkpoints/          # Resume checkpoints
│   └── logs/                 # Log files
│
└── schemas/              # JSON schemas for exports
```

## Configuration

Edit `config/production.yaml`:

```yaml
extraction:
  target_frames: 100        # Frames per video
  use_scene_detection: true # Best for slides
  skip_intro_seconds: 10    # Skip intros
  skip_credits_seconds: 30  # Skip credits

storage:
  local:
    keep_frames: true       # Save frames
    keep_audio: true        # Save audio segments

worlds:
  enabled: true
  selected:
    - world_everyday
    - world_formal
    - world_storytelling
    - world_proverbs
    - world_educational
```

## Commands

```bash
# Show progress
python scripts/run_extraction.py --status

# Process N videos
python scripts/run_extraction.py --limit 10

# Resume from checkpoint
python scripts/run_extraction.py --resume

# Retry failed videos
python scripts/run_extraction.py --retry-failed

# Dry run (list videos)
python scripts/run_extraction.py --dry-run

# Skip audio extraction
python scripts/run_extraction.py --no-audio
```

## Output

### Supabase Tables

| Table | Description |
|-------|-------------|
| `nko_sources` | Video metadata |
| `nko_frames` | Extracted frames |
| `nko_detections` | OCR results |
| `nko_trajectories` | Learning paths |
| `nko_audio_segments` | Audio for ASR |

### Local Files

Each processed video creates:

```
data/videos/{video_id}/
├── audio.m4a              # Full audio track
├── audio_segments/        # Scene-based segments
│   ├── segment_0001.m4a
│   └── ...
├── frames/                # Extracted frames
│   ├── frame_0001.jpg
│   └── ...
└── manifest.json          # All metadata
```

## Cost Estimate

For 522 videos (~60 minutes each):

| Component | Calculation | Cost |
|-----------|-------------|------|
| OCR (Pass 1) | 522 × 55 frames × $0.002 | $57 |
| Worlds (Pass 3) | 3000 phrases × 5 × $0.0001 | $2 |
| ASR (Pass 4) | 522 × 60 min × $0.006/min | $188 |
| **Total** | | **$247** |

Without ASR: **~$60**

## Graceful Shutdown

Press `Ctrl+C` during processing:
- Current video completes
- Checkpoint is saved
- Resume with `--resume`

## Troubleshooting

### Video download fails
```bash
# Retry with multiple strategies
python scripts/run_extraction.py --retry-failed
```

### Check pipeline status
```bash
python scripts/run_extraction.py --status
```

### Force run (override lock)
```bash
python scripts/run_extraction.py --force
```

