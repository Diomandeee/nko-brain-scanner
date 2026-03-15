# N'Ko Training Scripts

Local pipeline for extracting N'Ko content from YouTube videos using Gemini API.

## Overview

```
YouTube Video → yt-dlp → FFmpeg → Gemini OCR → 5-World Generation → Supabase
     │              │         │          │              │              │
     └─ Download ───┴─ Frames ┴─ N'Ko ───┴── Variants ──┴── Storage ───┘
```

## Cost Estimation

| Operation | Cost per Unit | Units per Video | Cost per Video |
|-----------|--------------|-----------------|----------------|
| Multimodal OCR | ~$0.002 | 50 frames | $0.10 |
| World Generation | ~$0.0001 | ~15 detections × 5 worlds | $0.0075 |
| **Total** | | | **~$0.11/video** |

For 522 videos: **~$57 total**

## Prerequisites

```bash
# Install dependencies
pip install scrapetube aiohttp pyyaml httpx

# Required tools
brew install yt-dlp ffmpeg  # macOS
# or: apt install yt-dlp ffmpeg  # Linux
```

## Environment Variables

```bash
# Required
export GEMINI_API_KEY="your-gemini-api-key"

# For Supabase storage (optional but recommended)
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_SERVICE_KEY="your-service-role-key"  # Required for inserts!
# Note: SUPABASE_ANON_KEY only works for reads due to RLS policies
```

## Usage

### Basic Usage (Local JSON Output)

```bash
# Process 5 videos from the channel
python nko_analyzer.py --limit 5

# Single video test
python nko_analyzer.py --video "https://www.youtube.com/watch?v=VIDEO_ID"

# Quick test with fewer frames
python nko_analyzer.py --limit 1 --max-frames 10
```

### With Supabase Storage

```bash
# Full pipeline with Supabase
python nko_analyzer.py --limit 5 --store-supabase

# Test with a single video
python nko_analyzer.py --video "URL" --store-supabase
```

### Skip World Generation (Cheaper, Faster)

```bash
# OCR only, no world variants
python nko_analyzer.py --limit 10 --skip-worlds
```

### Select Specific Worlds

```bash
# Only generate everyday and proverbs worlds
python nko_analyzer.py --worlds world_everyday world_proverbs
```

## Available Worlds

1. **world_everyday** - Casual conversation usage
2. **world_formal** - Official/formal writing
3. **world_storytelling** - Griot/oral tradition
4. **world_proverbs** - Mande wisdom sayings
5. **world_educational** - Teaching content

## Output

Results are saved to `analysis_results.json` (or custom path via `--output`):

```json
{
  "summary": {
    "videos_processed": 5,
    "total_frames": 250,
    "frames_with_nko": 75,
    "total_world_variants": 375
  },
  "results": [
    {
      "video_id": "xsUrdpKD5wM",
      "title": "ߒߞߏ ߦߌ߬ߘߊ߬ߞߏ...",
      "detections": [
        {
          "frame_index": 2,
          "nko_text": "ߒߞߏ ߘߐ߫ ߞߊ߬ߙߊ߲",
          "latin_transliteration": "N'Ko doŋ kaaraŋ",
          "english_translation": "N'Ko reading",
          "worlds": [
            {"world_name": "everyday", "variant_count": 3},
            {"world_name": "proverbs", "variant_count": 3}
          ]
        }
      ]
    }
  ]
}
```

## Files

- `nko_analyzer.py` - Main pipeline script
- `world_generator.py` - 5-world generation using Gemini text API
- `supabase_client.py` - Supabase database operations

## Troubleshooting

### "RLS policy violation" error

You're using the anon key instead of service role key. Set:
```bash
export SUPABASE_SERVICE_KEY="your-service-role-key"
```

### "No YAML module" warning

Install PyYAML:
```bash
pip install pyyaml
```

### Video download fails

Update yt-dlp:
```bash
pip install --upgrade yt-dlp
```

