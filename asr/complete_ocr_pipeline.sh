#!/bin/bash
# Complete OCR Pipeline — Process ALL 532 babamamadidiane videos
# Runs on Vast.ai A100. Downloads videos, extracts frames, runs Gemini OCR,
# aligns audio with on-screen text, produces ground-truth N'Ko training pairs.
#
# Usage (on Vast.ai):
#   bash complete_ocr_pipeline.sh
#
# Requires: GOOGLE_API_KEY (for Gemini), ffmpeg, yt-dlp

set -e

echo "=========================================="
echo "Complete OCR Pipeline — babamamadidiane"
echo "=========================================="

# Install deps
pip install -q google-generativeai yt-dlp soundfile 2>&1 | tail -3
apt-get update -qq && apt-get install -y -qq ffmpeg 2>&1 | tail -3

# Get the dynamic OCR pipeline
cd /workspace

# Check if pipeline script exists (deployed via SCP)
if [ ! -f dynamic_ocr_pipeline.py ]; then
    echo "ERROR: dynamic_ocr_pipeline.py not found. Deploy it first."
    exit 1
fi

# List all video IDs
echo "Listing all babamamadidiane videos..."
yt-dlp --flat-playlist --print id 'https://www.youtube.com/@babamamadidiane/videos' 2>/dev/null > all_video_ids.txt
TOTAL=$(wc -l < all_video_ids.txt)
echo "Total videos: $TOTAL"

# Check what's already processed
if [ -f results/dynamic_ocr/checkpoint.json ]; then
    PROCESSED=$(python3 -c "import json; d=json.load(open('results/dynamic_ocr/checkpoint.json')); print(len(set(d.get('processed',[]))))")
    echo "Already processed: $PROCESSED"
else
    PROCESSED=0
fi

REMAINING=$((TOTAL - PROCESSED))
echo "Remaining: $REMAINING videos"
echo ""

# Run the pipeline with resume
echo "Starting OCR pipeline (this may take several hours)..."
python3 -u dynamic_ocr_pipeline.py --all --resume 2>&1 | tee /workspace/ocr_pipeline.log

echo ""
echo "=========================================="
echo "OCR Pipeline Complete"
echo "=========================================="
echo "Pairs: $(wc -l < results/dynamic_ocr/dynamic_pairs.jsonl 2>/dev/null || echo 0)"
echo "Results: results/dynamic_ocr/dynamic_pairs.jsonl"
