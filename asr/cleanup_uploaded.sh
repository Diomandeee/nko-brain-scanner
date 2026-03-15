#!/bin/bash
# Auto-cleanup: delete local segments confirmed uploaded to Vast.ai
# Run periodically to prevent disk fill while downloads continue
SEGMENTS_DIR="/Users/mohameddiomande/Desktop/nko-brain-scanner/djoko_audio/segments"
VAST_SSH="ssh -p 28182 root@ssh2.vast.ai"

# Get dirs on Vast.ai
REMOTE_DIRS=$($VAST_SSH 'ls /workspace/segments/ 2>/dev/null' 2>/dev/null | grep -v "Welcome\|Have fun")

if [ -z "$REMOTE_DIRS" ]; then
    echo "$(date): Could not reach Vast.ai or no segments"
    exit 0
fi

DELETED=0
while IFS= read -r dir; do
    dir=$(echo "$dir" | tr -d '\r\n ')
    [ -z "$dir" ] && continue
    if [ -d "$SEGMENTS_DIR/$dir" ]; then
        rm -rf "$SEGMENTS_DIR/$dir"
        DELETED=$((DELETED + 1))
    fi
done <<< "$REMOTE_DIRS"

FREE=$(df -h /Users/mohameddiomande/ | tail -1 | awk '{print $4}')
REMAINING=$(find "$SEGMENTS_DIR" -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
echo "$(date): Cleaned $DELETED dirs, ${FREE} free, ${REMAINING} segments remaining"
