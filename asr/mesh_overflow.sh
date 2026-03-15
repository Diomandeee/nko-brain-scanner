#!/bin/bash
# Mesh Distributed Storage — Overflow segments across mesh nodes
#
# When Mac1 disk is low, distribute segments to Mac4/Mac5/cloud-vm.
# Segments are also pushed to Vast.ai for GPU transcription.
#
# Flow:
#   Mac1 (download) → Vast.ai (GPU transcription)
#                    → Mac4/Mac5 (overflow cache)
#                    → cloud-vm (long-term archive)
#
# Usage:
#   ./asr/mesh_overflow.sh              # Run once
#   ./asr/mesh_overflow.sh --watch      # Run continuously (every 5 min)
#   ./asr/mesh_overflow.sh --status     # Show mesh storage status

set -e

SEGMENTS_DIR="/Users/mohameddiomande/Desktop/nko-brain-scanner/djoko_audio/segments"
MIN_FREE_GB=4  # Trigger overflow when Mac1 has less than this
BATCH_SIZE=20  # Dirs to move per overflow cycle

# Mesh nodes with Tailscale SSH aliases
OVERFLOW_NODES=(
    "cloud-vm:/home/mohameddiomande/nko-segments"   # 102GB free, primary archive
    "mac4:/Users/mohameddiomande/nko-segments"        # 21GB free
    "mac5:/Users/mohameddiomande/nko-segments"        # 20GB free
)

# Vast.ai for transcription (separate from overflow)
VAST_SSH="ssh -p 28182 root@ssh2.vast.ai"
VAST_SEGMENTS="/workspace/segments"

get_free_gb() {
    df -g /Users/mohameddiomande/ 2>/dev/null | tail -1 | awk '{print $4}'
}

mesh_status() {
    echo "=== Mesh Storage Status ==="
    echo ""
    printf "%-12s %8s %8s %10s\n" "Node" "Free" "Segments" "Status"
    echo "-------------------------------------------"

    # Mac1 (local)
    local_free=$(get_free_gb)
    local_segs=$(find "$SEGMENTS_DIR" -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
    printf "%-12s %7sGB %8s %10s\n" "Mac1" "$local_free" "$local_segs" "$([ "$local_free" -lt "$MIN_FREE_GB" ] && echo 'LOW' || echo 'ok')"

    for node_spec in "${OVERFLOW_NODES[@]}"; do
        node="${node_spec%%:*}"
        path="${node_spec##*:}"
        free=$(ssh "$node" "df -g ~ 2>/dev/null | tail -1 | awk '{print \$4}'" 2>/dev/null || echo "?")
        segs=$(ssh "$node" "find $path -name '*.wav' 2>/dev/null | wc -l | tr -d ' '" 2>/dev/null || echo "0")
        printf "%-12s %7sGB %8s %10s\n" "$node" "$free" "$segs" "ok"
    done

    # Vast.ai
    vast_segs=$($VAST_SSH "find $VAST_SEGMENTS -name '*.wav' 2>/dev/null | wc -l" 2>/dev/null | grep -E "^[0-9]" || echo "?")
    vast_done=$($VAST_SSH "wc -l < /workspace/results/transcriptions.jsonl 2>/dev/null" 2>/dev/null | grep -E "^[0-9]" || echo "?")
    printf "%-12s %8s %8s %10s\n" "Vast.ai" "GPU" "$vast_segs" "${vast_done} done"

    echo ""
}

overflow_segments() {
    local free_gb=$(get_free_gb)

    if [ "$free_gb" -ge "$MIN_FREE_GB" ]; then
        echo "$(date): Mac1 has ${free_gb}GB free (above ${MIN_FREE_GB}GB threshold). No overflow needed."
        return 0
    fi

    echo "$(date): Mac1 LOW DISK (${free_gb}GB free). Overflowing segments..."

    # First, clean segments already on Vast.ai
    /Users/mohameddiomande/Desktop/nko-brain-scanner/asr/cleanup_uploaded.sh 2>/dev/null

    free_gb=$(get_free_gb)
    if [ "$free_gb" -ge "$MIN_FREE_GB" ]; then
        echo "$(date): Cleanup freed enough space (${free_gb}GB). No overflow needed."
        return 0
    fi

    # Still low - overflow to mesh nodes
    local dirs_to_move=$(ls -d "$SEGMENTS_DIR"/*/ 2>/dev/null | head -"$BATCH_SIZE")

    if [ -z "$dirs_to_move" ]; then
        echo "$(date): No segment dirs to overflow."
        return 0
    fi

    # Pick the node with most free space
    local best_node=""
    local best_free=0
    for node_spec in "${OVERFLOW_NODES[@]}"; do
        node="${node_spec%%:*}"
        path="${node_spec##*:}"
        free=$(ssh "$node" "df -g ~ 2>/dev/null | tail -1 | awk '{print \$4}'" 2>/dev/null || echo "0")
        if [ "$free" -gt "$best_free" ]; then
            best_free=$free
            best_node="$node"
            best_path="$path"
        fi
    done

    if [ -z "$best_node" ] || [ "$best_free" -lt 3 ]; then
        echo "$(date): No mesh node has enough space for overflow."
        return 1
    fi

    echo "$(date): Overflowing to $best_node ($best_path, ${best_free}GB free)..."

    # Create remote dir
    ssh "$best_node" "mkdir -p $best_path" 2>/dev/null

    # Move segments
    local moved=0
    for dir in $dirs_to_move; do
        dirname=$(basename "$dir")
        rsync -az "$dir" "${best_node}:${best_path}/" 2>/dev/null && rm -rf "$dir" && moved=$((moved + 1))
    done

    free_gb=$(get_free_gb)
    echo "$(date): Moved $moved dirs to $best_node. Mac1 now has ${free_gb}GB free."
}

watch_mode() {
    echo "Mesh overflow watch mode (checking every 5 min)..."
    while true; do
        overflow_segments
        sleep 300
    done
}

case "${1:-}" in
    --status) mesh_status ;;
    --watch)  watch_mode ;;
    *)        overflow_segments ;;
esac
