#!/bin/bash
# Autopilot: sync Quebec features → Sweden every 30 min, restart training when done
# Run on Mac1 while user is away

VASTAI_KEY=~/.ssh/id_vastai
SWEDEN_PORT=32176
QUEBEC_PORT=37310
SWEDEN_HOST="root@ssh8.vast.ai"
QUEBEC_HOST="root@ssh4.vast.ai"

sync_features() {
    echo "[$(date '+%H:%M')] Syncing Quebec pairs → Sweden..."

    # Pull Quebec pairs
    scp -i $VASTAI_KEY -P $QUEBEC_PORT $QUEBEC_HOST:/workspace/results/feature_pairs_babamamadidiane.jsonl /tmp/q_baba.jsonl 2>/dev/null
    scp -i $VASTAI_KEY -P $QUEBEC_PORT $QUEBEC_HOST:/workspace/results/feature_pairs_djoko.jsonl /tmp/q_djoko.jsonl 2>/dev/null

    # Pull Sweden pairs
    scp -i $VASTAI_KEY -P $SWEDEN_PORT $SWEDEN_HOST:/workspace/results/feature_pairs_djoko.jsonl /tmp/s_djoko.jsonl 2>/dev/null

    # Merge
    cat /tmp/s_djoko.jsonl /tmp/q_djoko.jsonl /tmp/q_baba.jsonl 2>/dev/null | sort -u > /tmp/merged.jsonl
    MERGED=$(wc -l < /tmp/merged.jsonl | tr -d ' ')

    # Push merged back to Sweden
    scp -i $VASTAI_KEY -P $SWEDEN_PORT /tmp/merged.jsonl $SWEDEN_HOST:/workspace/results/feature_pairs_djoko.jsonl 2>/dev/null

    # Check feature counts
    S_FEAT=$(ssh -i $VASTAI_KEY -p $SWEDEN_PORT $SWEDEN_HOST 'find /workspace/features -name "*.pt" | wc -l' 2>/dev/null | tr -d ' ')
    Q_FEAT=$(ssh -i $VASTAI_KEY -p $QUEBEC_PORT $QUEBEC_HOST 'find /workspace/features -name "*.pt" | wc -l' 2>/dev/null | tr -d ' ')

    # Check training
    TRAIN=$(ssh -i $VASTAI_KEY -p $SWEDEN_PORT $SWEDEN_HOST 'tail -1 /workspace/training4.log 2>/dev/null' 2>/dev/null)

    echo "  Merged pairs: $MERGED | Sweden features: $S_FEAT | Quebec features: $Q_FEAT"
    echo "  Training: $TRAIN"
}

check_training_done() {
    PROCS=$(ssh -i $VASTAI_KEY -p $SWEDEN_PORT $SWEDEN_HOST 'ps aux | grep concurrent_train | grep -v grep | wc -l' 2>/dev/null | tr -d ' ')
    if [ "$PROCS" = "0" ]; then
        echo "[$(date '+%H:%M')] Training finished! Restarting with more data..."
        ssh -i $VASTAI_KEY -p $SWEDEN_PORT $SWEDEN_HOST 'cd /workspace && nohup python3 -u concurrent_train.py \
            --features-dir /workspace/features \
            --pairs /workspace/results/feature_pairs_djoko.jsonl \
            --codebook /workspace/syllable_codebook.json \
            --min-features 100 --epochs 500 --batch-size 8 --lr 3e-5 \
            > /workspace/training_next.log 2>&1 &' 2>/dev/null
        echo "  New training round started (500 epochs, lr=3e-5)"
    fi
}

echo "=== Autopilot: Feature Sync + Training Monitor ==="
echo "Syncing every 30 min. Ctrl+C to stop."
echo ""

while true; do
    sync_features
    check_training_done
    echo ""
    sleep 1800
done
