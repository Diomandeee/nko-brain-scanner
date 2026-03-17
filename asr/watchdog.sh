#!/bin/bash
# NKo ASR Watchdog — Auto-recovery, checks every 10 min
KEY=~/.ssh/id_vastai
S_PORT=32176
P_PORT=32770
S="root@ssh8.vast.ai"
P="root@ssh5.vast.ai"

while true; do
    echo "[$(date '+%H:%M')] Checking..."

    # Sweden
    SP=$(ssh -i $KEY -o ConnectTimeout=10 -p $S_PORT $S 'ps aux | grep "char_level\|train_on_human" | grep -v grep | wc -l' 2>/dev/null | tr -d ' ')
    if [ "$SP" = "0" ] 2>/dev/null; then
        echo "  SWEDEN: Dead. Restarting..."
        ssh -i $KEY -o ConnectTimeout=10 -p $S_PORT $S 'cd /workspace && nohup python3 -u char_level_train.py --features-dir /workspace/human_features --pairs /workspace/bam_train_nko_fixed.jsonl --save-dir /workspace/human_model --epochs 200 --batch-size 8 --lr 3e-4 --min-features 100 > /workspace/human_char_training.log 2>&1 &' 2>/dev/null
        echo "  Restarted."
    else
        SL=$(ssh -i $KEY -o ConnectTimeout=10 -p $S_PORT $S 'tail -1 /workspace/human_char_training.log 2>/dev/null' 2>/dev/null)
        echo "  SWEDEN: OK ($SP) $SL"
    fi

    # Poland
    PP=$(ssh -i $KEY -o ConnectTimeout=10 -p $P_PORT $P 'ps aux | grep python | grep -v grep | wc -l' 2>/dev/null | tr -d ' ')
    if [ "$PP" = "0" ] 2>/dev/null; then
        PF=$(ssh -i $KEY -o ConnectTimeout=10 -p $P_PORT $P 'find /workspace/human_features -name "*.pt" 2>/dev/null | wc -l' 2>/dev/null | tr -d ' ')
        echo "  POLAND: Dead. Features: $PF"
        if [ "$PF" -gt 30000 ] 2>/dev/null; then
            ssh -i $KEY -o ConnectTimeout=10 -p $P_PORT $P 'cd /workspace && nohup python3 -u char_level_train.py --features-dir /workspace/human_features --pairs /workspace/bam_train_nko.jsonl --save-dir /workspace/char_model --epochs 200 --batch-size 8 --lr 3e-4 --min-features 100 > /workspace/char_training.log 2>&1 &' 2>/dev/null
            echo "  Restarted training."
        else
            ssh -i $KEY -o ConnectTimeout=10 -p $P_PORT $P 'cd /workspace && nohup python3 -u extract_human_features.py > /workspace/extract.log 2>&1 &' 2>/dev/null
            echo "  Restarted extraction."
        fi
    else
        PL=$(ssh -i $KEY -o ConnectTimeout=10 -p $P_PORT $P 'tail -1 /workspace/extract.log 2>/dev/null; tail -1 /workspace/v2_training.log 2>/dev/null' 2>/dev/null | tail -1)
        echo "  POLAND: OK ($PP) $PL"
    fi

    # Instance check
    R=$(vastai show instances 2>/dev/null | grep -c "running")
    [ "$R" -lt 2 ] && echo "  WARNING: Only $R instances running!"

    sleep 600
done
