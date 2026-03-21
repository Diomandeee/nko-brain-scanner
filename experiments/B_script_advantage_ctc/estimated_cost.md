# Experiment B: Cost Estimate

## Compute Requirements

| Task | Hardware | Time | Cost |
|------|----------|------|------|
| Train N'Ko CTC (50 epochs) | Vast.ai A100 80GB | ~2 hours | ~$2-3 |
| Train Latin CTC (50 epochs) | Vast.ai A100 80GB | ~2 hours | ~$2-3 |
| Feature extraction (if needed) | Vast.ai A100 80GB | ~1 hour | ~$1-2 |
| Evaluation + comparison | Same GPU instance | ~15 min | included |

## Vast.ai Pricing

- A100 80GB spot: ~$0.80-1.20/hr
- Two training runs + evaluation: ~4.5 hours
- Total GPU rental: ~$4-6
- With buffer for restarts/debugging: ~$8-10

## Data Notes

- If Whisper features are already extracted (from previous V3 training), skip feature extraction
- 37,306 audio segments already available from the main ASR pipeline
- Latin transcriptions need to be generated from existing Whisper output (free, local)

## Total: ~$8-10

Can be reduced to ~$5 if features are already cached from prior training runs.
