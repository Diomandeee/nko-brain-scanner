# Experiment C: Cost Estimate

## Compute Requirements

| Task | Hardware | Time | Cost |
|------|----------|------|------|
| SFT translation to N'Ko | Local (any machine) | ~5 min | $0 |
| LoRA training (N'Ko adapter) | Vast.ai A100 or Mac5 | ~1 hour | ~$1-2 |
| Evaluation (both adapters) | Local M4 or Mac5 | ~30 min | $0 |

## Notes

- Translation is pure string processing using the local N'Ko transliteration engine. No API calls.
- LoRA training can run on Mac5 via MLX if the dataset is small enough (~1000 examples).
  For larger datasets, Vast.ai A100 is faster.
- Evaluation requires loading both adapters sequentially and generating 50 responses each.

## Total: ~$1-3

Translation is free. Training is cheap. The main cost is GPU time for the LoRA training run.
