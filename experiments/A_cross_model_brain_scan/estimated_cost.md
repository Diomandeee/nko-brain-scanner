# Experiment A: Cost Estimate

## Compute Requirements

| Model | Hardware | Time | Cost |
|-------|----------|------|------|
| Qwen3-8B (8-bit) | Local M4 16GB | ~45 min | $0 |
| Llama-3.1-8B (8-bit) | Local M4 16GB | ~45 min | $0 |
| Gemma-3-12B (8-bit) | Local M4 or Mac5 | ~60 min | $0 |
| Qwen2-72B (4-bit) | Vast.ai A100 80GB | ~30 min | $2-3 |

## Vast.ai Pricing

- A100 80GB spot: ~$0.80-1.20/hr
- 30 min profiling run: ~$0.50-0.60
- Add ~10 min for model download and setup
- Total GPU rental: ~$0.70-0.90
- With buffer for retries: ~$2-3

## Total: ~$2-3

Three of four models run for free on local Apple Silicon.
Only the 72B model requires rented GPU time.
