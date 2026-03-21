# Experiment A: Script Invisibility Across Architectures

## Research Question

Is N'Ko's invisibility to language models universal, or specific to Qwen?

The original brain scan found that Qwen3-8B processes N'Ko with significantly higher sparsity,
lower entropy, and flatter kurtosis compared to English at every layer. This experiment tests
whether that pattern holds across architecturally different model families.

## Hypothesis

Script invisibility is a universal property of models trained predominantly on Latin and CJK data.
Every tested model will show the same activation deficit for N'Ko: higher sparsity (more dead neurons),
lower entropy (less information processing), and flatter kurtosis (less specialized circuits).

## Models

| Model | Params | Architecture | N'Ko in Training? | Compute |
|-------|--------|-------------|-------------------|---------|
| Qwen3-8B | 8B | Qwen (36 layers) | Unlikely (< 0.01%) | Local M4 |
| Llama-3.1-8B | 8B | LLaMA (32 layers) | No | Local M4 |
| Gemma-3-12B | 12B | Gemma (40 layers) | Unlikely | Local M4 / Mac5 |
| Qwen2-72B | 72B | Qwen (80 layers) | Unlikely | Vast.ai A100 |

## Method

For each model:
1. Load in 4-bit or 8-bit quantization (depending on available memory)
2. Feed 100 parallel English/N'Ko sentence pairs from `data/eval_pairs.jsonl`
3. At every layer, extract the hidden state and compute 4 metrics:
   - L2 norm (activation magnitude)
   - Shannon entropy (distribution spread)
   - Sparsity (fraction of near-zero neurons)
   - Kurtosis (circuit specialization)
4. Compute the "translation tax": ratio of N'Ko perplexity to English perplexity
5. Save per-layer JSON results

After all 4 models are scanned:
- Run `compare_models.py` to produce cross-model comparison table and figures
- Look for universal patterns vs model-specific quirks

## Data

`data/eval_pairs.jsonl` contains 100 parallel English/N'Ko pairs drawn from the project's
parallel corpus (`../../data/parallel_corpus.jsonl`). Same pairs used in the original brain scan.

## Running

```bash
# Scan each model (run sequentially or in parallel on different machines)
python3 run_brain_scan.py --model mlx-community/Qwen3-8B-8bit --output results/qwen3_8b.json
python3 run_brain_scan.py --model meta-llama/Llama-3.1-8B --output results/llama_8b.json
python3 run_brain_scan.py --model google/gemma-3-12b --output results/gemma_12b.json
python3 run_brain_scan.py --model Qwen/Qwen2-72B-Instruct --output results/qwen2_72b.json --quantize 4bit

# Compare all results
python3 compare_models.py --results-dir results/ --output results/comparison.json
```

## Expected Outputs

- `results/<model_name>.json` per model with per-layer metrics for English and N'Ko
- `results/comparison.json` with cross-model summary
- `results/figures/` with comparison plots (activation curves, sparsity heatmaps)

## Success Criteria

- If 4/4 models show the same activation deficit pattern for N'Ko: strong evidence for universal script invisibility
- If some models handle N'Ko better: indicates training data diversity matters more than architecture
- Either outcome is publishable

## Estimated Time

- 3 local models: ~2 hours each = ~6 hours
- Qwen2-72B on Vast.ai: ~30 minutes + setup
- Comparison analysis: ~5 minutes
