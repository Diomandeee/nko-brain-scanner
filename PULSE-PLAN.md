# Pulse Plan: NKO Brain Scanner — arXiv Sprint

## Goal
Execute the Evo3 master plan to take the NKO Brain Scanner from experimental results to arXiv preprint in 10 days. Fix critical evaluation issues, extend the tokenizer, build production artifacts, and write the paper.

## Source
Evo3 master plan at `~/Desktop/evo-cube-output/nko-brain-scanner-frontier/stage3-expand-master-plan.md`

## Known Problems (from Evo3 stress test)
1. **English eval broken**: Only 4 examples (141 tokens). Every English PPL number is noise.
2. **MLX embedding resize uncharted**: No `resize_token_embeddings()` in MLX. Need manual weight surgery on quantized model.
3. **Architecture mismatch**: Brain scan on Qwen2-72B (80 layers), fine-tuning on Qwen3-8B (36 layers). Need 8B brain scan.
4. **English degradation**: Fine-tuning may have caused catastrophic forgetting (unclear due to broken eval).
5. **No production serving**: Adapters exist but no inference API deployed.
6. **No HuggingFace artifacts**: Model, tokenizer, dataset not published.

## Wave 0: Pre-Flight (Iteration 1) ✅ COMPLETE
- [x] Check Mac5 disk space (27GB free)
- [x] Verify adapter files exist (19.4MB each)
- [x] Check embedding dtype on Mac5: uint32 (quantized, shape 151936x1024)
- [x] Smoke-test cross-script bridge: Bridge class works
- [x] Verify GCS download status (169 uploaded)

## Wave 1: Fix English Eval + True Baselines (Iterations 2-3) ✅ COMPLETE
- [x] Created `eval/build_eval_set.py`: 100 English + 100 N'Ko examples, SHA-256 dedup
- [x] Created `eval/run_corrected_profiler.py`: all 3 stages on frozen eval
- [x] Uploaded to Mac5 and ran profiler
- [x] Corrected results: NKo acc 32.8%, Eng PPL 3.80, translation tax 0.70x (-76%)
- [x] Updated blog + docs, committed (793f540)

## Wave 2 Track B: Fuse + Deploy (Iteration 3) ✅ COMPLETE
- [x] Fused 3-stage adapter: `~/nko-brain-scanner/fused-nko-qwen3/` (8.1GB on Mac5)
- [x] Tested inference: coherent text generation confirmed
- [x] MLX server deployed at :8150 (module: `mlx_lm.server`)
- [x] API endpoint documented in model_card.md

## Wave 2: Core Technical (Iterations 4-6)

### Track A: Tokenizer Extension ✅ MOSTLY COMPLETE
- [x] Created `tokenizer/extend_hf_tokenizer.py`: dequantize → constituent-mean init → re-quantize
- [x] Implemented constituent-mean embedding initialization (250/250 success, 0 fallback)
- [x] Tested on Mac5: model produces valid finite logits after resize (152192 vocab)
- [x] MLX resize WORKED: no need for Vast.ai fallback
- [x] Token compression confirmed: 29.6% fewer tokens on N'Ko eval (1955 → 1376)
- [x] Pre-training PPL baseline: 4.62 (original) vs 291 (extended, untrained embeddings)
- [ ] Retrain with extended tokenizer (500 iters, 3e-6 lr) — IN PROGRESS
- [ ] Run profiler on extended model

### Track B: Fuse + Deploy ✅ COMPLETE
- [x] Fused 3-stage adapter: `~/nko-brain-scanner/fused-nko-qwen3/` (8.1GB)
- [x] MLX server deployed at :8150 (module name: `mlx_lm.server`, not `mlx_lm.serve`)
- [x] Tested inference: coherent text generation confirmed
- [x] API endpoint documented in model_card.md

### Track C: Bayelemabaga Pilot ❌ DROPPED (kill criteria met)
- [x] Downloaded Bayelemabaga dataset (74,162 Bambara-French pairs from HF parquet)
- [x] Ran 500-pair pilot through cross-script bridge
- [x] Result: 500/500 converted but only 21.2% clean (no Latin leaks)
- [x] Bridge doesn't handle Bambara extended Latin chars: ɛ, ɔ, ʒ, ʔ, î
- [x] **DROPPED**: Clean rate 21.2% < 70% kill threshold. 106 clean pairs insufficient for SFT.

### Track D: Human Eval Prep
- [ ] Design evaluation protocol: 50 paired samples, 3-dimension Likert scale
- [ ] Build `eval/human_eval.py`: generates evaluation spreadsheet
- [ ] Mohamed recruits 3-5 N'Ko literate evaluators

## Wave 3: Integration (Iterations 7-8)

### Track A: 8B Brain Scan ✅ COMPLETE
- [x] Created `scanner/mlx_brain_scan_8b.py` with LayerCapture wrapper pattern
- [x] Ran on Mac5: 36 layers profiled (30 eng + 30 nko examples)
- [x] Generated 3 publication figures: L2 comparison, delta, sparsity
- [x] Key finding: Layers 0-27 frozen, 28-34 reduced (-103 ΔL2), Layer 35 spiked (+573)
- [x] `scanner/plot_brain_scan.py` generates figures from results JSON

### Track B: Demo ✅ PARTIAL
- [x] Built Gradio demo app with 5 tabs: generate, analyze, brain scan, results, about
- [x] Running on Mac5:7861
- [ ] Deploy to HuggingFace Spaces (requires fused model upload first)
- [ ] Add Supabase feedback logging

### Track C: HuggingFace Publication ✅ PARTIAL
- [x] Created model card (`model_card.md`) with full results, brain scan, citation
- [x] Created upload script (`scripts/upload_to_hf.py`)
- [ ] Run `huggingface-cli login` on Mac5 (requires user auth)
- [ ] Execute `python3 scripts/upload_to_hf.py --all` on Mac5

## Wave 4: Paper + arXiv (Iterations 9-10) ✅ PARTIAL
- [x] Created `paper/main.tex` with ACL template (8 sections)
- [x] Created `paper/references.bib` (11 references)
- [x] Brain scan section filled with actual results
- [x] Tables: 3-stage comparison, training config, NKo vs English
- [x] Figures referenced: brain_scan_l2_comparison.png, brain_scan_delta.png
- [ ] Final revision pass (polish prose, check numbers consistency)
- [ ] Compile LaTeX and verify PDF
- [ ] Submit to arXiv (cs.CL)
- [ ] Update blog post with arXiv link

## Success Criteria
- Corrected English PPL in 10-30 range (sane for Qwen3-8B)
- At least one of: tokenizer extension OR Bayelemabaga data expansion working
- 8B brain scan before/after figures generated
- arXiv preprint submitted
- Model published to HuggingFace Hub

## Kill Criteria
- If MLX embedding resize AND Vast.ai fallback both fail → drop tokenizer extension, proceed with MVP paper
- If corrected English eval shows no degradation → remove catastrophic forgetting section
- If cross-script bridge accuracy < 70% on Bambara → drop Bayelemabaga entirely
