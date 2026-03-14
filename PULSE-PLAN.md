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

## Wave 0: Pre-Flight (Iteration 1)
- [ ] Check Mac5 disk space (need 20GB free)
- [ ] Verify adapter files exist: `~/nko-brain-scanner/adapters-bpe/adapters.safetensors`
- [ ] Check embedding dtype on Mac5: `embed_tokens.weight.dtype` for Qwen3-8B-8bit
- [ ] Smoke-test cross-script bridge: convert 5 Bambara sentences to N'Ko
- [ ] Verify GCS download status (should be 96+ videos)

## Wave 1: Fix English Eval + True Baselines (Iterations 2-3)
- [ ] Create `eval/` directory in nko-brain-scanner
- [ ] Build `eval/build_eval_set.py`: generate 100 English eval examples from diverse sources (Dolly-15k, general knowledge, math)
- [ ] Build 100 N'Ko eval examples from held-out data (zero overlap with any training set)
- [ ] Format all examples identically: `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`
- [ ] Create `eval/run_corrected_profiler.py`: runs base, 2-stage, 3-stage on new eval sets
- [ ] Upload eval scripts + data to Mac5
- [ ] Run corrected profiler on Mac5
- [ ] Collect and analyze corrected results
- [ ] Update blog post with corrected English numbers

## Wave 2: Core Technical (Iterations 4-6)

### Track A: Tokenizer Extension
- [ ] Create `tokenizer/extend_hf_tokenizer.py`: add top 250 BPE tokens to Qwen3 tokenizer
- [ ] Implement constituent-mean embedding initialization
- [ ] Test on Mac5: verify model produces valid logits after resize
- [ ] If MLX resize fails: fall back to Vast.ai PyTorch approach
- [ ] Retrain with extended tokenizer (500 iters, 3e-6 lr)
- [ ] Run profiler on extended model

### Track B: Fuse + Deploy
- [ ] Fuse 3-stage adapter into base weights: `python3 -m mlx_lm.fuse`
- [ ] Start MLX server: `python3 -m mlx_lm.serve --port 8150`
- [ ] Test inference: send N'Ko and English prompts
- [ ] Document API endpoint and example usage

### Track C: Bayelemabaga Pilot
- [ ] Download Bayelemabaga dataset (47K Bambara-French pairs)
- [ ] Run 500-pair pilot through cross-script bridge (Latin Bambara → N'Ko)
- [ ] Mohamed reviews 100 converted pairs for accuracy
- [ ] Go/no-go decision on full 47K conversion

### Track D: Human Eval Prep
- [ ] Design evaluation protocol: 50 paired samples, 3-dimension Likert scale
- [ ] Build `eval/human_eval.py`: generates evaluation spreadsheet
- [ ] Mohamed recruits 3-5 N'Ko literate evaluators

## Wave 3: Integration (Iterations 7-8)

### Track A: 8B Brain Scan
- [ ] Create `scanner/mlx_brain_scan_comparison.py`: before/after activation profiling on 8B
- [ ] Run on Mac5: extract per-layer metrics for base vs fine-tuned
- [ ] Generate publication-quality comparison figures (the hero figure)
- [ ] Compare activation patterns: which layers learned N'Ko?

### Track B: Demo
- [ ] Build Gradio demo app with 4 tabs: generate, translate, analyze, compare
- [ ] Deploy to HuggingFace Spaces (upload fused model)
- [ ] Add Supabase feedback logging

### Track C: HuggingFace Publication
- [ ] Create model card for `Diomandeee/nko-qwen3-8b`
- [ ] Upload fused model + tokenizer to HF Hub
- [ ] Upload parallel corpus as `Diomandeee/nko-parallel-corpus` dataset
- [ ] Upload BPE vocabulary as artifact

## Wave 4: Paper + arXiv (Iterations 9-10)
- [ ] Create `paper/` directory with ACL template
- [ ] Write paper sections: Introduction, Background, Methodology, Results, Discussion, Conclusion
- [ ] Include figures: activation comparison, heatmap, training curves, tokenizer compression
- [ ] Include tables: 3-stage comparison, translation tax, BPE merges
- [ ] Final revision pass
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
