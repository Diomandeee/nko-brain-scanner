# Meta-Recursive Explorer Memory

## NKO Brain Scanner Project
- **Location**: `~/Desktop/nko-brain-scanner/` (GitHub repo: Diomandeee/nko-brain-scanner)
- **Model**: Qwen3-8B-8bit (MLX, Apple Silicon local training) + Qwen2-72B (Vast.ai brain scans)
- **Training**: 3-stage LoRA (CPT 2000 iters + SFT 1000 iters + BPE-aware SFT 1000 iters)
- **Best result**: 39.4% NKo token accuracy (base: 27%), perplexity 4.57 (base: 8.57)
- **BPE vocab**: 614 tokens (64 NKo chars + 512 BPE merges + 32 morphemes + 5 special + space)
- **Training data**: 21,240 combined_train + 2,361 combined_valid JSONL
- **Key insight**: Problem is data equity, not architecture. N'Ko has 1:100M token ratio vs English.

## Related Projects
- **NKo Unified Platform**: `~/Desktop/NKo/` - morphology (62K LOC), phonetics (34K), transliteration (25K), cultural tools
- **Cross-Script Bridge**: `~/Desktop/cross-script-bridge/` - IPA intermediary (N'Ko, Latin, Arabic)
- **LearnNKo**: `~/projects/LearnNKo/` - 933 video pipeline, NLLB fine-tuning, Next.js web app, iOS keyboard
- **NKo Wikipedia corpus**: 1,693 articles, 3.7M N'Ko chars in `data/nko_wikipedia_corpus.jsonl`

## Key File Paths
- Tokenizer: `tokenizer/tokenizer.py`, `tokenizer/bpe_vocab.json`
- Scanner: `scanner/activation_profiler.py`, `scanner/layer_duplicator.py`, `scanner/run_experiment.py`
- Training: `training/build_sft_data.py`, `training/build_cpt_data.py`
- Results: `results/profiler_bpe.json` (final 3-stage), `results/profiler_twostage.json`
- Blog: `blog/post.md` (500+ lines), `docs/index.md` (GitHub Pages)
