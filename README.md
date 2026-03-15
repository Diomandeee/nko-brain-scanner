# N'Ko Brain Scanner

**The Script That Machines Can't Read: Adapting Large Language Models for N'Ko**

A systematic study of how large language models process N'Ko (U+07C0-U+07FF), an alphabetic script used by 40+ million Manding-language speakers in West Africa. We perform activation profiling ("brain scanning"), train a three-stage adaptation pipeline, build a script-specific BPE tokenizer, and implement phonotactically-constrained decoding.

Paper: [ACL/EMNLP 2026 submission](paper/main.pdf)

## Key Results

| Metric | Base Qwen3-8B | Three-Stage Fine-Tuned | Change |
|--------|---------------|----------------------|--------|
| N'Ko Perplexity | 11.02 | **6.00** | -45.6% |
| N'Ko Token Accuracy | 23.0% | **32.8%** | +43% relative |
| Translation Tax (N'Ko/Eng PPL) | 2.90x | **0.70x** | **-76%** |
| English Top-1 Accuracy | 70.9% | 69.7% | -1.2pp |
| Syllable Validity (Constrained) | 89.8% | **100%** | +10.2pp |

After fine-tuning, the model processes N'Ko with *lower* perplexity than English, while English accuracy drops by only 1.2 percentage points.

## Six Contributions

1. **Activation Profiling (Brain Scan)**: Per-layer hidden state analysis reveals LoRA adaptation concentrates in top 8 layers, with reduced activation magnitudes in reasoning layers and +573 increase at output layer.

2. **Three-Stage Training Pipeline**: CPT (17,360 examples) + SFT (21,240) + BPE-aware (25,100) reduces translation tax from 2.90x to 0.70x on consumer hardware (Apple M4, 16GB).

3. **N'Ko BPE Tokenizer**: 512-merge tokenizer discovers linguistically valid subword units aligning with Manding grammatical particles. 2.75x compression, reducing token gap from 4x to 1.45x.

4. **Vocabulary Extension**: Quantized embedding surgery (151,936 to 152,192 tokens) + V2 LoRA adapter achieves 18.3% lower validation loss.

5. **Admissibility-Constrained Decoding**: 4-state FSM encoding N'Ko CV/CVN syllable structure improves validity from 89.8% to 100% as a logits processor.

6. **Open-Source Release**: Complete pipeline, model, tokenizer, and evaluation framework.

## Project Structure

```
nko-brain-scanner/
+-- paper/               # ACL/EMNLP 2026 paper (LaTeX)
+-- scanner/             # Brain scan activation profiling
|   +-- activation_profiler.py
|   +-- layer_duplicator.py
|   +-- heatmap_generator.py
|   +-- visualizer.py
+-- training/            # Three-stage training pipeline (MLX)
+-- tokenizer/           # N'Ko BPE tokenizer
|   +-- train_bpe.py
|   +-- train_morpheme_bpe.py
|   +-- eval_tokenizers.py
+-- constrained/         # Admissibility-constrained decoding
|   +-- nko_fsm.py       # 4-state syllable FSM
|   +-- logits_processor.py
|   +-- eval_admissibility.py
|   +-- gk_scorer.py     # Graph Kernel semantic scorer
+-- asr/                 # Syllable retriever (proof-of-concept ASR)
|   +-- syllable_retriever.py
|   +-- round_trip_eval.py
|   +-- audio_pipeline.py
+-- data/                # Corpora and codebooks
|   +-- syllable_codebook.json  # 3,024-entry N'Ko syllable codebook
|   +-- nko_bpe_vocab.json
|   +-- morpheme_bpe_vocab.json
+-- eval/                # Evaluation sets
+-- results/             # Experimental results (JSON)
+-- figures/             # Brain scan visualizations
+-- blog/                # Blog post
+-- model_card.md        # HuggingFace model card
```

## Reproduce

### Brain Scan (activation profiling)
```bash
# Requires GPU (A100 80GB for 72B, or M4 for 8B)
python3 -m scanner.run_experiment --experiment activation --model Qwen3-8B-8bit
```

### Three-Stage Training (consumer hardware)
```bash
# Stage 1: Continued Pre-Training
python3 -m mlx_lm lora --model mlx-community/Qwen3-8B-8bit \
    --data training/cpt/ --iters 2000 --learning-rate 1e-5

# Stage 2: Supervised Fine-Tuning
python3 -m mlx_lm lora --model mlx-community/Qwen3-8B-8bit \
    --adapter-path adapters-stage1/ --data training/sft/ --iters 1000

# Stage 3: BPE-Aware Training
python3 -m mlx_lm lora --model mlx-community/Qwen3-8B-8bit \
    --adapter-path adapters-stage2/ --data training/bpe/ --iters 1000
```

### BPE Tokenizer
```bash
python3 tokenizer/train_bpe.py --merges 512
python3 tokenizer/train_morpheme_bpe.py --merges 512 --compare-bpe
python3 tokenizer/eval_tokenizers.py
```

### Constrained Decoding
```bash
python3 constrained/eval_admissibility.py --model path/to/fused-model
```

### Evaluation
```bash
python3 run_twostage_profiler.py  # Perplexity + token accuracy
```

## Hardware & Cost

All training on Apple M4 MacBook with 16GB unified memory. Total training time: ~3 hours. Cloud cost: $1.72 (initial 72B brain scan on Vast.ai).

## Citation

```bibtex
@misc{diomande2026nko,
  title={The Script That Machines Can't Read: Adapting Large Language Models for N'Ko},
  author={Diomande, Mohamed},
  year={2026},
  url={https://github.com/Diomandeee/nko-brain-scanner}
}
```

## License

Apache-2.0
