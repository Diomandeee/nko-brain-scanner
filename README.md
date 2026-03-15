# N'Ko Brain Scanner

**The Script That Machines Can't Read: Adapting Large Language Models for N'Ko**

A systematic study of how large language models process N'Ko (U+07C0-U+07FF), an alphabetic script used by 40+ million Manding-language speakers in West Africa. We perform activation profiling ("brain scanning"), train a multi-stage adaptation pipeline, build a script-specific BPE tokenizer, implement phonotactically-constrained decoding, and design a retrieval-centric multimodal ASR architecture.

Paper: [ACL/EMNLP 2026 submission](paper/main.pdf)

## Key Results

### V1: Three-Stage Training (Base Vocabulary)

| Metric | Base Qwen3-8B | Three-Stage Fine-Tuned | Change |
|--------|---------------|----------------------|--------|
| N'Ko Perplexity | 11.02 | **6.00** | -45.6% |
| N'Ko Token Accuracy | 23.0% | **32.8%** | +43% relative |
| Translation Tax (N'Ko/Eng PPL) | 2.90x | **0.70x** | **-76%** |
| English Top-1 Accuracy | 70.9% | 69.7% | -1.2pp |
| Syllable Validity (Constrained) | 89.8% | **100%** | +10.2pp |

After fine-tuning, the model processes N'Ko with *lower* perplexity than English, while English accuracy drops by only 1.2 percentage points.

### V3: Extended Vocabulary + nicolingua (92K examples)

| Metric | V2 (Extended Vocab) | V3 (nicolingua) |
|--------|-------------------|-----------------|
| Training Val Loss | 3.506 | **3.275** (-6.6%) |
| Mode Collapse | 20/20 degenerate | **3/20** degenerate |
| Unconstrained Validity | --- | **99.8%** |
| Distinct-1 (diversity) | ~0 (collapsed) | **0.455** |

V3 fixes V2's mode collapse through 2.7x more training data (92,184 examples including 32,792 nicolingua parallel segments). Note: PPL comparisons between V1 (base vocab, 151,936 tokens) and V3 (extended vocab, 152,192 tokens) are not meaningful due to different tokenization.

## Seven Contributions

1. **Activation Profiling (Brain Scan)**: Per-layer hidden state analysis reveals LoRA adaptation concentrates in top 8 layers, with reduced activation magnitudes in reasoning layers and +573 increase at output layer.

2. **Three-Stage Training Pipeline**: CPT (17,360 examples) + SFT (21,240) + BPE-aware (25,100) reduces translation tax from 2.90x to 0.70x on consumer hardware (Apple M4, 16GB).

3. **N'Ko BPE Tokenizer**: 512-merge tokenizer discovers linguistically valid subword units aligning with Manding grammatical particles. 2.75x compression, reducing token gap from 4x to 1.45x. Morpheme-constrained variant improves boundary preservation by 5.6pp.

4. **Vocabulary Extension**: Quantized embedding surgery (151,936 to 152,192 tokens) + V2/V3 LoRA adapters. V3 trained on 92,184 examples including 32,792 nicolingua parallel segments.

5. **Admissibility-Constrained Decoding**: 4-state FSM encoding N'Ko CV/CVN syllable structure improves validity from 89.8% to 100% as a logits processor.

6. **Retrieval-Centric ASR Architecture**: Multimodal pipeline combining Whisper audio features, SigLIP visual features, and N'Ko text in a shared d=512 embedding space. Codebook retrieval + FSM-constrained beam search produces phonotactically valid N'Ko from speech.

7. **Open-Source Release**: Complete pipeline, model, tokenizer, ASR architecture, and evaluation framework.

## Project Structure

```
nko-brain-scanner/
+-- paper/               # ACL/EMNLP 2026 paper (LaTeX)
+-- scanner/             # Brain scan activation profiling
|   +-- mlx_activation_profiler.py
|   +-- mlx_brain_scan_8b.py
|   +-- compare_profiles.py
|   +-- eval_translation_tax.py
+-- training/            # Training pipeline (MLX LoRA)
|   +-- build_v3_data.py     # V3 data builder (92K examples)
|   +-- sft_builder.py
|   +-- build_cpt_data.py
|   +-- build_bpe_data.py
+-- tokenizer/           # N'Ko BPE tokenizer
|   +-- train_bpe.py         # Standard 512-merge BPE
|   +-- train_morpheme_bpe.py # Morpheme-constrained BPE
|   +-- eval_tokenizers.py   # Comparison evaluation
|   +-- morpheme_tokenizer.py
|   +-- build_vocab_extension.py
|   +-- extend_hf_tokenizer.py
+-- constrained/         # Admissibility-constrained decoding
|   +-- nko_fsm.py           # 4-state syllable FSM
|   +-- logits_processor.py  # MLX logits processor
|   +-- eval_admissibility.py
|   +-- gk_scorer.py         # Graph Kernel semantic scorer
+-- asr/                 # Retrieval-centric multimodal ASR
|   +-- audio_pipeline.py    # YouTube download + VAD segmentation
|   +-- audio_encoder.py     # Whisper encoder (frozen features)
|   +-- speaker_diarizer.py  # pyannote speaker clustering
|   +-- scene_encoder.py     # SigLIP visual feature extraction
|   +-- joint_embedding.py   # Shared embedding space (d=512)
|   +-- syllable_retriever.py # Codebook retrieval + FSM beam search
|   +-- train_asr.py         # Multi-loss training loop
|   +-- round_trip_eval.py   # Round-trip accuracy evaluation
+-- nko_core/            # N'Ko language core
|   +-- __init__.py          # Unicode, phonology, morphology
+-- data/                # Corpora and codebooks
|   +-- syllable_codebook.json  # 3,024-entry codebook
|   +-- nko_bpe_vocab.json
|   +-- morpheme_bpe_vocab.json
|   +-- nicolingua/          # WMT 2023 parallel corpus (32,792 segments)
+-- eval/                # Evaluation scripts
|   +-- run_corrected_profiler.py  # Base/2-stage/3-stage comparison
|   +-- run_v3_profiler.py         # V3 fused model evaluation
|   +-- run_v3_generation.py       # Mode collapse check (20 prompts)
+-- results/             # Experimental results (JSON)
+-- figures/             # Brain scan visualizations
+-- blog/                # Blog post
+-- model_card.md        # HuggingFace model card
```

## Reproduce

### Brain Scan (activation profiling)
```bash
python3 scanner/mlx_brain_scan_8b.py
```

### Three-Stage Training (consumer hardware)
```bash
# Stage 1: Continued Pre-Training (17,360 examples)
python3 -m mlx_lm lora --model mlx-community/Qwen3-8B-8bit \
    --data training/cpt/ --iters 2000 --learning-rate 1e-5

# Stage 2: Supervised Fine-Tuning (21,240 examples)
python3 -m mlx_lm lora --model mlx-community/Qwen3-8B-8bit \
    --adapter-path adapters-stage1/ --data training/sft/ --iters 1000

# Stage 3: BPE-Aware Training (25,100 examples)
python3 -m mlx_lm lora --model mlx-community/Qwen3-8B-8bit \
    --adapter-path adapters-stage2/ --data training/bpe/ --iters 1000
```

### V3 Training (92K examples with nicolingua)
```bash
# Build V3 data (combines all sources + nicolingua parallel corpus)
python3 training/build_v3_data.py

# Train V3 adapter on extended-vocabulary model
python3 -m mlx_lm lora -c v3_training_config.yaml
```

### BPE Tokenizer
```bash
python3 tokenizer/train_bpe.py --merges 512
python3 tokenizer/train_morpheme_bpe.py --merges 512 --compare-bpe
python3 tokenizer/eval_tokenizers.py
```

### Constrained Decoding
```bash
python3 constrained/eval_admissibility.py --model path/to/fused-model --num-samples 50
```

### ASR Pipeline
```bash
# Smoke tests
python3 asr/joint_embedding.py
python3 asr/speaker_diarizer.py
python3 asr/round_trip_eval.py

# Training (synthetic data for pipeline validation)
python3 asr/train_asr.py --synthetic --epochs 50

# Audio processing
python3 asr/audio_pipeline.py path/to/video.mp4 --output segments/
```

### Evaluation
```bash
# V1-V3 comparison
python3 eval/run_corrected_profiler.py
python3 eval/run_v3_profiler.py

# Generation quality (mode collapse check)
python3 eval/run_v3_generation.py
```

## Training Data Summary

| Source | Examples | Description |
|--------|----------|-------------|
| N'Ko Wikipedia CPT | 17,360 | Text completion (300-char sliding window) |
| SFT Instructions | 4,312 | Cultural knowledge, script teaching, grammar |
| BPE-Aware | 3,860 | Merge boundary + word boundary completion |
| Vocabulary Extension | 33,912 | Extended vocab training data |
| nicolingua (WMT 2023) | 32,792 | N'Ko-French-English parallel segments |
| V3 Combined | 92,184 | All sources unified for V3 adapter |

## Hardware & Cost

All training on Apple M4 with 16GB unified memory (Mac5). Total training time: ~6 hours across all stages. Cloud cost: $1.72 (initial 72B brain scan on Vast.ai).

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
