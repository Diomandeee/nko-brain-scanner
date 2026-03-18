# N'Ko Brain Scanner

**The Script That Machines Can't Read: Adapting Large Language Models for N'Ko**

A systematic study of how large language models process N'Ko (U+07C0-U+07FF), an alphabetic script used by 40+ million Manding-language speakers in West Africa. We perform activation profiling ("brain scanning"), train a multi-stage adaptation pipeline, build a script-specific BPE tokenizer, implement phonotactically-constrained decoding, and design a retrieval-centric multimodal ASR architecture.

Paper: [ACL/EMNLP 2026 submission](paper/main.pdf)

## Key Results

### ASR: First Audio-to-N'Ko System (Training, Epoch 90/200)

The world's first ASR system that outputs N'Ko script directly from audio. No prior system does this. All existing Bambara ASR (MALIBA-AI, Meta MMS, Google USM) outputs Latin transliteration only.

**Architecture**: Whisper large-v3 (frozen encoder) + 4x downsample + char-level BiLSTM CTC (5.4M params) + 65 N'Ko character classes + FSM syllable validation post-decoding.

**Training data**: bam-asr-early (37h, 37,306 human-labeled samples) with cross-script bridge (Latin→N'Ko transliteration).

**Current results** (epoch 76 of 200, loss still dropping):

| Metric | Value | Note |
|--------|-------|------|
| Val Loss | 0.430 | Dropping ~0.015/epoch |
| N'Ko CER | 54.1% | Down from 61.5% at epoch 16 |
| Round-trip WER | 93.4% | Includes N'Ko→Latin conversion penalty |
| Compute cost | ~$3 | RTX 4090 at $0.26/hr |

Sample prediction at epoch 76 (6/9 words correct):
```
Gold: ߞߋ ߡߎߛߋ ߕߐ ߛߏߙߏ ߛߋߢߊߟߌ ߟߊ ߞʔߎ ߓʔߊ ߘߍߟߌ
Pred: ߞߋ ߡߎߛߋ ߕߐ ߛߏߙߏ ߛߋߢߊߟߌ ߟߊ ߞߎ ߓߊ ߘߌߜߌ
```

Blog: [From Dead Circuits to Living Speech](blog/asr-breakthrough.md)

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

## Eight Contributions

1. **First Audio-to-N'Ko ASR System**: Char-level CTC on frozen Whisper features produces N'Ko script directly from audio. No prior system does this. 5.4M parameters, trained for ~$3 on a single RTX 4090.

2. **Cross-Script Training Bridge**: Deterministic Latin Bambara→N'Ko transliteration enables training on existing Latin-labeled corpora without any N'Ko-labeled audio. Handles IPA normalization, NFD decomposition, and digraph resolution.

3. **Activation Profiling (Brain Scan)**: Per-layer hidden state analysis of Qwen2-72B reveals 3-4x "translation tax" for N'Ko: weaker activations, higher entropy, dead reasoning circuits. Circuit duplication (RYS) produces zero improvement for N'Ko across all 55 configurations.

4. **Three-Stage Training Pipeline**: CPT (17,360 examples) + SFT (21,240) + BPE-aware (25,100) reduces translation tax from 2.90x to 0.70x on consumer hardware (Apple M4, 16GB).

5. **N'Ko BPE Tokenizer**: 512-merge tokenizer discovers linguistically valid subword units aligning with Manding grammatical particles. 2.75x compression, reducing token gap from 4x to 1.45x.

6. **Admissibility-Constrained Decoding**: 4-state FSM encoding N'Ko CV/CVN syllable structure achieves 100% phonotactic validity. Used both as LLM logits processor and ASR post-processor.

7. **Phonetic Transparency Hypothesis**: N'Ko's 1:1 phoneme-to-character mapping makes it a structurally easier CTC target than Latin orthography. The same design advantage that LLMs can't exploit (data starvation) directly benefits CTC decoders.

8. **Open-Source Release**: Complete pipeline, models, tokenizer, ASR architecture, brain scan tools, cross-script bridge, and evaluation framework.

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
|   +-- build_sft_data_v3.py  # V3 data builder (92K examples)
|   +-- build_sft_data.py     # SFT data builder
|   +-- build_cpt_data.py     # CPT sliding window data
|   +-- build_sft_data_v2.py  # V2 extended vocab data
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
+-- asr/                 # ASR pipeline (char-level CTC)
|   +-- char_level_train.py      # Char-level CTC training (primary, 5.4M params)
|   +-- extract_human_features.py # Whisper feature extraction from bam-asr-early
|   +-- eval_bam_test.py         # CER/WER evaluation on test set (1,463 samples)
|   +-- submit_leaderboard.py    # MALIBA-AI benchmark submission
|   +-- postprocess.py           # CTC decode + syllable segmentation + FSM validation
|   +-- retrieval_asr_v2.py      # V2 Transformer architecture (36M params)
|   +-- watchdog.sh              # Auto-recovery daemon for GPU instances
|   +-- stream_with_features.py  # Streaming pipeline (YouTube→features→transcribe)
|   +-- train_on_human_data.py   # bam-asr-early fine-tuning
|   +-- train_retrieval_asr.py   # Retrieval-based ASR training
|   +-- audio_pipeline.py        # YouTube download + VAD segmentation
|   +-- audio_encoder.py         # Whisper encoder (frozen features)
|   +-- bridge_to_nko.py         # Latin Bambara → N'Ko cross-script bridge
|   +-- dynamic_ocr_pipeline.py  # Scene-adaptive OCR for teaching videos
|   +-- download_djoko.py        # Djoko + babamamadidiane audio downloader
|   +-- eval_vs_maliba.py        # WER evaluation vs MALIBA-AI baseline
|   +-- speaker_diarizer.py      # pyannote speaker clustering
|   +-- scene_encoder.py         # SigLIP visual feature extraction
|   +-- joint_embedding.py       # Shared embedding space (d=512)
|   +-- syllable_retriever.py    # Codebook retrieval + FSM beam search
|   +-- round_trip_eval.py       # Round-trip accuracy evaluation
|   +-- vastai_pipeline.py       # Vast.ai distributed training orchestration
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
+-- blog/                # Research blog posts
|   +-- post.md              # Part 1: The Script That Machines Can't Read (brain scan)
|   +-- index.md             # Summary: Adapting LLMs for N'Ko
|   +-- asr-breakthrough.md  # Part 2: From Dead Circuits to Living Speech (ASR)
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
python3 training/build_sft_data_v3.py

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
# Train on human-labeled bam-asr-early (Vast.ai, RTX 4090)
python3 asr/char_level_train.py --epochs 100 --batch-size 8
python3 asr/train_on_human_data.py --epochs 100 --batch-size 4

# Download YouTube data
python3 asr/download_djoko.py --limit 100          # Djoko episodes
python3 asr/download_djoko.py --channel babamamadidiane --limit 50

# OCR: extract N'Ko text from teaching video frames (Gemini 3 Flash)
python3 asr/dynamic_ocr_pipeline.py --limit 5

# Cross-script bridge: Latin Bambara → N'Ko
python3 asr/bridge_to_nko.py --input results/transcriptions.jsonl --output results/nko_pairs.jsonl

# Evaluate vs MALIBA-AI baseline (45.73% WER)
python3 asr/eval_vs_maliba.py --model-path results/best_retrieval_asr.pt

# Smoke tests
python3 asr/joint_embedding.py
python3 asr/round_trip_eval.py
```

### Evaluation
```bash
# V1-V3 comparison
python3 eval/run_corrected_profiler.py
python3 eval/run_v3_profiler.py

# Generation quality (mode collapse check)
python3 eval/run_v3_generation.py
```

## ASR Pipeline

### Architecture

The ASR model uses a char-level CTC approach rather than the original syllable codebook.

The original design targeted 3,024 syllable classes. In practice, 99.2% of them never appear in training data. That causes the model to memorize rare patterns rather than generalize. The fix: predict individual N'Ko characters (65 targets) and let the FSM assemble valid syllables post-decoding.

**Stack:**
1. Whisper large-v3 encoder (frozen) — 1,280-dim audio features per 10ms frame
2. 4x temporal downsample — compresses 1500 frames to 375, keeps CTC alignment tractable
3. BiLSTM CTC head — 2 layers, 512 hidden units, 65 output classes (N'Ko Unicode range U+07C0-U+07FF + space)
4. FSM syllable assembler — 4-state machine validates CV/CVN structure on decoded characters

Total trainable params: 5.4M. Whisper stays frozen.

### Training Data

**Primary (human-labeled):**
- `bam-asr-early` (RobotsMali / FarmRadio): 37h, 37,306 samples. Bambara radio speech with manual Latin transcriptions. Cross-script bridge converts Latin to N'Ko targets.

**Pseudo-labeled (YouTube):**
- Djoko (Koman Diabate): 1,461 episodes. Whisper transcribes Latin Bambara, bridge converts to N'Ko.
- babamamadidiane: 532 teaching videos. Teacher reads N'Ko text on screen. OCR pipeline extracts the text; audio windows are aligned per slide transition.

### OCR Pipeline

Teaching videos have slides with N'Ko text. Rather than sampling fixed frames, the pipeline uses FFmpeg scene detection to find slide transitions, then runs Gemini 3 Flash on one frame per scene. Each (audio window, N'Ko text) pair is a ground-truth alignment: the teacher is literally explaining the text on screen.

### Cross-Script Bridge

`bam-asr-early` has Latin Bambara transcriptions, not N'Ko. The bridge maps Latin graphemes to N'Ko characters, preserves tone marks (acute/grave convert to N'Ko combining marks), and validates output against the syllable FSM. Pairs that produce no N'Ko characters are dropped.

### Evaluation

Target: beat MALIBA-AI 45.73% WER on 500 blind samples from the FarmRadio corpus. Metrics: CER on N'Ko output, WER on round-trip Latin transliteration, SVR (syllable validity rate).

## Datasets

| Dataset | Hours | License | Role |
|---------|-------|---------|------|
| bam-asr-early (RobotsMali) | 37h | CC-BY-4.0 | Primary training |
| afvoices | 159h | CC-BY-4.0 | Available for scaling |
| kunkado | 39h | CC-BY-SA-4.0 | Radio Bambara |
| Djoko (YouTube) | ~480h | Fair use / research | Pseudo-labeled training |
| babamamadidiane (YouTube) | ~50h | Fair use / research | OCR-aligned teaching pairs |
| MALIBA-AI benchmark | 500 samples | Evaluation only | Blind WER eval |

All YouTube data is processed locally and not redistributed.

## Training Data Summary (Text Model)

| Source | Examples | Description |
|--------|----------|-------------|
| N'Ko Wikipedia CPT | 17,360 | Text completion (300-char sliding window) |
| SFT Instructions | 4,312 | Cultural knowledge, script teaching, grammar |
| BPE-Aware | 3,860 | Merge boundary + word boundary completion |
| Vocabulary Extension | 33,912 | Extended vocab training data |
| nicolingua (WMT 2023) | 32,792 | N'Ko-French-English parallel segments |
| V3 Combined | 92,184 | All sources unified for V3 adapter |

## Hardware & Cost

Text model training on Apple M4 with 16GB unified memory (Mac5). Total training time: ~6 hours across all stages. Cloud cost: $1.72 (initial 72B brain scan on Vast.ai).

ASR training on Vast.ai: RTX 4090 + RTX 3090. Char-level CTC training runs at ~4 samples/sec on 4090. Full bam-asr-early training (~100 epochs) takes approximately 8-12 hours per run.

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
