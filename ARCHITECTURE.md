# NKo Brain Scanner — Unified Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      NKO UNIFIED PLATFORM                          │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   RESEARCH   │  │  INFERENCE   │  │   TRAINING   │              │
│  │              │  │              │  │              │              │
│  │ Brain Scan   │  │ Constrained  │  │ Data Pipeline│              │
│  │ Activation   │  │ Generation   │  │ LoRA Trainer │              │
│  │ Profiling    │  │ FSM Decode   │  │ Eval Suite   │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                 │                       │
│  ┌──────▼─────────────────▼─────────────────▼───────┐              │
│  │              CORE LINGUISTIC LAYER                │              │
│  │         (imported from ~/Desktop/NKo/)            │              │
│  │                                                   │              │
│  │  phonetics.py  transliterate.py  morphology.py   │              │
│  │  ┌──────────┐  ┌──────────────┐  ┌────────────┐  │              │
│  │  │ 7 vowels │  │ N'Ko↔Latin   │  │ 28 morpheme│  │              │
│  │  │ 27 cons  │  │ N'Ko↔Arabic  │  │ types      │  │              │
│  │  │ 5 tones  │  │ via IPA      │  │ conjugator │  │              │
│  │  │ syllabify│  │ intermediary │  │ compounds  │  │              │
│  │  └──────────┘  └──────────────┘  └────────────┘  │              │
│  └──────────────────────┬───────────────────────────┘              │
│                         │                                          │
│  ┌──────────────────────▼───────────────────────────┐              │
│  │              TOKENIZER LAYER                      │              │
│  │                                                   │              │
│  │  Standard BPE (512 merges, 614 vocab)            │              │
│  │  Morpheme-Aware BPE (107 merges, 160 vocab)      │              │
│  │  HF Extension (151,936 → 152,192 vocab)          │              │
│  │  N'Ko Syllable Codebook (~3,640 entries)          │              │
│  └──────────────────────┬───────────────────────────┘              │
│                         │                                          │
│  ┌──────────────────────▼───────────────────────────┐              │
│  │              MODEL LAYER                          │              │
│  │                                                   │              │
│  │  Qwen3-8B-8bit (base)                            │              │
│  │  + V1 Adapter (4,312 examples, val loss 4.29)    │              │
│  │  + V2 Adapter (33,912 examples, val loss 3.506)  │              │
│  │  Fused: fused-nko-qwen3-v2 (152,192 vocab)      │              │
│  └──────────────────────────────────────────────────┘              │
│                                                                     │
│  ┌──────────────────────────────────────────────────┐              │
│  │              ASR LAYER (V5 — planned)             │              │
│  │                                                   │              │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────────────┐ │              │
│  │  │ Audio   │  │ Visual   │  │ N'Ko Text       │ │              │
│  │  │Encoder  │  │ Encoder  │  │ Encoder         │ │              │
│  │  │(Whisper)│  │(InternVL)│  │(BPE + FSM)      │ │              │
│  │  └────┬────┘  └────┬─────┘  └────┬────────────┘ │              │
│  │       └─────────┬──┘─────────────┘               │              │
│  │          Joint Embedding Space (d=512)            │              │
│  │                  │                                │              │
│  │          N'Ko Syllable Codebook Retrieval         │              │
│  │          + FSM-Constrained Assembly               │              │
│  └──────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
nko-brain-scanner/
├── ARCHITECTURE.md           # This file
├── paper/
│   ├── main.tex              # ACL/EMNLP paper (8 pages)
│   └── references.bib
│
├── nko_core/                 # Unified imports from ~/Desktop/NKo/
│   ├── __init__.py           # Re-exports: phonetics, transliterate, morphology
│   ├── phonetics.py          # → nko.phonetics (symlink or thin wrapper)
│   ├── transliterate.py      # → nko.transliterate
│   ├── morphology.py         # → nko.morphology
│   └── syllable_codebook.py  # NEW: Enumerate all valid N'Ko syllables
│
├── scanner/                  # Activation profiling (brain scan)
│   ├── activation_profiler.py
│   ├── mlx_activation_profiler.py
│   ├── heatmap_generator.py
│   ├── compare_profiles.py
│   ├── eval_translation_tax.py
│   └── plot_brain_scan.py
│
├── constrained/              # Admissibility-constrained decoding
│   ├── nko_fsm.py            # 4-state syllable FSM
│   ├── logits_processor.py   # MLX logits_processor + repetition penalty
│   ├── gk_scorer.py          # Graph Kernel semantic scoring (optional)
│   └── eval_admissibility.py # Constrained vs unconstrained comparison
│
├── tokenizer/                # N'Ko tokenization
│   ├── tokenizer.py          # Base BPE implementation
│   ├── train_bpe.py          # Standard BPE training (512 merges)
│   ├── morpheme_tokenizer.py # Morpheme-aware BPE
│   ├── train_morpheme_bpe.py # Morpheme-constrained training
│   ├── build_vocab.py        # Vocabulary construction
│   ├── build_vocab_extension.py  # HF vocab extension builder
│   ├── extend_hf_tokenizer.py   # Quantized embedding surgery
│   └── eval_tokenizers.py    # Head-to-head comparison
│
├── training/                 # Data pipelines + training scripts
│   ├── build_cpt_data.py     # Continued pre-training data
│   ├── build_sft_data.py     # SFT dataset (V1)
│   ├── build_sft_data_v2.py  # SFT dataset (V2, 33,912 examples)
│   └── corpus/               # Raw corpus data
│       ├── wikipedia/        # N'Ko Wikipedia (3.7M chars)
│       └── nicolingua/       # WMT 2023 (130K segments, TODO)
│
├── asr/                      # V5: Retrieval-centric N'Ko ASR (NEW)
│   ├── __init__.py
│   ├── audio_pipeline.py     # YouTube → audio extraction + VAD
│   ├── speaker_diarizer.py   # pyannote speaker clustering
│   ├── scene_encoder.py      # InternVL keyframe captioning
│   ├── audio_encoder.py      # Whisper encoder (frozen features)
│   ├── joint_embedding.py    # Shared embedding space + projectors
│   ├── syllable_retriever.py # Codebook retrieval + FSM assembly
│   ├── round_trip_eval.py    # N'Ko → Latin vs MALIBA-AI comparison
│   └── train_asr.py          # Multi-loss training loop
│
├── eval/                     # Evaluation framework
│   ├── build_eval_set.py     # 100 English + 100 N'Ko frozen eval
│   ├── nko_eval.jsonl        # N'Ko eval examples
│   ├── english_eval.jsonl    # English eval examples
│   ├── run_corrected_profiler.py  # PPL + accuracy profiling
│   └── youtube_to_training_data.py # YouTube scraping pipeline
│
├── results/                  # Evaluation outputs
│   ├── admissibility_comparison.json
│   └── tokenizer_comparison.json
│
├── figures/                  # Paper figures
│   ├── brain_scan_l2_comparison.png
│   └── brain_scan_delta.png
│
└── data/                     # Processed data
    ├── nko-corpus.jsonl      # Wikipedia corpus
    ├── bpe_vocab.json        # Standard BPE vocabulary
    └── morpheme_bpe_vocab.json  # Morpheme BPE vocabulary
```

## Module Dependencies

```
nko_core/
  ├── phonetics     ← ~/Desktop/NKo/nko/phonetics.py
  ├── transliterate  ← ~/Desktop/NKo/nko/transliterate.py
  ├── morphology     ← ~/Desktop/NKo/nko/morphology.py
  └── syllable_codebook (NEW, uses phonetics)

constrained/
  ├── nko_fsm        ← uses nko_core.phonetics (VOWEL_CHARS, CONSONANT_CHARS)
  └── logits_processor ← uses nko_fsm

tokenizer/
  ├── train_bpe      ← uses nko_core.phonetics (tone attachment)
  ├── morpheme_tokenizer ← uses nko_core.morphology (MorphologicalAnalyzer)
  └── extend_hf_tokenizer ← uses tokenizer.train_bpe (BPE vocab)

asr/ (V5)
  ├── syllable_retriever ← uses nko_core.syllable_codebook + constrained.nko_fsm
  ├── audio_pipeline     ← uses nko_core.transliterate (round-trip eval)
  └── joint_embedding    ← uses nko_core.phonetics (feature extraction)

training/
  ├── build_sft_data_v2 ← uses nko_core.transliterate, nko_core.morphology
  └── build_cpt_data    ← uses nko_core.phonetics (text validation)
```

## Task Backlog

### Completed
- [x] T5: Embedding extension pipeline (151,936 → 152,192)
- [x] T6: YouTube-to-training-data pipeline
- [x] T7: V2 LoRA training (val loss 3.506)
- [x] T8: Constrained decoding (100% syllable validity)
- [x] T9: Paper update (8 pages, ACL format)

### Active
- [ ] T10: Benchmark vs Bambara/N'Ko baselines + nicolingua corpus
- [ ] T11: V3 Bambara ASR → N'Ko script
- [ ] T12: V4 N'Ko LM-fused ASR rescoring
- [ ] T13: V5 Retrieval-centric N'Ko ASR (Djoko series)

### Infrastructure
- [ ] T14: Unify nko_core/ imports (symlinks to ~/Desktop/NKo/)
- [ ] T15: Build syllable codebook (~3,640 entries)
- [ ] T16: Djoko audio extraction pipeline (959 episodes)
- [ ] T17: nicolingua corpus download + preprocessing

## Key Numbers

| Metric | Value |
|--------|-------|
| N'Ko Unicode range | U+07C0–U+07FF |
| N'Ko vowels | 7 |
| N'Ko consonants | 26 |
| Tone marks | 5 |
| Possible CV syllables | 182 |
| Possible CVN syllables | ~546 |
| Total tonal syllables | ~3,640 |
| Qwen3 base vocab | 151,936 |
| Extended vocab | 152,192 (+250 N'Ko BPE + 6 padding) |
| V2 training examples | 33,912 |
| V2 best val loss | 3.506 |
| Constrained syllable validity | 100% (vs 89.8% unconstrained) |
| Djoko episodes | 959+ |
| Djoko total audio | ~320 hours |
| MALIBA-AI SOTA WER | 45.73% |
| nicolingua corpus | 130,850 parallel segments |
| Wikipedia corpus | 3.7M characters (1,693 articles) |
