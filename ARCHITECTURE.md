# NKo Brain Scanner — Unified Architecture

## Anti-Laziness Rules (enforced)

1. **No thin wrappers.** `nko_core/__init__.py` handles all imports from `~/Desktop/NKo/` via `sys.path`. No separate `phonetics.py`, `transliterate.py`, `morphology.py` wrapper files. If `from nko_core import phonetics` works, no wrapper is needed.
2. **No premature release.** HuggingFace upload happens AFTER mode collapse is fixed and the model generates coherent N'Ko text. Not before.
3. **Architecture matches disk.** Every file listed below exists. Every number is current. If reality changes, this doc gets updated.

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
│  │  │ 26 cons  │  │ N'Ko↔Arabic  │  │ types      │  │              │
│  │  │ 5 tones  │  │ via IPA      │  │ conjugator │  │              │
│  │  │ syllabify│  │ intermediary │  │ compounds  │  │              │
│  │  └──────────┘  └──────────────┘  └────────────┘  │              │
│  └──────────────────────┬───────────────────────────┘              │
│                         │                                          │
│  ┌──────────────────────▼───────────────────────────┐              │
│  │              TOKENIZER LAYER                      │              │
│  │                                                   │              │
│  │  Standard BPE (512 merges, 614 vocab)            │              │
│  │  Morpheme-Aware BPE (158 merges, 206 vocab)      │              │
│  │  HF Extension (151,936 → 152,192 vocab)          │              │
│  │  N'Ko Syllable Codebook (3,024 entries)           │              │
│  └──────────────────────┬───────────────────────────┘              │
│                         │                                          │
│  ┌──────────────────────▼───────────────────────────┐              │
│  │              MODEL LAYER                          │              │
│  │                                                   │              │
│  │  Qwen3-8B-8bit (base)                            │              │
│  │  + V1 Adapter (CPT+SFT+BPE, val loss 4.29)      │              │
│  │  + V2 Adapter (33,912 ex, val loss 3.506)        │              │
│  │  + V3 Adapter (92,184 ex, PENDING TRAINING)      │              │
│  │  Fused V2: on Mac5 (MODE COLLAPSE — DO NOT USE)  │              │
│  └──────────────────────────────────────────────────┘              │
│                                                                     │
│  ┌──────────────────────────────────────────────────┐              │
│  │              ASR LAYER (retrieval-centric)        │              │
│  │                                                   │              │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────────────┐ │              │
│  │  │ Audio   │  │ Visual   │  │ N'Ko Text       │ │              │
│  │  │Encoder  │  │ Encoder  │  │ Encoder         │ │              │
│  │  │(Whisper)│  │ (SigLIP) │  │(BPE + FSM)      │ │              │
│  │  └────┬────┘  └────┬─────┘  └────┬────────────┘ │              │
│  │       └─────────┬──┘─────────────┘               │              │
│  │          Joint Embedding Space (d=512)            │              │
│  │                  │                                │              │
│  │          N'Ko Syllable Codebook Retrieval         │              │
│  │          + FSM-Constrained Assembly               │              │
│  └──────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

## Directory Structure (matches disk)

```
nko-brain-scanner/
├── ARCHITECTURE.md              # This file
├── README.md                    # Project overview + reproduction steps
├── model_card.md                # HuggingFace model card (DO NOT UPLOAD UNTIL V3 WORKS)
├── requirements.txt             # Python dependencies
├── run_twostage_profiler.py     # PPL + accuracy profiling (top-level entry point)
├── estimate_cost.py             # Cloud compute cost estimation
├── download_results.sh          # Pull results from Mac5
├── setup_vastai.sh              # Vast.ai GPU setup (for brain scan)
├── PULSE-PLAN.md                # Execution plan history
│
├── nko_core/                    # Unified imports from ~/Desktop/NKo/
│   ├── __init__.py              # sys.path manipulation + re-exports (phonetics, transliterate, morphology)
│   │                            # NO thin wrapper files — __init__.py handles everything
│   └── syllable_codebook.py     # Enumerate all valid N'Ko syllables (3,024 entries)
│
├── scanner/                     # Activation profiling (brain scan)
│   ├── __init__.py
│   ├── activation_profiler.py   # GPU-based profiling (nnsight, for 72B on A100)
│   ├── mlx_activation_profiler.py  # MLX-based profiling (for 8B on M4)
│   ├── mlx_brain_scan_8b.py     # 8B brain scan with LayerCapture wrapper
│   ├── heatmap_generator.py     # Activation heatmap visualization
│   ├── compare_profiles.py      # Before/after LoRA comparison
│   ├── eval_translation_tax.py  # NKo/English PPL ratio measurement
│   ├── plot_brain_scan.py       # Publication-quality brain scan figures
│   ├── layer_duplicator.py      # Layer analysis utilities
│   ├── run_experiment.py        # Experiment orchestrator
│   └── visualizer.py            # General visualization utilities
│
├── constrained/                 # Admissibility-constrained decoding
│   ├── __init__.py
│   ├── nko_fsm.py               # 4-state syllable FSM (START→ONSET→NUCLEUS→CODA)
│   ├── logits_processor.py      # MLX logits processor + repetition penalty
│   ├── gk_scorer.py             # Graph Kernel semantic scoring
│   └── eval_admissibility.py    # Constrained vs unconstrained comparison
│
├── tokenizer/                   # N'Ko tokenization
│   ├── __init__.py
│   ├── tokenizer.py             # Base BPE implementation
│   ├── train_bpe.py             # Standard BPE training (512 merges)
│   ├── morpheme_tokenizer.py    # Morpheme-aware BPE implementation
│   ├── train_morpheme_bpe.py    # Morpheme-constrained BPE training
│   ├── build_vocab.py           # N'Ko vocabulary construction from corpus
│   ├── build_vocab_extension.py # HF vocab extension token list builder
│   ├── extend_hf_tokenizer.py   # Quantized embedding surgery (dequant→extend→requant)
│   ├── eval_tokenizers.py       # Head-to-head tokenizer comparison
│   ├── bpe_vocab.json           # Standard BPE vocabulary (614 tokens)
│   ├── morpheme_bpe_vocab.json  # Morpheme BPE vocabulary (206 tokens)
│   └── vocab.json               # Raw vocabulary data
│
├── training/                    # Data pipelines + training data
│   ├── build_cpt_data.py        # Continued pre-training data builder
│   ├── build_sft_data.py        # SFT dataset V1
│   ├── build_sft_data_v2.py     # SFT dataset V2 (33,912 examples)
│   ├── build_sft_data_v3.py     # SFT dataset V3 (92,184 examples, nicolingua-expanded)
│   ├── adapter_config.json      # LoRA adapter configuration
│   ├── new_nko_tokens.json      # Extended vocabulary token definitions
│   ├── vocab_extension.json     # Vocab extension metadata
│   ├── cpt_train.jsonl          # CPT training data (17,360 examples)
│   ├── cpt_valid.jsonl          # CPT validation data
│   ├── train.jsonl              # V1 SFT training data
│   ├── valid.jsonl              # V1 SFT validation data
│   ├── train_v2.jsonl           # V2 SFT training data
│   ├── valid_v2.jsonl           # V2 SFT validation data
│   ├── combined_train.jsonl     # V1 combined (CPT + SFT)
│   ├── combined_valid.jsonl     # V1 combined validation
│   ├── combined_train_v2.jsonl  # V2 combined (CPT + SFT)
│   └── combined_valid_v2.jsonl  # V2 combined validation
│
├── asr/                         # Retrieval-centric N'Ko ASR
│   ├── __init__.py
│   ├── audio_pipeline.py        # YouTube → audio extraction + VAD (EXISTS)
│   ├── syllable_retriever.py    # Codebook retrieval + FSM assembly (EXISTS)
│   ├── round_trip_eval.py       # N'Ko → Latin round-trip accuracy (EXISTS)
│   ├── generate_synthetic_eval.py # Synthetic eval pair generation (EXISTS)
│   ├── speaker_diarizer.py      # pyannote speaker clustering + VADOnly fallback
│   ├── scene_encoder.py         # SigLIP keyframe feature extraction (d=512)
│   ├── audio_encoder.py         # Whisper encoder (frozen features)
│   ├── joint_embedding.py       # Shared embedding space (d=512) + contrastive/retrieval loss
│   └── train_asr.py             # Multi-loss training loop (contrastive + retrieval)
│
├── eval/                        # Evaluation framework
│   ├── build_eval_set.py        # 100 English + 100 N'Ko frozen eval builder
│   ├── nko_eval.jsonl           # N'Ko eval examples (frozen, never in training)
│   ├── english_eval.jsonl       # English eval examples (frozen)
│   ├── run_corrected_profiler.py # PPL + accuracy profiling (corrected methodology)
│   ├── run_v3_profiler.py       # V3 fused model evaluation
│   └── run_v3_generation.py     # V3 mode collapse check (20 prompts)
│
├── data/                        # Corpora, codebooks, processed data
│   ├── __init__.py
│   ├── build_corpus.py          # Corpus construction utilities
│   ├── scrape_nko_wiki.py       # N'Ko Wikipedia scraper
│   ├── nko_wikipedia_corpus.jsonl  # Wikipedia JSONL (1,693 articles)
│   ├── nko_wikipedia_corpus.txt    # Wikipedia plain text (3.7M chars)
│   ├── nko_merged_corpus.txt    # Merged corpus (Wikipedia + other sources)
│   ├── parallel_corpus.jsonl    # Parallel translation pairs
│   ├── syllable_codebook.json   # 3,024-entry N'Ko syllable codebook
│   ├── nko_lexicon.json         # N'Ko lexicon (GK seed data)
│   ├── nko_collocations.json    # N'Ko word collocations
│   ├── gk_seed_checkpoint.json  # Graph Kernel seeding progress
│   ├── synthetic_eval_pairs.json # Synthetic evaluation pairs
│   ├── sft_v3_combined.jsonl    # V3 combined training data (92,184 examples)
│   ├── nicolingua/              # WMT 2023 N'Ko parallel corpus
│   │   ├── raw/
│   │   │   └── oldi_seed_nko.jsonl  # Raw OLDI seed data
│   │   ├── train.jsonl          # Processed train split (29,513 examples)
│   │   └── test.jsonl           # Processed test split (3,279 examples)
│   └── bayelemabaga/            # Bambara-French parallel data (pilot)
│       └── ...                  # 74,162 pairs (21.2% clean rate — DROPPED)
│
├── results/                     # Evaluation outputs (JSON)
│   ├── activation_profiles/     # Raw activation profile data
│   ├── figures/                 # Result visualization figures
│   ├── heatmaps/                # Brain scan heatmap images
│   ├── profiler_corrected.json  # Corrected 3-stage profiler results
│   ├── profiler_comparison.json # V1 profiler comparison
│   ├── profiler_twostage.json   # Two-stage profiler results
│   ├── profiler_bpe.json        # BPE-stage profiler results
│   ├── benchmark_comparison.json # V2 benchmark comparison
│   ├── brain_scan_8b.json       # 8B brain scan (36 layers)
│   ├── admissibility_comparison.json # Constrained vs unconstrained eval
│   ├── admissibility_eval.json  # Basic admissibility eval
│   ├── admissibility_eval_gen.json # Generative admissibility eval
│   ├── fsm_validation.json      # FSM validation results
│   ├── tokenizer_comparison.json # Tokenizer head-to-head results
│   ├── round_trip_eval.json     # ASR round-trip accuracy
│   ├── translation_tax.json     # Translation tax measurement
│   └── v2_20_prompts.json       # V2 generative eval (SHOWS MODE COLLAPSE)
│
├── figures/                     # Publication figures (PNG)
│   ├── brain_scan_l2_comparison.png
│   ├── brain_scan_delta.png
│   └── brain_scan_sparsity.png
│
├── paper/                       # ACL/EMNLP 2026 paper (LaTeX)
│   ├── main.tex                 # Paper source (8 pages)
│   ├── references.bib           # Bibliography (16 entries)
│   ├── acl.sty                  # ACL style file
│   ├── acl_natbib.bst           # ACL bibliography style
│   ├── main.pdf                 # Compiled paper
│   └── nko-brain-scanner.pdf    # Alternate compiled copy
│
├── probes/                      # Probing experiments
│   ├── __init__.py
│   ├── scoring.py               # Probe scoring utilities
│   ├── math_probes.json         # Mathematical probes
│   └── semantic_probes.json     # Semantic probes
│
├── demo/                        # Gradio demo app
│   └── app.py                   # 5-tab demo (generate, analyze, brain scan, results, about)
│
├── blog/                        # Blog post
│   ├── post.md                  # Blog content
│   └── assets/                  # Blog images
│
├── docs/                        # Documentation site
│   ├── _config.yml              # Jekyll config
│   ├── index.md                 # Landing page
│   └── assets/                  # Site assets
│
└── scripts/                     # Utility scripts
    ├── upload_to_hf.py          # HuggingFace upload (DO NOT RUN UNTIL V3 WORKS)
    ├── seed_gk_nko.py           # Graph Kernel N'Ko data seeding
    └── bayelemabaga_pilot.py    # Bayelemabaga cross-script bridge pilot
```

## Module Dependencies

```
nko_core/
  __init__.py handles ALL imports from ~/Desktop/NKo/:
  ├── phonetics     ← ~/Desktop/NKo/nko/phonetics.py (via sys.path)
  ├── transliterate  ← ~/Desktop/NKo/nko/transliterate.py (via sys.path)
  ├── morphology     ← ~/Desktop/NKo/nko/morphology.py (via sys.path)
  └── syllable_codebook.py (uses phonetics internally)

constrained/
  ├── nko_fsm        ← uses nko_core.phonetics (VOWEL_CHARS, CONSONANT_CHARS)
  ├── logits_processor ← uses nko_fsm
  ├── gk_scorer      ← uses Graph Kernel API (localhost:8001)
  └── eval_admissibility ← uses logits_processor + gk_scorer

tokenizer/
  ├── train_bpe      ← uses nko_core.phonetics (tone attachment)
  ├── morpheme_tokenizer ← uses nko_core.morphology (MorphologicalAnalyzer)
  ├── extend_hf_tokenizer ← uses tokenizer.train_bpe (BPE vocab)
  └── eval_tokenizers ← uses tokenizer + morpheme_tokenizer

asr/
  ├── syllable_retriever ← uses nko_core.syllable_codebook + constrained.nko_fsm
  ├── audio_pipeline     ← uses nko_core.transliterate (round-trip eval)
  ├── round_trip_eval    ← uses syllable_retriever + nko_core
  ├── joint_embedding    ← numpy-based (framework-independent), d=512
  └── train_asr          ← uses joint_embedding + syllable_retriever

training/
  ├── build_sft_data_v3 ← uses nko_core.transliterate, nko_core.morphology, data/nicolingua/
  ├── build_sft_data_v2 ← uses nko_core.transliterate, nko_core.morphology
  └── build_cpt_data    ← uses nko_core.phonetics (text validation)
```

## Task Backlog

### Completed
- [x] T1: Brain scan activation profiling (72B on A100 + 8B on M4)
- [x] T2: Three-stage training V1 (CPT + SFT + BPE, val loss 4.29)
- [x] T3: Constrained decoding FSM (100% syllable validity)
- [x] T4: BPE tokenizer (512 merges, 614 vocab)
- [x] T5: Morpheme-aware BPE tokenizer (158 merges, 206 vocab)
- [x] T6: Embedding extension pipeline (151,936 → 152,192 vocab)
- [x] T7: V2 LoRA training (33,912 examples, val loss 3.506, BUT mode collapse)
- [x] T8: Paper draft (8 pages, ACL format, compiles clean)
- [x] T9: Corrected eval sets (100 English + 100 N'Ko, proper methodology)
- [x] T10: nicolingua corpus download + preprocessing (32,792 examples)
- [x] T11: V3 SFT data builder (92,184 combined examples)
- [x] T12: GK N'Ko word seeding (2,100+ words, 493 loaded by scorer)
- [x] T13: Syllable codebook generation (3,024 entries)
- [x] T14: GK semantic scorer (gk_scorer.py, 250 lines)
- [x] T15: nko_core/ unified imports (__init__.py handles everything, no wrappers)

### Active (critical path: T16 → T17 → T18 → T19)
- [ ] T16: Train V3 adapter on Mac5 (8 LoRA layers, 92K examples, target: fix mode collapse)
- [ ] T17: Evaluate V3 generation diversity (20-prompt generative eval, check for collapse)
- [ ] T18: GK-augmented training (integrate knowledge graph signal into training loss)
- [x] T19: Build ASR layer (5 files built: speaker_diarizer, scene_encoder, audio_encoder, joint_embedding, train_asr)
- [ ] T20: Paper revision (add V3 results, ASR section, fix all numbers)
- [ ] T21: Human evaluation (recruit 3-5 N'Ko literate evaluators)
- [ ] T22: HuggingFace upload (model + dataset + model card) — LAST STEP

### Blocked
- T17 blocked by T16 (need trained V3 to evaluate)
- T18 blocked by T16 (need V3 baseline before adding GK signal)
- T20 blocked by T17, T19 (need results to write about)
- T22 blocked by T17 (model must generate coherent N'Ko before release)

## Key Numbers (verified against actual data)

| Metric | Value | Source |
|--------|-------|--------|
| N'Ko Unicode range | U+07C0-U+07FF | Unicode standard |
| N'Ko vowels | 7 | nko_core.phonetics |
| N'Ko consonants | 26 | nko_core.phonetics |
| Tone marks | 5 | nko_core.phonetics |
| Syllable codebook entries | 3,024 | data/syllable_codebook.json |
| Qwen3 base vocab | 151,936 | model config |
| Extended vocab | 152,192 (+250 N'Ko BPE + 6 padding) | tokenizer/extend_hf_tokenizer.py |
| Standard BPE merges | 512 | tokenizer/bpe_vocab.json |
| Standard BPE vocab | 614 (64 base + 512 merges + 32 morpheme + 6 special) | tokenizer/bpe_vocab.json |
| Morpheme BPE merges | 158 | tokenizer/morpheme_bpe_vocab.json |
| Morpheme BPE vocab | 206 | tokenizer/morpheme_bpe_vocab.json |
| V1 training examples | CPT 17,360 + SFT 21,240 + BPE 25,100 | training/*.jsonl |
| V2 training examples | 33,912 | training/combined_train_v2.jsonl |
| V3 training examples | 92,184 | data/sft_v3_combined.jsonl |
| V2 best val loss | 3.506 | results/benchmark_comparison.json |
| N'Ko PPL (base → V1) | 11.02 → 6.00 (-45.6%) | results/profiler_corrected.json |
| Translation tax (base → V1) | 2.90x → 0.70x (-76%) | results/profiler_corrected.json |
| English accuracy drop | -1.2pp (70.9% → 69.7%) | results/profiler_corrected.json |
| Constrained syllable validity | 100% (vs 89.8% unconstrained) | results/admissibility_comparison.json |
| nicolingua downloaded | 32,792 examples (29,513 train + 3,279 test) | data/nicolingua/ |
| Wikipedia corpus | 3.7M characters (1,693 articles) | data/nko_wikipedia_corpus.txt |
| GK words seeded | 2,100+ | data/gk_seed_checkpoint.json |
| V2 mode collapse | CONFIRMED: every N'Ko prompt produces degenerate repetition | results/v2_20_prompts.json |
| Djoko episodes | 959+ | external source |
| Djoko total audio | ~320 hours | external source |
| MALIBA-AI SOTA WER | 45.73% | literature |

## Mac5 Model Paths

| Path | Description | Status |
|------|-------------|--------|
| `~/nko-brain-scanner/extended-nko-qwen3/` | Extended vocab base model | EXISTS |
| `~/nko-brain-scanner/adapters-extended/` | V2 adapter (8 LoRA layers) | EXISTS, MODE COLLAPSE |
| `~/nko-brain-scanner/fused-extended-nko-qwen3/` | Fused V2 model (5.4GB) | EXISTS, MODE COLLAPSE |
| `~/nko-brain-scanner/fused-nko-qwen3/` | Fused V1 model (8.1GB) | EXISTS, working |
| `~/nko-brain-scanner/adapters-v3/` | V3 adapter (8 LoRA layers, 92K examples) | TRAINING IN PROGRESS |
| `~/nko-brain-scanner/fused-v3-nko-qwen3/` | Fused V3 model | PENDING (after training) |
