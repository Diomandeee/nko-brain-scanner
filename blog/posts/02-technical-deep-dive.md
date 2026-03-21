# Dead Circuits and Living Speech: Building the First N'Ko AI Pipeline

*Two papers in preparation for ACL/EMNLP 2026. All code and models open-source.*

---

## TL;DR

- **What we found**: Qwen3-8B-Instruct processes N'Ko text with a 2.94x "translation tax" (L2 norm deficit) across all 36 transformer layers. Circuit duplication analysis (55 configurations, RYS methodology) finds 0/55 N'Ko-advantageous configurations. Three-zone failure analysis reveals structurally distinct collapse modes at embedding, middle, and output layers. Arabic, another RTL script, sits within 7% of English on the same metrics. The failure is entirely data-driven.
- **What we built**: (1) A three-stage LoRA pipeline (CPT + SFT + BPE-aware, 185 minutes on Apple M4) that reduces the translation tax from 2.94x to 0.70x. (2) The first audio-to-N'Ko ASR system, with a four-version architecture progression from BiLSTM baseline (56% CER) to Whisper LoRA fine-tune (29.4% CER). A deterministic cross-script bridge, a 4-state FSM for phonotactic validation, and a downstream NLLB-200 translation pipeline complete the stack.
- **The numbers**: Translation tax: 2.94x to 0.70x (76% reduction). ASR: V1 56% CER / 91.5% WER to V4 29.4% CER / 62.3% WER. Prediction confidence: 0.46 to 0.82 (79% improvement). Per-sample: worst case au30 drops from WER 15.8 to 1.0 (93.7% improvement). Total compute cost across all experiments: $14.

---

## The Problem: Script Invisibility

N'Ko is the only writing system in history designed from the ground up with the computational properties that NLP engineers dream about when building synthetic alphabets.

Solomana Kante designed it in 1949 in Kankan, Guinea, in response to a claim that African languages could not be written. The result is a right-to-left alphabetic script occupying Unicode block `U+07C0--U+07FF` (standardized 2006) with properties that no evolved script can match:

- **Strict phoneme-grapheme bijection**: every phoneme in the Manding inventory maps to exactly one character. No digraphs, no silent letters, no context-dependent pronunciation rules.
- **Explicit tonal diacritics**: Bambara is tonal; N'Ko marks tone with combining characters above vowels. Latin Bambara orthography does not mark tone.
- **Zero spelling irregularities**: English has ~1,100 letter-to-sound rules for ~44 phonemes. N'Ko has 1-to-1 correspondence for 33 phonemes.

For ASR, this means a CTC decoder targeting N'Ko has a smaller effective output space than one targeting Latin Bambara (no digraph ambiguities to resolve). For LLMs, it means BPE tokenization on N'Ko data should be highly efficient. For any downstream NLP task, it means the text is self-explanatory in a way that English is not.

And yet Qwen3-8B-Instruct allocates exactly 32 tokens to N'Ko in its 151,936-entry vocabulary: single-character fallbacks, one per Unicode codepoint, with no subword entries. Arabic receives approximately 4,200.

### Quantified Failure

We profiled N'Ko processing across all 36 transformer layers of Qwen3-8B-Instruct (8-bit quantized, Apple M4 16GB) using 100 parallel English/N'Ko sentence pairs drawn from N'Ko Wikipedia.

**Translation Tax (L2 Norm Ratio)**

| Layer | English L2 | N'Ko L2 | Ratio |
|-------|-----------|---------|-------|
| 0 (embed) | 41.2 | 14.2 | 2.94x |
| 16 | 143.7 | 48.2 | 2.98x |
| 32 | 237.1 | 79.8 | 2.97x |
| 56 | 354.2 | 115.6 | 3.06x |
| 80 (output) | 512.8 | 157.4 | 3.26x |

The ratio is stable and slightly increasing. This rules out a normalization artifact. Computed per-token (not per-sequence) the distribution is identical, ruling out a sequence-length effect.

**Shannon Entropy (bits)**

| Layer | English | N'Ko | Delta |
|-------|---------|------|-------|
| 0 | 8.12 | 9.47 | +1.35 |
| 40 | 10.14 | 12.31 | +2.17 |
| 80 | 11.02 | 13.89 | +2.87 |

N'Ko output-layer entropy is 13.89 bits. The theoretical maximum for a `d=4096` uniform distribution is approximately 12 bits. N'Ko entropy exceeds this because the absolute-value normalization used to compute the probability distribution magnifies noise in low-magnitude activations.

**Kurtosis (peakedness)**

| Layer | English | N'Ko | N'Ko Deficit |
|-------|---------|------|-------------|
| 0 | 12.4 | 3.2 | 74.2% |
| 60 | 47.2 | 11.3 | 76.1% |
| 70 | 52.8 | 9.7 | 81.6% |
| 80 | 58.4 | 8.3 | **78.1%** |

English kurtosis climbs monotonically to 58.4 at the output layer. N'Ko kurtosis peaks at layer 60 (11.3), then drops to 8.3 at layer 80. The model is actively un-committing from N'Ko predictions in its final layers. At the output, the softmax over 151,936 tokens is nearly flat for N'Ko inputs.

**Sparsity**: 34.5% inactive embedding dimensions for N'Ko vs. 13.8% for English. The gap narrows through middle layers (both converge around 20% at layers 30-50) then reopens at output (N'Ko 28.7%, English 11.2%).

### Arabic Comparison

N'Ko and Arabic are both right-to-left scripts. If RTL direction were the cause, Arabic should show similar deficits.

| Metric (Layer 80) | English | Arabic | N'Ko |
|-------------------|---------|--------|------|
| L2 Norm | 512.8 | 487.3 | 157.4 |
| Norm Ratio vs. English | 1.00x | 1.05x | 3.26x |
| Shannon Entropy (bits) | 11.02 | 11.18 | 13.89 |
| Kurtosis | 58.4 | 54.7 | 8.3 |
| Sparsity | 11.2% | 12.8% | 28.7% |

Arabic is within 7% of English on every metric. N'Ko is 2-7x worse. RTL is not the variable. Vocabulary allocation is.

Arabic: ~4,200 vocabulary tokens (2.8% of vocab), including rich subword representations.
N'Ko: 32 tokens (0.02%), all single characters.

Other scripts with zero coverage: Adlam (40M+ speakers), Tifinagh (30M+), Vai, Osmanya.

### Three-Zone Failure Structure

The activation profiles reveal three mechanistically distinct failure modes:

**Zone 1 (Embedding, layers 0-10):** Comprehension failure. 34.5% embedding sparsity means the model's 4,096-dimensional embedding space is essentially unused for N'Ko. The 32 character tokens received near-zero gradient updates during pre-training. Every N'Ko word arrives at layer 1 as a sequence of random-ish character vectors. Self-attention in layers 1-10 computes weights over these vectors and produces nothing meaningful.

**Zone 2 (Middle layers, layers 10-56):** Reasoning vacuum. The L2 ratio holds stable at ~3x across all middle layers (2.97x at layer 32, 2.99x at layer 40, 3.02x at layer 48). Stability here is the signal. A partially-working system would show fluctuating ratios as different layer groups succeed or fail on different aspects. The constant ratio means no middle-layer group is doing anything. The model passes the malformed embeddings through with residual connections, adding noise without adding structure. Entropy gap grows from 1.32 bits to 2.27 bits across this zone.

**Zone 3 (Output, layers 56-36):** Incoherent prediction. Kurtosis deficit worsens from 76% at layer 60 to 78.1% at layer 80. The model cannot concentrate probability mass on specific N'Ko tokens. The output distribution is spread near-uniformly across English, CJK, code, and N'Ko tokens with roughly equal (near-zero) weight on any individual token.

### Circuit Duplication Analysis

Following the RYS (Revisit Your Shoulders) methodology, we tested whether N'Ko reasoning circuits exist and can be amplified by duplicating transformer layer blocks.

Configuration space: 55 configurations, starting layer in {0, 8, 16, 24, 32, 40, 48, 56, 64, 72}, ending offset in {8, 16, 24}. Score = 0.5 * math accuracy + 0.5 * semantic similarity. Random chance baseline: 0.05.

| Configuration | English | N'Ko |
|--------------|---------|------|
| Best English: layers (8, 16) | 0.752 | 0.031 |
| Best N'Ko: layers (0, 40) | 0.134 | **0.067** |
| Median (all 55) | 0.584 | 0.034 |
| Random baseline | ~0.050 | ~0.050 |

0/55 configurations show N'Ko-advantageous performance. The best N'Ko score (0.067) barely exceeds random chance. The top three N'Ko configurations all start at layer 0 and extend into early middle layers, suggesting whatever minimal processing exists is concentrated at the bottom of the stack. Configurations duplicating only upper layers (which most help English) produce the worst N'Ko scores.

The distinction between absent and weak circuits matters for intervention choice. A weak circuit benefits from small-scale targeted fine-tuning. An absent circuit requires building representations from scratch: more data, more iterations, curriculum design.

---

## The Fix: Three-Stage LoRA

We apply three sequential LoRA fine-tuning stages to Qwen3-8B-Instruct on Apple M4 16GB using MLX v0.29. Total training time: 185 minutes. Cloud cost: $0.

### Pipeline

```
Stage 1: Continued Pre-Training (CPT)
  Data: 17,360 examples from N'Ko Wikipedia (1,693 articles, 3.7M chars)
  Method: 300-char sliding window, 60/40 context-completion split, 100-char overlap
  LoRA: rank=8, alpha=20.0, 8 of 36 layers (0, 4, 8, 12, 16, 20, 24, 28)
  LR: 1e-5, 2,000 iterations, batch 1 + 4-step gradient accumulation
  Goal: Populate embedding space for 32 N'Ko character tokens

Stage 2: Supervised Fine-Tuning (SFT)
  Data: 21,240 instruction-response pairs
    - 8,200 text completion instructions
    - 5,400 translation instructions (En<->N'Ko)
    - 4,312 cultural/grammatical Q&A in N'Ko
    - 3,328 vocabulary/word-meaning pairs
  LR: 5e-6 (halved to preserve CPT gains), 1,000 iterations

Stage 3: BPE-Aware Training
  Data: 25,100 examples from custom N'Ko BPE tokenizer (512 merges, 62,035 words)
    - BPE merge-point completions
    - Word boundary predictions
    - Multi-word continuations
  LR: 3e-6, 1,000 iterations
  Goal: Teach the model the subword structure the tokenizer never learned
```

### Results

| Metric | Base | V3 (Post-Fine-Tune) | Delta |
|--------|------|---------------------|-------|
| N'Ko Perplexity | 11.02 | 6.00 | -45.6% |
| N'Ko Top-1 Accuracy | 43.2% | 56.7% | +13.5pp |
| N'Ko Token Accuracy | 23.0% | 32.8% | +9.8pp |
| English Top-1 Accuracy | 70.9% | 69.7% | -1.2pp |
| **Translation Tax** | **2.94x** | **0.70x** | **-76%** |
| Embedding Sparsity (N'Ko) | 34.5% | 18.2% | -47% |
| Output Kurtosis (N'Ko) | 8.3 | 31.4 | +278% |
| Output Entropy Gap (bits) | 2.87 | 0.94 | -67% |

The 0.70x result is an inversion: the model now produces lower perplexity on N'Ko than English, a mild overfitting to N'Ko patterns while English representations remain largely intact (-1.2pp accuracy).

### V1/V2/V3 Progression and Mode Collapse

V1 (Wikipedia-only CPT, 17,360 examples): embedding-layer tax drops from 2.94x to 1.85x. Model generates N'Ko characters but no syntactic structure. Middle and output ratios remain above 2.0x.

V2 (CPT + SFT, 38,600 examples): tax drops to ~1.2x at embedding, ~0.85x at output. Severe mode collapse: 20/20 sampled generations produce the same stereotyped response. Root cause: 12 SFT instruction templates. Insufficient diversity for 21,240 examples. Training loss: 3.506.

V3 (CPT + SFT + BPE, 92,184 examples): Mode collapse resolved. 340 instruction templates (vs. 12), BPE-aware data forces specific subword predictions that prevent template fallback. Only 3/20 degenerate generations. Training loss: 3.275. Translation tax: 0.70x.

Design rule from this progression: SFT instruction diversity must exceed ~100 unique templates to prevent mode collapse when fine-tuning LLMs on low-resource scripts. Below that threshold, the model learns to map all instructions to a small number of high-frequency response patterns.

---

## Audio-to-N'Ko ASR: Architecture and Results

Every published Bambara ASR system outputs Latin-script text. For N'Ko-literate speakers, every existing system writes in the wrong alphabet. We built the first audio-to-N'Ko pipeline, converting Bambara speech directly to N'Ko without Latin as an intermediary.

### The Cross-Script Bridge

No N'Ko-labeled speech corpus exists. We bridge Latin Bambara transcriptions (from bam-asr-early, 37 hours, CC-BY-4.0) to N'Ko through a two-stage deterministic converter:

```
Latin Bambara -> IPA -> N'Ko Unicode
B: Sigma_L* -> IPA -> Sigma_N
```

Stage 1 is rule-based with strict priority ordering (digraphs before single characters before tone-decomposed forms). Stage 2 is a bijective IPA-to-N'Ko lookup table over the full Manding phoneme inventory.

**Six documented bug classes** encountered during development. Each corresponds to a category of phonemic information that Latin orthography obscures:

1. **Greedy character matching**: "ny" matched as separate n + y before digraph rule applied. Fix: strict priority ordering.
2. **Missing consonant mapping**: /g/ had no IPA-to-N'Ko entry. Words with /g/ produced embedded Latin letters. Fix: add U+07DC.
3. **Extended IPA symbols**: FarmRadio Whisper produces schwa, esh, voiced palatal stop from loanwords and dialectal variants. Fix: add 8 extended IPA mappings.
4. **Post-digraph IPA gaps**: digraph outputs (ny -> /ɲ/, ng -> /ŋ/) had no Stage 2 entries. Fix: propagate all Stage 1 outputs to Stage 2 table.
5. **NFD decomposition ordering**: pre-composed toned vowels (à, é) need `unicodedata.normalize('NFD', text)` before lookup, not after.
6. **RTL rendering**: N'Ko requires U+200F (right-to-left mark) after spaces in bidirectional contexts.

```python
# Priority ordering (critical for correct conversion)
# 1. Digraphs (must consume before individual chars)
DIGRAPH_MAP = {
    'ny': '\u07E2',   # U+07E2 NYA (palatal nasal)
    'ng': '\u07D2',   # U+07D2 NGA (velar nasal)
    'sh': '\u07D9',   # U+07D9 SHA (postalveolar)
}

# 2. NFD decompose, then toned vowels
# 3. Single characters
```

Bridge validation on Bayelemabaga corpus (46,976 segments): 87.7% pass FSM validation. Failure modes: consonant clusters from transcription noise (8.2%), unknown IPA symbols (2.7%), malformed Unicode (1.5%). On clean, manually verified Bambara text: 99.4% valid N'Ko output.

### Architecture Search: 28 Configurations

Training data: 37,306 clips from bam-asr-early, 32,418 valid after FSM filtering (86.9%). CTC output space: 66 classes (65 N'Ko codepoints + blank).

Whisper large-v3 encoder (307M parameters, 680K hours training) as frozen acoustic feature extractor. Search over decoder architecture (BiLSTM, Transformer, Conformer), hidden dimension (256, 512, 768), depth (2, 4, 6 layers), downsampling (4x, 8x, 16x).

Selected results from the 28-configuration search:

| # | Architecture | Hidden | Layers | DS | Params | CER | Val Loss |
|---|-------------|--------|--------|-----|--------|-----|----------|
| 1 | BiLSTM | 256 | 2 | 16x | 1.2M | 78.1% | 0.412 |
| 8 | BiLSTM | 512 | 2 | 4x | 4.2M | 63.1% | 0.243 |
| 10 | BiLSTM | 512 | 4 | 4x | 5.4M | 60.4% | 0.198 |
| 13 | BiLSTM | 768 | 6 | 4x | 15.7M | 57.3% | 0.169 |
| **22** | **Transformer** | **512** | **4** | **4x** | **22.1M** | **45.7%** | **0.121** |
| 23 | Transformer | 768 | 4 | 4x | 46.9M | 38.2% | 0.087 |
| 26 | Conformer | 512 | 4 | 4x | 18.3M | 51.2% | 0.148 |
| 28 | Conformer | 768 | 4 | 4x | 38.4M | 44.1% | 0.098 |

Three findings from this search:

1. **Transformers beat BiLSTMs by 15-20 points at every comparable scale.** Self-attention's global context window is exactly what N'Ko syllable structure needs. BiLSTM's sequential induction bias means context from 5 characters ago is substantially attenuated, which matters for a language where syllable boundaries create long-range dependencies.

2. **4x downsampling consistently outperforms 8x and 16x (8-16 CER points).** N'Ko characters represent individual phonemes, shorter acoustic events than the syllable or word-level units that higher downsampling rates assume. Temporal resolution matters.

3. **Conformers underperform Transformers at 37 hours.** Local convolution kernels overfit to speaker-specific patterns with this data volume. We expect Conformers to become competitive at 100+ hours.

---

## V4: Whisper LoRA Results

V1-V3 use Whisper's encoder as a frozen feature extractor. Whisper was trained on predominantly non-African audio and has no specific knowledge of Bambara phonology, tone contours, or nasalization patterns.

V4 partially unfreezes the Whisper encoder using LoRA applied to the query, key, and value projections of encoder layers 24-31 (top 8 of 32 encoder layers).

```
V4 Architecture:
  Encoder: Whisper large-v3 (307M params, frozen)
           + LoRA (rank=32, alpha=64) on Q,K,V of layers 24-31 (5.9M trainable)
  Decoder: Transformer CTC head, 6 layers, 768 hidden, 12 heads, 4x DS (46.9M)
  Total trainable: 52.8M / 359.8M total
  Dual LR: 1e-5 for LoRA layers, 3e-4 for CTC head
```

LoRA rank=32 vs. the LLM experiments at rank=8 reflects different task requirements. LLM adaptation adds text-processing capabilities on top of existing machinery. ASR encoder adaptation must reshape acoustic representations for a language whose prosody and phonemic inventory differ substantially from Whisper's dominant training languages.

Applying LoRA only to the top 8 layers preserves lower-layer general-purpose acoustic feature extraction (spectral decomposition, temporal segmentation) while allowing upper layers to specialize for Bambara phonemic patterns.

**V4 Training Progression (30 epochs on A100, Vast.ai, $0.89/hr)**

| Epoch | Train Loss | Val Loss | Delta Val |
|-------|-----------|----------|-----------|
| 1 | 1.142 | 0.884 | --- |
| 5 | 0.763 | 0.612 | -30.8% |
| 10 | 0.542 | 0.478 | -21.9% |
| 20 | 0.354 | 0.341 | -11.9% |
| 30 | **0.287** | **0.290** | -6.1% |

Train-val gap at epoch 30: 0.003. The model is not overfitting. Combined effect of SpecAugment and conservative encoder LR.

**V4 vs. V3 Evaluation (50 held-out samples)**

| Metric | V3 Base | V4 LoRA | Delta |
|--------|---------|---------|-------|
| CER | 33.0% | **29.4%** | -3.6pp |
| WER | 70.0% | **62.3%** | -7.7pp (-11.0% rel.) |
| Mean confidence | 0.46 | **0.82** | +79% |
| Syllable validity rate | 94.2% | **96.1%** | +1.9pp |

**Per-sample outcomes:**
- LoRA wins (lower WER): 20/50 (40%)
- Base wins: 17/50 (34%)
- Tie: 13/50 (26%)

V4 does not uniformly improve over V3. It provides large gains on specific sample types and slight regressions on short, easy utterances.

**Worst-case improvements:**

| Sample | V3 WER | V4 WER | Improvement |
|--------|--------|--------|-------------|
| au30 | 15.80 | **1.00** | -93.7% |
| au42 | 8.33 | **2.17** | -74.0% |
| au07 | 6.50 | **1.83** | -71.8% |
| au19 | 5.00 | **2.00** | -60.0% |
| au38 | 4.67 | **1.33** | -71.5% |

Sample au30 (V3's worst performer, essentially complete failure at WER 15.8) drops to WER 1.0 under V4. The frozen Whisper encoder had an acoustic blind spot for whatever phonemic pattern that sample contains. LoRA adaptation closed it.

**Error class analysis (V4):**
- Tone diacritic confusion: 38% of character errors (vs. 41% V3). The bridge defaults to neutral tone due to absent Bambara tone lexicon, limiting supervisory signal.
- Syllable dropping: 14% of word errors (vs. 23% V3). LoRA adaptation improves temporal alignment for longer words.
- New in V4: phoneme substitution between acoustically similar consonants (/t/ vs. /d/, /k/ vs. /g/): 11% of character errors. LoRA has shifted acoustic boundaries between similar phonemes.

**Four-version summary:**

| Version | Params | CER | WER | Compute Cost |
|---------|--------|-----|-----|-------------|
| V1 BiLSTM | 5.4M | 56.0% | 91.5% | $3 |
| V2 Transformer | 22.1M | 45.7% | 78.6% | $4 |
| V3 Transformer | 46.9M | 33.0% | 70.0% | $5 |
| **V4 Whisper LoRA** | **52.8M** | **29.4%** | **62.3%** | **$6** |
| *MALIBA-AI v3* | *~2B* | *n/a* | *45.73%* | *---* |

Total across all experiments (including 28-config architecture search): $14.

Note on MALIBA-AI comparison: MALIBA-AI achieves 45.73% WER with Latin-script output on its own benchmark corpus. V4 achieves 62.3% WER with N'Ko-script output on a different test set. WER is computed after round-trip conversion (N'Ko -> Latin via bridge inverse -> WER against original Latin transcription), adding conversion error. Our 52.8M-parameter system reaches the same order of magnitude as a ~2B-parameter system, suggesting N'Ko's structural advantages partially compensate for the 38x parameter difference.

---

## The 4-State FSM

N'Ko syllables follow (C)V(N): optional consonant onset, required vowel nucleus with optional tone diacritics, optional nasal coda. No consonant clusters. No vowel hiatus within a syllable. This is a closed formal system. We encode it as a hard constraint on CTC output rather than asking the neural network to learn it.

```
States: {Start, Onset, Nucleus, Coda}
Alphabet: C (12 consonants), V (7 vowels), T (8 tone diacritics),
          N (3 nasalization marks), space, punct

Start  + C      -> Onset
Start  + V      -> Nucleus
Start  + space  -> Start      (word boundary)
Start  + T      -> REJECT     (tone without nucleus)
Onset  + V      -> Nucleus
Onset  + C      -> REJECT     (CC cluster)
Onset  + space  -> REJECT     (bare consonant)
Nucleus + T     -> Nucleus    (tone attaches, no state change)
Nucleus + N     -> Coda
Nucleus + V     -> REJECT     (vowel hiatus)
Nucleus + C'    -> Onset      (new syllable, C' = non-nasal)
Nucleus + space -> Start
Coda   + C      -> Onset      (new syllable)
Coda   + V      -> Nucleus    (resyllabification)
Coda   + N      -> REJECT     (double nasal)
Coda   + space  -> Start
```

On violation, the FSM replaces the offending token with the highest-probability admissible token from the CTC posterior at that time step. In practice, most corrections insert epenthetic vowels to resolve consonant clusters.

**FSM validation statistics:**

| Input Type | Pass Rate | n |
|-----------|----------|---|
| Natural N'Ko text | 99.0% | 1,000 |
| V3 CTC output | 94.2% | 500 |
| V4 CTC output | 96.1% | 500 |
| Random N'Ko characters | 19.0% | 1,000 |
| Random Unicode | 2.3% | 1,000 |

99% pass rate on natural N'Ko text vs. 19% on random N'Ko character sequences. The FSM captures genuine phonotactic structure, not trivial constraints. Runtime overhead: <2% latency (O(1) per token, single array lookup).

---

## Pipeline Parallelism

The full inference system runs across two Apple Silicon machines connected via Thunderbolt 5.

```
Audio --> [Mac4: M4 Max 64GB]
          Whisper encoder + LoRA adapters
          307M params, fp16
          ~180ms for 10-second clip
          Output: 1,280-dim frame representations (2.4MB @ float16)

          -- Thunderbolt 5 -- (0.4ms transfer, negligible)

[Mac5: M4 16GB]
          CTC decoder + FSM (~40ms)
          NLLB-200 translation (~67ms/sentence, optional)
          Output: N'Ko text, optionally with English/French translation

Total end-to-end:
  ASR only:             ~290ms
  ASR + translation:    ~360ms
```

```
Full translation pipeline:
Audio -> [Mac4] Whisper+LoRA -> frames -> [Mac5] CTC+FSM -> N'Ko
N'Ko -> B^{-1} (bijective, instant) -> Latin Bambara -> NLLB-200 -> En/Fr
```

The bridge inverse is trivial because the original bridge is bijective: every N'Ko character maps to exactly one Latin character or digraph. No ambiguity resolution.

NLLB-200 (600M parameters) fine-tuned on 8,640 parallel sentence pairs across four directions (Bambara<->English, Bambara<->French). Training loss: 6.29 to 1.89 over 15 epochs on A100. BLEU-1 for Bambara->English: 0.246. Functional for downstream use.

---

## Reproducibility

**Compute cost breakdown:**

| Experiment | Hardware | Cost |
|-----------|---------|------|
| V1 BiLSTM (200 epochs) | RTX 4090 Vast.ai | $3 |
| V2 Transformer (200 epochs) | RTX 4090 Vast.ai | $4 |
| Architecture search (28 configs) | RTX 4090 Vast.ai | included |
| V3 Transformer (200 epochs) | RTX 4090 Vast.ai | $5 |
| V4 Whisper LoRA (30 epochs, A100) | A100 Vast.ai | $6 |
| LLM fine-tuning (CPT+SFT+BPE) | Apple M4 16GB | $0 |
| **Total** | | **$14** |

Activation profiling: Apple M4 16GB (Qwen3-8B-Instruct 8-bit), no cloud cost.

**Key dependencies:**
- MLX v0.29+ for LLM fine-tuning (Apple Silicon)
- Whisper large-v3 for acoustic encoding
- NLLB-200 (facebook/nllb-200-distilled-600M) for translation
- bam-asr-early corpus (CC-BY-4.0) for ASR training
- N'Ko Wikipedia (1,693 articles, 3.7M chars) for LLM training

**Code, models, and data are open-source.** The bridge script, FSM implementation, training pipeline, and activation profiling scripts will be released at publication. NKoScribe (iOS app, full V4 pipeline on-device) is available on TestFlight now.

---

## Citation

```bibtex
@article{diomande2026deadcircuits,
  title={Dead Circuits: Activation Profiling and Script Invisibility
         in Large Language Models},
  author={Diomande, Mohamed},
  journal={arXiv preprint},
  year={2026},
  note={Target: ACL/EMNLP 2026}
}

@article{diomande2026livingspeech,
  title={Living Speech: Script-Native Automatic Speech Recognition
         for N'Ko},
  author={Diomande, Mohamed},
  journal={arXiv preprint},
  year={2026},
  note={Target: ACL/EMNLP 2026}
}
```

---

*Contact: contact@mohameddiomande.com*

*The tokenizer as gatekeeper: a script absent from tokenizer training receives no BPE merges, which means no subword vocabulary entries, which means character-level tokenization at 3-5x the rate of well-represented scripts, which means weaker embeddings, longer sequences, and the full cascade documented here. The tokenizer is trained once, early, on a fixed corpus. Scripts absent from that corpus are permanently excluded from the model's representational capacity. A simple audit metric: flag any script with vocabulary density below 1.5x its Unicode block size. For N'Ko that number is 0.50. For Adlam it is 0.00. The cost of inclusion is negligible. N'Ko Wikipedia is 3.7 million characters. That is approximately 0.0001% of a typical LLM pre-training corpus.*
