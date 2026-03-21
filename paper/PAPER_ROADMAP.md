# N'Ko Research Paper Roadmap

## Overview

Six papers forming a coherent research arc: from diagnosing why AI fails on N'Ko, to building systems that work, to proving structural advantages, to connecting language identity with personalized AI and blockchain provenance.

Each paper builds on the previous. Together they constitute a thesis-level body of work on indigenous script computing.

---

## Paper 1: Dead Circuits (WRITTEN)

**Title:** Dead Circuits: Activation Profiling and Script Invisibility in Large Language Models

**Status:** Draft complete (1,305 lines). Model identity corrected to Qwen3-8B. Claims audited.

**File:** `paper/paper1_dead_circuits.tex`

**Core finding:** LLMs have zero functional circuits for N'Ko. Translation tax of 2.90x, 0/55 reasoning configurations, 85.8% kurtosis deficit. Three-stage LoRA reduces tax to 0.70x.

**Venue:** ACL/EMNLP 2026

**Dependencies:** None. Standalone diagnostic paper.

**Open issue:** Data is from Qwen3-8B. Experiment A (cross-model scan) will either validate universality or require revision.

---

## Paper 2: Living Speech (WRITTEN)

**Title:** Living Speech: Script-Native Automatic Speech Recognition for N'Ko

**Status:** Draft complete (1,504 lines). V4 LoRA results included. Eval numbers updated.

**File:** `paper/paper2_living_speech.tex`

**Core finding:** First audio-to-N'Ko ASR. 46.9M-param CTC decoder on frozen Whisper achieves 33% CER. V4 Whisper LoRA (merged encoder) improves confidence 12x. Cross-script bridge recovers phonemic information Latin orthography suppresses. 4-state FSM guarantees phonotactic validity. Total compute: $14.

**Venue:** Interspeech 2026 or ACL 2026

**Dependencies:** None. Standalone systems paper.

**Open issue:** V4 eval shows marginal WER improvement (11%) but massive confidence improvement (79%). Experiment B would strengthen the structural advantage claim.

---

## Paper 3: Script Invisibility Across Architectures (FROM EXPERIMENT A)

**Title:** Script Invisibility Is Universal: Activation Profiling Across Four LLM Families

**Status:** Not started. Scaffolded in `experiments/A_cross_model_brain_scan/`.

**Hypothesis:** The translation tax and dead circuit phenomenon observed in Qwen3-8B is not model-specific. It is a universal consequence of training data composition, reproducible across architecturally distinct model families.

**Models to test:**
- Qwen3-8B (existing data, free)
- Llama-3.1-8B (free on Apple Silicon)
- Gemma-3-12B (free on Apple Silicon or Mac5)
- Qwen2-72B ($2-3 on Vast.ai, validates the original Paper 1 framing)

**Key metrics per model:**
- Translation tax (L2 norm ratio English/N'Ko at each layer)
- Entropy inflation profile
- Kurtosis deficit at output layer
- Sparsity at embedding layer
- Circuit duplication yield (RYS methodology, 55 configurations per model)
- Tokenizer analysis: how many N'Ko tokens in each model's vocabulary

**Novel contribution:** No prior work compares script-specific activation profiles across model families. The comparison between Arabic (RTL, well-resourced) and N'Ko (RTL, zero-resourced) isolates data starvation from directional bias.

**Estimated cost:** ~$3 (only the 72B model needs cloud compute)

**Venue:** NeurIPS 2026 or ICLR 2027

**Dependencies:** Paper 1 (establishes methodology)

---

## Paper 4: Script Design Affects ASR Accuracy (FROM EXPERIMENT B)

**Title:** Does Script Design Matter? A Controlled Comparison of CTC Decoding for Engineered vs Evolved Orthographies

**Status:** Not started. Scaffolded in `experiments/B_script_advantage_ctc/`.

**Hypothesis:** Given identical model architecture, training data, and compute budget, a CTC decoder targeting N'Ko (bijective phoneme-to-character mapping) achieves lower character error rate than a CTC decoder targeting Latin Bambara (digraph-based, ambiguous mapping).

**Experimental design:**
- Same architecture: CharASR V3 (46.9M params, h768, L6, Conv1d stride-4)
- Same data: 37,306 bam-asr-early clips
- Same encoder: Whisper large-v3 (frozen)
- Same training schedule: 200 epochs, AdamW, cosine decay
- Decoder A: 66-class output (65 N'Ko codepoints + blank)
- Decoder B: ~40-class output (26 Latin + digraph tokens + blank)
- Evaluation: CER on aligned test set (same audio, two reference transcriptions)

**Controls:**
- Output vocabulary size difference is a confound. Address by also testing a 66-class Latin decoder (pad with unused tokens) to isolate vocab size from script structure.
- The bridge introduces conversion error. Address by using human-annotated N'Ko references for a 100-sample subset.

**Novel contribution:** No prior work has tested whether the design properties of a target script measurably affect ASR accuracy. This is the first controlled experiment isolating script structure from all other variables.

**Estimated cost:** ~$10 (two training runs on Vast.ai)

**Venue:** Interspeech 2026 or ICASSP 2027

**Dependencies:** Paper 2 (establishes architecture and baseline)

---

## Paper 5: Script as Thought (FROM EXPERIMENT C)

**Title:** Script as Thought: Indigenous Script as Internal Representation in Personalized Language Models

**Status:** Not started. Scaffolded in `experiments/C_nko_cognitive_twin/`.

**Hypothesis:** A personalized language model (cognitive twin) trained on a bilingual speaker's data exhibits different behavioral patterns when its LoRA adapter is trained on N'Ko-encoded versions of the speaker's utterances compared to English-encoded versions. The N'Ko encoding produces measurably different outputs on culturally grounded prompts.

**Experimental design:**
- Base model: Cognitive twin (currently training on Vast.ai)
- Adapter A: English SFT data (35 advantage-weighted trajectory examples)
- Adapter B: Same data translated to N'Ko via the bridge
- Evaluation metrics:
  - Perplexity on held-out prompts (50 English, 50 N'Ko)
  - Token compression ratio (N'Ko vs English for same semantic content)
  - Behavioral divergence score (cosine similarity of hidden states on identical prompts)
  - Cultural grounding score: accuracy on 20 Manding cultural knowledge questions
  - Response style analysis: sentence length, vocabulary diversity, code-switching frequency

**Novel contribution:** No prior work has examined whether a personalized model's behavior changes when its training representation uses the speaker's indigenous script rather than a colonial orthography. This bridges NLP, cultural computing, and personalization research.

**Estimated cost:** ~$3

**Venue:** FAccT 2026 or AIES 2027 (AI ethics and society)

**Dependencies:** Cognitive twin training completion (Vast.ai instance 33195812). Paper 1 (motivates why script matters for internal representation).

---

## Paper 6: Inscribing Knowledge (FROM EXPERIMENTS D + E)

**Title:** Inscribing Knowledge: Blockchain Provenance for Compressed Linguistic Representations in Indigenous Scripts

**Status:** Not started. Scaffolded in `experiments/D_sigil_compression/` and `experiments/E_epoch_inscription/`.

**Hypothesis:** The derivation chain from raw conversation to blockchain inscription (conversation → SFT curation → N'Ko translation → sigil compression → on-chain inscription) produces a verifiable, compressed knowledge representation where each transformation step is auditable and each inscription carries provenance linking it to its source.

**Experimental design:**
- Input: 1,000 conversation turns from Supabase (112K+ available)
- Layer 0: Raw English text (average tokens per turn via GPT-4 tokenizer)
- Layer 1: Concept extraction (top-k keywords per turn)
- Layer 2: N'Ko transliteration (character count)
- Layer 3: Sigil compression (10-sigil encoding from brain scanner)
- Layer 4: EPOCH inscription (Clarity contract call on Stacks)
- Metrics: compression ratio at each layer, information retention (can you reconstruct meaning from sigils?), inscription cost per knowledge unit, provenance verification time

**The 10 sigils:**
Each sigil is a N'Ko character pattern that the brain scanner identified as activating specific semantic circuits. They serve as category anchors for compressing arbitrary text into a fixed-alphabet encoding. The paper formalizes the sigil assignment algorithm and measures its information-theoretic properties.

**Novel contribution:** First work combining indigenous script compression with blockchain provenance for AI knowledge chains. Connects computational linguistics, information theory, and decentralized systems.

**Estimated cost:** ~$0.10 (local compute + minimal STX gas fees)

**Venue:** ACM CCS 2026 (blockchain track) or a new venue at the intersection of blockchain and linguistics

**Dependencies:** Experiments C and D. Paper 5 (establishes the N'Ko representation). EPOCH contracts (already deployed).

---

## Timeline and Parallelism

```
Month 1 (NOW):
  ├── A: Cross-model brain scan (can start immediately, ~2 days)
  ├── B: CTC script advantage (can start immediately, ~3 days)
  ├── D: Sigil compression (can start immediately, ~1 day)
  └── Wait: Cognitive twin training completes

Month 2:
  ├── C: N'Ko cognitive twin (after twin training, ~2 days)
  ├── E: EPOCH inscription (after C + D, ~1 day)
  ├── Paper 3: Write from Experiment A results
  └── Paper 4: Write from Experiment B results

Month 3:
  ├── Paper 5: Write from Experiment C results
  ├── Paper 6: Write from Experiments D + E results
  └── Submit Papers 3 + 4 to venues

Month 4:
  └── Submit Papers 5 + 6 to venues
```

## Total Cost

| Experiment | Compute | Notes |
|------------|---------|-------|
| A | ~$3 | Only 72B needs cloud |
| B | ~$10 | Two full training runs |
| C | ~$3 | One LoRA training run |
| D | $0 | All local |
| E | ~$0.10 | STX gas fees |
| **Total** | **~$16** | Less than lunch |

## Paper Count

| Paper | Status | Experiment |
|-------|--------|------------|
| 1. Dead Circuits | WRITTEN | Existing data |
| 2. Living Speech | WRITTEN | Existing data |
| 3. Script Invisibility Across Architectures | SCAFFOLDED | A |
| 4. Script Design Affects ASR | SCAFFOLDED | B |
| 5. Script as Thought | SCAFFOLDED | C |
| 6. Inscribing Knowledge | SCAFFOLDED | D + E |

Six papers. $16 total compute. One researcher. One script. One argument across six angles: N'Ko's design advantages are real, measurable, universal, and actionable.

---

*"N'Ko" means "I say" in all Manding languages. These papers are the evidence for what Solomana Kante said in 1949: this script was built to work.*
