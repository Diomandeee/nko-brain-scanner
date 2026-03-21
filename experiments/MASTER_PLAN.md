# N'Ko Brain Scanner: Experiment Master Plan

## Overview

Five experiments that extend the original brain scan findings into a complete research program.
The original brain scan revealed that N'Ko is functionally invisible to Qwen3-8B: higher sparsity,
lower entropy, flatter kurtosis compared to English across all 36 layers. These experiments test
whether that finding generalizes, whether it matters for downstream tasks, and whether the
properties of N'Ko script can be turned into computational advantages.

## Experiments

### A. Cross-Model Brain Scan ("Script Invisibility Across Architectures")
- **Directory**: `A_cross_model_brain_scan/`
- **Question**: Is N'Ko invisible to all models, or just Qwen?
- **Method**: Run the activation profiling pipeline on 4 models (Qwen3-8B, Llama-3.1-8B, Gemma-3-12B, Qwen2-72B). Compare translation tax, entropy, kurtosis, sparsity.
- **Hypothesis**: Script invisibility is universal. Every model trained predominantly on Latin/CJK data will show the same activation deficit for N'Ko.
- **Status**: SCAFFOLDED

### B. CTC Script Advantage ("Controlled Comparison")
- **Directory**: `B_script_advantage_ctc/`
- **Question**: Does N'Ko's phonetic transparency give it an advantage for ASR?
- **Method**: Train the same CharASR architecture (V3, 46.9M params) on the same 37,306 audio samples. One decoder outputs N'Ko characters (65 classes + blank). The other outputs Latin Bambara characters (~40 classes + blank). Compare CER on aligned test sets.
- **Hypothesis**: N'Ko CTC will achieve lower CER because its 1:1 phoneme-to-character mapping eliminates the ambiguity that Latin introduces.
- **Status**: SCAFFOLDED

### C. N'Ko Cognitive Twin
- **Directory**: `C_nko_cognitive_twin/`
- **Question**: What happens when you retrain a cognitive twin to think in N'Ko?
- **Method**: Take cognitive twin SFT data (English), translate to N'Ko via the bridge, train a parallel LoRA adapter. Compare behavioral evaluation, token compression, perplexity.
- **Hypothesis**: The N'Ko twin will produce more compressed representations (fewer tokens for same semantics) due to N'Ko's agglutinative morphology.
- **Status**: SCAFFOLDED (blocked by cognitive twin training completion)

### D. Sigil Compression
- **Directory**: `D_sigil_compression/`
- **Question**: Can N'Ko sigils compress conversational knowledge more efficiently than BPE?
- **Method**: Take 1000 conversation turns, extract key concepts, map through N'Ko transliteration, assign to 10 sigil anchors. Measure compression ratio vs English BPE tokenization.
- **Hypothesis**: Sigil encoding achieves 50-100x compression over raw BPE because it maps to semantic categories, not lexical tokens.
- **Status**: SCAFFOLDED

### E. EPOCH Inscription
- **Directory**: `E_epoch_inscription/`
- **Question**: Can the full derivation chain (conversation -> SFT -> N'Ko -> sigil -> blockchain) preserve knowledge provenance?
- **Method**: Build the 5-layer derivation chain. Each layer produces an artifact. The final inscription carries a provenance hash. Uses existing EPOCH Clarity contracts.
- **Hypothesis**: On-chain inscription with N'Ko-encoded provenance creates a verifiable, culturally-rooted knowledge record that no centralized system can replicate.
- **Status**: SCAFFOLDED (depends on C and D)

## Dependency Graph

```
A (cross-model brain scan)       [independent]
B (CTC script advantage)         [independent]
C (N'Ko cognitive twin)          [blocked by cognitive twin training]
D (sigil compression)            [independent]
E (EPOCH inscription)            [depends on C and D]
```

Parallel execution plan:
- **Phase 1** (can start now): A, B, D run in parallel
- **Phase 2** (after cognitive twin): C
- **Phase 3** (after C and D): E

## Estimated Total Cost

| Experiment | Compute | Cost |
|------------|---------|------|
| A: Cross-Model | Qwen3-8B, Llama-3.1-8B, Gemma-3-12B local (M4). Qwen2-72B on Vast.ai A100 ~30min | ~$2-3 |
| B: CTC Script | Two training runs on Vast.ai A100 ~2h each | ~$10 |
| C: Cognitive Twin | Translation local. LoRA training on Vast.ai ~1h | ~$3 |
| D: Sigil Compression | All local computation | $0 |
| E: EPOCH Inscription | Local + STX gas fees | ~$1-5 |
| **Total** | | **~$16-21** |

## Timeline

| Week | Activities |
|------|-----------|
| Week 1 | Run A (local models), start B (Vast.ai) |
| Week 1 | Run D (local, fast) |
| Week 2 | Run A (Qwen2-72B on Vast.ai), analyze B results |
| Week 2+ | Run C when cognitive twin training is complete |
| Week 3+ | Run E after C and D are done |

## Running an Experiment

Each experiment has:
- `README.md` with full design document
- Python scripts with `--help` for all arguments
- `estimated_cost.md` with cost breakdown
- A corresponding blog draft in `../blog/`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Blog Posts

Each experiment has a companion blog draft in `../blog/`:
- [Experiment A: Cross-Model Brain Scan](../blog/experiment-a-cross-model-brain-scan.md)
- [Experiment B: Script Advantage](../blog/experiment-b-script-advantage.md)
- [Experiment C: N'Ko Cognitive Twin](../blog/experiment-c-nko-cognitive-twin.md)
- [Experiment D: Sigil Compression](../blog/experiment-d-sigil-compression.md)
- [Experiment E: EPOCH Inscription](../blog/experiment-e-epoch-inscription.md)

All drafts follow the voice of the original brain scan blog post: narrative, technical, accessible, no AI-isms.
