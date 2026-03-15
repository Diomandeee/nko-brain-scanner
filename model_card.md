---
license: apache-2.0
language:
  - nqo
  - en
  - bm
base_model: mlx-community/Qwen3-8B-8bit
tags:
  - nko
  - manding
  - low-resource
  - african-languages
  - brain-scan
  - lora
  - mlx
  - constrained-decoding
  - bpe-tokenizer
datasets:
  - custom
pipeline_tag: text-generation
---

# NKo-Qwen3-8B-V2: Three-Stage Fine-Tuned Model for N'Ko Script

A Qwen3-8B model adapted for N'Ko script processing through a three-stage pipeline (CPT + SFT + BPE-aware training), with vocabulary extension (250 N'Ko BPE tokens) and admissibility-constrained decoding via a syllable FSM.

## Key Results

| Metric | Base | V1 (3-Stage) | V2 (Extended Vocab) | Change |
|--------|------|-------------|-------------------|--------|
| N'Ko Perplexity | 11.02 | **6.00** | --- | -45.6% |
| N'Ko Token Accuracy | 23.0% | **32.8%** | --- | +43% rel |
| Translation Tax | 2.90x | **0.70x** | --- | **-76%** |
| English Top-1 Acc | 70.9% | 69.7% | --- | -1.2pp |
| Val Loss | 4.290 | --- | **3.506** | -18.3% |
| Syllable Validity | 89.8% | --- | **100%** (FSM) | +10.2pp |

## Models Included

### V1: Three-Stage Adapter (Base Vocabulary)
- **Training**: CPT (17,360 examples) + SFT (21,240) + BPE-aware (25,100)
- **Config**: LoRA rank 8, scale 20.0, top 8 layers
- **Result**: N'Ko PPL 11.02 -> 6.00, Translation Tax 2.90x -> 0.70x
- **Strength**: Higher generation diversity

### V2: Extended Vocabulary Adapter
- **Base model**: Vocabulary extended from 151,936 to 152,192 tokens (+250 N'Ko BPE tokens)
- **Extension method**: Dequantize-extend-requantize embedding surgery
- **Training**: 33,912 examples, 2,000 iterations, LoRA rank 8 (4 layers)
- **Result**: Val loss 3.506 (18.3% lower than V1)
- **Token reduction**: 29.6% fewer tokens for N'Ko text

## Constrained Decoding

The package includes a 4-state finite-state machine (FSM) that enforces N'Ko CV/CVN syllable structure during generation:

```python
from constrained.logits_processor import NKoAdmissibilityProcessor

# Create processor from tokenizer
processor = NKoAdmissibilityProcessor(tokenizer)

# Use as logits processor during generation
logits = processor(tokens, logits)  # Masks inadmissible tokens to -inf
```

FSM states: START -> ONSET (consonant) -> NUCLEUS (vowel) -> CODA (nasal)

Results on 50 N'Ko prompts:
- Unconstrained: 89.8% valid syllables, 9.4 tok/s
- FSM-Constrained: **100%** valid syllables, 5.4 tok/s

## N'Ko BPE Tokenizer

512-merge tokenizer trained on N'Ko Wikipedia (62,035 word occurrences):
- **Compression**: 2.75x (reduces token gap from 4x to 1.45x)
- **Linguistically valid**: Top merges correspond to Manding grammatical particles
- **Vocab**: 614 tokens (64 base + 512 merges + 32 morpheme + 6 special)

Morpheme-aware variant also included (158 effective merges, 206-token vocabulary, 0.941 morpheme boundary preservation).

## Brain Scan

Per-layer activation profiling reveals:
- **Layers 0-27 (Frozen)**: Zero activation change under LoRA
- **Layers 28-34 (Adaptation)**: Reduced L2 norms (-38 to -104), more efficient encoding
- **Layer 35 (Output)**: +573 L2 increase, sharper N'Ko predictions

## Training Hardware

- **Device**: Apple M4 MacBook, 16GB unified memory
- **Framework**: MLX v0.29 with mlx_lm
- **Total time**: ~3 hours (all three stages)
- **Cloud cost**: $0 (consumer hardware only)

## Limitations

- Mode collapse during extended N'Ko generation (repetitive output dominated by frequent tokens)
- Only 4 LoRA layers adapted on 8B model, insufficient for diverse generation
- No task-level N'Ko benchmarks exist; evaluated on perplexity and token accuracy only
- No human evaluation conducted
- WMT 2023 nicolingua corpus (130,850 segments) not yet integrated

## Usage

```python
from mlx_lm import load, generate

# Load fused model
model, tokenizer = load("Diomandeee/nko-qwen3-8b-v2")
response = generate(model, tokenizer, prompt="What is N'Ko script?", max_tokens=200)

# Or with constrained decoding
from constrained.logits_processor import NKoAdmissibilityProcessor
processor = NKoAdmissibilityProcessor(tokenizer)
# Use with generate_step() loop
```

## Syllable Retriever (ASR Proof-of-Concept)

A 3,024-entry syllable codebook (23C x 7V x 5T x 2N across V/VN/CV/CVN patterns) with cosine k-NN retrieval and FSM-constrained beam search achieves 100% round-trip accuracy on synthetic embeddings.

## Citation

```bibtex
@misc{diomande2026nko,
  title={The Script That Machines Can't Read: Adapting Large Language Models for N'Ko},
  author={Diomande, Mohamed},
  year={2026},
  url={https://github.com/Diomandeee/nko-brain-scanner}
}
```

## Links

- **Paper**: [ACL/EMNLP 2026 submission](https://github.com/Diomandeee/nko-brain-scanner/tree/main/paper)
- **Code**: [github.com/Diomandeee/nko-brain-scanner](https://github.com/Diomandeee/nko-brain-scanner)
- **Blog**: [diomandeee.github.io/nko-brain-scanner](https://diomandeee.github.io/nko-brain-scanner/)
- **N'Ko Wikipedia**: [nqo.wikipedia.org](https://nqo.wikipedia.org)
