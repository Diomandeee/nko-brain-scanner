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
  - retrieval-asr
datasets:
  - custom
  - nicolingua
pipeline_tag: text-generation
---

# NKo-Qwen3-8B-V3: N'Ko Script Adaptation with nicolingua Integration

A Qwen3-8B model adapted for N'Ko script processing through multi-stage training (CPT + SFT + BPE-aware + vocabulary extension + nicolingua integration), with admissibility-constrained decoding via a syllable FSM and a retrieval-centric multimodal ASR architecture.

## Key Results

| Metric | Base | V1 (3-Stage) | V2 (Extended) | V3 (nicolingua) |
|--------|------|-------------|---------------|-----------------|
| N'Ko Perplexity | 11.02 | **6.00** | --- | 62.89 |
| Translation Tax | 2.90x | **0.70x** | --- | 7.11x |
| Val Loss | 4.290 | --- | 3.506 | **3.275** |
| Training Examples | --- | 25,100 | 33,912 | **92,184** |
| LoRA Layers | --- | 8 | 8 | 8 |
| Syllable Validity | 89.8% | --- | 100% (FSM) | 99.8% / 100% (FSM) |
| Mode Collapse | --- | No | Yes (20/20) | **No (3/20)** |

**Note**: V3's higher N'Ko perplexity is an artifact of vocabulary extension. The extended model (152,192 tokens) tokenizes N'Ko differently than the base model (151,936 tokens), making PPL scores non-comparable across vocabulary sizes. V3's key contribution is fixing V2's mode collapse while maintaining the extended vocabulary's superior training loss (3.275 vs V1's 4.290).

## Models Included

### V1: Three-Stage Adapter (Base Vocabulary)
- **Training**: CPT (17,360) + SFT (21,240) + BPE-aware (25,100) = 25,100 total
- **Config**: LoRA rank 8, scale 20.0, top 8 layers, lr 1e-5/5e-6/3e-6
- **Result**: N'Ko PPL 11.02 -> 6.00, Translation Tax 2.90x -> 0.70x
- **Strength**: Higher generation diversity on base vocabulary

### V2: Extended Vocabulary Adapter
- **Base model**: Vocabulary extended from 151,936 to 152,192 tokens (+250 N'Ko BPE tokens)
- **Extension method**: Dequantize-extend-requantize embedding surgery
- **Training**: 33,912 examples, 2,000 iterations, LoRA rank 8, 8 layers
- **Result**: Val loss 3.506 (18.3% lower than V1)
- **Limitation**: Mode collapse on generation (repetitive output)

### V3: nicolingua-Expanded Adapter
- **Base model**: Same extended vocabulary model as V2
- **Training**: 92,184 examples (all V2 data + 32,792 nicolingua parallel segments)
- **Config**: 3,000 iterations, batch 1, grad_accum 8, lr 1e-5, max_seq_len 384
- **Best checkpoint**: Iter 1500 (val loss 3.275, 6.6% lower than V2's 3.506)
- **Result**: Mode collapse fixed (3/20 degenerate vs V2's 20/20). Unconstrained syllable validity 99.8%. Avg Distinct-1 0.455, Distinct-2 0.621.
- **Tradeoff**: Extended vocabulary tokenization changes N'Ko token distributions, making PPL non-comparable with V1 (base vocab). V3 N'Ko PPL on frozen eval = 62.89 vs V1's 6.00, but this reflects tokenization differences, not generation quality.

## Constrained Decoding

4-state finite-state machine (FSM) enforces N'Ko CV/CVN syllable structure during generation:

```python
from constrained.logits_processor import NKoAdmissibilityProcessor

processor = NKoAdmissibilityProcessor(tokenizer)
logits = processor(tokens, logits)  # Masks inadmissible tokens to -inf
```

FSM states: START -> ONSET (consonant) -> NUCLEUS (vowel) -> CODA (nasal)

Results on 50 N'Ko prompts (V3 fused model):
- Unconstrained: 99.8% valid syllables, 9.7 tok/s
- FSM-Constrained: **100%** valid syllables, 5.9 tok/s

V1 base model comparison (50 prompts):
- Unconstrained: 89.8% valid syllables, 9.4 tok/s
- FSM-Constrained: **100%** valid syllables, 5.4 tok/s

## N'Ko BPE Tokenizer

512-merge tokenizer trained on N'Ko Wikipedia (62,035 word occurrences):
- **Compression**: 2.75x (reduces token gap from 4x to 1.45x)
- **Linguistically valid**: Top merges correspond to Manding grammatical particles
- **Vocab**: 614 tokens (64 base + 512 merges + 32 morpheme + 6 special)

Morpheme-aware variant: 158 effective merges, 206-token vocabulary, 0.941 morpheme boundary preservation (+5.6pp over standard BPE).

## Retrieval-Centric ASR Architecture

Multimodal pipeline for N'Ko speech recognition:
1. **Audio Encoder**: Frozen Whisper features
2. **Scene Encoder**: SigLIP visual context (d=512)
3. **Joint Embedding Space**: Contrastive + retrieval loss training
4. **Syllable Retriever**: 3,024-entry codebook + FSM-constrained beam search

100% round-trip accuracy on synthetic embeddings. Designed for low-resource settings where transcribed speech data is scarce.

## Brain Scan

Per-layer activation profiling reveals:
- **Layers 0-27 (Frozen)**: Zero activation change under LoRA
- **Layers 28-34 (Adaptation)**: Reduced L2 norms (-38 to -104), more efficient encoding
- **Layer 35 (Output)**: +573 L2 increase, sharper N'Ko predictions

## Training Hardware

- **Device**: Apple M4, 16GB unified memory
- **Framework**: MLX v0.29 with mlx_lm
- **Total time**: ~6 hours (all stages across V1-V3)
- **Cloud cost**: $1.72 (initial 72B brain scan)

## Limitations

- V2 model exhibits mode collapse during extended N'Ko generation; V3 targets this with 2.7x more training data
- Evaluated on perplexity and token accuracy only (no task-level N'Ko benchmarks exist)
- No human evaluation conducted
- ASR architecture validated with synthetic data only; real audio evaluation pending

## Usage

```python
from mlx_lm import load, generate

# Load fused model
model, tokenizer = load("Diomande/nko-qwen3-8b-v3")
response = generate(model, tokenizer, prompt="ߒߞߏ ߦߋ߫ ߡߎ߲߬", max_tokens=200)
```

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
