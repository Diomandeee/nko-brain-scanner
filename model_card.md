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
datasets:
  - custom
pipeline_tag: text-generation
---

# NKo-Qwen3-8B: Three-Stage Fine-Tuned Model for N'Ko Script

A Qwen3-8B model fine-tuned for N'Ko (ߒߞߏ) script processing through a three-stage pipeline: continued pre-training on N'Ko Wikipedia, supervised fine-tuning on instruction data, and BPE-aware subword training.

## Key Results

| Metric | Base | Fine-Tuned | Change |
|--------|------|------------|--------|
| N'Ko Perplexity | 11.02 | **6.00** | -45.6% |
| N'Ko Top-1 Accuracy | 43.2% | **56.7%** | +13.5pp |
| N'Ko Token Accuracy | 23.0% | **32.8%** | +9.8pp |
| English Top-1 Accuracy | 70.9% | 69.7% | -1.2pp |
| Translation Tax (NKo/Eng PPL) | 2.90x | **0.70x** | **-76%** |

The model processes N'Ko with lower perplexity than English after fine-tuning, while English accuracy drops by only 1.2 percentage points.

## Training

### Three-Stage Pipeline

| Stage | Data | Iterations | LR |
|-------|------|------------|-----|
| 1. Continued Pre-Training | 17,360 Wikipedia examples (3.7M N'Ko chars) | 2,000 | 1e-5 |
| 2. Supervised Fine-Tuning | 21,240 combined examples | 1,000 | 5e-6 |
| 3. BPE-Aware Training | 25,100 examples (incl. 3,860 BPE-focused) | 1,000 | 3e-6 |

### Configuration
- **Base model**: Qwen3-8B (mlx-community/Qwen3-8B-8bit)
- **Method**: LoRA (rank 8, scale 20.0, top 8 layers)
- **Hardware**: Apple M4, 16GB unified memory
- **Framework**: MLX v0.29 with mlx_lm
- **Total training time**: ~3 hours
- **Total cost**: $0 (consumer hardware)

## Brain Scan

We performed per-layer activation profiling comparing the base and fine-tuned models:

- **Layers 0-27 (Frozen)**: Zero activation change. LoRA only modifies top 8 layers.
- **Layers 28-34 (Adaptation Zone)**: Reduced L2 norms (-38 to -104). More efficient N'Ko encoding.
- **Layer 35 (Output)**: Massive increase (+573). Sharper, more confident predictions.

## N'Ko Script

N'Ko (ߒߞߏ) is an alphabetic writing system created in 1949 by Solomana Kante for the Manding language family. It is used by over 40 million speakers across Guinea, Mali, Cote d'Ivoire, and neighboring countries. Unicode block: U+07C0-U+07FF.

Key properties:
- 1:1 phoneme-to-character mapping (zero spelling exceptions)
- Explicit tonal diacritics
- Right-to-left writing direction
- 27 base characters + combining marks

## Limitations

- Cannot hold coherent conversations in N'Ko (generates more accurate tokens but still produces repetitive extended text)
- Tokenizer still uses character-level fallback for N'Ko (32 tokens in vocabulary, all single characters)
- Only evaluated on perplexity and token accuracy (no task-level benchmarks exist for N'Ko)
- No human evaluation of generation quality

## Usage

```python
from mlx_lm import load, generate

model, tokenizer = load("Diomandeee/nko-qwen3-8b")
response = generate(model, tokenizer, prompt="What is N'Ko script?", max_tokens=200)
print(response)
```

Or with the LoRA adapter on the base model:

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3-8B-8bit",
                        adapter_path="Diomandeee/nko-qwen3-8b-adapter")
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

- **Paper**: [Blog post with full narrative](https://diomandeee.github.io/nko-brain-scanner/)
- **Code**: [github.com/Diomandeee/nko-brain-scanner](https://github.com/Diomandeee/nko-brain-scanner)
- **N'Ko Wikipedia**: [nqo.wikipedia.org](https://nqo.wikipedia.org)
