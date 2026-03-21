# Experiment C: N'Ko Cognitive Twin

## Research Question

What happens when a cognitive twin trained on English conversation patterns is retrained
to operate in N'Ko? Does the representation compress? Does behavior change?

## Hypothesis

The N'Ko cognitive twin will produce:
1. **More compressed representations**: N'Ko's agglutinative morphology packs more meaning
   per character than English, so fewer tokens should encode equivalent semantics.
2. **Lower perplexity on Manding cultural content**: The script was designed for this domain.
3. **Measurable behavioral divergence**: The twin's personality should shift subtly when
   its internal representation changes scripts.

## Prerequisites

- Cognitive twin SFT training data (English) must exist on Mac5
- The transliteration bridge (`~/Desktop/NKo/nko/transliterate.py`) must be functional
- Base model (Qwen3-8B or similar) must be available

## Method

### Step 1: Translate SFT Data to N'Ko

Take the cognitive twin's English SFT JSONL (user/assistant message pairs).
For each pair:
- If the content is in Bambara/Manding Latin: transliterate to N'Ko using the bridge
- If the content is in English: phonetically transliterate to N'Ko (experimental)
- Preserve the SFT message structure

Output: `data/nko_sft.jsonl` with N'Ko versions of all training examples.

### Step 2: Train N'Ko LoRA Adapter

Train a LoRA adapter on the N'Ko SFT data using the same hyperparameters as the
English cognitive twin adapter. Same base model, same rank, same alpha.

### Step 3: Compare

Load both adapters (English and N'Ko) and run the same 50 evaluation prompts.
Measure:
- **Perplexity**: How confidently does each twin generate?
- **Token count**: How many tokens does each twin use for equivalent responses?
- **Response similarity**: How similar are the outputs? (BLEU, cosine similarity)
- **Behavioral evaluation**: Does the N'Ko twin exhibit different personality traits?

## Scripts

- `translate_sft_to_nko.py`: Convert SFT JSONL from English/Latin to N'Ko
- `compare_twins.py`: Evaluate both adapters side-by-side
- `eval_prompts.jsonl`: 50 evaluation prompts

## Running

```bash
# Step 1: Translate
python3 translate_sft_to_nko.py \
    --input ~/sft_training_data.jsonl \
    --output data/nko_sft.jsonl

# Step 2: Train (on Mac5 or Vast.ai)
python3 -m mlx_lm lora \
    --model mlx-community/Qwen3-8B-8bit \
    --data data/ \
    --train \
    --num-layers 8 \
    --iters 1000

# Step 3: Compare
python3 compare_twins.py \
    --model mlx-community/Qwen3-8B-8bit \
    --english-adapter ~/adapters-english/ \
    --nko-adapter ~/adapters-nko/ \
    --prompts eval_prompts.jsonl \
    --output results/twin_comparison.json
```

## Blocking Dependencies

This experiment is blocked by completion of the cognitive twin's English SFT training.
The N'Ko adapter must be trained on data that parallels the English adapter's training data
for the comparison to be valid.

## Success Criteria

- Token compression ratio > 1.2x (N'Ko uses fewer tokens than English for same content)
- N'Ko perplexity < English perplexity on Manding cultural prompts
- Measurable but not catastrophic behavioral divergence (the twin should still be recognizable)
