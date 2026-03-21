# Experiment B: CTC Script Advantage, Controlled Comparison

## Research Question

Does N'Ko's phonetic transparency give it a measurable advantage for speech recognition?

## Hypothesis

N'Ko CTC will achieve lower Character Error Rate (CER) than Latin Bambara CTC because:

1. N'Ko has a 1:1 phoneme-to-character mapping. Every sound maps to exactly one character.
   The CTC decoder only needs to learn 65 classes (64 N'Ko code points + blank).

2. Latin Bambara has digraphs (ny, ng, gb), ambiguous mappings (c can be /k/ or /s/),
   and no tone marking. The decoder needs ~40 classes but faces higher ambiguity per class.

3. The same audio input, decoded through a phonetically transparent alphabet, should produce
   fewer alignment errors than decoding through a phonetically opaque alphabet.

## Design

**Controlled variable**: Everything is identical except the output vocabulary.

| Component | N'Ko Version | Latin Version |
|-----------|-------------|---------------|
| Audio data | 37,306 Bambara/Manding segments | Same |
| Encoder | Whisper Large V3 (frozen) | Same |
| Architecture | CharASR V3 (46.9M params) | Same architecture, different head |
| Head dimensions | 1280 -> 768 -> 65 classes | 1280 -> 768 -> ~40 classes |
| Loss function | CTC | CTC |
| Training schedule | Cosine warmup, 50 epochs | Same |
| Test set | 10% holdout, aligned pairs | Same audio, Latin transcriptions |

The N'Ko transcriptions come from the existing training pipeline (Whisper output bridged to N'Ko).
The Latin transcriptions come from the same Whisper output in Latin Bambara.

## Scripts

- `train_nko_ctc.py`: Train the N'Ko CTC decoder. Outputs N'Ko characters.
- `train_latin_ctc.py`: Train the Latin CTC decoder. Outputs Latin Bambara characters.
- `compare_scripts.py`: Load both trained models, evaluate on aligned test set, produce CER/WER comparison.

## Running

```bash
# On Vast.ai A100:

# 1. Train N'Ko version
python3 train_nko_ctc.py \
    --features-dir /data/features/ \
    --pairs /data/pairs.jsonl \
    --epochs 50 \
    --checkpoint-dir checkpoints/nko/

# 2. Train Latin version
python3 train_latin_ctc.py \
    --features-dir /data/features/ \
    --pairs /data/pairs_latin.jsonl \
    --epochs 50 \
    --checkpoint-dir checkpoints/latin/

# 3. Compare
python3 compare_scripts.py \
    --nko-checkpoint checkpoints/nko/best.pt \
    --latin-checkpoint checkpoints/latin/best.pt \
    --test-pairs /data/test_pairs_aligned.jsonl \
    --features-dir /data/features/
```

## Expected Outputs

- `checkpoints/nko/best.pt`, `checkpoints/latin/best.pt`
- `results/comparison.json` with CER, WER, and per-sample breakdowns
- Convergence curves showing training dynamics for both scripts

## Success Criteria

- If N'Ko CER < Latin CER: phonetic transparency advantage confirmed
- If N'Ko CER ~ Latin CER: script design doesn't affect ASR (surprising)
- If N'Ko CER > Latin CER: Latin's smaller class count outweighs phonetic ambiguity
