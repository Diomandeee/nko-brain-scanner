# Experiment D: Sigil Compression and Knowledge Encoding

## Research Question

Can N'Ko sigils compress conversational knowledge more efficiently than standard
BPE tokenization? How much information survives the compression?

## Background: The 10 Sigils

The N'Ko brain scanner project defines 10 "sound sigils," each an N'Ko character
mapped to a semantic category derived from activation pattern analysis:

| Sigil | Character | Name | Meaning |
|-------|-----------|------|---------|
| 1 | ߛ | stabilization | Dispersion decreased |
| 2 | ߜ | dispersion | Spread increased |
| 3 | ߕ | transition | Change point |
| 4 | ߙ | return | Re-entry to basin |
| 5 | ߡ | dwell | Sustained stay |
| 6 | ߚ | oscillation | Rapid alternation |
| 7 | ߞ | recovery | Return latency |
| 8 | ߣ | novelty | New basin |
| 9 | ߠ | place_shift | Location change |
| 10 | ߥ | echo | Pattern match |

These sigils represent dynamic patterns, not static concepts. The hypothesis is that
any conversation turn can be characterized by a small sequence of these patterns.

## Hypothesis

Sigil encoding achieves 50-100x compression over raw BPE tokenization because it maps
to semantic dynamics rather than lexical tokens. A 500-token conversation turn can be
encoded as 3-5 sigils that capture its information-theoretic shape.

## Method

1. **Sample**: Extract 1000 conversation turns from Supabase (or from local JSONL)
2. **Tokenize**: Count BPE tokens using GPT-4's tokenizer (tiktoken, cl100k_base)
3. **Transliterate**: Convert each turn to N'Ko via the transliteration bridge
4. **Extract concepts**: Pull top-k keywords/concepts from each turn
5. **Assign sigils**: Map each turn to a sigil sequence based on:
   - Content stability (ߛ) vs content change (ߕ)
   - Topic repetition (ߥ) vs novelty (ߣ)
   - Sustained focus (ߡ) vs rapid switching (ߚ)
   - Recovery from confusion (ߞ) vs dispersion (ߜ)
6. **Measure compression**:
   - English BPE token count
   - N'Ko character count (after transliteration)
   - Sigil sequence length
   - Compute ratios

## Scripts

- `sigil_encoder.py`: Full pipeline from text to sigil sequence
- `measure_compression.py`: Compute and visualize compression ratios

## Running

```bash
# Encode and measure
python3 sigil_encoder.py \
    --input conversations.jsonl \
    --output results/sigil_encoded.jsonl

# Analyze compression
python3 measure_compression.py \
    --input results/sigil_encoded.jsonl \
    --output results/compression_stats.json
```

## Expected Outputs

- `results/sigil_encoded.jsonl`: Each turn with original text, N'Ko transliteration, sigil sequence
- `results/compression_stats.json`: Compression ratios, distributions, summary table
- Compression ratio table printed to stdout

## Success Criteria

- Sigil encoding produces 50-100x compression over BPE
- Sigil sequences are consistent (same topic produces similar sigils)
- Information loss is bounded (reconstructibility analysis)
