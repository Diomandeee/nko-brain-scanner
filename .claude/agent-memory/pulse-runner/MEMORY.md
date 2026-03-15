# Pulse Runner Memory — nko-brain-scanner

## Project: NKo Brain Scanner
- **Path**: ~/Desktop/nko-brain-scanner/
- **Purpose**: N'Ko ASR pipeline and language model fine-tuning

## Key Data Sources for N'Ko
- **OLDI-Seed (HuggingFace)**: `openlanguagedata/oldi_seed`, filter `iso_639_3 == 'nqo'` -> 6,193 N'Ko Wikipedia entries
- **parallel_corpus.jsonl**: 460 English-N'Ko pairs (data/)
- **bayelemabaga/pilot_pairs.jsonl**: 500 Bambara-French-N'Ko triples (data/)
- **bayelemabaga/pilot_sft.jsonl**: 1,001 pre-formatted SFT examples (data/)
- **nko_wikipedia_corpus.jsonl**: 1,693 articles with N'Ko text (data/)
- **LearnNKo ML data**: ~/projects/LearnNKo/ml/data/ (8,817 Bambara-French pairs, 17,648 training examples)
- **NKo library**: ~/Desktop/NKo/nko/data/ (corpus, cultural, keyboard vocab)
- **Cross-script bridge**: ~/Desktop/NKo (nko.transliterate) converts Latin Bambara -> N'Ko script

## Data Format
- SFT format: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
- V2 combined: training/combined_train_v2.jsonl (33,912 examples)
- V3 combined: data/sft_v3_combined.jsonl (92,184 examples)

## Data Acquisition Gotchas
- **masakhane/nicolingua-nko**: Does NOT exist on HuggingFace
- **OPUS nqo**: Only wikimedia has en-nqo (80 pairs, too small to be useful)
- **WMT23 nko**: statmt.org path returns 404
- **facebook/flores**: Uses deprecated dataset scripts, cannot load with latest HF datasets lib
- **Gated datasets**: sudoping01/nko-tts, nko-asr require access approval
- **HF datasets lib**: `trust_remote_code` no longer supported, script-based datasets fail
- **opustools install**: Requires --break-system-packages on macOS with system Python 3.14

## N'Ko Unicode
- Range: U+07C0 to U+07FF
- Script detection: `any(0x07C0 <= ord(ch) <= 0x07FF for ch in text)`
- Sentence separators: `.` and `߸` (N'Ko comma)

## State Management
- Divergent rail state: ~/.claude/state/divergent_rail_state.json
- Phase 1 critical track: COMPLETE (2026-03-15)
