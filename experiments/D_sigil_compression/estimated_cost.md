# Experiment D: Cost Estimate

## Compute Requirements

| Task | Hardware | Time | Cost |
|------|----------|------|------|
| Sigil encoding (1000 turns) | Local (any machine) | ~2 min | $0 |
| Compression measurement | Local | ~30 sec | $0 |
| Visualization | Local | ~10 sec | $0 |

## Notes

- Entire experiment runs locally. No GPU needed.
- tiktoken (GPT-4 tokenizer) is a pip install, no API calls.
- N'Ko transliteration is local string processing.
- Sigil assignment is rule-based, no ML inference.

## Total: $0

This is the cheapest experiment. Pure computation, no external services.
