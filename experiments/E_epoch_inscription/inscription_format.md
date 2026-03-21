# EPOCH Inscription Format Specification

> Updated 2026-03-20 from actual `derivation_chain.py` output (10 test inscriptions verified).

## The 5-Layer Derivation Chain

Each text passes through 5 transformation layers. Every layer produces content
and a SHA-256 hash. The final inscription carries the complete provenance chain.

```
Layer 0: Raw Text
    Input:  English text string
    Output: SHA-256(text)

Layer 1: Concept Extraction
    Input:  Raw text
    Method: TF-IDF keyword extraction (stop-word removal, frequency * log(word_length))
    Output: Pipe-delimited sorted keywords, e.g. "expect|impossible"
    Hash:   SHA-256(keyword_string)

Layer 2: N'Ko Transliteration
    Input:  Raw text
    Method: NkoTransliterator.convert(text, target="nko") via IPA bridge
    Output: N'Ko script text, e.g. "ߘߋߣʔߕ ߍߞߛߔߍߗߕ θߍ ߌߡߔߋߛߛߌߓߟߍ"
    Hash:   SHA-256(nko_text)
    Extra:  confidence score (0.0-1.0), IPA intermediate

Layer 3: Sigil Compression
    Input:  Raw text + optional previous text
    Method: extract_features() -> assign_sigils() from Experiment D
    Output: 1-5 N'Ko sigil characters, e.g. "ߣ" or "ߣߞ"
    Hash:   SHA-256(sigil_string)
    Extra:  compression_ratio (original_chars / sigil_count)

Layer 4: EPOCH Inscription Payload
    Input:  Layers 0-3
    Method: Assemble hash chain, compute provenance hash, build Clarity params
    Output: Full JSON payload + compact inscription string
    Hash:   SHA-256(canonical_payload_json)
```

## Hash Algorithm

All hashes use SHA-256 on UTF-8 encoded strings.

The **provenance hash** is computed as:

```
provenance_hash = SHA-256(hash_layer_0 | hash_layer_1 | hash_layer_2 | hash_layer_3)
```

where `|` is the literal pipe character separating hex-encoded hashes.

Example hash chain input:
```
5a27373a...608477|b4b74d45...d051f0|ee40d022...1b6912|1e855078...6cff68
```

## Output Format (Per Entry)

Each entry in the output JSONL file has this structure:

```json
{
  "input_text": "Don't expect the impossible",
  "layers": [
    {
      "layer": 0,
      "name": "raw_text",
      "content": "Don't expect the impossible",
      "hash": "5a27373ae051050e...",
      "char_count": 27,
      "word_count": 4
    },
    {
      "layer": 1,
      "name": "concept_extraction",
      "content": "expect|impossible",
      "hash": "b4b74d45d6908e05...",
      "keywords": ["impossible", "expect"],
      "keyword_count": 2
    },
    {
      "layer": 2,
      "name": "nko_transliteration",
      "content": "ߘߋߣʔߕ ߍߞߛߔߍߗߕ θߍ ߌߡߔߋߛߛߌߓߟߍ",
      "hash": "ee40d02219f78da0...",
      "ipa_intermediate": "donʔt ekspetʃt θe impossible",
      "confidence": 1.0,
      "nko_char_count": 27
    },
    {
      "layer": 3,
      "name": "sigil_compression",
      "content": "ߣ",
      "hash": "1e855078521d3e4b...",
      "sigil_count": 1,
      "compression_ratio": 27.0
    },
    {
      "layer": 4,
      "name": "epoch_inscription",
      "content": "{...full payload JSON...}",
      "hash": "a7acc136abb5d9dd...",
      "compact_inscription": "NKP1|ߣ|bf4c0b379c52c37d|1.00",
      "compact_bytes": 29,
      "full_payload": {
        "version": 1,
        "type": "knowledge-provenance",
        "sigils": "ߣ",
        "provenance_hash": "bf4c0b379c52c37d...",
        "layer_hashes": ["hash0", "hash1", "hash2", "hash3"],
        "hash_chain_input": "hash0|hash1|hash2|hash3",
        "confidence": 1.0,
        "timestamp": 1774056568,
        "keywords": ["impossible", "expect"],
        "nko_text": "ߘߋߣʔߕ ..."
      },
      "clarity_params": {
        "nko-text": "ߘߋߣʔߕ ...",
        "inscription-hash": "bf4c0b379c52c37d...",
        "claim-type": "novelty",
        "sigil": "ߣ",
        "confidence": 10000,
        "density": 1000,
        "basin-id": "knowledge-provenance",
        "depth": 4
      }
    }
  ],
  "summary": {
    "original_chars": 27,
    "original_words": 4,
    "keywords": ["impossible", "expect"],
    "nko_chars": 27,
    "sigil_count": 1,
    "sigils": "ߣ",
    "compression_ratio": 27.0,
    "confidence": 1.0,
    "provenance_hash": "bf4c0b379c52c37d...",
    "compact_inscription": "NKP1|ߣ|bf4c0b379c52c37d|1.00",
    "compact_bytes": 29,
    "clarity_claim_type": "novelty"
  }
}
```

## Compact Inscription Format

```
NKP1|<sigils>|<hash-prefix-16>|<confidence>
```

Fields:
- `NKP1`: Protocol identifier ("N'Ko Knowledge Provenance v1")
- `<sigils>`: 1-5 N'Ko sigil characters (UTF-8, 3 bytes each)
- `<hash-prefix-16>`: First 16 hex chars of the provenance hash
- `<confidence>`: Transliteration confidence (2 decimal places)

Observed sizes from 10-sample test run:
- Single sigil: 29 bytes (e.g., `NKP1|ߣ|bf4c0b379c52c37d|1.00`)
- Two sigils: 31 bytes (e.g., `NKP1|ߣߞ|7a43f8dc2a3c8366|1.00`)
- Average: 29.2 bytes

## Clarity Contract Parameters

The `clarity_params` field maps directly to the `inscribe` function in
`nko-inscription.clar`:

```clarity
(define-public (inscribe
    (nko-text (string-utf8 1024))       ;; N'Ko transliteration (Layer 2)
    (inscription-hash (buff 32))         ;; Provenance hash (Layer 4)
    (claim-type (string-ascii 20))       ;; Sigil claim type name
    (sigil (string-utf8 4))              ;; Sigil character(s) (Layer 3)
    (confidence uint)                    ;; Scaled 0-10000 (Layer 2)
    (density uint)                       ;; Bits/stroke scaled x1000
    (basin-id (string-ascii 64))         ;; "knowledge-provenance"
    (depth uint))                        ;; Number of layer hashes (4)
```

Claim type mapping (sigil name -> Clarity constant):
| Sigil | Char | Claim Type |
|-------|------|------------|
| stabilization | ߛ | stabilize |
| dispersion | ߜ | dispersion |
| transition | ߕ | transition |
| return | ߙ | return |
| dwell | ߡ | dwell |
| oscillation | ߚ | oscillation |
| recovery | ߞ | recovery |
| novelty | ߣ | novelty |
| place_shift | ߠ | place-shift |
| echo | ߥ | echo |

## Verification

Given an inscription JSONL file:

```bash
python3 verify_chain.py --input results/sample_inscriptions.jsonl --verbose
```

The verifier checks:
1. Each layer's hash matches SHA-256(content) for layers 0-3
2. Layer 4's `layer_hashes` array matches actual layer hashes
3. Provenance hash = SHA-256(hash0|hash1|hash2|hash3)
4. The `hash_chain_input` string matches the computed chain
5. Compact inscription hash prefix matches provenance hash[:16]
6. Layer 4 payload hash matches SHA-256(canonical JSON)

Result: `VALID` or `BROKEN (at: <list of failed checks>)`

## Test Results (2026-03-20)

10-sample run from `data/parallel_corpus.jsonl`:
- All 10 chains verified VALID
- Average compact inscription: 29.2 bytes
- Average full payload: 881 bytes
- Average compression ratio: 29.0x
- Average transliteration confidence: 0.97
- Claim type distribution: 100% novelty (short proverb texts)

## Gas Cost Estimate

See `estimated_cost.md` for detailed gas analysis based on actual payload sizes.
