# EPOCH Inscription Format Specification

## On-Chain Record Structure

### Compact Form (for memo fields, max 34 bytes)

```
NKP1|<sigils>|<hash-prefix-16>|<confidence>
```

Example:
```
NKP1|ߛߡ|a3f8b2c1d9e7f456|0.95
```

Fields:
- `NKP1`: Protocol identifier ("N'Ko Knowledge Provenance v1")
- `<sigils>`: 1-5 N'Ko sigil characters (UTF-8, 3 bytes each)
- `<hash-prefix-16>`: First 16 hex chars of provenance hash
- `<confidence>`: Transliteration confidence (2 decimal places)

Maximum size: 4 + 1 + 15 + 1 + 16 + 1 + 4 = 42 bytes
(Exceeds 34-byte memo. Use contract storage instead.)

### Full Form (for contract storage)

```json
{
    "version": 1,
    "type": "knowledge-provenance",
    "sigils": "ߛߡ",
    "provenance_hash": "a3f8b2c1d9e7f456...",
    "layer_hashes": [
        "hash_of_raw_text",
        "hash_of_sft_curation",
        "hash_of_nko_translation",
        "hash_of_sigil_compression"
    ],
    "confidence": 0.95,
    "timestamp": 1711234567
}
```

### Clarity Contract Function

```clarity
(define-public (inscribe-knowledge
    (sigils (string-utf8 20))
    (provenance-hash (buff 32))
    (confidence uint)
    (layer-count uint))
  (begin
    (map-set knowledge-records
      { inscriber: tx-sender, id: (var-get next-id) }
      {
        sigils: sigils,
        provenance-hash: provenance-hash,
        confidence: confidence,
        layer-count: layer-count,
        block-height: block-height,
        timestamp: (unwrap-panic (get-block-info? time (- block-height u1)))
      })
    (var-set next-id (+ (var-get next-id) u1))
    (ok (var-get next-id))))
```

### Hash Algorithm

All hashes use SHA-256. The provenance hash is computed as:

```
provenance = SHA-256(hash_layer_0 | hash_layer_1 | hash_layer_2 | hash_layer_3)
```

where `|` is string concatenation of hex-encoded hashes separated by `|`.

### Verification

Given an inscription ID:
1. Read on-chain record (sigils + provenance hash)
2. Obtain the original text from off-chain storage
3. Rebuild the derivation chain locally
4. Compare computed provenance hash with on-chain record
5. Match = provenance verified. Mismatch = tampering detected.

### Gas Cost Estimate

- STX transfer with memo: ~0.001 STX ($0.001)
- Contract call with storage: ~0.01-0.05 STX ($0.01-0.05)
- Batch of 100 inscriptions: ~1-5 STX ($1-5)
