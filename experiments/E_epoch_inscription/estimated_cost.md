# EPOCH Inscription: Gas Cost Estimates

> Based on actual inscription payload sizes from 10-sample test run (2026-03-20).
> Source data: `data/parallel_corpus.jsonl` (N'Ko proverbs, 4-8 words each).

## Stacks Gas Model

Stacks transaction costs have two components:
1. **Execution cost**: CPU cycles for contract function execution
2. **Storage cost**: Bytes written to chain state (maps, data vars)

The `inscribe` function in `nko-inscription.clar` writes to:
- `inscriptions` map (main storage)
- `claim-counts` map (counter update)
- `basins` map (visit count increment, if basin exists)
- 3 data vars (`chain-head`, `total-inscriptions`, `rate-window-count`)
- 1 print event (for chainhook indexing)

## Per-Inscription Storage Breakdown

Based on the `inscriptions` map schema and actual test data:

| Field | Type | Fixed/Variable | Bytes |
|-------|------|----------------|-------|
| Key: `index` | uint | Fixed | 16 |
| `nko-text` | string-utf8 1024 | Variable | 43-74 (avg 54) |
| `inscription-hash` | buff 32 | Fixed | 32 |
| `prev-hash` | buff 32 | Fixed | 32 |
| `claim-type` | string-ascii 20 | Variable | 7-11 (avg 8) |
| `sigil` | string-utf8 4 | Variable | 3-15 (avg 4) |
| `confidence` | uint | Fixed | 16 |
| `stacks-block` | uint | Fixed | 16 |
| `lexicon-version` | uint | Fixed | 16 |
| `density` | uint | Fixed | 16 |
| `basin-id` | string-ascii 64 | Variable | 22 |
| `depth` | uint | Fixed | 16 |
| `inscriber` | principal | Fixed | 33 |
| **Total** | | | **~280 bytes avg** |

## Observed Payload Sizes (10-Sample Run)

```
Compact inscription (on-chain memo):
  Min:  29 bytes
  Max:  31 bytes
  Avg:  29.2 bytes

N'Ko text (UTF-8):
  Min:  43 bytes
  Max:  74 bytes
  Avg:  54.4 bytes

Full JSON payload (off-chain reference):
  Min:  864 bytes
  Max:  913 bytes
  Avg:  881.0 bytes
```

## Gas Estimates

### Contract Call (inscribe function)

Stacks contract calls have a base fee plus per-byte storage costs.
Current Stacks fee market (as of March 2026):

| Metric | Value |
|--------|-------|
| Base transaction fee | ~0.001 STX |
| Contract call overhead | ~0.005 STX |
| Storage write cost | ~0.00001 STX/byte |
| Map write overhead | ~0.002 STX per map operation |
| Print event cost | ~0.001 STX |

**Per inscription total estimate:**

```
Base fee:              0.001 STX
Contract call:         0.005 STX
Storage (280 bytes):   0.003 STX
Map writes (3 maps):   0.006 STX
Data var updates (3):  0.003 STX
Print event:           0.001 STX
-------------------------------
Total:                ~0.019 STX per inscription
```

Rounding up for fee market variance: **~0.02 STX per inscription**.

### Batch Estimates

| Batch Size | Est. Gas (STX) | Est. Cost (USD at $1/STX) |
|------------|----------------|---------------------------|
| 1 | 0.02 | $0.02 |
| 10 | 0.20 | $0.20 |
| 100 | 2.00 | $2.00 |
| 460 (full parallel corpus) | 9.20 | $9.20 |
| 1,000 | 20.00 | $20.00 |

### Rate Limit Constraint

The `nko-inscription.clar` contract enforces a rate limit of 100 inscriptions
per 144-block window (~24 hours). At max throughput:

```
100 inscriptions/day * 0.02 STX = 2.0 STX/day
Monthly (30 days): ~60 STX/month
Annual (365 days): ~730 STX/year
```

### Comparison: Memo vs Contract Storage

| Method | Max Size | Cost | Verifiable? |
|--------|----------|------|-------------|
| STX transfer with memo | 34 bytes | ~0.001 STX | Hash prefix only |
| Contract storage (inscribe) | 1024+ bytes | ~0.02 STX | Full chain + on-chain verify |
| Contract + off-chain JSON | Unlimited | ~0.02 STX + hosting | Full chain, payload off-chain |

The compact inscription (29 bytes avg) fits within the 34-byte memo field,
but only carries a hash prefix. The contract storage approach provides full
on-chain verification via the `verify-link` read-only function.

### Recommendation

For the parallel corpus (460 entries):
- **Contract storage**: ~9.20 STX total. Full on-chain provenance.
- **Memo-only**: ~0.46 STX total. Partial verification (hash prefix only).
- **Hybrid**: Inscribe sigil + provenance hash on-chain, store full payload in
  IPFS/Arweave with CID reference. Best of both worlds.

Given current treasury balance, contract storage for 460 entries is feasible
and provides the strongest provenance guarantee.

## Notes

- These estimates are for **testnet** validation. Mainnet fees may be 2-5x higher
  during congestion periods.
- The `nko-inscription.clar` contract is already deployed to testnet (12/12
  contracts deployed per EPOCH CLAUDE.md).
- No actual blockchain interaction was performed for this estimate. All numbers
  are derived from payload size analysis and published Stacks fee schedules.
- Longer texts (full sentences vs. proverbs) will have larger N'Ko text fields,
  increasing per-inscription cost by ~0.001 STX per additional 100 UTF-8 bytes.
