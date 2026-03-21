# Experiment E: EPOCH Inscription, Blockchain Knowledge Provenance

## Research Question

Can the full derivation chain (conversation -> SFT curation -> N'Ko translation ->
sigil compression -> blockchain inscription) create a verifiable, culturally-rooted
knowledge provenance record?

## Background

EPOCH is a Clarity smart contract protocol on Stacks blockchain (see ~/Desktop/crypto-gpu-fund/).
It provides on-chain storage for compute contributions, time logs, and agent registries.
This experiment extends EPOCH to inscribe knowledge artifacts with N'Ko-encoded provenance.

## The 5-Layer Derivation Chain

Each conversation turn passes through 5 transformations. Each layer produces an artifact
and a hash. The final inscription carries the complete provenance chain.

```
Layer 0: Raw Conversation
    -> Original English/Bambara text
    -> SHA-256 hash of raw text

Layer 1: SFT Curation
    -> Structured as {"messages": [{"role": ..., "content": ...}]}
    -> SHA-256 hash of curated JSONL line

Layer 2: N'Ko Translation
    -> Transliterated via IPA bridge
    -> SHA-256 hash of N'Ko text
    -> Confidence score from transliterator

Layer 3: Sigil Compression
    -> Reduced to 1-5 sigil characters
    -> SHA-256 hash of sigil sequence
    -> Compression ratio

Layer 4: EPOCH Inscription
    -> On-chain record with all 4 prior hashes
    -> STX transaction ID
    -> Block height and timestamp
```

## Hypothesis

On-chain inscription with N'Ko-encoded provenance creates a verifiable knowledge record that:
1. Is immutable (blockchain guarantees)
2. Is culturally meaningful (N'Ko script preserves the origin language)
3. Is compressed (sigil encoding reduces storage cost)
4. Has full provenance (every transformation is hash-linked)

## Scripts

- `derivation_chain.py`: Build the full 5-layer chain for a single turn
- `inscription_format.md`: Specification for the on-chain inscription format

## Prerequisites

- Experiments C and D must be completed (translation + sigil encoding)
- STX wallet with testnet/mainnet tokens
- Access to Stacks blockchain (testnet or mainnet)

## Running

```bash
# Build derivation chain (does NOT inscribe yet)
python3 derivation_chain.py \
    --input sample_turn.json \
    --output results/derivation_chain.json

# To actually inscribe (requires STX):
# python3 derivation_chain.py --input sample_turn.json --inscribe --network testnet
```

## Success Criteria

- Complete derivation chain with all 5 layers and 4 intermediate hashes
- Inscription payload fits within Stacks memo field (34 bytes) or uses contract storage
- Round-trip verification: given an inscription ID, reconstruct and verify all hashes
