# Putting Knowledge On-Chain in the Script It Was Meant For

*A blockchain provenance system for AI knowledge artifacts, encoded in N'Ko.*

---

## The Problem with Knowledge Without Provenance

Every time an AI system learns something from a conversation, that knowledge goes somewhere. It gets stored in a database, baked into fine-tuning data, surfaced later as a retrieved memory. The content persists. The provenance often doesn't.

Provenance is the chain of custody: where did this knowledge come from? What transformations did it pass through? Who verified it? When was it produced? These questions matter for the same reason they matter in any other domain where knowledge is reused. A medical finding cited without its source. A legal precedent referenced without the original case. A training example fed to a model without tracking which conversation it derived from.

For AI systems that learn from ongoing human interaction, provenance is a live problem. The cognitive twin trained on hundreds of conversations has no internal record of which specific exchanges shaped which behaviors. The ASR model fine-tuned on N'Ko audio has no permanent link between its weights and the original recordings. The system's knowledge is real, but it's floating, unanchored to anything verifiable.

Experiment E is an attempt to anchor it.

---

## EPOCH and the Derivation Chain

EPOCH is a Clarity smart contract protocol built on the Stacks blockchain. The original application was compute contribution tracking: when a machine contributes GPU time to a shared training job, the contribution gets recorded on-chain with a timestamp and a verifiable signature. The blockchain provides immutability. Once a record is written, it can't be altered.

Experiment E extends that infrastructure to knowledge artifacts.

The core concept is a five-layer derivation chain. Each conversation turn passes through five transformations in sequence. Each transformation produces an artifact, and each artifact gets hashed. The final on-chain inscription carries the complete chain: four intermediate hashes plus the original content hash.

Layer 0 is the raw conversation. The original English or Bambara text, exactly as it was produced. We take a SHA-256 hash of this raw text. This hash is the root of the provenance chain. If anyone later wants to verify that an on-chain record derives from a specific conversation, they can hash the conversation text and compare.

Layer 1 is SFT curation. The raw turn gets structured as a supervised fine-tuning example, a JSON object with a messages array and roles. This is the form the data takes before training. We hash the curated JSONL line. The transformation from raw text to curated SFT format involves editorial decisions: which turns are worth keeping, how to frame the assistant side, what context to include. The hash records the state after those decisions.

Layer 2 is N'Ko translation. The curated text passes through the transliteration bridge and becomes N'Ko. We hash the N'Ko text and record the transliterator's confidence score. This layer is where the knowledge artifact gains its cultural encoding. The N'Ko script is not a decoration on top of the data. It's a representation choice that carries information about linguistic origin.

Layer 3 is sigil compression. The N'Ko text gets passed to the sigil encoder from Experiment D and reduced to 1-5 sigil characters. We hash the sigil sequence and record the compression ratio. The sigils carry the semantic shape of the turn without the full content.

Layer 4 is the EPOCH inscription. All four prior hashes, plus the sigil sequence itself, get written to the Stacks blockchain. The transaction ID, block height, and block timestamp come back from the chain. These form the final anchor. The knowledge artifact now has a permanent, verifiable record.

---

## What the Inscription Actually Contains

The on-chain record has to fit within practical constraints. Stacks blockchain transactions have a memo field with a 34-byte limit. For shorter entries, the complete sigil sequence and the chain of hashes can be packed into the memo. For longer records, the inscription uses contract storage, where there's no hard byte limit but costs scale with storage used.

A complete derivation chain record looks like this in its minimal form:

```
{
  "raw_hash": "sha256:...",
  "sft_hash": "sha256:...",
  "nko_hash": "sha256:...",
  "sigil_hash": "sha256:...",
  "sigils": "ߛߕߣ",
  "nko_confidence": 0.94,
  "compression_ratio": 87.4
}
```

The sigils, in this case three characters representing stabilization, transition, and novelty, are the actual encoded knowledge artifact. The hashes are the provenance chain. The metadata fields describe the quality of the translation and compression.

Anyone with access to the original conversation can verify the chain. Hash the raw text, compare to raw_hash. Translate to N'Ko with the bridge, hash the result, compare to nko_hash. Run the sigil encoder, hash the output, compare to sigil_hash. If all four hashes match, the on-chain record authentically represents this conversation turn.

---

## The Dependencies

This experiment is downstream of the two that precede it. The N'Ko translation step (Experiment C) needs to produce a working transliteration bridge with reliable confidence scores. The sigil compression step (Experiment D) needs to produce consistent, reproducible sigil encodings. If either of those systems is producing noisy or inconsistent output, the provenance chain breaks down.

The blockchain layer is, in some ways, the simplest part. Stacks has a well-documented API. Clarity contracts are deterministic. Transaction submission and confirmation are straightforward. The hard parts are the upstream components.

There's also a cost consideration. Testnet inscription is free. Mainnet inscription costs STX, the Stacks native token. For a research project, testnet is sufficient to validate the system. But the long-term vision is mainnet inscription, where the records are truly permanent and not dependent on a testnet that might be reset.

---

## Why Any of This Is Worth Doing

The combination of properties this system creates is unusual. Immutable (the blockchain guarantees it). Culturally meaningful (the N'Ko script preserves the origin language and cultural context). Compressed (sigil encoding reduces what needs to be stored). Fully traceable (every transformation is hash-linked, so you can always go from inscription back to source).

No existing knowledge management system has all four properties simultaneously. Most have one or two. Databases are traceable but mutable. Files are compressed but not immutable. Blockchain records are immutable but typically not culturally meaningful in the way a language-specific encoding would be.

This experiment is testing whether a system with all four properties is practically buildable. Not just theoretically possible. Actually buildable, with the existing tools, on existing infrastructure.

The answer will come from running the pipeline end to end on a real conversation turn and checking whether the round-trip verification works.

---

## Results

Experiment in progress. Results will be published here when available.
