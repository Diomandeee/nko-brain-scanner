# Can 10 Characters Encode a Lifetime of Conversation?

*Experiment D: Sigil Compression*

Solomana Kante designed N'Ko with 27 base characters. We found 10 of them that matter more than the rest.

When we ran the brain scan on a language model processing N'Ko, certain character patterns activated specific semantic circuits more strongly than others. Not all characters are equal. Some carry more meaning per activation than their neighbors. We call these the 10 sigils: semantic anchor points in the N'Ko character space.

The question this experiment asks: can you use those 10 sigils to compress arbitrary text into a fixed, culturally grounded encoding? And if so, how much information do you lose?

## The Setup

We took 460 entries from our Bambara-English parallel corpus. Proverbs, cultural knowledge, everyday sentences. Each one exists in English, IPA, and N'Ko.

For each entry, we ran it through three encoding layers:

Layer 1: English text, tokenized with a standard BPE tokenizer (the same kind GPT-4 uses). This gives us the baseline token count.

Layer 2: N'Ko transliteration. Every English word gets mapped through IPA to its N'Ko equivalent. This expands the character count because N'Ko is character-level while BPE is subword-level.

Layer 3: Sigil compression. Each N'Ko sequence gets mapped to its nearest sigil anchor based on the phonetic and semantic properties of the characters. The 10 sigils act like categories: every piece of text gets assigned to the sigil that best represents its semantic content.

## The Results

| Layer | Total Count | Avg per Turn | Compression vs Sigil |
|-------|-------------|-------------|---------------------|
| English BPE | 2,410 tokens | 5.2 | 2.8x larger |
| N'Ko characters | 8,854 chars | 19.2 | 10.2x larger |
| Sigils | 865 units | 1.9 | 1.0x (baseline) |

The 10 sigils compress 460 conversation turns into 865 sigil units. That is 2.8 times more compact than BPE tokenization. Each turn averages 1.9 sigils.

The sigil frequency distribution is not uniform:

- **Novelty**: 52.6% of all assignments. Most text introduces new concepts.
- **Recovery**: 29.9%. A significant portion of conversation is about returning to known ground.
- **Echo**: 15.4%. Repetition and reinforcement.
- **Transition**: 2.0%. Rare but structurally important.
- **Stabilization**: 0.1%. Almost never triggered.

The heavy skew toward novelty and recovery tells us something about the corpus. Proverbs and cultural knowledge are mostly about introducing wisdom (novelty) or calling back to tradition (recovery). A conversation corpus would likely show a different distribution, with more transition and echo.

## What This Means

2.8x compression over BPE is meaningful. It means that if you wanted to inscribe your conversation history on a blockchain, or store it in a memory system with strict size constraints, sigil encoding gives you nearly 3x more content per unit of storage.

But compression ratio is only one dimension. The more interesting question is reconstructibility: can you go backwards from sigils to meaning? A ZIP file compresses text too, but the compressed form carries no semantic information. Sigils are different. Each one has a name and a meaning. "Novelty" tells you something that "0x7F3A" does not.

The next experiment (E) takes these sigil encodings and inscribes them on the Stacks blockchain with full provenance hashing. Each inscription links back through the derivation chain to the original text. The compression happens in N'Ko. The provenance happens on-chain. The meaning travels with the data.

## Reproducibility

All code and data for this experiment are at `experiments/D_sigil_compression/` in the repository. The sigil encoder runs locally with zero compute cost. The 460-entry parallel corpus is included.

Total cost: $0.
