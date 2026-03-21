# N'Ko Research Experiments

This is the working index for the N'Ko brain scanner research project. The project started with a single question: what happens inside a language model when it processes a script it was never trained to read? That question opened into several more, each one running as a separate experiment.

The posts below are organized in two sections: published writing and active experiments. The experiments are in progress. Results pages will be updated as data comes in.

---

## Active Experiments

### [Does Every AI Have the Same Blind Spot?](experiment-a-cross-model-brain-scan.md)
**Experiment A: Cross-Model Brain Scan**

The original brain scan found that Qwen3-8B processes N'Ko with measurably less activation than English at every layer. This experiment runs the same scan on four architecturally different models to find out whether that deficit is universal or specific to one model family.

---

### [Does Script Design Affect How Machines Hear?](experiment-b-script-advantage.md)
**Experiment B: CTC Script Advantage**

N'Ko has a 1:1 phoneme-to-character mapping. Latin Bambara has digraphs, ambiguous mappings, and no tone marking. Everything else held equal: same audio, same encoder, same model architecture. This experiment asks whether the cleaner script produces fewer ASR errors.

---

### [What If Your AI Twin Thought in Your Mother Tongue?](experiment-c-nko-cognitive-twin.md)
**Experiment C: N'Ko Cognitive Twin**

A cognitive twin is a language model fine-tuned on one person's conversation history. The existing twin was trained in English. This experiment retrains it on the same conversations transliterated to N'Ko, then compares the two twins on token compression, perplexity, and behavioral divergence.

---

### [Can 10 Characters Encode a Lifetime of Conversation?](experiment-d-sigil-compression.md)
**Experiment D: Sigil Compression**

Ten N'Ko characters, each mapped to a semantic pattern (stabilization, transition, novelty, recovery, and six others), form a compression layer for conversational knowledge. This experiment measures how much information survives when a 500-token conversation turn is reduced to a 3-character sigil sequence.

---

### [Putting Knowledge On-Chain in the Script It Was Meant For](experiment-e-epoch-inscription.md)
**Experiment E: EPOCH Inscription**

A five-layer pipeline takes a raw conversation turn, curates it as SFT data, transliterates it to N'Ko, compresses it to sigils, then inscribes the full derivation chain on the Stacks blockchain. Each step produces a hash. The final inscription is immutable, culturally encoded, compressed, and fully traceable.

---

## Published Writing

### [The Script That Machines Can't Read](post.md)
The original brain scan post. Qwen3-8B processes N'Ko text with significantly higher sparsity, lower entropy, and flatter kurtosis than English at every layer. This post explains the methodology, the findings, and what they mean for language model development.

---

### [Dead Circuits and Living Speech: Building the First N'Ko AI Pipeline](technical-deep-dive.md)
The technical deep dive. Two papers in preparation. All code and models open-source. Covers the full pipeline from data collection to model training to evaluation.

---

### [The Script That Was Built for AI (And That AI Has Never Heard Of)](blog-nko-asr-pipeline.md)
The ASR pipeline post. How we built a speech recognition system for N'Ko starting from near-zero training data, a retrieval-centric architecture, and a 3,024-entry codebook derived from activation analysis.

---

### [When AI Can't See Your Language](blog-nko-dead-circuits-accessible.md)
The accessible version of the brain scan findings. Written for general audiences. Covers the translation tax, the three failure zones, and what it means that a 151,936-word vocabulary contains only 32 N'Ko entries.

---

## About This Project

The N'Ko brain scanner project is a research effort to understand and address AI's treatment of low-resource scripts. N'Ko is the test case because it's an unusually well-designed writing system, phonetically precise, consistent, and purpose-built for the languages it represents, that has been almost entirely excluded from the training data of major language models.

The experiments above are not just about N'Ko. They're about what happens to any language community when the dominant AI systems are trained predominantly on other languages, and what the path toward fixing that looks like.

All code is open-source. All data (where legally shareable) is published. All results, positive and negative, get written up here.
