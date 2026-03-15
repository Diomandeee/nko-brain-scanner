# The Script That Machines Can't Read

**Adapting Large Language Models for N'Ko**

*Mohamed Diomande, March 2026*

---

N'Ko is an alphabetic script used by over 40 million Manding-language speakers across West Africa. It has a Unicode block (U+07C0-U+07FF), a Wikipedia with thousands of articles, and a vibrant literary tradition. But when you feed N'Ko text to state-of-the-art language models, they choke. Not subtly. Catastrophically.

This is the story of how I taught a language model to read N'Ko, on a laptop, for less than $2.

## The Problem

When I ran Qwen3-8B on N'Ko text, the model produced garbled output. Its N'Ko perplexity was 11.02 compared to 3.8 for English. That's a 2.90x "translation tax," meaning the model found N'Ko nearly three times harder to predict than English. N'Ko token accuracy sat at 23%, and about 10% of generated syllables were phonotactically invalid.

The root cause: N'Ko occupies just 64 codepoints in Unicode. During pre-training, these 64 characters compete with tens of thousands of CJK, Latin, and Cyrillic tokens for model capacity. The model learns to represent N'Ko, barely, through its general multilingual abilities, but it never develops deep understanding.

## Brain Scanning

Before fine-tuning, I wanted to understand what was happening inside the model. I built an "activation profiler" (we call it a brain scanner) that records the hidden state norms at every transformer layer while the model processes English vs. N'Ko text.

The brain scan revealed something interesting:

- **Layers 0-27**: N'Ko activations were systematically lower than English across all frozen layers. The model was processing N'Ko with less "confidence" at every stage.
- **Layer 35 (output)**: N'Ko activations were lower than English, confirming the model's uncertainty about N'Ko predictions.

This gave us a clear target: we needed to boost N'Ko representations specifically in the upper layers where language-specific processing happens.

## Three-Stage Training Pipeline

I designed a three-stage LoRA fine-tuning pipeline, all running on an Apple M4 with 16GB of unified memory:

**Stage 1: Continued Pre-Training (CPT)**
17,360 examples from N'Ko Wikipedia, using a 300-character sliding window. This teaches the model the basic statistical patterns of N'Ko text. Learning rate: 1e-5, 2000 iterations.

**Stage 2: Supervised Fine-Tuning (SFT)**
21,240 examples covering cultural knowledge, grammar explanations, script teaching, and translation pairs. This teaches the model to follow N'Ko-specific instructions. Learning rate: 5e-6, 1000 iterations.

**Stage 3: BPE-Aware Training**
25,100 examples where the model practices completing text at subword merge boundaries and word boundaries. This teaches alignment between the tokenizer's byte-level decisions and linguistic structure. Learning rate: 3e-6, 1000 iterations.

Total training time: about 2 hours on a consumer laptop.

## The Brain Scan After Training

Running the brain scanner again after fine-tuning revealed the adaptation pattern:

- **Layers 0-27**: Zero change. LoRA was applied to the top 8 layers only, and these frozen layers showed identical activations.
- **Layers 28-34**: *Reduced* L2 norms (decreases of -38 to -104). The model learned more *efficient* N'Ko encoding, using less activation energy to represent the same information.
- **Layer 35 (output)**: A dramatic +573 increase in L2 norm. The model became much more confident in its N'Ko predictions.

The translation tax dropped from 2.90x to 0.70x. After fine-tuning, the model found N'Ko *easier* to predict than English.

## Building a Script-Specific Tokenizer

Standard BPE tokenizers waste capacity on N'Ko. Qwen3's tokenizer produces about 4x more tokens for the same N'Ko content compared to English. I trained a 512-merge BPE tokenizer specifically for N'Ko on 62,035 word occurrences from Wikipedia.

The results were linguistically interesting. The top merges naturally discovered Manding grammatical particles: postpositions, conjunctions, and common morphemes. The tokenizer achieved 2.75x compression, reducing the token gap from 4x to 1.45x.

I also built a morpheme-constrained variant that prevents merges from crossing morpheme boundaries. This achieved 0.941 morpheme boundary preservation vs. 0.891 for standard BPE, at the cost of some compression.

## Vocabulary Extension and V3

To integrate the N'Ko BPE tokens into the model, I performed vocabulary extension surgery: dequantizing the embedding matrices, extending from 151,936 to 152,192 tokens, then requantizing. This gave the model 256 new N'Ko-specific tokens.

I then trained a V3 adapter on 92,184 examples, incorporating 32,792 parallel segments from the nicolingua corpus (WMT 2023 N'Ko-French-English parallel data). V3 achieved the lowest validation loss (3.275) and fixed the mode collapse problem that plagued V2, where the model would generate repetitive output. Only 3 of 20 test prompts produced degenerate responses, compared to 20/20 for V2.

## Constrained Decoding

Even with fine-tuning, the model occasionally produces N'Ko syllables that violate the language's phonotactic rules. N'Ko follows a strict CV/CVN (consonant-vowel, consonant-vowel-nasal) syllable structure.

I implemented a 4-state finite-state machine as a logits processor:

```
START -> ONSET (consonant) -> NUCLEUS (vowel) -> CODA (nasal) -> back to START
```

The FSM masks inadmissible tokens to negative infinity at each generation step. This guarantees 100% valid syllable structure with about 40% throughput reduction (from 9.7 to 5.9 tokens/sec).

With V3, the model already produces 99.8% valid syllables without the constraint, so the FSM acts as a safety net rather than a crutch.

## ASR Architecture

I also designed a retrieval-centric ASR architecture for N'Ko speech recognition. Instead of the standard end-to-end approach (which requires large amounts of transcribed speech data that doesn't exist for N'Ko), the system:

1. Extracts frozen Whisper audio features
2. Extracts SigLIP visual context from video frames
3. Projects both into a shared d=512 embedding space
4. Retrieves N'Ko syllables from a 3,024-entry codebook
5. Applies FSM-constrained beam search for phonotactically valid output

This achieves 100% round-trip accuracy on synthetic embeddings. Real audio evaluation is pending the collection of transcribed N'Ko speech data.

## What I Learned

**Data diversity matters more than data volume.** V2 had 33,912 examples and mode-collapsed. V3 had 92,184 examples from diverse sources and worked. The nicolingua parallel corpus was key.

**Vocabulary extension has tradeoffs.** The extended vocabulary improved training loss but changed tokenization enough to make perplexity scores non-comparable with the base model. This is an honest limitation, not a failure.

**Consumer hardware is sufficient.** All training ran on an M4 Mac with 16GB RAM. The only cloud cost was $1.72 for an initial brain scan of the larger 72B model on Vast.ai.

**Brain scanning works.** Activation profiling gave actionable insight into where adaptation was happening and whether it was working. The "efficient encoding + confident output" pattern (reduced L2 in reasoning layers, increased L2 at output) is a clean signal.

## Reproducing This Work

Everything is open source: the training pipeline, evaluation scripts, tokenizer, constrained decoder, ASR architecture, and all results.

- **Code**: [github.com/Diomandeee/nko-brain-scanner](https://github.com/Diomandeee/nko-brain-scanner)
- **Model**: [huggingface.co/Diomande/nko-qwen3-8b-v3](https://huggingface.co/Diomande/nko-qwen3-8b-v3) *(upload pending)*
- **Paper**: [ACL/EMNLP 2026 submission](https://github.com/Diomandeee/nko-brain-scanner/tree/main/paper)

N'Ko has 40 million speakers and a rich literary tradition. It deserves first-class support from language models. This is a start.
