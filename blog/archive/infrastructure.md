# 656,000 Lines of Code for a Script Most AI Has Never Seen

*Mohamed Diomande, March 2026*

---

Here's a number that hit me when I ran `wc -l` across the project: 656,000 lines. Python, Swift, TypeScript. All of it for a single writing system that most language models handle worse than random noise.

The script is N'Ko (ߒߞߏ). You've probably never heard of it. That's the whole problem.

---

## The Script

In 1949, a self-taught linguist named Solomana Kante designed N'Ko from scratch. He spoke Manding, a language family with 40 million speakers across West Africa. Bambara in Mali, Dioula in Cote d'Ivoire, Mandinka in The Gambia, Maninka in Guinea. Arabic script and Latin script both existed for these languages. Neither fit.

Arabic doesn't capture Manding's vowel distinctions. Latin doesn't encode its tonal system. The word "ba" means mother, goat, or river depending on tone. In Latin script, all three look identical. In N'Ko, each has a different mark.

Kante built something precise. 27 base characters, each mapping to exactly one sound. No silent letters. No irregular spellings. Explicit diacritical marks for all three tonal levels. It reads right-to-left. The name means "I say" in every Manding language.

75 years later, N'Ko has a Unicode block (U+07C0 to U+07FF), a Wikipedia with 1,693 articles, and a growing literary community. It also has almost zero coverage in any major language model.

When I fed N'Ko text to Qwen3-8B, perplexity was 11.02. English sat at 3.8. That 2.90x gap is what I call the translation tax. The model was working nearly three times as hard, and still getting 77% of tokens wrong.

So I decided to fix it.

---

## What We Built

The fix wasn't just a fine-tuned model. To do this right, you need infrastructure all the way down. Here's what 656,000 lines actually contains.

### Core NLP (the foundation everything depends on)

Before any model training, I needed solid linguistic machinery. All of this lives in `~/Desktop/NKo/nko/` and gets imported via `nko_core/__init__.py`.

**phonetics.py**: 7 vowels, 26 consonants, 5 tone marks, syllabification logic.

**transliterate.py**: Bidirectional bridges. N'Ko to Latin, N'Ko to Arabic, N'Ko to IPA, and back. The IPA layer is the pivot, so any pair is composable.

**morphology.py**: 1,346 lines. 28 morpheme types. Conjugator. Compound analyzer. This is the hardest part, because Manding morphology is agglutinative and the rules compound quickly.

All three modules are imported via `sys.path` in `__init__.py`. No thin wrapper files. If `from nko_core import phonetics` works, no separate file is needed.

### The iOS Keyboard

There's a full iOS keyboard with a prediction engine. The core intelligence layer is 42,000 lines. It handles right-to-left input, tone mark insertion, morpheme-aware prediction, and a ranking system that learns from user corrections.

Building a right-to-left keyboard in iOS SwiftUI is not obvious. The text engine fights you at every step. SwiftUI's default text direction assumptions are baked in deep.

### The Web Learning Platform

A Next.js app. 328 TypeScript files. Spaced repetition system (SRS), structured lessons, dictionary with audio, writing practice mode. This is how you teach the script to someone who has never seen it.

### The Video Extraction Pipeline

N'Ko has one major video resource: Djoko, a TV show with 959+ episodes and around 320 hours of audio. That's a real corpus, if you can extract it.

The pipeline goes: YouTube download, then frame extraction, then Gemini OCR on frames that contain N'Ko text, then 5 "world" generation variants. The world variants give you different pedagogical framings of the same source material. The OCR is noisy. Scene detection filters frames where the on-screen text is static vs. transitioning.

### The Cross-Script Bridge

A mapping layer between writing systems. N'Ko, Latin, Arabic, IPA, and phonetic romanization all share a common intermediate representation. This is what makes the morphological analyzer usable for speakers who only read Latin script, and what powers the ASR round-trip eval.

### The Brain Scanner

Before training anything, I wanted to see inside the model. I built `mlx_brain_scan_8b.py`: a `LayerCapture` wrapper around Qwen3-8B that records hidden state L2 norms at every transformer layer, per token, for N'Ko vs. English text.

The scan costs almost nothing on a Mac. For the 72B model (which I ran on a Vast.ai A100 for an hour), the total cloud bill was $1.72.

What the scan showed:

Layers 0 to 27: N'Ko activations were systematically lower than English at every frozen layer. The model was processing N'Ko with less signal throughout. Layer 35 (output): N'Ko activations were lower than English, confirming the model was uncertain about its N'Ko predictions.

After three-stage LoRA training:

Layers 0 to 27: Zero change. These layers were frozen, and they stayed frozen. Layers 28 to 34: Reduced L2 norms (decreases of -38 to -104). More efficient N'Ko encoding. Less activation energy for the same information. Layer 35: Up +573. The model got confident.

The translation tax went from 2.90x to 0.70x. After training, the model found N'Ko easier to predict than English.

### The Three-Stage LoRA Pipeline

The training runs entirely on an Apple M4 with 16GB unified memory.

Stage 1 is continued pre-training on 17,360 examples from N'Ko Wikipedia. 300-character sliding windows. 2,000 iterations. This teaches the base statistical texture of N'Ko.

Stage 2 is supervised fine-tuning on 21,240 examples: cultural content, grammar explanations, script teaching, translation pairs. 1,000 iterations.

Stage 3 is BPE-aware training on 25,100 examples where the model practices completing text at subword merge boundaries. 1,000 iterations. This aligns the tokenizer's decisions with linguistic structure.

V1 adapter: val loss 4.29, translation tax 0.70x. Clean.

V2: 33,912 examples, val loss 3.506, mode collapsed. Every N'Ko prompt produced degenerate repetition. 20 of 20 test prompts failed. Confirmed and documented.

V3: 92,184 examples from diverse sources, including 32,792 segments from the nicolingua corpus (WMT 2023 N'Ko-French-English parallel data). Val loss 3.275. Only 3 of 20 prompts showed degenerate output. Mode collapse effectively resolved.

The difference between V2 and V3 wasn't volume. It was diversity. More data from the same distribution doesn't help. Different data does.

### The Syllable Codebook and FSM

N'Ko syllable structure is strict: consonant-vowel, or consonant-vowel-nasal. That's it. No clusters, no complex codas.

I enumerated every phonotactically valid N'Ko syllable: 3,024 entries. That's the complete set. Mathematically exhausted.

The constrained decoder is a 4-state FSM: START, ONSET (consonant), NUCLEUS (vowel), CODA (nasal). At each generation step, it masks tokens that would produce an invalid syllable to negative infinity. Result: 100% valid syllable output. Cost: 40% throughput reduction (9.7 to 5.9 tokens per second).

With V3, the model generates 99.8% valid syllables without the constraint. The FSM is now a safety net.

### The Tokenizer Work

Qwen3's tokenizer produces about 4x more tokens for N'Ko text than for equivalent English. I trained a custom 512-merge BPE tokenizer on 62,035 N'Ko word occurrences from Wikipedia.

The top merges naturally discovered Manding grammatical particles. The tokenizer learned postpositions and conjunctions before it learned anything about specific words. That's correct behavior.

Compression achieved: 2.75x. Token gap closed: from 4x to 1.45x.

There's also a morpheme-constrained variant that prevents merges from crossing morpheme boundaries. It preserves 94.1% of morpheme boundaries vs. 89.1% for standard BPE. You pay a small compression penalty.

To wire the custom BPE into the model, I ran embedding surgery: dequantize the embedding matrices, extend from 151,936 to 152,192 tokens, requantize. The model now has 256 dedicated N'Ko tokens.

### The ASR Architecture

Transcribed N'Ko speech data almost doesn't exist. MALIBA-AI holds the state-of-the-art with a 45.73% word error rate, trained on what little exists. End-to-end seq2seq doesn't work when you have 10 hours of transcribed audio.

The architecture I built goes a different direction. It's retrieval-centric.

Frozen Whisper audio encoder extracts audio features. SigLIP extracts visual keyframe features from video. Both get projected into a shared 512-dimensional embedding space. At inference, the system retrieves N'Ko syllables from the 3,024-entry codebook and applies FSM-constrained beam search to assemble valid output.

The model has 4.5 million parameters. Smaller than a rounding error on GPT-4.

On synthetic round-trip evaluation: 100% accuracy. Real audio eval is waiting on more transcribed data.

---

## The Hardware

Everything above ran on consumer hardware. The M4 Mac with 16GB handles training, inference, tokenizer work, codebook generation, and the web platform. Mac5 (also M4, 16GB) runs the MLX server for the fused model.

The only cloud spend was $1.72 for one hour on a Vast.ai A100, which I needed to run the 72B brain scan. Everything else stayed local.

This is not a lab project. No GPU cluster. No research compute budget.

---

## The Brain Scan Discovery

The activation profiling result is the clearest finding in the project. The gap between N'Ko and English activations across all 36 layers of the model is not subtle. It's systematic, measurable, and it tells you exactly what fine-tuning needs to do.

The "efficient encoding plus confident output" pattern in the post-training scan is what healthy multilingual adaptation looks like. Lower L2 in the reasoning layers means the model isn't working as hard to process the script. Higher L2 at the output layer means it's more certain about its predictions.

This pattern was not something I expected going in. It emerged from the data and pointed directly at the mechanism.

---

## What's Next

V3 is the current adapter. It's good. Mode collapse is resolved. But the paper still needs the V3 results folded in, and the ASR system needs real audio to validate.

The model will go to HuggingFace when it generates coherent N'Ko text reliably across diverse prompts. Not before.

Human evaluation is the real gate. I need 3 to 5 N'Ko literate evaluators to tell me whether the output is actually correct, not just phonotactically valid.

---

## The Numbers, Flat

- 656,000 lines of code total
- 42,000 lines in the keyboard prediction engine
- 328 TypeScript files in the learning platform
- 1,346 lines in the morphological analyzer
- 3,024 syllable codebook entries (complete set)
- 92,184 training examples (V3)
- 3.7M characters of N'Ko Wikipedia corpus
- 2.90x translation tax before training
- 0.70x translation tax after V1
- 100% FSM syllable validity
- 4.5M parameter ASR model
- $1.72 cloud spend total

---

## Links

**Code**: [github.com/Diomandeee/nko-brain-scanner](https://github.com/Diomandeee/nko-brain-scanner)

**Model** (upload pending V3 human eval): [huggingface.co/Diomande/nko-qwen3-8b-v3](https://huggingface.co/Diomande/nko-qwen3-8b-v3)

**Paper** (ACL/EMNLP 2026 submission): [github.com/Diomandeee/nko-brain-scanner/tree/main/paper](https://github.com/Diomandeee/nko-brain-scanner/tree/main/paper)

N'Ko has 40 million speakers and 75 years of growing literary tradition. It deserves real infrastructure, not afterthought support. This is what that infrastructure looks like.
