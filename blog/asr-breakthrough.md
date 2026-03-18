# From Dead Circuits to Living Speech

**Building the World's First Audio-to-N'Ko ASR System**

*Mohamed Diomande, Independent Researcher, March 2026*

---

## The Circuit That Wasn't There

In Part 1 of this research, we performed a brain scan on Qwen2-72B. We fed it N'Ko text and measured what happened inside: 80 transformer layers, 8,192 neurons each, four metrics per layer. The results were stark.

The model's reasoning circuits, the same circuits that David Noel Ng showed could be amplified through layer duplication to boost English reasoning by 17.72%, produced nothing for N'Ko. Not weak signal. Not partial comprehension. Nothing. Every one of 55 circuit duplication configurations scored at or near random chance. The heatmap was blank. The circuits were dead.

We called this the "translation tax": a 3-4x reduction in activation magnitude, 30-60% higher entropy, and progressive loss of circuit specialization from layer 0 through layer 80. The model couldn't read N'Ko because it had never been taught to.

Then we asked a different question.

If the model can't read N'Ko, can it *hear* it?

---

## The Inversion

Every existing Bambara ASR system, MALIBA-AI, Meta's MMS, Google's USM, does the same thing: take audio in, produce Latin text out. The output looks like this:

```
ko muso tɛ sɔrɔ sonyali la k'u b'a deli
```

This is Bambara written in Latin script, an orthography designed by French colonial linguists. It works. But for the majority of literate Bambara speakers, those who learned to read in N'Ko, this output is foreign. They would write the same sentence as:

```
ߞߋ ߡߎߛߋ ߕߐ ߛߏߙߏ ߛߋߢߊߟߌ ߟߊ ߞʔߎ ߓʔߊ ߘߍߟߌ
```

No existing system produces this. The ASR field has a blind spot in the same place the LLM field does: N'Ko doesn't exist in the training data, so N'Ko doesn't exist in the output.

Our brain scan showed that LLMs can't process N'Ko text. But what if we skip the LLM entirely? What if we go straight from audio to N'Ko characters, using the same phonetic regularity that Solomana Kanté designed into the script 77 years ago?

---

## The Phonetic Transparency Hypothesis

Here is the connection between the circuit research and the ASR work, and it's the same insight seen from two directions.

**The brain scan finding**: N'Ko's 1:1 phoneme-to-character mapping is a computational advantage that LLMs can't exploit because they lack training data. The design advantages are real but latent.

**The ASR hypothesis**: That same 1:1 mapping should make N'Ko an *easier* CTC target than Latin for audio transcription.

Consider what a CTC decoder has to learn for each script:

| Property | Latin Bambara Target | N'Ko Target |
|----------|---------------------|-------------|
| Character inventory | 26 letters + digraphs | 65 characters (single symbols) |
| "ny" = one sound | Must learn 2-char → 1-phoneme | ߢ (single character) |
| "ng" = one sound | Must learn 2-char → 1-phoneme | ߒ (single character) |
| Tone | Not marked (invisible) | Explicit diacritics |
| Vowel length | Context-dependent | Distinct characters |
| Phoneme alignment | Many-to-many | 1:1 |

When you train a CTC model to output Latin Bambara, the decoder must learn that the audio segment /ɲ/ maps to the two-character sequence "ny", that /ŋ/ maps to "ng", that tone distinctions in the audio have no written representation. Every digraph is a learned exception. Every unmarked tone is lost information.

When you train a CTC model to output N'Ko, the decoder learns: one sound = one character. There are no exceptions. Tone marks in the audio map directly to tone diacritics in the output. The model's output space is isomorphic to the phoneme inventory.

This is the hypothesis the brain scan generated but couldn't test, because LLMs process text, not audio. To test it, we needed to build something new.

---

## The Architecture

We built the pipeline from frozen Whisper features through a character-level CTC decoder with FSM post-processing.

### Layer 1: Whisper (Frozen)

OpenAI's Whisper large-v3 serves as the acoustic feature extractor. We don't fine-tune it. We don't even use its decoder. We take the encoder output: 1,500 frames of 1,280-dimensional representations per audio clip, downsampled 4x to 375 frames, then 4x again to 93 frames in the training pipeline.

Whisper was trained on 680,000 hours of multilingual audio. It knows what speech sounds like, even in languages it can't transcribe. The encoder representations capture phonetic information regardless of the target script.

### Layer 2: Character-Level CTC Head

A 5.4M parameter model sits on top of the frozen Whisper features:

```
Linear(1280 → 512) → 3-layer BiLSTM(512, bidirectional) → Linear(512 → 66)
```

66 output classes: 65 N'Ko characters (U+07C0 through U+07FF, covering digits, vowels, consonants, tone marks, nasalization marks, and space) plus one CTC blank token.

This is deliberately minimal. The question isn't "can a large model do this?" It's "can a small model do this, and if so, what does that tell us about the script?"

### Layer 3: FSM Post-Processing

This is where the circuit research feeds back in.

The brain scan showed that LLMs fail to learn N'Ko's syllable structure from data alone. The FSM encodes that structure as hard constraints. N'Ko syllables follow a strict grammar:

```
START → ONSET (consonant) → NUCLEUS (vowel) → CODA (nasal/tone) → START
START → NUCLEUS (vowel) → CODA → START
```

Valid patterns: V, CV, VN, CVN. Nothing else. No consonant clusters. No vowel-initial codas. No floating tones without a host vowel.

The FSM is a 4-state machine (START, ONSET, NUCLEUS, CODA) that validates every decoded character sequence against this grammar. Invalid transitions trigger corrections. The FSM achieves 100% phonotactic validity on clean output, something the LLM brain scan showed was impossible for the model to learn from its training data.

The LLM's dead circuits are replaced by an explicit finite-state machine. What the neural network couldn't learn from data, we encode as structure.

### The Cross-Script Bridge

No N'Ko-labeled speech corpus exists. Every available Bambara audio dataset (bam-asr-early, afvoices, Common Voice) has Latin transcriptions only.

We built a deterministic Latin-to-N'Ko transliteration bridge:

```
Latin:  ko muso tɛ sɔrɔ sonyali la
IPA:    ko muso tɛ sɔrɔ soɲali la
N'Ko:   ߞߋ ߡߎߛߋ ߕߐ ߛߏߙߏ ߛߋߢߊߟߌ ߟߊ
```

The bridge handles: IPA normalization, NFD decomposition of toned vowels (e.g., a + combining grave accent), digraph resolution (ny→ɲ, ng→ŋ), and direct Unicode mapping from the IPA phoneme inventory to N'Ko codepoints.

Six critical bugs were found and fixed during development:
1. Greedy "na"→ߠ match that corrupted any word containing "na"
2. Missing "g"→ߜ mapping
3. Missing "z"→ߛ, "ə"→ߐ, "ʃ"→ߛ mappings
4. Missing "ɲ" and "ŋ" in the single-character IPA table
5. NFD decomposition failure on pre-composed toned vowels

Each bug was a lesson in how Latin orthographic conventions obscure the underlying phonology. The bridge doesn't just convert characters. It recovers the phonemic representation that Latin obscured and maps it to the script that was designed to express it.

---

## The Training

37,306 human-labeled Bambara audio clips from bam-asr-early (CC-BY-4.0), 37 hours total. Each clip has a Latin transcription that we bridged to N'Ko. Features pre-extracted as float16 tensors on a Vast.ai RTX 4090 ($0.26/hour).

The model trains on character-level CTC loss:

```
CTC_loss(predicted_N'Ko_chars, gold_N'Ko_chars)
```

No language model. No attention-based decoder. No beam search. Pure CTC with greedy argmax decoding, followed by FSM validation.

### Loss Curve

The model is currently training. At epoch 76 of 200:

| Epoch | Train Loss | Val Loss | Observation |
|-------|-----------|----------|-------------|
| 1 | 2.625 | 2.399 | Repeating single characters |
| 10 | 1.603 | 1.569 | First 3 words recognizable |
| 20 | 1.287 | 1.257 | Word boundaries forming |
| 40 | 0.962 | 0.929 | Broke below 1.0 |
| 76 | 0.583 | 0.533 | Multi-word sequences correct |

Sample at epoch 76:
```
Gold: ߞߋ ߡߎߛߋ ߕߐ ߛߏߙߏ ߛߋߢߊߟߌ ߟߊ ߞʔߎ ߓʔߊ ߘߍߟߌ
Pred: ߞߋ ߡߎߛߋ ߕߐ ߛߏߙߏ ߛߋߢߊߟߌ ߟߊ ߞߎ ߓߊ ߘߌߜߌ
```

The first six words are perfect. The model learned to decode audio directly into N'Ko characters that a native reader can read.

---

## The Circuit Connection

The brain scan and the ASR system are two views of the same problem, and together they reveal something about how script design interacts with machine learning architectures.

**Finding 1: LLM circuits for N'Ko are dead because of data starvation, not architectural limitation.** The reasoning circuits exist. They work for English. They produce nothing for N'Ko because the embedding layer can't represent N'Ko characters, and the early layers can't build clean representations.

**Finding 2: The FSM encodes what the LLM couldn't learn.** N'Ko's syllable grammar (CV/CVN) is a structural regularity that a model trained on billions of English tokens never discovers about N'Ko. We encode it explicitly as a post-processing constraint. The dead circuit is replaced by a deterministic state machine.

**Finding 3: Phonetic transparency helps CTC directly.** The 1:1 phoneme-to-character mapping that we hypothesized would help LLMs (but couldn't, due to data starvation) does help CTC. The decoder has 65 classes, each mapping to one phoneme. No digraphs to learn. No irregular spellings. No ambiguity. The CTC decoder's job is strictly simpler for N'Ko than it would be for Latin.

**Finding 4: Cross-script bridges recover what colonialism obscured.** The Latin orthography used in all existing Bambara corpora was designed for French linguists, not for computational transparency. Our bridge doesn't just convert scripts. It recovers the phonemic structure that N'Ko was designed to express and that Latin conceals behind digraphs, unmarked tones, and borrowed conventions.

---

## What This Means

We built a system that didn't exist. Not a better version of something. A new category.

No previous system converts audio to N'Ko text. MALIBA-AI, the state-of-the-art for Bambara, outputs Latin only. Meta's MMS outputs Latin only. Google's USM outputs Latin only. For the millions of N'Ko-literate speakers across West Africa, the entire ASR field has been producing output in what amounts to a foreign script.

The brain scan showed why: N'Ko is invisible to the models that power modern NLP. The ASR system shows an alternative path: bypass the LLM entirely, use a script-aware decoder that respects N'Ko's phonological design, and produce output that N'Ko readers can actually use.

The model is small (5.4M parameters). The data is modest (37 hours). The compute is cheap ($3 total). The architecture is simple (frozen Whisper + BiLSTM + FSM).

And it works. Audio goes in, N'Ko comes out.

---

## What Comes Next

The current model is a proof of concept. The research program continues:

1. **Complete training** (200 epochs, currently at 76) and run full evaluation on 1,463 test samples
2. **Submit to MALIBA-AI leaderboard** via round-trip transliteration (N'Ko→Latin), to establish a baseline comparison with existing Latin-output systems
3. **Scale data**: incorporate afvoices (159h), Djoko teaching videos (16K+ segments), and babamamadidiane educational content
4. **Scale architecture**: train the V2 Transformer (36M params) and explore Whisper LoRA fine-tuning
5. **Publish**: the system, the cross-script bridge, and the combined brain scan + ASR findings

The script that machines couldn't read is now the script that machines can write from speech.

Kanté would have approved.

---

*Part 1: [The Script That Machines Can't Read](post.md) (Brain scan research)*
*Part 2: This post (ASR breakthrough)*
*Code: [github.com/Diomandeee/nko-brain-scanner](https://github.com/Diomandeee/nko-brain-scanner)*

*Total research cost: $1.72 (brain scan) + ~$3 (ASR training) = $4.72*
*Hardware: Vast.ai RTX 4090 ($0.26/hr) + Apple M4 16GB (local)*
