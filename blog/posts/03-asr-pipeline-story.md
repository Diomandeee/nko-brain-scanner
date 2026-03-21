# The Script That Was Built for AI (And That AI Has Never Heard Of)

In 1949, a self-taught scholar in Kankan, Guinea named Solomana Kante designed a writing system from scratch. He was responding to a claim that African languages could not be written. He took that as a challenge and spent years analyzing the phonological structure of Manding languages, then built a script that encodes their sounds with a precision that linguists and engineers spend careers trying to achieve in synthetic alphabets.

Every phoneme gets exactly one character. Every character represents exactly one phoneme. Tone is marked explicitly. No digraphs. No silent letters. No exceptions.

The script is called N'Ko, which means "I say" in every Manding language. It has 40 million speakers across Guinea, Mali, Cote d'Ivoire, Senegal, Burkina Faso, and diaspora communities worldwide.

No AI system has ever heard from them.

That is the paradox I spent the last several months trying to fix. Here is what I found.

---

## What Happens When a 72B Model Sees N'Ko

Before building anything, I wanted to understand the problem precisely. Not "N'Ko is underrepresented in training data" as a vague statement, but a precise measurement of what that underrepresentation looks like inside a large language model.

I ran what I'm calling a brain scan: activation profiling of Qwen2-72B-Instruct processing 100 parallel English/N'Ko sentence pairs. Same factual content, both languages, through all 81 transformer layers. I measured what the model's internal representations look like at each layer for each script.

The numbers are striking.

Qwen2-72B has 151,936 vocabulary entries. Arabic, another right-to-left script, has over 4,200 dedicated entries including rich subword representations. N'Ko gets exactly 32 entries: single Unicode codepoints, one per character, with no subword tokens. Every N'Ko word gets decomposed into individual characters. The model processes "kankan" as seven separate tokens rather than one.

What does that do to the model's internal representations?

**The translation tax.** At the embedding layer, English activations have an average L2 norm of 41.2. N'Ko activations have an average L2 norm of 14.2. That is a 2.90x ratio. The model is spending roughly 3x less activation energy on N'Ko text at the first layer. This ratio does not recover as you go deeper. It worsens: by the output layer, it reaches 3.26x. Every layer that should be building up richer representations is instead amplifying the initial deficit.

**Entropy inflation.** Shannon entropy measures how diffuse or concentrated the model's internal activations are. High entropy means the model is spreading attention across everything, not committing to anything. For English, entropy at the output layer is 11.02 bits. For N'Ko, it is 13.89 bits, close to the theoretical maximum for that dimension size, which means the model is essentially randomly distributing probability mass across its vocabulary when it tries to output N'Ko.

**The kurtosis collapse.** This one is the most telling number. Kurtosis measures how peaked the activation distribution is. In a model that knows what it is doing, kurtosis climbs at the output layers: the model gets more and more confident as it approaches a prediction. For English, kurtosis goes from 12.4 at embedding to 58.4 at the output layer. A clean upward climb.

For N'Ko, kurtosis peaks at layer 60 (11.3), then *drops* to 8.3 at the output. The model is actively un-committing. At the exact moment it should be concentrating on a prediction, it is spreading back out toward maximum entropy. The 85.8% kurtosis deficit at the output layer is not a model that is unsure which N'Ko character to pick. It is a model that is unsure whether to pick N'Ko characters at all.

**Dead circuits, not weak ones.** To test whether there were any N'Ko reasoning circuits worth amplifying, I ran a circuit duplication analysis: 55 different configurations of duplicating transformer layer blocks, following the RYS (Revisit Your Shoulders) methodology. The idea is that duplicating a layer amplifies whatever computation that layer is performing. If N'Ko had working circuits, duplicating them would improve performance.

Zero out of 55 configurations showed N'Ko-advantageous performance. The best N'Ko score across all 55 configurations was 0.067, barely above random chance at 0.05. The circuits are not weak. They are absent.

The comparison with Arabic makes this clear. Arabic is also right-to-left. Arabic is also Semitic. When you run the same profiling on Arabic, the activation profiles look like English: high L2 norms, concentrated entropy, climbing kurtosis at the output layer. The difference between Arabic and N'Ko is not script direction. It is pre-training data. Arabic has 4,200+ vocabulary entries because Arabic text was in the training data. N'Ko has 32 because it was not.

This is what I mean by script invisibility: not a bug in the architecture, not a fundamental limitation of the model, but an artifact of which scripts got included in the data.

---

## Building the First N'Ko ASR, From Zero

The problem with building a Bambara ASR system that outputs N'Ko is that no N'Ko-labeled speech corpus exists. Every existing Bambara ASR system outputs Latin script. The FarmRadio Bambara Whisper model, the MALIBA-AI models, the sudoping01 models. All Latin output. You cannot train an N'Ko ASR system if you have no N'Ko training targets.

So the first thing to build was a bridge.

### The Cross-Script Bridge

The bridge converts existing Latin Bambara transcriptions to N'Ko. The idea is straightforward: Latin Bambara maps to IPA (International Phonetic Alphabet), then IPA maps to N'Ko codepoints. N'Ko has a bijective phoneme-grapheme mapping, so the IPA-to-N'Ko conversion is a clean lookup table.

```
B: Latin Bambara -> IPA -> N'Ko Unicode
```

The Latin-to-IPA step requires priority ordering because Bambara uses digraphs. "ny" is a single phoneme (the palatal nasal), not the letter "n" followed by "y". "ng" is the velar nasal, not two consonants. You have to consume digraphs before single characters or you get garbage output.

```python
# Priority 1: digraphs first
DIGRAPH_RULES = {
    'ny': 'ɲ',   # palatal nasal -> N'Ko U+07E2
    'ng': 'ŋ',   # velar nasal -> N'Ko U+07D2
    'sh': 'ʃ',   # voiceless postalveolar -> N'Ko U+07D9
}

# Priority 2: toned vowels (requires NFD decomposition first)
# Priority 3: single characters
```

During development I ran into six distinct bug classes. None of them were typical programming bugs. They were all places where Latin orthographic conventions for Bambara conceal information that N'Ko was designed to express.

Bug 1 was greedy character matching (the digraph problem above). Bug 2 was a missing consonant: the phoneme /g/ had no IPA-to-N'Ko entry, so any word with /g/ produced a Latin "g" embedded in otherwise valid N'Ko output. Bug 3 was extended IPA symbols from FarmRadio's transcription model that covered loanwords and dialectal variants not in the standard Bambara inventory. Bug 4 was that fixing Bug 1 revealed a gap in Bug 3: digraph outputs from Stage 1 had no Stage 2 entries. Bug 5 was NFD decomposition: pre-composed toned vowels (à, é) need Unicode canonical decomposition before lookup. Bug 6 was RTL rendering: N'Ko is right-to-left, so spaces between N'Ko words need U+200F (right-to-left mark) to render correctly in LTR-dominant environments.

Each one of these bugs is a small document of a place where colonial orthography obscures phonemic structure that N'Ko was designed to express. The bridge does not just convert scripts. It recovers information that 20th century French linguistic conventions encoded away.

After fixing all six, the bridge produces valid N'Ko output for 99.4% of clean Bambara text.

### The FSM: Writing Grammar as Code

N'Ko syllables follow a strict structure: (C)V(N). Optional consonant onset, required vowel nucleus, optional nasal coda. No consonant clusters. No vowel hiatus within a syllable.

This is a formal, closed system. There is no reason to ask a neural network to learn it from data when you can specify it exactly. I built a 4-state finite-state machine:

```
States: {Start, Onset, Nucleus, Coda}

Start  + consonant -> Onset
Start  + vowel     -> Nucleus
Onset  + vowel     -> Nucleus
Onset  + consonant -> REJECT (no CC clusters)
Nucleus + tone_mark -> Nucleus (tone attaches)
Nucleus + nasal    -> Coda
Nucleus + consonant -> Onset (new syllable)
Coda   + consonant -> Onset (new syllable)
Coda   + nasal     -> REJECT (no double nasal)
```

When the FSM hits an invalid transition, it does not discard the character. It replaces it with the highest-probability admissible token from the CTC decoder's posterior distribution at that time step. In practice most corrections insert a vowel to resolve consonant clusters, epenthetic vowels that the acoustic signal probably carried but the neural network missed.

The FSM adds less than 2% latency. It guarantees 100% phonotactically valid output. Natural N'Ko text passes at 99%. Random N'Ko character sequences pass at 19%. The FSM knows the difference.

---

## Four Architectures: V1 Through V4

I started with frozen Whisper large-v3 as the acoustic encoder (307M parameters, trained on 680,000 hours of audio) and trained only the CTC decoder on top of its feature representations. Training data: 37,306 audio clips from the bam-asr-early corpus, about 37 hours of Bambara speech. Latin transcriptions fed through the bridge for N'Ko targets.

**V1: BiLSTM, 56% CER.** Three layers of bidirectional LSTM with 512 hidden dimensions. 5.4M trainable parameters. Validation loss 0.143. The error analysis was instructive: V1 dropped syllables in multi-syllabic words because BiLSTM's sequential induction bias means context from 5 characters ago is substantially attenuated. For a language where syllable structure creates long-range dependencies, this matters a lot. Tone diacritic errors on 78% of toned outputs.

Garbage. But informative garbage.

**Architecture search: 28 configurations.** Before scaling up I ran a systematic search over architecture family (BiLSTM, Transformer, Conformer), hidden dimension (256, 512, 768), depth (2, 4, 6 layers), and temporal downsampling (4x, 8x, 16x). Three findings from this search:

1. Transformers beat BiLSTMs by 15-20 points at every comparable scale. Self-attention's global context window is exactly what N'Ko syllable structure needs and BiLSTM cannot provide.

2. 4x downsampling consistently beats 8x and 16x. N'Ko characters represent individual phonemes, shorter acoustic events than the syllable or word-level units that higher downsampling rates assume.

3. Conformers underperform Transformers at low data volume. The local convolution kernels overfit to speaker-specific patterns with only 37 hours of training data.

**V3: Transformer fullpower, 33% CER.** 6 Transformer layers, 768 hidden dimension, 12 attention heads, 4x downsampling. 46.9M trainable parameters. SpecAugment for regularization (without it, the model overfit catastrophically after epoch 80). Trained 200 epochs.

The loss curve had a clear progression through four phases: character discovery (epochs 1-10, model learns to emit individual N'Ko characters instead of blank tokens), word formation (10-40, learns word boundaries and recognizable Bambara words), sentence structure (40-100, multi-word sequences and first tone diacritic predictions), refinement (100-200, diminishing returns on edge cases).

By epoch 200: 33% CER, 70% WER. A 23-point CER improvement over V1.

**V4: Whisper LoRA, 29.4% CER.** V1 through V3 use Whisper's encoder as a frozen feature extractor. The representations are powerful but generic. Whisper was trained on predominantly non-African audio. It has no specific knowledge of Bambara phonology, Bambara tone contours, or Bambara nasalization patterns.

V4 partially unfreezes the Whisper encoder using LoRA (rank=32, alpha=64) applied to the query, key, and value projection matrices of Transformer layers 24-31 (the top 8 of 32 encoder layers). This adds 5.9M trainable parameters to the encoder. Dual learning rates: 1e-5 for the LoRA layers (conservative, preserve pre-trained representations) and 3e-4 for the CTC head.

30 epochs on A100. Validation loss goes from 0.884 to 0.290. Train-val gap at epoch 30: 0.003. The model is not overfitting.

The aggregate results:

| Version | Params | CER | WER | Cost |
|---------|--------|-----|-----|------|
| V1 BiLSTM | 5.4M | 56.0% | 91.5% | $3 |
| V2 Transformer | 22.1M | 45.7% | 78.6% | $4 |
| V3 Transformer | 46.9M | 33.0% | 70.0% | $5 |
| V4 Whisper LoRA | 52.8M | 29.4% | 62.3% | $6 |

Total compute cost for all experiments: $14.

But the aggregate numbers undersell what V4 actually does. The most dramatic result is per-sample. Sample au30, V3's worst-performing sample (WER 15.8, essentially complete failure), improves to WER 1.0 under V4. A 93.7% improvement on a single sample. The frozen encoder had an acoustic blind spot for whatever phonemic pattern that sample contains. LoRA adaptation filled it.

Mean prediction confidence jumps from 0.46 (V3) to 0.82 (V4). The model is not just predicting slightly better characters. It is predicting with nearly twice the certainty.

---

## The Pipeline: Two Mac Minis and Thunderbolt 5

The full inference system runs across two Apple Silicon machines connected via Thunderbolt 5.

Mac4 (M4 Max, 64GB RAM) handles the Whisper encoder with LoRA adapters. For a typical 10-second audio clip, it runs the forward pass through 307M encoder parameters and produces 1,280-dimensional frame representations. That takes about 180ms.

Those frame representations, about 2.4MB serialized at float16, transfer to Mac5 over Thunderbolt 5. Transfer latency: 0.4ms. Negligible.

Mac5 (M4, 16GB RAM) runs the CTC decoder (46.9M parameters, about 40ms), FSM post-processing (under 1ms), and if translation is requested, NLLB-200 (67ms per sentence).

End-to-end: about 290ms for ASR only, 360ms with translation. Real-time for conversational speech.

The translation pipeline is:

```
Audio -> [Mac4] Whisper+LoRA -> frames -> [Mac5] CTC+FSM -> N'Ko
N'Ko -> bridge inverse (bijective) -> Latin Bambara -> NLLB-200 -> English/French
```

The bridge inverse is trivial because the original bridge is bijective. Every N'Ko character maps to exactly one Latin character or digraph. No ambiguity to resolve. Just reverse the lookup table.

NLLB-200 was fine-tuned on 8,640 parallel sentence pairs across four language directions (Bambara to/from English and French). Training loss dropped from 6.29 to 1.89 over 15 epochs. The translation quality (BLEU-1 = 0.246 for Bambara to English) is modest but functional. Speak Bambara, see N'Ko text, see English translation. Under 300ms, end to end.

---

## Why This Matters

A child in Kankan, Guinea, who speaks Maninka and reads N'Ko cannot dictate a text message, search the web, or interact with any AI system in their own script. Every voice interface, every language model, every autocomplete, every keyboard suggestion responds in Latin orthography that French colonial linguists designed in the 20th century. Not for the 40 million people who actually speak and read these languages. For the administrative apparatus that governed them.

The cognitive cost of this compounds across education, commerce, healthcare, and creative expression. Every keystroke in a foreign alphabet is a small friction. Decades of small frictions add up.

Solomana Kante spent years building a script that encodes the Manding phoneme inventory with the precision that NLP engineers dream of when designing synthetic alphabets. Regular tokenization. Minimal CTC output space. Unambiguous spelling. Explicit tone. He built all of this in 1949 with no computers, no training data, no benchmark evaluations. Just a systematic study of the sounds of the languages he grew up with and the insight that they deserved to be written correctly.

The script has been waiting for AI to catch up to it.

This is the first time a machine has written in that script from speech.

---

## What Comes Next

A few concrete next steps, in order of likely impact.

**More training data.** The afvoices dataset from RobotsMali contains 612 hours of Bambara speech. We trained V1-V4 on 37 hours. Scaling to 612 hours is the single highest-leverage move. I expect Conformer architectures (which underperformed Transformers at 37 hours but should outperform them with sufficient data) to become competitive at that scale.

**Tone-labeled data.** The primary error class in V4 is tone diacritic confusion: 38% of character errors are predicted on the correct consonant-vowel pair with the wrong combining mark. The bridge defaults to neutral tone because no comprehensive Bambara tone lexicon exists to annotate training data. Building even a partial tone lexicon for common lexical items would substantially reduce this error class.

**Beam search.** V1-V4 use greedy CTC decoding. Beam search with N'Ko language model scoring would improve word-level accuracy, particularly for multi-syllabic words where the CTC alignment posterior is flat.

**iOS app.** NKoScribe is on TestFlight right now. The full pipeline (V4 encoder + CTC decoder + FSM + translation) runs on-device. You speak Bambara. You see N'Ko. You can send it.

The LoRA adaptation work will continue. There are still samples where V3's frozen encoder outperforms V4's adapted one, which means there are acoustic patterns the adaptation has not fully addressed. The current V4 was trained for 30 epochs with 37 hours of data. More data, more epochs, larger LoRA rank applied to more encoder layers.

The LLM side of this (paper 1, the brain scan) showed that a three-stage LoRA pipeline reduced the translation tax from 2.90x to 0.70x in the language model, getting N'Ko representations from near-random to nearly competitive with English. The same approach should continue yielding returns on the ASR side.

Solomana Kante designed a script that any phonologist would envy, for a language spoken by tens of millions of people, and spent his life making sure it survived. The least we can do is make sure the machines can read it.

---

*The two research papers (Dead Circuits: Activation Profiling and Script Invisibility in Large Language Models, and Living Speech: Script-Native Automatic Speech Recognition for N'Ko) are in preparation for ACL/EMNLP 2026. The ASR models and bridge code will be released publicly. NKoScribe is available on TestFlight. Contact: contact@mohameddiomande.com*
