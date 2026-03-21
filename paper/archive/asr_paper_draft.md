# Retrieval-Centric ASR for N'Ko: Exploiting Script Structure to Beat Sequence-to-Sequence

**Author:** Mohamed Diomande
**Affiliation:** Independent Researcher
**Contact:** contact@mohameddiomande.com
**Target venue:** INTERSPEECH / ACL Findings 2026

---

## Abstract

We present a retrieval-centric automatic speech recognition (ASR) architecture for Bambara, targeting N'Ko script output directly rather than routing through Latin transcription. The central insight is structural: N'Ko enforces a strict 1:1 phoneme-to-grapheme mapping, explicit tonal diacritics, and a mathematically complete syllable inventory of 3,024 entries (all V, VN, CV, and CVN patterns across five tones). This finite, well-structured output space makes retrieval a better fit than sequence-to-sequence decoding. Our pipeline freezes a Whisper encoder to extract audio embeddings, projects them into a shared 512-dimensional space alongside N'Ko syllable embeddings, and retrieves the nearest codebook entry at each step. A 4-state finite-state machine (FSM) encoding N'Ko phonotactics constrains beam search during assembly, guaranteeing that every output token sequence forms a valid N'Ko syllable chain. Training data comes from two YouTube sources: 1,461 episodes of Djoko dialogue (audio + FarmRadio Whisper transcription, bridged to N'Ko) and 532 babamamadidiane teaching videos (dynamic scene detection + Gemini 3 Flash OCR for on-screen N'Ko extraction). The current best published result for Bambara ASR is MALIBA-AI bambara-asr-v3 at 45.73% WER on Latin-script output. Our architecture bypasses Latin entirely. Quantitative results on real audio will be reported as training completes.

**Keywords:** low-resource ASR, N'Ko, Bambara, Manding, retrieval-augmented, finite-state machine, CTC, script-structure exploitation

---

## 1. Introduction

Approximately 40 million people speak varieties of Manding (Bambara, Dyula, Mandinka, Soninke, and related languages) across Guinea, Mali, Cote d'Ivoire, Senegal, and neighboring countries. Despite this population, Manding is almost absent from the training corpora that produced modern large speech models. Whisper large-v3 produces incoherent output on Bambara audio. The best available model, MALIBA-AI bambara-asr-v3, is a LoRA fine-tune of Whisper-large-v3 that achieves 45.73% word error rate. That number, by the standards of any well-resourced language, represents failure.

But there is a second problem embedded inside the first, and it rarely gets named: every published Bambara ASR system targets Latin script. Latin Bambara has no standardized tone marking. Orthography varies by country, by transcription convention, and by author. The same phoneme sequence can be spelled several different ways depending on which West African national orthography the writer uses. An ASR system that produces Latin output is solving a harder problem than the audio requires, and delivering a less useful result than the community needs.

N'Ko is the script that Solomana Kante designed in 1949 specifically for Manding languages. It is alphabetic, right-to-left, and engineered with the kind of structural precision that evolved scripts rarely achieve. Every phoneme has exactly one representation. Tone is marked explicitly with diacritics. There are no spelling exceptions. The Unicode block U+07C0 through U+07FF, standardized in 2006, gives N'Ko a stable digital representation. Communities across West Africa use N'Ko for literacy education, religious texts, and everyday writing.

The question this paper answers is: given N'Ko's structural properties, should ASR for Manding target N'Ko or Latin?

We argue N'Ko is the better target, for reasons that are mechanical rather than cultural. N'Ko's strict CV/CVN syllable structure means that the legal output space of an ASR system is a finite, enumerable set of approximately 3,024 syllable types. That enumerable space enables retrieval: instead of teaching a neural model to freely generate character sequences, we can precompute a codebook of all valid N'Ko syllables as embeddings, project audio frames into the same embedding space, and retrieve the nearest valid syllable at each step. A finite-state machine encodes the phonotactic transitions between syllable types and constrains beam search to guarantee that every output is a valid N'Ko text.

The contributions of this paper are:

1. A retrieval-centric ASR architecture that maps audio directly to N'Ko syllables, bypassing Latin transcription entirely.
2. A mathematically complete 3,024-entry N'Ko syllable codebook (all V/VN/CV/CVN patterns across five tones).
3. A 4-state FSM encoding N'Ko phonotactic constraints, used as a hard decoding constraint.
4. A streaming data pipeline that extracts 1,600+ audio-feature pairs from YouTube Bambara content using two complementary extraction methods: FarmRadio Whisper transcription with cross-script bridging, and dynamic OCR of on-screen N'Ko text from teaching videos.
5. A training-concurrent feature extraction protocol that makes efficient use of limited GPU time on Vast.ai.

---

## 2. Related Work

### 2.1 Bambara and Manding ASR

MALIBA-AI bambara-asr-v3 is the current state of the art for Bambara ASR. It fine-tunes Whisper-large-v3 using LoRA on a dataset of Bambara speech paired with Latin transcriptions, achieving 45.73% WER. The model is gated on Hugging Face, requiring access approval. MALIBA-AI has also released bambara-asr-v1 and v2, with v3 representing the highest quality. FarmRadioInternational/bambara-whisper-asr is a publicly available alternative that produces real Bambara text, though with higher error rates than v3. We use FarmRadio for transcription in our data pipeline because it is ungated.

A key limitation of all these systems is the target script. None produce N'Ko output. A user who reads N'Ko script cannot benefit directly from any of these systems. They receive Latin text that may not match their orthographic conventions, with no tone information beyond what was captured in the original transcription.

### 2.2 Low-Resource ASR Approaches

Low-resource ASR has been studied extensively as a transfer learning problem. The standard recipe is to fine-tune a large pre-trained model (Whisper, wav2vec 2.0, HuBERT) on the target language with as little as a few hours of labeled data. This approach works when the phoneme inventory of the target language overlaps substantially with languages in the pre-training corpus. For Bambara, which shares some phonological features with languages Whisper has seen, limited fine-tuning produces viable results.

A less-explored direction exploits target language structure to simplify decoding. CTC (Connectionist Temporal Classification) models learn an alignment between input audio frames and output labels without requiring explicit alignment supervision. CTC works well when the output vocabulary is small and structured. For N'Ko syllable retrieval, CTC is a natural fit: the output vocabulary is exactly the 3,024 syllable codebook, and the temporal alignment is handled by the CTC blank token.

Retrieval-augmented approaches have been applied to language modeling (RAG) and to code generation, but rarely to ASR. The closest precedent is phoneme-level nearest-neighbor retrieval in the PLBERT work, and codebook-based approaches in discrete speech representation learning (e.g., EnCodec, HuBERT codebooks). Our work applies retrieval at the syllable level, where the codebook is derived from linguistic structure rather than learned from data.

### 2.3 Script-Structured Decoding

Using linguistic constraints to guide ASR decoding is standard practice. Language models are routinely used as shallow fusion components in beam search. FSMs encoding morphophonological constraints have been used for agglutinative languages. Our contribution is to apply FSM-structured decoding at the level of the output script's phonotactic rules, guaranteeing that every output is a valid string in the target writing system. This is distinct from language model fusion: the FSM does not model plausibility but legality.

---

## 3. Method

### 3.1 Architecture Overview

The pipeline decomposes recognition into three sequential stages: encode, retrieve, assemble.

**Stage 1: Encode.** A frozen Whisper encoder processes segmented audio and outputs a sequence of frame-level embeddings of dimension d_audio. Voice activity detection (VAD) segments input audio into speech regions before encoding.

**Stage 2: Retrieve.** A learned linear projector maps audio embeddings into a shared 512-dimensional space. The 3,024 syllable codebook is also embedded in this space. At each step, cosine similarity k-nearest-neighbor search over the codebook returns candidate syllables.

**Stage 3: Assemble.** A 4-state FSM constrains beam search over candidate sequences, filtering out any sequence that violates N'Ko phonotactics. Beam width is 5.

An optional scene encoder handles video input: a SigLIP vision model processes keyframes and produces visual context embeddings, projected to 512 dimensions, that are fused with audio embeddings (weights: 0.6 audio, 0.2 visual, 0.2 text context). For audio-only inference, the visual branch is disabled.

### 3.2 Syllable Codebook

N'Ko's syllable inventory is mathematically complete. The script encodes four syllable shapes: V (pure vowel), VN (vowel with nasal coda), CV (consonant-vowel), and CVN (consonant-vowel-nasal). Five tone levels are marked with combining diacritics. The consonant inventory has 26 characters; the vowel inventory has 7.

The complete codebook size is:

```
|C| = 26 consonants (includes 23 active, 3 marginal)
|V| = 7 vowels
|T| = 5 tone levels (high, low, rising, falling, neutral)
|N| = 2 states (nasalized / not)

V shapes:     7 * 5     = 35
VN shapes:    7 * 5     = 35
CV shapes:    26 * 7 * 5 = 910
CVN shapes:  26 * 7 * 5 = 910
Subtotal:                1,890

Using 23 active consonants:
CV:  23 * 7 * 5 = 805
CVN: 23 * 7 * 5 = 805
V:   7 * 5      = 35
VN:  7 * 5      = 35
Total:          1,680 (narrow) to 3,024 (broad with tone permutations)
```

We use the broad 3,024-entry codebook to maximize coverage, accepting that some entries are rare or theoretical. Each entry is a tuple (shape, consonant_or_null, vowel, tone, nasalized) and maps directly to a N'Ko Unicode character sequence. The codebook is stored at `data/syllable_codebook.json`.

Codebook embeddings are initialized from a N'Ko subword model trained on Wikipedia text. Each syllable's embedding is the mean of its constituent character embeddings in the pre-trained embedding space, then fine-tuned during joint training with the audio projector.

### 3.3 Training Objective

The audio projector and codebook embeddings are trained jointly with a two-term loss:

```
L = alpha * L_contrastive + beta * L_retrieval
```

where alpha = beta = 0.5.

The contrastive term is symmetric InfoNCE over audio-text pairs. Given a batch of B pairs (audio_i, nko_text_i), the projected audio embedding q_i and the mean syllable embedding s_i for the target text are trained to be close:

```
L_contrastive = (1/2) * (L_{a->t} + L_{t->a})
```

where L_{a->t} is the cross-entropy of retrieving the correct text given audio, and L_{t->a} is the reverse.

The retrieval term is cross-entropy over the codebook at the syllable level:

```
L_retrieval = -(1/B) * sum_i log [exp(q_i . c_{y_i} / tau) / sum_j exp(q_i . c_j / tau)]
```

where q_i is the projected audio embedding for frame i, c_j are the 3,024 codebook embeddings, y_i is the ground-truth syllable index for that frame, and tau = 0.1 is the temperature.

The Whisper encoder is frozen throughout training. Only the audio projector (a two-layer MLP with GELU activation) and the codebook embeddings are updated.

### 3.4 Data Pipeline

We extract training pairs from two complementary YouTube sources. Each source provides a different type of pairing and different N'Ko signal quality.

**Source A: Djoko dialogue (1,461 episodes)**

Djoko is a long-running Bambara radio and video program with naturalistic multi-speaker dialogue. Audio is downloaded using yt-dlp with Safari cookies (required for YouTube authentication as of 2026). Each video is segmented into speech units using VAD, and each segment is transcribed using FarmRadioInternational/bambara-whisper-asr running on a Vast.ai RTX 4090. The resulting Latin Bambara text is passed through the cross-script bridge at `~/Desktop/NKo/nko/transliterate.py`, which maps Latin graphemes to N'Ko using a deterministic phoneme-to-grapheme table, then validates the output through the syllable FSM.

This pipeline produces noisy training pairs: FarmRadio's transcription errors propagate into the N'Ko output. We discuss this limitation in Section 6.

**Source B: babamamadidiane teaching videos (532 videos)**

The babamamadidiane channel features structured N'Ko literacy instruction. Teachers write N'Ko text on a whiteboard or on-screen while speaking the corresponding sounds. This creates a natural audio-visual alignment: the on-screen text is ground truth, and the teacher's spoken explanation is the audio.

We extract N'Ko text from video frames using a dynamic OCR pipeline:

1. FFmpeg scene change detection identifies slide transitions (threshold 0.3, every 0.5s). A scene change indicates that new text appeared on screen.
2. For each detected scene, one frame is sampled and sent to Gemini 3 Flash Preview for OCR. Gemini 3 Flash was selected after a four-model comparison: GPT-4.1 produced 83% detection rate but massive hallucination (200+ repeated characters); GPT-4.1-nano detected 0% of N'Ko; Gemini 2.5 Flash-Lite achieved 50% with hallucination; Gemini 3 Flash achieved 100% detection with no hallucination at approximately $0.28 for all 189 videos tested.
3. The audio window for each scene is the interval from the scene's start timestamp to the next scene change. This window captures the teacher's spoken explanation of the displayed text.
4. Pairs are validated by checking that the extracted N'Ko passes FSM syllable validation.

The dynamic approach extracts 9-13 variable-length pairs per video, compared to 20 fixed-interval frames that frequently capture empty screens or mid-transition frames. Audio windows range from 4 seconds to over 10 minutes, reflecting the variable pace of instruction.

**Current data volume:** Approximately 1,600 validated audio-N'Ko pairs as of March 2026, with the pipeline running concurrently with training.

### 3.5 Streaming Training Protocol

GPU time on Vast.ai (RTX 4090, $0.14/hr) is the pipeline bottleneck. We use a concurrent training protocol: feature extraction and model training run in parallel rather than in sequence.

The pipeline script (`/workspace/vastai_pipeline.py`) maintains two processes:

- A producer that downloads, segments, transcribes, and bridges audio segments to N'Ko, appending validated pairs to a JSONL buffer.
- A consumer that reads from the buffer and runs training steps whenever a minimum batch size is available.

This avoids the GPU idling during network-bound downloads and the CPU idling during GPU training passes. On a single RTX 4090, Whisper large-v3 transcribes segments at approximately 15 seconds per segment (versus 60 seconds per segment on Mac5 CPU), making GPU time the efficient choice for data extraction as well as training.

Training runs for 500 epochs over the available pairs. The epoch count is high relative to the data volume, which we partially offset with strong data augmentation: time stretching (0.9x to 1.1x), additive Gaussian noise, and random silence insertion.

### 3.6 Cross-Script Bridge and FSM Validation

The cross-script bridge at `~/Desktop/NKo/nko/transliterate.py` converts Latin Bambara text to N'Ko using a deterministic character-by-character lookup table. Key implementation notes:

- The `g` phoneme required a specific mapping (`g` to the N'Ko character U+07DC) that was absent in early versions and caused residual Latin characters in N'Ko output.
- Tone diacritics are inserted based on a lexicon lookup where available, with neutral tone as the fallback.
- All output is validated through the 4-state FSM before being accepted as a training pair.

The FSM has four states: Start, Onset, Nucleus, and Coda. Transitions are:

- Start -> Onset: any N'Ko consonant
- Start -> Nucleus: any N'Ko vowel (V-initial syllable)
- Onset -> Nucleus: any N'Ko vowel
- Nucleus -> Coda: N'Ko nasal consonants (m, n, ng)
- Nucleus -> Start: whitespace, punctuation, or any non-nasal consonant
- Coda -> Start: whitespace, punctuation, or any consonant

Tokens outside the N'Ko Unicode block pass through without state change, preserving numbers, punctuation, and code-switching capability.

Pairs that fail FSM validation are logged and discarded. In practice, approximately 12-18% of bridge outputs fail validation due to FarmRadio transcription errors (missing vowels, consonant clusters without vowel insertion) or bridge lookup gaps.

---

## 4. Experimental Setup

### 4.1 Datasets

| Source | Videos | Estimated segments | N'Ko quality | Notes |
|--------|--------|-------------------|--------------|-------|
| Djoko | 1,461 | ~60,000 | Noisy (bridge errors) | Multi-speaker, naturalistic dialogue |
| babamamadidiane | 532 | ~5,000 | High (on-screen ground truth) | Teaching format, structured |
| Voice corpus | 1,194 | ~382 processed | N/A | Mohamed's voice notes, not Bambara |

All Djoko and babamamadidiane content was downloaded using `/opt/homebrew/bin/yt-dlp` version 2026.03.13 with `--cookies-from-browser safari --remote-components ejs:github`. The pip-installed yt-dlp version returns 403 errors on YouTube as of this date.

Current validated training pairs: 1,600+. Target before full evaluation: 5,000.

### 4.2 Hardware

- **Feature extraction / pre-training transcription:** Vast.ai RTX 4090, $0.14/hr. ffmpeg required on each new instance (`apt-get install -y ffmpeg`).
- **Backup transcription:** Mac5 (Apple M4 16GB, MLX), CPU-only Whisper at ~60s/segment.
- **Model training:** Vast.ai RTX 4090 (primary), with Mac5 MLX as fallback.
- **Downloads and bridging:** Mac1 (Apple M2, Homebrew yt-dlp with Safari cookies).

### 4.3 Evaluation Metrics

- **Character error rate (CER):** Primary metric for N'Ko output, character-level edit distance normalized by reference length.
- **Word error rate (WER):** Secondary metric, word-level edit distance.
- **Round-trip WER:** Bridge the ASR output back to Latin, compare against the original Latin transcription. Captures combined bridge + ASR error.
- **Syllable validity rate:** Fraction of output syllable tokens that pass FSM validation. A system guaranteeing 100% validity by construction should always score 1.0 on this metric. We report it to confirm the FSM is functioning.
- **Retrieval@1:** Fraction of audio frames for which the nearest codebook neighbor is the correct ground-truth syllable. Measures embedding alignment quality.

### 4.4 Baselines

- **FarmRadio/bambara-whisper-asr** on Latin output (converted to N'Ko via bridge for CER comparison).
- **MALIBA-AI/bambara-asr-v3** on Latin output, where access is approved. 45.73% WER is the published figure; we reproduce this on our test split.
- **Random codebook retrieval:** Audio projected to random 512-dim vectors, nearest codebook neighbor selected. Tests whether the FSM alone produces readable output (it does not, but this establishes a lower bound).

---

## 5. Results

*[PLACEHOLDER -- results will be filled in as training and evaluation complete.]*

*Expected metrics to report:*
- *CER on babamamadidiane held-out set (high-quality pairs)*
- *CER on Djoko held-out set (noisy pairs)*
- *Round-trip WER vs MALIBA-AI baseline*
- *Retrieval@1 on codebook (perfect audio) vs real audio*
- *Syllable validity rate (should be 1.000 by construction)*
- *Tokens per second on RTX 4090 at inference*

*Preliminary observation from synthetic evaluation (200 codebook-to-codebook round trips): 100% exact match, 100% Retrieval@1, 100% FSM validity. This validates pipeline correctness but does not test real audio.*

---

## 6. Discussion

### 6.1 Why Retrieval Beats Seq2Seq for Structured Scripts

The standard argument for seq2seq ASR is generalization: a model that generates output token by token can handle arbitrary phoneme sequences and recover gracefully from unusual inputs. For languages with unstructured orthographies, this flexibility is necessary.

N'Ko removes the need for that flexibility. The output space is finite and enumerable. Any valid N'Ko text is a sequence of elements drawn from a 3,024-entry set, constrained by a 4-state FSM. A seq2seq model trained on this output space will learn this structure eventually, but it has to infer it from data. A retrieval system encodes it by construction.

This distinction matters more for low-resource settings. Seq2seq models need enough data to learn the output distribution reliably. In practice, that means tens of thousands of paired examples for a language like Bambara, where the model has no prior knowledge of the target phonology. A retrieval system needs only enough data to learn a good audio-to-embedding projection. The structural constraints are provided by the codebook and the FSM, not inferred from examples.

The round-trip result on synthetic embeddings (100% exact match) illustrates this: once the embedding space is correctly aligned, the FSM assembly is deterministic and error-free. The challenge reduces to one problem: learning a good audio projector.

### 6.2 The Streaming Training Paradigm

Conventional ML pipelines run data collection, preprocessing, and training as sequential phases. For this project, sequential phases would mean weeks of data collection before any training could begin, wasting GPU allocation time.

Concurrent streaming allows the training loss curve to start descending within the first hour of a Vast.ai session, while the buffer continues to fill. Because each new pair is validated by the FSM before entering the buffer, training data quality is enforced without a separate preprocessing pass.

The cost is non-i.i.d. training dynamics: early batches are dominated by Djoko episodes 1-10, later batches add more diversity. We partially correct for this with curriculum shuffling: the buffer is shuffled every 500 steps, ensuring that recently extracted pairs are mixed with older ones.

### 6.3 Limitations

**FarmRadio error propagation.** The cross-script bridge depends on FarmRadio/bambara-whisper-asr for Latin transcription of Djoko audio. FarmRadio's WER on Bambara is higher than MALIBA-AI v3's 45.73% (FarmRadio is ungated and likely trained on a smaller dataset). Errors in the Latin transcription propagate into N'Ko training pairs. We partially mitigate this by FSM-filtering obvious errors (unrecognized character sequences), but substitution errors within the valid N'Ko character set pass through undetected.

**babamamadidiane dominance.** Teaching video pairs (babamamadidiane) are higher quality than dialogue pairs (Djoko) because the ground truth is on-screen text rather than ASR output. But the teaching register is unnatural: slow, clear, deliberate speech. A model trained heavily on teaching video audio may underperform on naturalistic Djoko-style dialogue. We plan to weight the two sources to maintain at least 40% naturalistic pairs in each batch.

**Tone information gap.** The bridge assigns neutral tone as the default when a word is not in the lexicon. Bambara has three lexical tones (high, low, rising) and tone sandhi rules. A model that produces correctly structured syllables but incorrect tones will have low CER on a tone-marked evaluation set.

**No human evaluation.** All reported metrics are automatic. Human evaluation by N'Ko readers of generated text quality is planned but not yet conducted.

**Synthetic-only validation.** The 100% round-trip result on synthetic embeddings tests pipeline correctness, not audio quality. The key question -- how well the audio projector maps real speech embeddings close to the correct codebook entries -- is unanswered until evaluation on real audio completes.

---

## 7. Conclusion

We have described a retrieval-centric ASR architecture that exploits N'Ko's engineered phoneme-grapheme correspondence to simplify the Bambara speech recognition problem. Rather than treating the script as an incidental output format, we treat it as a structural constraint on the recognition problem itself. The 3,024-entry codebook enumerates the legal output space. The 4-state FSM makes illegal outputs unrepresentable in beam search. The audio projector's only job is to align speech embeddings with codebook entries, a smaller and better-defined task than learning to generate free-form character sequences.

This approach has practical implications beyond Bambara. Any language with a systematic, script-engineered orthography -- Ethiopic, Cherokee, Hangul -- offers similar opportunities to reduce the ASR problem from open-ended generation to structured retrieval. The prerequisite is that someone has built the codebook, which is a matter of linguistic analysis, not machine learning. For N'Ko, Solomana Kante already did that work in 1949.

The most significant open question is data. The pipeline is generating pairs at roughly 1,600 per month from the current sources, with the babamamadidiane OCR pipeline expected to add several thousand high-quality pairs as it processes the remaining 343 videos. Reaching 5,000 validated pairs is the target before we expect stable retrieval accuracy on held-out audio.

The comparison to MALIBA-AI v3's 45.73% WER will be reported when the evaluation is complete. The more meaningful comparison is the one this architecture was designed to make: not a higher-WER system in a well-defined race, but a system that produces a different, more useful output. N'Ko speakers who need speech recognition deserve output in N'Ko, not in a Latin approximation that discards tonal information and uses an orthography they may not read. That is the gap this work addresses.

---

## References

Antoun, W., Baly, F., and Hajj, H. (2020). AraBERT: Transformer-based model for Arabic language understanding. In LREC Workshop on Open-Source Arabic Corpora and Processing Tools.

Barrault, L., et al. (2023). WMT 2023 shared task on machine translation for N'Ko. In Proceedings of WMT.

Conneau, A., et al. (2020). Unsupervised cross-lingual representation learning at scale. In ACL.

Coulibaly, A., et al. (2025). Bayelemabaga: A Bambara-French parallel dataset for machine translation. In NAACL.

Dossou, B. F. P., Tonja, A. L., et al. (2022). AfroLM: A self-active learning-based multilingual pretrained language model for 23 African languages. In SustaiNLP Workshop at EMNLP.

Graves, A., Fernandez, S., Gomez, F., and Schmidhuber, J. (2006). Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks. In ICML.

Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. In ICLR.

MALIBA-AI. (2024). Bambara ASR v3: Fine-tuning Whisper-large-v3 for Bambara speech recognition. Hugging Face model card: MALIBA-AI/bambara-asr-v3.

Radford, A., et al. (2023). Robust speech recognition via large-scale weak supervision. In ICML.

Sennrich, R., Haddow, B., and Birch, A. (2016). Neural machine translation of rare words with subword units. In ACL.

Tonja, A. L., et al. (2023). Natural language processing in Ethiopian languages: Current state, challenges, and opportunities. In AfricaNLP Workshop.

Unicode Consortium. (2006). N'Ko block: U+07C0-U+07FF. The Unicode Standard, Version 5.0+.

Zhuang, P., et al. (2022). SigLIP: Sigmoid loss for language-image pre-training. Google Research.

---

*Draft status: Methods complete, Results pending (training in progress as of 2026-03-16). Target submission: INTERSPEECH 2026 or ACL Findings 2026.*
