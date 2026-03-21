# Does Script Design Affect How Machines Hear?

*N'Ko was built for phonetic precision. This experiment asks whether that precision shows up in speech recognition.*

---

## The Design Question Nobody Asked

When Solomana Kante designed N'Ko in 1949, he made a rule that no other major writing system fully follows: every sound gets exactly one character. Every character represents exactly one sound. No exceptions, no digraphs, no context-dependent pronunciations. If you hear it, you can write it. If you see it, you can say it.

This was a design choice for human readers. Kante wanted a script that West African children could learn without memorizing exceptions. A script where literacy was achievable in weeks, not years.

But there's a second beneficiary of that design decision that Kante couldn't have anticipated: automatic speech recognition systems.

Modern ASR works by solving an alignment problem. You have a stream of audio. You have a sequence of characters. You need to learn which sounds map to which characters. The less ambiguous that mapping, the easier the alignment problem. The easier the alignment problem, the fewer errors the decoder makes.

N'Ko's 1:1 phoneme-to-character mapping is about as clean as an alignment problem gets. You have 64 characters in the N'Ko Unicode block, each one representing one sound. The CTC decoder needs to learn 65 classes: 64 characters plus the blank token.

Latin Bambara, the same language written in the Latin alphabet, is different. It has digraphs: "ny" for the palatal nasal, "ng" for the velar nasal, "gb" for the labial-velar stop. A single sound becomes two characters. The decoder has to learn that "n" followed by "y" is not two sounds but one. It also has no tone marking, so tonal distinctions that change word meaning disappear from the transcription entirely.

Experiment B is a direct test of whether that design difference translates into measurable performance.

---

## The Controlled Comparison

The experimental design is tightly controlled. Everything is identical except the output vocabulary.

Both systems use the same audio: 37,306 Bambara and Manding speech segments collected for the N'Ko ASR pipeline. Both systems use the same encoder: Whisper Large V3, frozen, extracting audio features without any fine-tuning. Both systems use the same architecture: CharASR V3, a character-level ASR decoder with 46.9 million parameters. Both systems train on the same schedule: 50 epochs with cosine warmup. Both evaluate on the same 10% holdout set.

The only difference is what the decoder is predicting. The N'Ko version predicts N'Ko characters (65 output classes including blank). The Latin version predicts Latin Bambara characters (roughly 40 classes including blank). Both are evaluated on the same audio. The N'Ko test transcriptions are N'Ko. The Latin test transcriptions are Latin.

This is a clean comparison. If N'Ko CER comes in lower than Latin CER, we can attribute it to the script design difference and nothing else. There's no confounding from data quantity, model architecture, or training procedure. Those are all held constant.

The primary metric is CER: Character Error Rate. CER measures what fraction of characters in the predicted transcription are wrong, counting insertions, deletions, and substitutions. A CER of 0.15 means 15% of characters in the output are incorrect. For context, production-quality ASR systems typically aim for CER below 0.05 on well-represented languages.

We also report WER (Word Error Rate) and per-sample breakdowns. WER is often more meaningful for practical use because a single character error can corrupt an entire word. But for comparing script designs, CER is the more direct measure of alignment quality.

---

## What the Hypothesis Actually Claims

The hypothesis is not that N'Ko ASR will be good in absolute terms. The underlying audio data is from a low-resource language, the training set is smaller than what production systems use, and the model is a controlled research system rather than a production deployment.

The hypothesis is comparative. Given identical everything else, the N'Ko decoder should make fewer character-level alignment errors than the Latin decoder. The phonetic transparency of the script should reduce alignment ambiguity, and that reduction should show up in the error numbers.

There are three possible outcomes. If N'Ko CER is lower, that confirms the phonetic transparency advantage. If the two are approximately equal, that means script design doesn't affect ASR at the level we're measuring, which would be a genuinely surprising result worth publishing. If N'Ko CER is higher, that suggests Latin's smaller class count (40 vs 65) outweighs the disambiguation benefit of phonetic transparency, which would flip our assumption about what makes ASR easy.

All three outcomes are informative. The experiment is designed to get a clean answer.

---

## The Bigger Picture

This experiment isn't just about N'Ko. The question of whether script design affects ASR quality applies to any low-resource language community deciding how to digitize their language.

Many languages have multiple competing scripts. Bambara is written in both N'Ko and Latin. Hausa is written in both Latin and Ajami (Arabic-derived). Uyghur uses both Arabic script and Latin. When a community is building technology for their language, the choice of which script to build ASR for is a real decision with real consequences.

If phonetically transparent scripts produce lower error rates with the same training data, that's an argument for investing in N'Ko ASR over Latin Bambara ASR. It means you can get more performance per labeled audio hour when you choose the right script.

The stakes aren't purely academic.

---

## Results

Experiment in progress. Results will be published here when available.
