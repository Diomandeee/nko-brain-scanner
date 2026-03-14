# The Script That Machines Can't Read

*How a 72-billion-parameter AI reveals the cost of digital language exclusion*

---

In 1949, Solomana Kante sat down to design a writing system from scratch. He wanted something that would capture the sounds of Manding languages, the way they actually work, not shoehorned into Latin letters or Arabic script. What he created was N'Ko: 27 base characters, each mapping to exactly one sound. Explicit tone marks. No silent letters. No irregular spellings. A writing system built with the same precision you'd use to design a programming language.

Seventy-seven years later, we pointed a 72-billion-parameter AI at Kante's script. We wanted to know: does a well-designed writing system make it easier for machines to think?

The answer surprised us.

## The Clue

In October 2024, David Noel Ng published a paper called "Reasoning Yield from Stacking" (RYS). He discovered something strange about Qwen2-72B, a 72-billion-parameter language model with 80 transformer layers. When he duplicated 7 specific middle layers, so those layers ran twice in the forward pass, the model's multi-step reasoning improved by 17.72%. No retraining. No new data. Just running the same circuits twice.

His explanation: those 7 layers contain the model's "reasoning circuits." Running them twice is like letting someone re-read a tricky sentence. The computation gets another pass through the logic.

But Ng only tested with English. His paper never asked: does the *input script* affect how well those circuits work?

That's where N'Ko comes in.

## The Hypothesis

N'Ko is what English wishes it were. Consider:

| Property | N'Ko | English |
|----------|------|---------|
| Phoneme mapping | 1:1 (each character = one sound) | Many-to-many ("ough" = 6+ pronunciations) |
| Tone marking | Explicit diacritics | None (context-dependent) |
| Morphology | Agglutinative (predictable word-building) | Irregular (go/went, mouse/mice) |
| Spelling rules | Zero exceptions | Exceptions outnumber rules |
| Unicode range | 64 characters (U+07C0-U+07FF) | 26 letters + thousands of combinations |

If a model's reasoning circuits benefit from clean, predictable input, then N'Ko's designed regularity should produce cleaner activation patterns than English's evolved irregularity. The reasoning circuits should have an easier job.

That was the hypothesis.

## The Brain Scanner

We built a pipeline to scan the "brain" of Qwen2-72B as it processes text. Two experiments, one model, two scripts.

**Experiment 1: Activation Profiling.** Feed the model 100 parallel sentences in English and N'Ko. At every one of the 80 layers, measure how the neurons respond. Four metrics:
- **L2 Norm**: How hard is this layer working? (activation magnitude)
- **Shannon Entropy**: How spread out is the attention? (certainty vs confusion)
- **Sparsity**: How many neurons are doing nothing? (efficiency)
- **Kurtosis**: How focused are the active neurons? (specialization)

**Experiment 2: Circuit Duplication Heatmap.** Replicate Ng's technique: for every possible block of layers, duplicate it and measure whether reasoning improves or degrades. Do this for English input. Then do it again for N'Ko input. Compare the heatmaps.

We ran both experiments on a single A100 80GB GPU, with the model quantized to 4-bit precision. Total compute cost: $1.72 on Vast.ai.

## Experiment 1: The Translation Tax

This is what we found.

![Activation profiles: English vs N'Ko across 80 layers](assets/activation_comparison.png)

*Per-layer activation profiles for English (blue) and N'Ko (orange). Each panel shows a different metric across all 80 transformer layers.*

The four panels tell one story: **the model doesn't know how to read N'Ko.**

**L2 Norm (top-left):** English activations are 3-4x larger than N'Ko across the entire network. The model processes English with high-magnitude signals from start to finish. For N'Ko, the signal is permanently weak. Layer 51, right in the heart of the reasoning zone, shows the starkest difference: English = 2,093, N'Ko = 684.

**Shannon Entropy (top-right):** N'Ko starts at maximum entropy (~12 bits) in the earliest layers. The model is maximally uncertain about what it's looking at. English starts low (~4.5 bits) and climbs gradually. By layer 20, they converge, but the damage is done. The early layers failed to build clean representations of N'Ko.

**Sparsity (bottom-left):** At layer 0, 35% of neurons are near-zero for N'Ko versus 14% for English. The embedding layer can barely activate anything for N'Ko characters. It's as if the model is squinting at a script it can barely see.

**Kurtosis (bottom-right):** English maintains highly focused, specialized circuits (kurtosis ~7,500) throughout the network. N'Ko starts at near-zero kurtosis and slowly climbs, suggesting the model eventually falls back on general-purpose language processing. But it never builds the sharp, specialized circuits that English enjoys.

We call this the **translation tax**: the computational cost of processing a language the model wasn't trained on. For English, the model has built optimized highways. For N'Ko, every layer is a dirt road.

## Experiment 2: The Heatmaps

If the activation profiles show the problem, the heatmaps show the consequence.

![Circuit Duplication Heatmaps: English vs N'Ko](assets/heatmap_coarse.png)

*Left: English heatmap showing which layer duplications improve reasoning. Center: N'Ko heatmap. Right: Difference (green = N'Ko advantage, pink = English advantage). Each point represents one (start layer, end layer) duplication configuration.*

**English (left panel):** Rich with signal. The model's reasoning circuits are visible as a warm band in layers 0-56. The best configuration, duplicating layers 8-16, achieves a combined score of **0.752**. Even duplicating late layers (56-72) still helps. The model has a wide "improvement zone" where layer duplication boosts performance.

**N'Ko (center panel):** Nearly blank. Every single configuration scores near zero. The best N'Ko configuration, (0, 40), scores **0.067**, which is noise. Duplicating any set of layers, from any starting point to any ending point, does nothing for N'Ko processing.

**Difference (right panel):** All pink. English advantage everywhere. Not a single configuration where N'Ko outperforms English. Not one.

The reasoning circuits that Ng discovered are real. They work. But they need something to reason *about*. For English, the early layers build clean, high-magnitude representations that feed perfectly into the reasoning zone. For N'Ko, the signal never arrives.

## What This Actually Means

Our original hypothesis was wrong. We expected N'Ko's phonological regularity to produce cleaner activation patterns. It didn't, but not because the hypothesis was flawed. It was because we were asking the wrong question.

The right question isn't "does script design affect reasoning circuits?" It's "does the model have the training data to *use* those design advantages?"

Qwen2-72B was trained on trillions of tokens. The vast majority were in English, Chinese, and a handful of major languages. N'Ko is spoken by over 40 million Manding speakers across West Africa, but its digital corpus is tiny. Wikipedia has ~6,000 N'Ko articles. The model probably saw a few thousand N'Ko tokens during training, compared to hundreds of billions of English tokens.

The result is a structural gap that no architectural trick can fix:

1. **Layers 0-10** (Comprehension): The model can't parse N'Ko. Entropy is maximal, sparsity is high, activations are weak. It's trying to read a script it barely recognizes.

2. **Layers 10-56** (Reasoning): The reasoning circuits exist and work for English. But for N'Ko, the upstream signal is too weak and noisy. You can't reason about something you can't read.

3. **Layers 56-80** (Generation): With no coherent reasoning to synthesize, the model produces near-random output for N'Ko prompts.

Layer duplication amplifies what's already there. If the reasoning circuits are processing clean signals, amplification helps. If they're processing noise, amplification just produces more noise.

## The Bigger Picture

Solomana Kante designed N'Ko so that West Africans could write their own languages with precision and dignity. The script is elegant, consistent, and computationally clean. It has everything a language model should love.

But the model doesn't love it, because the model doesn't know it.

This is the paradox of AI language equity: the languages that would benefit most from AI, the ones with designed writing systems, rich oral traditions, and growing digital presence, are the ones AI understands least. Not because of any linguistic deficiency. Because of data.

The reasoning circuits in Qwen2-72B are language-agnostic in architecture but language-specific in training. They can reason in any language, in theory. In practice, they can only reason in languages they've been fed.

The fix isn't better architecture. It's better data. If Qwen2-72B had been trained on even a few million N'Ko tokens, from Wikipedia, from cultural texts, from the rich corpus of Manding proverbs and greetings, those early layers would learn to build clean representations of N'Ko characters. The reasoning circuits would engage. And N'Ko's phonological regularity might finally become the computational advantage it was designed to be.

Until then, the most precisely designed writing system in the world remains invisible to the most powerful language models. Not because they can't see it. Because nobody taught them to look.

---

## Methodology

- **Model:** Qwen2-72B-Instruct, loaded in BitsAndBytes 4-bit (NF4) quantization on a single A100 80GB GPU
- **Experiment 1:** 100 parallel sentence pairs (NKO/English), hidden states extracted at all 81 layers (embedding + 80 transformer layers), 4 metrics computed per layer
- **Experiment 2:** 55 coarse-grained (i,j) configurations (step=8), 8 probes per config (4 math + 4 semantic), scored with distribution-based partial credit
- **Total compute:** ~2 hours on Vast.ai, $1.72
- **Code:** Open-sourced at [github.com/Diomandeee/nko-brain-scanner](https://github.com/Diomandeee/nko-brain-scanner)

## References

1. Ng, David Noel. "Reasoning Yield from Stacking (RYS)." October 2024.
2. Kante, Solomana. N'Ko Writing System. 1949.
3. Unicode Consortium. "N'Ko Block: U+07C0-U+07FF." The Unicode Standard.
4. Yang et al. "Qwen2 Technical Report." Alibaba Group, 2024.
