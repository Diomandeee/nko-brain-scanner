# Does Every AI Have the Same Blind Spot?

*Testing whether N'Ko invisibility is a universal property of language models, or a quirk of one.*

---

## The Original Finding

The first N'Ko brain scan found something uncomfortable.

Qwen3-8B, an 8-billion-parameter model trained on trillions of tokens, processed N'Ko text with measurably less activation than English at every single layer. More dead neurons. Less information being distributed. Flatter circuits. The model wasn't failing because N'Ko is difficult. It was failing because it had barely seen the script in training.

The technical name for this is an activation deficit. When a layer processes unfamiliar input, fewer neurons fire. The ones that do fire produce weaker signals. The result is a flatter, sparser activation profile across all 4,096 hidden dimensions. You can measure this with four numbers: L2 norm (how loudly is the layer speaking?), Shannon entropy (how spread out is the information?), sparsity (what fraction of neurons are essentially turned off?), and kurtosis (how specialized are the active circuits?).

For N'Ko, all four metrics pointed the same direction. The model was running on reduced capacity. It was processing N'Ko text with the cognitive equivalent of one hand behind its back.

But Qwen3-8B is one model, from one company, trained in one particular way.

Does every AI have this same blind spot?

---

## What We're Testing

Experiment A asks a direct question: is N'Ko's invisibility specific to Qwen, or is it a structural property of models trained on data where N'Ko barely exists?

The hypothesis is that it's structural. Every tested model will show the same activation deficit for N'Ko, because all of them were trained predominantly on Latin and CJK text. The gap isn't about architecture. It's about data.

To test this, we're running the same brain scan on four models from different families:

**Qwen3-8B** (8B parameters, Qwen architecture, 36 layers) is the baseline from the original experiment. It gives us a replication check before we start comparing.

**Llama-3.1-8B** (8B parameters, LLaMA architecture, 32 layers) is Meta's open model. Different architecture, different training data composition, similar scale. Meta has published that Llama 3.1 was trained on 15 trillion tokens with an intentional push toward multilingual coverage. How that translated into actual N'Ko representation is an open question.

**Gemma-3-12B** (12B parameters, Gemma architecture, 40 layers) is Google's model. Slightly larger, different layer count, different tokenizer. Google's data pipelines are separate from both Alibaba's and Meta's.

**Qwen2-72B** (72B parameters, Qwen architecture, 80 layers) is nine times larger than the base Qwen3 model. If scale alone fixes the problem, the 72B version should show a narrower N'Ko gap. If it doesn't, that tells us something important: more parameters trained on the same skewed data distribution doesn't fix the underlying problem.

All four models are fed the same 100 parallel English/N'Ko sentence pairs from the project's corpus. Same sentences, same metrics, same method. The only variable is the model.

---

## How the Scan Works

For each model, we load it in 4-bit or 8-bit quantization depending on available memory. The three smaller models run locally on an Apple M4. The 72B model runs on a Vast.ai A100, where we can load it in 4-bit with enough head room to run inference without exhausting VRAM.

At every layer, we extract the full hidden state tensor and compute the four core metrics. The hidden state is the model's internal representation at that stage of processing, a vector of 4,096 numbers (or more, for the larger models) that encodes what the model understands about the input so far.

We also compute the translation tax: the ratio of N'Ko perplexity to English perplexity. Perplexity is a direct measure of how confused the model is. If the model has fully learned a language, perplexity is low. If the model is essentially guessing token by token, perplexity is high. The ratio tells us how much harder N'Ko is for each model, independent of the model's overall quality.

After all four scans are done, a comparison script pulls all the results together and looks for patterns. Does every model show a sparsity spike at the same layers? Does the translation tax scale with model size? Are the activation curves shaped similarly, or does each architecture fail in a different way?

---

## Why the Answer Matters

There are two possible outcomes, and both are useful.

If all four models show the same activation deficit, that's strong evidence the problem is in the training data, not the architecture. No amount of architectural innovation will fix N'Ko invisibility if training datasets keep underrepresenting the script. The fix has to be data-level.

If some models handle N'Ko better than others, that points to something else. Maybe certain architectures generalize better across scripts. Maybe certain training data compositions matter more than total scale. Maybe the tokenizer design affects how well the model can handle a right-to-left script with a compact Unicode block.

Either outcome changes how you think about solving the problem.

---

## Results

Experiment in progress. Results will be published here when available.
