# What If Your AI Twin Thought in Your Mother Tongue?

*A cognitive twin trained on English conversation gets retrained to operate in N'Ko. This is what we're looking for.*

---

## What a Cognitive Twin Is

A cognitive twin is a language model fine-tuned on a specific person's conversation history. Not a general assistant. A specific, personalized model that has seen your questions, your reasoning patterns, your way of framing problems. It learns to respond the way you think, not the way a generic chatbot does.

The technical implementation involves collecting your conversation data as supervised fine-tuning (SFT) examples, message pairs where the user side is a question or prompt and the assistant side is the kind of response you'd actually want. You train a LoRA adapter on that data, a small set of weight updates that shift a base model's behavior toward your patterns without replacing everything the base model knows.

The result is something that feels distinctly familiar. It has your vocabulary preferences. It matches your register. It picks up on the framing you use for certain kinds of problems. It's not magic. It's pattern matching over your actual language use. But pattern matching done well, at scale, on a model that already understands language deeply, produces something that feels personal in a way generic models don't.

All of the existing cognitive twin work was done in English. Every training example was English. Every evaluation prompt was English. The model learned to reason in English because that's the language the training data was in.

Experiment C asks a different question: what happens when the training data is in N'Ko?

---

## The Three Predictions

The hypothesis has three distinct parts, and each one is independently falsifiable.

The first prediction is token compression. N'Ko is an agglutinative script with a compact Unicode block. Agglutinative languages pack semantic content efficiently: affixes attach to roots to build meaning, reducing the number of separate tokens needed to express a concept compared to analytical languages like English. If the N'Ko twin has internalized this compression, it should produce equivalent responses in fewer tokens. We're setting a threshold: a compression ratio greater than 1.2x counts as a confirmed compression effect.

The second prediction is cultural perplexity. Perplexity measures how confident the model is about its own output. Low perplexity means the model is generating text it has essentially seen before, operating in familiar territory. High perplexity means it's working harder, building each token from less certain foundations. The N'Ko cognitive twin was trained on data that includes Manding cultural context, greetings, proverbs, social structures that the English twin never saw. On prompts that draw on Manding culture, the N'Ko twin should show lower perplexity. It should be more at home.

The third prediction is behavioral divergence. This is the most speculative of the three, and the hardest to measure. The hypothesis is that the internal representation change from English to N'Ko produces a subtle but measurable shift in the twin's behavior. Not a personality replacement. The twin should still be recognizable. But the way it structures responses, the concepts it reaches for first, the register it defaults to, those should shift when the model's entire internal vocabulary is encoded in a different script.

---

## The Method

Step one is translation. We take the existing English SFT training data and convert it to N'Ko. Content in Manding or Bambara Latin gets transliterated directly using the bridge script, which handles the phonetic mapping from Latin-alphabet Manding to N'Ko characters. Content in English gets phonetically transliterated to N'Ko, which is an experimental approach that essentially encodes English sounds using N'Ko character forms.

This step is where the most uncertainty lives. The transliteration bridge works well for Manding content because it was designed for that mapping. Phonetically encoding English in N'Ko characters is experimental. The resulting text is phonetically valid N'Ko, in the sense that you could read it aloud and hear English. But it's not conventional N'Ko writing. We're curious whether the model can still learn from it.

Step two is training. We run a LoRA adapter training pass on the N'Ko SFT data using the same hyperparameters as the English adapter. Same base model (Qwen3-8B), same LoRA rank and alpha, same number of training iterations, same learning rate schedule. The only change is the data.

Step three is comparison. We run both adapters (English and N'Ko) against the same 50 evaluation prompts and measure perplexity, token count, response similarity (using BLEU and cosine similarity), and behavioral evaluation using a structured rubric.

---

## The Blocking Dependency

This experiment can't start until the English cognitive twin's SFT training is complete. The comparison only makes sense if both adapters were trained on parallel data, the same underlying conversations, one in English and one in N'Ko. Training the N'Ko adapter on a different set of examples would contaminate the comparison: you wouldn't know if differences came from the script or from the training examples.

This dependency is deliberate, not a limitation. The whole point is to isolate the script variable. Everything else has to be controlled.

---

## Why This Matters Beyond the Research

The framing of "AI that thinks in your mother tongue" sounds like a product pitch. But the research question underneath it is real.

Most people who speak Manding languages also speak French or English. They navigate multiple languages constantly. Their inner voice, the way they think when they're not performing for an audience, is often in the language they grew up speaking. For many Manding speakers, that's not English.

A cognitive twin trained in English models the English-language version of your thinking. It captures your patterns in your second or third language. The N'Ko cognitive twin experiment is a first attempt at asking whether modeling your thinking in your first language produces something measurably different.

The answer might be no. The behavioral divergence prediction might fail entirely. The twin might turn out to be structurally identical regardless of which script the weights were updated on.

But it might not be no. And finding out is worth doing.

---

## Results

Experiment in progress. Results will be published here when available.
