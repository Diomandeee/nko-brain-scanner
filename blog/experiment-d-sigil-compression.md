# Can 10 Characters Encode a Lifetime of Conversation?

*N'Ko sigils as a compression layer for conversational knowledge. This is what we're testing.*

---

## The Compression Problem

Every conversation you have with an AI system generates tokens. A short exchange might be 200 tokens. A deep technical session can run to 10,000 or more. The session ends. Those tokens get stored in a database. Later, when you need context from that conversation, you load the tokens back into a model's context window.

This works, but it's expensive. Context windows have limits. Loading full conversation history costs compute. At scale, across thousands of sessions over months or years, the storage and retrieval costs become substantial.

There's a deeper problem than cost, though. Raw tokens are not meaning. A 3,000-token conversation contains a lot of repetition, preamble, clarification, filler. The actual informational content, the things that changed your understanding or produced a useful output, might be representable in far fewer bits. The question is: what's the right compression function?

Standard compression (gzip, LZ4) removes byte-level redundancy. It doesn't understand semantics. It will compress "the model improved after retraining" and "my cat sat on the mat" at roughly the same rate because both have similar byte patterns. You want something that compresses based on meaning, not bytes.

This is where N'Ko sigils come in.

---

## The 10 Sigils

The N'Ko brain scanner project produced a side effect from its activation analysis: a set of 10 characters from the N'Ko Unicode block, each mapped to a semantic category derived from patterns in how the model's internal activations behave across conversation turns.

These aren't arbitrary labels. Each sigil represents a dynamic pattern, a characterization of how information is moving:

**ߛ (stabilization)**: The conversation is settling. Dispersion decreased. The model found its footing on this topic.

**ߜ (dispersion)**: The conversation is spreading out. Multiple threads opening simultaneously.

**ߕ (transition)**: A change point. The topic, the register, or the framing shifted.

**ߙ (return)**: Re-entry to familiar territory. The conversation came back to something it had covered before.

**ߡ (dwell)**: Sustained engagement. The conversation stayed on one thing for longer than expected.

**ߚ (oscillation)**: Rapid alternation. Back-and-forth between two positions or framings.

**ߞ (recovery)**: Return after confusion. The model had to correct course, and did.

**ߣ (novelty)**: New basin. Something in the conversation had no prior pattern to match against.

**ߠ (place shift)**: A location change in the conceptual space. Not a topic change, but a change in how the topic is being approached.

**ߥ (echo)**: Pattern match. This turn closely mirrors something that came before.

The hypothesis is that any conversation turn can be characterized by a short sequence of these patterns. A 500-token turn that represents a moment of recovery from confusion, followed by stabilization on a new framing, might be encoded as ߞߛ. Two characters. The information-theoretic shape of the conversation survives in 2 characters that would otherwise require 500 tokens to store.

---

## What We're Measuring

The compression experiment has three stages.

First, we sample 1,000 conversation turns from the project's conversation history. These are real turns from real sessions, not synthetic data. We tokenize them with GPT-4's tokenizer (tiktoken, cl100k_base) to get a baseline BPE token count for each turn.

Second, we run the transliteration bridge to convert each turn to N'Ko. This gives us a second representation of each turn and a second measurement: how many N'Ko characters does the turn become? N'Ko's compact Unicode block and regular phoneme-to-character mapping mean the character count will differ from the English character count, and the comparison is informative on its own.

Third, we run the sigil encoder. For each turn, it extracts the top-k concepts (what is this turn actually about?), analyzes content stability versus change, detects topic repetition versus novelty, identifies sustained focus versus rapid switching, and flags recovery patterns. It maps all of this to a sigil sequence of 1 to 5 characters.

The metric we care about most is the compression ratio: BPE token count divided by sigil sequence length. If the hypothesis holds, the ratio should be in the range of 50x to 100x. A 500-token turn becomes 5 sigils. A 200-token turn becomes 3 sigils. If the ratio is consistently below 20x, the sigils aren't compressing enough. If it's above 200x, we're almost certainly losing too much information.

The second metric is consistency. The same topic should produce similar sigils. Two turns that are both about debugging a failing test should produce overlapping sigil sequences even if the specific code and error messages are completely different. If the sigil assignments are inconsistent, the encoding isn't capturing the underlying semantic pattern.

The third metric is reconstructibility. Can you describe what kind of conversation produced a given sigil sequence, even if you can't recover the exact text? This is measured with human evaluation: given only the sigil sequence, how well can someone characterize the conversation? Perfect reconstruction is not the goal. Bounded information loss is.

---

## Why N'Ko Specifically

You could build this kind of semantic compression system using any symbol set. You could use emoji. You could use arbitrary integers. You could use ASCII characters.

N'Ko is a specific choice with a specific rationale.

N'Ko characters are Unicode characters in a block that was designed to be self-contained and semantically coherent. The script has a visual grammar: characters flow right to left, combining marks attach to base characters, the visual structure carries information about morphological relationships. Using N'Ko sigils means the compression system is using a symbol set with structural regularity, not a grab-bag of arbitrary codes.

There's also a cultural argument. The sigil system is being built for a cognitive twin trained on data from Manding-speaking contexts. Using characters from a script designed for Manding languages to encode the semantic patterns of conversations in that domain creates a kind of alignment between the representation layer and the content layer. The symbols belong to the domain they're describing.

Whether that alignment has practical consequences is an open question. But it's worth testing.

---

## Results

Experiment in progress. Results will be published here when available.
