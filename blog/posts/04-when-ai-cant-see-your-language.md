# When AI Can't See Your Language

In 1949, a man named Solomana Kante sat down in Kankan, Guinea, and did something that most linguists said was impossible.

He designed a writing system.

Not borrowed, not adapted from something else. Built from scratch, character by character, for the Manding languages of West Africa. He did it because someone had made a public claim that African languages were inherently unsuitable for writing. Kante, who was self-taught and spoke seven languages, took that as a challenge.

He spent years studying the sounds of Bambara, Maninka, and Dioula. He listened to how the languages actually worked. And then he built a writing system perfectly calibrated to them.

Every sound gets exactly one character. Every character represents exactly one sound. Tones are written out. No exceptions, no weird spelling rules, no silent letters. If you know the script, you know how to pronounce any word. If you can pronounce a word, you can write it.

He called it N'Ko. In every Manding language, that means "I say."

The script has over 40 million speakers today. It's used for education, trade, religious texts, personal letters, and signs across Guinea, Mali, Cote d'Ivoire, Senegal, Burkina Faso, and diaspora communities worldwide.

No AI system had ever learned to read it.

That's what I set out to fix. Here's what happened when I looked inside one of the most powerful AI models in the world and asked it to process text written in N'Ko.

---

## The Experiment

I ran something I started calling a brain scan.

Take one of the most capable AI language models available right now. Feed it 100 sentences. Some in English, some in N'Ko. Same meaning in both. Then watch what happens inside the model at every stage of processing.

Modern AI language models are built like layers of analysis stacked on top of each other. The first layer reads the raw characters. The middle layers think about what the characters mean. The final layers decide what to say back. I measured what the model's internal computations look like at each of those stages for English versus N'Ko.

The results were striking.

Before I get to them, here is one fact you need to understand: this model, called Qwen2-72B-Instruct, has a vocabulary of 151,936 entries. Think of the vocabulary as the model's dictionary. Arabic, another non-Western writing system, has over 4,200 entries in that dictionary. Arabic words, Arabic syllables, Arabic common phrases.

N'Ko has 32. Just the individual characters. No words. No syllables. No common phrases. Every single N'Ko word has to be spelled out character by character, like if someone asked you to read English but had never taught you that "the" is a single word, and made you read it as three separate letters every single time.

---

## The Translation Tax

Here is the simplest way I can describe what we found.

Imagine you are lifting a heavy box. The model, when reading English, lifts with full strength. We can measure that effort. When reading N'Ko, the model lifts with about a third of its strength. Not because N'Ko is heavier. Because the model was never trained to lift it.

We call this the translation tax. The model expends 2.9 times less effort on N'Ko text than on English text, right from the very first layer of analysis. And it gets worse as you go deeper. By the time the model is trying to figure out what to say back, it's working at 3.26 times less capacity for N'Ko than for English.

This is not how it should work. If the model understood N'Ko, both numbers should be about equal. The gap tells you the model is not really understanding N'Ko. It is guessing.

But we can be even more specific about how it fails. We found three distinct failure zones, and they correspond to three different things the model needs to do to understand any text.

---

## Three Zones of Failure

Think of reading any sentence as having three jobs.

**Job one:** Read the words and recognize what they are. For English, the model does this well. It has rich internal representations for English words, built up from seeing enormous amounts of English text during training. For N'Ko, it has almost nothing. The model reads each N'Ko character and produces something close to a random guess about what kind of character this is. The embeddings, which is what we call the model's internal representation of each character, are sparse and weak. Many of the model's 4,096 internal dimensions are just inactive, sitting at zero. For English, only about 14% of those dimensions are inactive at the reading stage. For N'Ko, it's 34.5%. More than twice as many.

**Job two:** Think about what the words mean in context. Figure out grammar. Build up an understanding of what the sentence is saying. For English, you can watch this happening layer by layer inside the model. It builds richer and richer representations as you go deeper. For N'Ko, the model is essentially just passing the original weak readings through without adding much. There's no thinking happening. We confirmed this with a separate experiment where we tried to amplify whatever N'Ko reasoning circuits existed in the model. We tested 55 different ways of doing this. All 55 failed. There is nothing to amplify. The circuits are not weak. They're absent.

**Job three:** Produce an answer. Pick the right words. For English, the model gets more and more confident as it approaches a prediction. If you measure how "peaked" its probability distribution is (is it pretty sure it's this word, or is it spreading bets across many words), that peaking increases all the way to the final layer. For N'Ko, the model actually gets less confident at the end. It peaks a little bit around the middle of processing, then spreads back out before it has to commit to an answer. At the moment it should be most certain, it pulls back. The result is that its probability mass is spread almost uniformly across its entire 151,936-word vocabulary. It cannot tell a difference between an N'Ko word, a Chinese character, some code, or an English word. It has no idea.

---

## The Fix Was Easy

This is the part that should make you angry, or maybe hopeful, depending on your disposition.

After documenting all of that failure, I trained the model to do better. On a laptop. In three hours.

The training used text from N'Ko Wikipedia, about 3.7 million characters, alongside some structured instruction data. The whole thing cost zero dollars in cloud computing because it ran on the laptop's processor.

After training, the translation tax dropped from 2.94x to 0.70x. That last number means the model is now working harder on N'Ko than on English. It has slightly over-calibrated, which is a known side effect of this kind of training, but the point is: the model went from barely functional to performing better than it does on English on the same tasks.

The embedding sparsity dropped by 47%. The model's output layer went from producing a nearly flat probability distribution to producing concentrated, peaked predictions. The 78.1% kurtosis deficit, which measures how much the model was un-committing at the final layer, shrank by 67%.

Three hours. On a laptop. Nearly 76% of the problem, closed.

The problem was never the script. The problem was always the data.

---

## What This Means for 40 Million People

A child in Kankan, Guinea, who speaks Maninka and reads N'Ko cannot send a voice message in her own writing system. Cannot search the internet in her own script. Cannot talk to any AI assistant in the writing that her community uses.

Every voice interface, every smart keyboard, every autocomplete, every language model responds in Latin characters. The letters designed for French colonial administrators in the 20th century, not for the people who actually speak and read these languages.

This is not a small inconvenience. It compounds. Every time you have to switch to a foreign writing system, there is friction. Friction in education, because your textbooks assume one script but your teacher writes in another. Friction in commerce, because your receipts and contracts are in something other than your mother tongue's script. Friction in self-expression, because the tools that everyone uses for everything were built for someone else.

Solomana Kante understood this. He spent his life not just designing N'Ko but teaching it, standardizing it, keeping it alive. He died in 1987. The writing system he built in response to a claim of African linguistic inferiority is now used across six countries and has Wikipedia in it.

And AI has ignored it for its entire existence.

The hard part is not technical. We proved that. Three hours and a laptop is all it takes to fix the problem for one script. The hard part is awareness. The hard part is that the people building these systems have to decide it matters.

---

## N'Ko Is Not Alone

N'Ko is one example of a broader pattern.

Adlam is a writing system designed in 1989 by two brothers from Guinea for the Fulani language. Fulani has around 40 million speakers across West Africa and into Central Africa. The Adlam script was specifically designed to have the same kind of phonemic precision as N'Ko, one character for every sound, no ambiguity. The major AI models have zero vocabulary entries for Adlam. Not even the 32 individual characters that N'Ko gets. Zero.

Tifinagh is the ancient script of the Amazigh people, the indigenous people of North Africa, sometimes called Berbers. Thirty million speakers. Zero vocabulary entries.

Osmanya was designed in the 1920s for the Somali language, which has around 16 million speakers. Zero.

These scripts share something besides their absence from AI training data. They were all designed by indigenous communities, often in explicit response to colonial pressure to use Western or Arabic scripts instead. Their absence from AI systems is a second chapter of the same story. The first chapter was the external pressure to not write in these scripts. The second chapter is that the digital tools built for writing mostly exclude them.

The method we used for N'Ko works for all of them. A bridge to convert existing Latin text into the target script. A training run on the available text. Three hours. The cost of not doing it is paid by 40 million people at a time.

---

## What We Actually Built

This is not just about making an AI model read N'Ko text.

We also built the first system that can listen to someone speaking Bambara and write down what they said in N'Ko.

Not in Latin characters, the way every other speech recognition system for Bambara works. In N'Ko. The writing system that actually belongs to the language.

It works in real time. You speak Bambara. The system writes what you said in N'Ko script. If you want, it also translates to English or French, under 300 milliseconds from when you stop speaking to when you see the text.

That has never existed before. This is the first time a machine has written in N'Ko from speech.

The app is called NKoScribe and it is on TestFlight right now.

---

Solomana Kante built a writing system from scratch in 1949 because someone said it couldn't be done.

The writing system he built has properties that computer scientists dream about. Every phoneme maps to exactly one character. No digraphs, no silent letters, no exceptions. Explicit tone marking. Completely unambiguous. If you trained an AI system to write in N'Ko, you would find it easier than training one to write in English, because English spelling is genuinely chaotic and N'Ko is not.

He built all of that without computers, without training data, without any of the tools we now consider essential for this kind of work. Just a systematic study of the languages he grew up with and the belief that they deserved to be written down correctly.

The least we can do is make sure the machines can read it.
