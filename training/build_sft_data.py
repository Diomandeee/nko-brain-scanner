#!/usr/bin/env python3
"""
Build SFT training data for N'Ko fine-tuning from all available sources.

Generates chat-format JSONL for MLX LoRA training:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Sources:
  1. parallel_corpus.jsonl (460 pairs)
  2. nko-wikipedia.txt (4,146 lines) -> completion tasks
  3. proverbs-unified.json (62 proverbs) -> interpretation tasks
  4. greetings-unified.json (23 protocols) -> conversation tasks
  5. blessings-unified.json (29 entries) -> cultural tasks
  6. nko-cultural.txt (154 lines) -> reading comprehension
  7. nko-vocabulary.txt (457 lines) -> vocabulary tasks

Output: training/train.jsonl, training/valid.jsonl (90/10 split)
"""

import json
import random
from pathlib import Path

NKO_DATA = Path.home() / "Desktop" / "NKo" / "nko" / "data"
BRIDGE_DATA = Path.home() / "Desktop" / "cross-script-bridge" / "data"
SCANNER_DATA = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent

random.seed(42)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_lines(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def msg(user_text, assistant_text):
    """Create a chat message pair."""
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def extract_parallel_corpus():
    """460 parallel NKo/English pairs -> translation tasks in both directions."""
    examples = []
    path = SCANNER_DATA / "parallel_corpus.jsonl"
    with open(path) as f:
        for line in f:
            entry = json.loads(line.strip())
            en = (entry.get("english", "") or "").strip()
            nko = (entry.get("nko", "") or "").strip()
            if not en or not nko:
                continue

            # EN -> NKO
            examples.append(msg(
                f"Translate this to N'Ko script: {en}",
                nko
            ))
            # NKO -> EN
            examples.append(msg(
                f"Translate this N'Ko text to English: {nko}",
                en
            ))
            # With IPA if available
            ipa = (entry.get("ipa", "") or "").strip()
            if ipa:
                examples.append(msg(
                    f"What is the IPA transcription of this N'Ko text: {nko}",
                    ipa
                ))

    print(f"  Parallel corpus: {len(examples)} examples")
    return examples


def extract_wikipedia():
    """N'Ko Wikipedia text -> completion and comprehension tasks."""
    examples = []
    path = NKO_DATA / "corpus" / "nko-wikipedia.txt"
    lines = load_lines(path)

    # Group lines into paragraphs (2-5 lines each)
    paragraphs = []
    current = []
    for line in lines:
        if len(line) < 5:  # skip very short lines
            continue
        current.append(line)
        if len(current) >= 3:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))

    for para in paragraphs:
        if len(para) < 20:
            continue

        # Completion task: give first half, predict second half
        words = para.split()
        if len(words) >= 6:
            mid = len(words) // 2
            prefix = " ".join(words[:mid])
            suffix = " ".join(words[mid:])
            examples.append(msg(
                f"Continue this N'Ko text: {prefix}",
                suffix
            ))

        # Reading comprehension: "Read this N'Ko passage"
        examples.append(msg(
            f"Read the following N'Ko text and provide a summary in English: {para}",
            f"This is a passage in N'Ko script about a topic from the N'Ko Wikipedia. "
            f"The text contains {len(words)} words in Manding language written in N'Ko script."
        ))

    print(f"  Wikipedia: {len(examples)} examples")
    return examples


def extract_proverbs():
    """62 proverbs -> interpretation, cultural context, translation tasks."""
    examples = []
    path = NKO_DATA / "cultural" / "proverbs-unified.json"
    data = load_json(path)

    proverbs = []
    if isinstance(data, dict):
        for section in data.values():
            if isinstance(section, list):
                proverbs.extend(section)
    elif isinstance(data, list):
        proverbs = data

    for p in proverbs:
        nko = (p.get("text_nko") or "").strip()
        latin = (p.get("text_latin") or "").strip()
        en = (p.get("english_translation") or "").strip()
        meaning = (p.get("meaning") or "").strip()
        literal = (p.get("literal_translation") or "").strip()

        if not en:
            continue

        if nko:
            # Interpret proverb
            examples.append(msg(
                f"What does this N'Ko proverb mean? {nko}",
                f"{en}. {meaning}" if meaning else en
            ))
            # Translate proverb
            examples.append(msg(
                f"Translate this N'Ko proverb to English: {nko}",
                en
            ))

        if latin and en:
            # Latin transliteration -> English
            examples.append(msg(
                f"What does the Manding proverb '{latin}' mean?",
                f"{en}. {meaning}" if meaning else en
            ))

        if nko and literal:
            examples.append(msg(
                f"Give the literal translation of: {nko}",
                literal
            ))

    print(f"  Proverbs: {len(examples)} examples")
    return examples


def extract_greetings():
    """23 greeting protocols -> conversation, cultural etiquette tasks."""
    examples = []
    path = NKO_DATA / "cultural" / "greetings-unified.json"
    data = load_json(path)

    greetings = data if isinstance(data, list) else data.get("greetings", [])
    for g in greetings:
        nko = (g.get("text_nko", "") or "").strip()
        en = (g.get("english_translation", "") or "").strip()
        response = (g.get("expected_response") or g.get("response") or "").strip()
        context = (g.get("context") or g.get("usage_context") or "").strip()

        if not nko:
            continue

        if en:
            examples.append(msg(
                f"How do you say '{en}' in N'Ko?",
                nko
            ))
            examples.append(msg(
                f"What does this N'Ko greeting mean? {nko}",
                en
            ))

        if response:
            resp_nko = ""
            if isinstance(response, dict):
                resp_nko = response.get("text_nko", "")
            elif isinstance(response, str):
                resp_nko = response

            if resp_nko:
                examples.append(msg(
                    f"Someone greets you with: {nko}. How do you respond in N'Ko?",
                    resp_nko
                ))

        if context:
            ctx = context if isinstance(context, str) else str(context)
            examples.append(msg(
                f"When would you use this N'Ko greeting: {nko}?",
                ctx
            ))

    print(f"  Greetings: {len(examples)} examples")
    return examples


def extract_blessings():
    """29 blessings -> cultural/religious tasks."""
    examples = []
    path = NKO_DATA / "cultural" / "blessings-unified.json"
    data = load_json(path)

    blessings = data if isinstance(data, list) else data.get("blessings", [])
    for b in blessings:
        nko = (b.get("text_nko", "") or "").strip()
        en = (b.get("english_translation", "") or "").strip()
        life_event = (b.get("life_event") or b.get("usage") or "").strip()

        if not nko or not en:
            continue

        examples.append(msg(
            f"Translate this N'Ko blessing: {nko}",
            en
        ))

        if life_event:
            evt = life_event if isinstance(life_event, str) else str(life_event)
            examples.append(msg(
                f"What N'Ko blessing is appropriate for {evt}?",
                f"{nko} ({en})"
            ))

    print(f"  Blessings: {len(examples)} examples")
    return examples


def extract_cultural_text():
    """Raw N'Ko cultural text -> reading tasks."""
    examples = []
    path = NKO_DATA / "corpus" / "nko-cultural.txt"
    lines = load_lines(path)

    for line in lines:
        if len(line) < 10:
            continue
        examples.append(msg(
            f"What does this N'Ko text say? {line}",
            f"This is a N'Ko cultural text. The text reads: {line}"
        ))

    print(f"  Cultural text: {len(examples)} examples")
    return examples


def extract_vocabulary():
    """Vocabulary definitions -> word-level tasks."""
    examples = []
    path = NKO_DATA / "corpus" / "nko-vocabulary.txt"
    lines = load_lines(path)

    for line in lines:
        # Lines typically have format: NKo_word - definition or NKo_word (transliteration) = meaning
        if len(line) < 5:
            continue

        # Check if line contains NKO characters
        has_nko = any(0x07C0 <= ord(ch) <= 0x07FF for ch in line)
        if has_nko:
            examples.append(msg(
                f"Define this N'Ko vocabulary word: {line.split()[0] if line.split() else line}",
                line
            ))

    print(f"  Vocabulary: {len(examples)} examples")
    return examples


def extract_concepts():
    """Cultural concepts -> explanation tasks."""
    examples = []
    path = NKO_DATA / "cultural" / "cultural-concepts.json"
    if not path.exists():
        return examples

    data = load_json(path)
    concepts = data if isinstance(data, list) else data.get("concepts", [])

    for c in concepts:
        nko = (c.get("text_nko", "") or "").strip()
        en = (c.get("english_translation", "") or "").strip()
        usage = (c.get("usage", "") or "").strip()

        if not nko or not en:
            continue

        explanation = en
        if usage:
            explanation = f"{en}. {usage}"

        examples.append(msg(
            f"What is the Manding cultural concept '{nko}'?",
            explanation
        ))

    print(f"  Concepts: {len(examples)} examples")
    return examples


def extract_keyboard_vocab():
    """Keyboard vocabulary -> quick reference tasks."""
    examples = []
    path = NKO_DATA / "keyboard" / "vocabulary.json"
    if not path.exists():
        return examples

    data = load_json(path)
    for category, entries in data.items():
        if not isinstance(entries, dict):
            continue
        for nko_word, info in entries.items():
            if isinstance(info, dict):
                meaning = info.get("meaning", info.get("english", ""))
                transliteration = info.get("transliteration", "")
            elif isinstance(info, str):
                meaning = info
                transliteration = ""
            else:
                continue

            if meaning:
                examples.append(msg(
                    f"What does '{nko_word}' mean in N'Ko?",
                    f"{meaning}" + (f" (transliteration: {transliteration})" if transliteration else "")
                ))

    print(f"  Keyboard vocab: {len(examples)} examples")
    return examples


def main():
    print("Building SFT training data from all N'Ko sources...\n")

    all_examples = []

    all_examples.extend(extract_parallel_corpus())
    all_examples.extend(extract_wikipedia())
    all_examples.extend(extract_proverbs())
    all_examples.extend(extract_greetings())
    all_examples.extend(extract_blessings())
    all_examples.extend(extract_cultural_text())
    all_examples.extend(extract_vocabulary())
    all_examples.extend(extract_concepts())
    all_examples.extend(extract_keyboard_vocab())

    # Shuffle
    random.shuffle(all_examples)

    # Split 90/10
    split = int(len(all_examples) * 0.9)
    train = all_examples[:split]
    valid = all_examples[split:]

    # Write
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(OUT_DIR / "valid.jsonl", "w") as f:
        for ex in valid:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_examples)} examples")
    print(f"Train: {len(train)} | Valid: {len(valid)}")
    print(f"Written to: {OUT_DIR / 'train.jsonl'}")
    print(f"            {OUT_DIR / 'valid.jsonl'}")

    # Stats
    nko_count = sum(
        1 for ex in all_examples
        if any(0x07C0 <= ord(ch) <= 0x07FF for ch in ex["messages"][0]["content"])
        or any(0x07C0 <= ord(ch) <= 0x07FF for ch in ex["messages"][1]["content"])
    )
    print(f"\nExamples containing N'Ko script: {nko_count}/{len(all_examples)} ({nko_count/len(all_examples)*100:.0f}%)")


if __name__ == "__main__":
    main()
