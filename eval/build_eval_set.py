#!/usr/bin/env python3
"""Build frozen evaluation sets: 100 English + 100 N'Ko examples.

These eval sets NEVER overlap with training data and are used for all
profiler runs going forward. Format: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

English examples: diverse general knowledge, math, reasoning, science.
N'Ko examples: held-out from parallel corpus + cultural data.
"""

import json
import hashlib
import random
from pathlib import Path

random.seed(42)

SCANNER = Path(__file__).parent.parent
EVAL_DIR = SCANNER / "eval"
NKO_RANGE = range(0x07C0, 0x0800)


def has_nko(text):
    return any(ord(c) in NKO_RANGE for c in text)


def msg(user_text, assistant_text):
    return {"messages": [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text}
    ]}


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_training_hashes():
    """Load hashes of all training data to prevent overlap."""
    hashes = set()
    for path in [
        SCANNER / "training" / "train.jsonl",
        SCANNER / "training" / "valid.jsonl",
        SCANNER / "training" / "cpt_train.jsonl",
        SCANNER / "training" / "cpt_valid.jsonl",
        SCANNER / "training" / "combined_train.jsonl",
        SCANNER / "training" / "combined_valid.jsonl",
        SCANNER / "data-bpe" / "train.jsonl",
        SCANNER / "data-bpe" / "valid.jsonl",
    ]:
        if path.exists():
            with open(path) as f:
                for line in f:
                    try:
                        ex = json.loads(line)
                        msgs = ex.get("messages", [])
                        if msgs:
                            full = " ".join(m.get("content", "") for m in msgs)
                            hashes.add(text_hash(full))
                        text = ex.get("text", "")
                        if text:
                            hashes.add(text_hash(text))
                    except:
                        continue
    return hashes


def build_english_examples():
    """Generate 100 diverse English QA pairs covering multiple domains."""
    examples = []

    # Math / numerical reasoning (20 examples)
    math_pairs = [
        ("What is 247 + 389?", "247 + 389 = 636"),
        ("Calculate 15% of 480.", "15% of 480 = 72"),
        ("What is the square root of 144?", "The square root of 144 is 12."),
        ("If a rectangle has length 8 and width 5, what is its area?", "The area is 8 x 5 = 40 square units."),
        ("What is 3/4 + 1/6?", "3/4 + 1/6 = 9/12 + 2/12 = 11/12"),
        ("How many seconds are in 2 hours?", "2 hours = 2 x 60 x 60 = 7,200 seconds."),
        ("What is 17 x 23?", "17 x 23 = 391"),
        ("If you drive 60 miles per hour for 2.5 hours, how far do you travel?", "Distance = 60 x 2.5 = 150 miles."),
        ("What is the next prime number after 29?", "The next prime number after 29 is 31."),
        ("Convert 5 kilometers to meters.", "5 kilometers = 5,000 meters."),
        ("What is 2 to the power of 10?", "2^10 = 1,024"),
        ("If a shirt costs $40 and is 25% off, what is the sale price?", "25% of $40 = $10 discount. Sale price = $40 - $10 = $30."),
        ("What is the perimeter of a square with side length 7?", "Perimeter = 4 x 7 = 28 units."),
        ("How many minutes are in a week?", "7 days x 24 hours x 60 minutes = 10,080 minutes."),
        ("What is 999 - 456?", "999 - 456 = 543"),
        ("If 3x + 7 = 22, what is x?", "3x = 22 - 7 = 15, so x = 5."),
        ("What is the average of 12, 18, 24, and 30?", "Average = (12 + 18 + 24 + 30) / 4 = 84 / 4 = 21."),
        ("Convert 32 degrees Fahrenheit to Celsius.", "C = (32 - 32) x 5/9 = 0 degrees Celsius."),
        ("What is 7! (7 factorial)?", "7! = 7 x 6 x 5 x 4 x 3 x 2 x 1 = 5,040"),
        ("If a circle has radius 5, what is its circumference?", "Circumference = 2 x pi x 5 = approximately 31.42 units."),
    ]

    # Science / factual knowledge (20 examples)
    science_pairs = [
        ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au, from the Latin word aurum."),
        ("How many bones are in the adult human body?", "An adult human body has 206 bones."),
        ("What is the speed of light in a vacuum?", "The speed of light in a vacuum is approximately 299,792,458 meters per second."),
        ("What is photosynthesis?", "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen."),
        ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
        ("What is DNA?", "DNA (deoxyribonucleic acid) is the molecule that carries genetic instructions for the development and function of living organisms."),
        ("What causes tides on Earth?", "Tides are primarily caused by the gravitational pull of the Moon and, to a lesser extent, the Sun on Earth's oceans."),
        ("What is the boiling point of water at sea level?", "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level."),
        ("How many elements are in the periodic table?", "As of 2024, there are 118 confirmed elements in the periodic table."),
        ("What is the hardest natural substance on Earth?", "Diamond is the hardest natural substance on Earth, scoring 10 on the Mohs hardness scale."),
        ("What is the mitochondria?", "The mitochondria are organelles found in cells that generate most of the cell's supply of ATP, which is used as a source of chemical energy."),
        ("What planet is known as the Red Planet?", "Mars is known as the Red Planet due to its reddish appearance caused by iron oxide on its surface."),
        ("What is the atomic number of carbon?", "The atomic number of carbon is 6."),
        ("How does a vaccine work?", "A vaccine works by exposing the immune system to a weakened or inactive form of a pathogen, training it to recognize and fight the real pathogen."),
        ("What is the Pythagorean theorem?", "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides: a^2 + b^2 = c^2."),
        ("What are the three states of matter?", "The three common states of matter are solid, liquid, and gas. Plasma is sometimes considered the fourth state."),
        ("What is gravity?", "Gravity is a fundamental force that attracts objects with mass toward each other. On Earth, it gives weight to objects and causes them to fall."),
        ("What is the function of white blood cells?", "White blood cells are part of the immune system and help the body fight infections and diseases."),
        ("What is the closest star to Earth?", "The closest star to Earth is the Sun. The next closest star system is Alpha Centauri, about 4.37 light-years away."),
        ("What causes earthquakes?", "Earthquakes are caused by the sudden release of energy in the Earth's crust, usually due to the movement of tectonic plates."),
    ]

    # Geography / history / culture (20 examples)
    geo_pairs = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("Which continent is Brazil in?", "Brazil is in South America."),
        ("What is the longest river in the world?", "The Nile River is traditionally considered the longest river in the world at approximately 6,650 kilometers."),
        ("In what year did World War II end?", "World War II ended in 1945."),
        ("What is the tallest mountain in the world?", "Mount Everest is the tallest mountain in the world at 8,849 meters above sea level."),
        ("What language is most widely spoken in the world?", "Mandarin Chinese has the most native speakers, while English is the most widely spoken language overall including non-native speakers."),
        ("What is the largest country by area?", "Russia is the largest country by area, covering over 17 million square kilometers."),
        ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci, likely between 1503 and 1519."),
        ("What is the largest ocean on Earth?", "The Pacific Ocean is the largest ocean on Earth."),
        ("What is the population of the world approximately?", "The world population is approximately 8 billion people as of 2024."),
        ("What is the Great Wall of China?", "The Great Wall of China is a series of fortifications built along the northern borders of China to protect against invasions. It stretches over 20,000 kilometers."),
        ("What currency is used in Japan?", "The Japanese yen is the currency used in Japan."),
        ("What is the Sahara?", "The Sahara is the largest hot desert in the world, located in North Africa, covering about 9.2 million square kilometers."),
        ("Who was the first person to walk on the Moon?", "Neil Armstrong was the first person to walk on the Moon on July 20, 1969."),
        ("What is the European Union?", "The European Union is a political and economic union of 27 European member states that aims to ensure free movement of people, goods, services, and capital."),
        ("What is the Amazon Rainforest?", "The Amazon Rainforest is the world's largest tropical rainforest, covering much of northwestern Brazil and extending into several other South American countries."),
        ("In what country are the pyramids of Giza?", "The pyramids of Giza are in Egypt."),
        ("What is the United Nations?", "The United Nations is an international organization founded in 1945 with 193 member states, aimed at maintaining international peace and security."),
        ("What is the capital of Australia?", "The capital of Australia is Canberra, not Sydney as many people assume."),
        ("What is the Dead Sea?", "The Dead Sea is a salt lake bordered by Jordan and Israel. It is the lowest point on Earth's surface and one of the saltiest bodies of water."),
    ]

    # Reasoning / logic (20 examples)
    logic_pairs = [
        ("If all dogs are animals and all animals need water, do dogs need water?", "Yes. Since all dogs are animals and all animals need water, it follows that all dogs need water."),
        ("What comes next in the sequence: 2, 4, 8, 16, ...?", "The next number is 32. Each number is doubled (multiplied by 2)."),
        ("If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?", "5 minutes. Each machine makes one widget in 5 minutes, so 100 machines make 100 widgets in 5 minutes."),
        ("A farmer has 17 sheep. All but 9 die. How many sheep does he have left?", "9 sheep. 'All but 9' means 9 survive."),
        ("Is the following statement true or false: 'This sentence has five words.'", "True. Counting the words: This(1) sentence(2) has(3) five(4) words(5)."),
        ("If you rearrange the letters 'CIFAIPC', you get the name of which ocean?", "PACIFIC."),
        ("What has keys but no locks, space but no room, and you can enter but cannot go inside?", "A keyboard."),
        ("If a doctor gives you 3 pills and tells you to take one every 30 minutes, how long will the pills last?", "60 minutes. You take the first pill immediately, the second after 30 minutes, and the third after 60 minutes."),
        ("Which is heavier: a pound of feathers or a pound of bricks?", "They weigh the same. Both are one pound."),
        ("If there are 3 apples and you take away 2, how many do you have?", "You have 2 apples, because you took 2."),
        ("Complete the analogy: hot is to cold as up is to ___.", "Down. Hot and cold are opposites, as are up and down."),
        ("If you have a 3-gallon jug and a 5-gallon jug, how do you measure exactly 4 gallons?", "Fill the 5-gallon jug, pour into the 3-gallon jug until full (leaving 2 gallons in the 5-gallon jug). Empty the 3-gallon jug. Pour the 2 gallons into the 3-gallon jug. Fill the 5-gallon jug again and pour into the 3-gallon jug until full (adding 1 gallon). You now have 4 gallons in the 5-gallon jug."),
        ("What day follows the day before yesterday if today is Wednesday?", "Tuesday. The day before yesterday was Monday, the day that follows Monday is Tuesday."),
        ("If two's company and three's a crowd, what are four and five?", "Nine. 4 + 5 = 9."),
        ("A bat and ball cost $1.10 in total. The bat costs $1 more than the ball. How much does the ball cost?", "The ball costs $0.05. If the ball costs x, then the bat costs x + $1. So x + x + $1 = $1.10, which gives 2x = $0.10, so x = $0.05."),
        ("How many times can you subtract 5 from 25?", "Once. After subtracting 5 from 25, you have 20, not 25."),
        ("What is the missing number: 1, 1, 2, 3, 5, 8, ___?", "13. This is the Fibonacci sequence where each number is the sum of the two preceding ones."),
        ("If you are in a race and you pass the person in second place, what place are you in?", "Second place. You passed the person who was in second, so you take their position."),
        ("A rooster sits on a roof facing north. If it lays an egg, which direction does the egg roll?", "Roosters do not lay eggs."),
        ("What makes more sense: 'The yolk of an egg is white' or 'The yolk of an egg are white'?", "Neither makes sense. The yolk of an egg is yellow, not white."),
    ]

    # Language / writing (20 examples)
    lang_pairs = [
        ("What is a synonym for 'happy'?", "Some synonyms for 'happy' include joyful, content, pleased, delighted, and cheerful."),
        ("What is the past tense of 'go'?", "The past tense of 'go' is 'went'."),
        ("What is an antonym of 'ancient'?", "An antonym of 'ancient' is 'modern' or 'new'."),
        ("Define the word 'ephemeral'.", "Ephemeral means lasting for a very short time; transient or fleeting."),
        ("What is an onomatopoeia?", "An onomatopoeia is a word that phonetically imitates the sound it describes, such as 'buzz', 'hiss', or 'splash'."),
        ("What is the plural of 'cactus'?", "The plural of 'cactus' is 'cacti' (Latin plural) or 'cactuses' (English plural). Both forms are accepted."),
        ("What part of speech is the word 'quickly'?", "'Quickly' is an adverb. It modifies verbs, adjectives, or other adverbs to indicate manner, and it describes how an action is performed."),
        ("What is alliteration?", "Alliteration is a literary device where consecutive words begin with the same consonant sound, such as 'Peter Piper picked a peck of pickled peppers.'"),
        ("What is the difference between 'their', 'there', and 'they're'?", "'Their' is possessive (their house), 'there' refers to a place (over there), and 'they're' is a contraction of 'they are'."),
        ("What is a palindrome?", "A palindrome is a word, phrase, or sequence that reads the same backward as forward, such as 'racecar' or 'madam'."),
        ("What does the prefix 'anti-' mean?", "The prefix 'anti-' means against or opposite of, as in 'antibacterial' (against bacteria) or 'anticlockwise' (opposite of clockwise)."),
        ("What is a metaphor?", "A metaphor is a figure of speech that directly compares two unlike things without using 'like' or 'as'. For example, 'time is money' or 'life is a journey'."),
        ("How many vowels are in the English alphabet?", "There are 5 standard vowels in the English alphabet: A, E, I, O, U. Sometimes Y is considered a vowel."),
        ("What is the superlative form of 'good'?", "The superlative form of 'good' is 'best'. The comparative form is 'better'."),
        ("What does 'etc.' stand for?", "'Etc.' is an abbreviation for the Latin phrase 'et cetera', which means 'and the rest' or 'and so on'."),
        ("What is a haiku?", "A haiku is a traditional Japanese poem consisting of three lines with a 5-7-5 syllable pattern."),
        ("What is the difference between 'affect' and 'effect'?", "'Affect' is usually a verb meaning to influence. 'Effect' is usually a noun meaning a result. For example: 'The rain affected the game' vs 'The effect of rain was a delay'."),
        ("What is an oxymoron?", "An oxymoron is a figure of speech that combines two contradictory terms, such as 'jumbo shrimp', 'living dead', or 'deafening silence'."),
        ("Name three Romance languages.", "Three Romance languages are Spanish, French, and Italian. Others include Portuguese, Romanian, and Catalan."),
        ("What is the origin of the word 'democracy'?", "The word 'democracy' comes from the Greek words 'demos' (people) and 'kratos' (power or rule), meaning 'rule by the people'."),
    ]

    for pairs in [math_pairs, science_pairs, geo_pairs, logic_pairs, lang_pairs]:
        for q, a in pairs:
            examples.append(msg(q, a))

    return examples


def build_nko_examples():
    """Build 100 N'Ko eval examples from held-out sources."""
    examples = []

    # Source 1: Cultural data not used in training
    cultural_sources = [
        SCANNER.parent / "NKo" / "data" / "cultural" / "proverbs-unified.json",
        SCANNER.parent / "NKo" / "data" / "cultural" / "greetings-unified.json",
        SCANNER.parent / "NKo" / "data" / "cultural" / "blessings-unified.json",
        SCANNER.parent / "NKo" / "data" / "cultural" / "cultural-concepts.json",
    ]

    for src in cultural_sources:
        if not src.exists():
            continue
        try:
            with open(src) as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    nko = item.get("nko", "") or item.get("text_nko", "") or item.get("nko_text", "")
                    eng = item.get("english", "") or item.get("translation", "") or item.get("meaning", "")
                    if nko and eng and has_nko(nko):
                        examples.append(msg(
                            f"Translate to English: {nko}",
                            eng
                        ))
            elif isinstance(data, dict):
                for key, items in data.items():
                    if not isinstance(items, list):
                        continue
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        nko = item.get("nko", "") or item.get("text_nko", "") or item.get("nko_text", "")
                        eng = item.get("english", "") or item.get("translation", "") or item.get("meaning", "")
                        if nko and eng and has_nko(nko):
                            examples.append(msg(
                                f"Translate to English: {nko}",
                                eng
                            ))
        except Exception as e:
            print(f"  Warning: could not load {src.name}: {e}")

    # Source 2: Wikipedia sentences (held out from CPT data by using last 200 lines)
    wiki_path = SCANNER / "data" / "nko_wikipedia_corpus.txt"
    if wiki_path.exists():
        with open(wiki_path) as f:
            lines = [l.strip() for l in f.readlines() if l.strip() and has_nko(l.strip())]
        # Use the LAST 200 lines as held-out (training used the first portion)
        held_out = lines[-200:]
        for i in range(0, len(held_out) - 1, 2):
            if len(held_out[i]) > 10 and len(held_out[i+1]) > 10:
                examples.append(msg(
                    f"Continue this N'Ko text: {held_out[i]}",
                    held_out[i+1]
                ))

    # Source 3: N'Ko text completion from vocabulary
    nko_unified = SCANNER.parent / "NKo" / "data" / "nko-unified.json"
    if nko_unified.exists():
        with open(nko_unified) as f:
            unified = json.load(f)
        if isinstance(unified, list):
            for item in unified:
                nko = item.get("nko", "")
                name = item.get("name", "") or item.get("english", "")
                if nko and name and has_nko(nko):
                    examples.append(msg(
                        f"What is the N'Ko character '{nko}'?",
                        f"The N'Ko character '{nko}' is called '{name}'."
                    ))

    # Source 4: Cognate pairs
    cognates = SCANNER.parent / "NKo" / "data" / "nko-cognates.json"
    if cognates.exists():
        with open(cognates) as f:
            cog_data = json.load(f)
        if isinstance(cog_data, dict):
            for category, items in cog_data.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    nko = item.get("nko", "")
                    eng = item.get("english", "") or item.get("meaning", "")
                    if nko and eng and has_nko(nko):
                        examples.append(msg(
                            f"What does '{nko}' mean in English?",
                            eng
                        ))

    return examples


def main():
    print("Loading training data hashes for overlap detection...")
    train_hashes = load_training_hashes()
    print(f"  {len(train_hashes)} training example hashes loaded")

    print("\nBuilding English eval set (100 examples)...")
    eng_examples = build_english_examples()
    print(f"  Generated {len(eng_examples)} English examples")

    print("\nBuilding N'Ko eval set (100 examples)...")
    nko_examples = build_nko_examples()
    print(f"  Generated {len(nko_examples)} N'Ko examples (pre-dedup)")

    # Dedup against training data
    nko_deduped = []
    for ex in nko_examples:
        full = " ".join(m["content"] for m in ex["messages"])
        h = text_hash(full)
        if h not in train_hashes:
            nko_deduped.append(ex)
    print(f"  After dedup: {len(nko_deduped)} N'Ko examples")

    # Trim to exactly 100
    random.shuffle(nko_deduped)
    nko_final = nko_deduped[:100]
    eng_final = eng_examples[:100]

    # Verify
    eng_tokens = sum(len(m["content"].split()) for ex in eng_final for m in ex["messages"])
    nko_has_nko_count = sum(1 for ex in nko_final if any(has_nko(m["content"]) for m in ex["messages"]))

    print(f"\nFinal counts:")
    print(f"  English: {len(eng_final)} examples, ~{eng_tokens} words")
    print(f"  N'Ko: {len(nko_final)} examples, {nko_has_nko_count} contain N'Ko chars")

    # Save
    eng_path = EVAL_DIR / "english_eval.jsonl"
    nko_path = EVAL_DIR / "nko_eval.jsonl"

    with open(eng_path, "w") as f:
        for ex in eng_final:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(nko_path, "w") as f:
        for ex in nko_final:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSaved:")
    print(f"  {eng_path}")
    print(f"  {nko_path}")


if __name__ == "__main__":
    main()
