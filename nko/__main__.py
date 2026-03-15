"""
python -m nko — N'Ko Unified Platform CLI

Usage:
    python -m nko transliterate <text> [--to nko|latin|arabic]
    python -m nko phonetics <text>
    python -m nko morphology <word>
    python -m nko proverb [random|search <query>]
    python -m nko culture <tool> [subcommand] [args]
    python -m nko stats
    python -m nko version

Cultural Tools (python -m nko culture):
    proverbs    Browse, search, and filter Manding proverbs
    calendar    Cultural and religious calendar
    sigils      N'Ko Sound Sigils — audio signatures
    blessings   Blessings, condolences, congratulations
    greetings   Multi-turn Manding greeting protocol
    clans       Manding clan/family database (jamu)
    concepts    Cultural concepts, titles, kinship terms
    stats       Cultural data statistics
"""

import sys


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("--help", "-h"):
        print(__doc__.strip())
        return

    cmd = args[0]

    if cmd == "version":
        from nko import __version__
        print(f"nko {__version__}")

    elif cmd == "transliterate":
        from nko.transliterate import transliterate, detect_script
        if len(args) < 2:
            print("Usage: python -m nko transliterate <text> [--to nko|latin|arabic]")
            sys.exit(1)
        # Parse --to target
        target = "latin"
        text_parts = []
        i = 1
        while i < len(args):
            if args[i] == "--to" and i + 1 < len(args):
                target = args[i + 1]
                i += 2
            else:
                text_parts.append(args[i])
                i += 1
        text = " ".join(text_parts)
        if not text:
            print("No text provided.")
            sys.exit(1)
        result = transliterate(text, target=target)
        src = detect_script(text)
        print(f"[{src.value} → {target}] {result}")

    elif cmd == "phonetics":
        from nko.phonetics import NKoPhonetics
        if len(args) < 2:
            print("Usage: python -m nko phonetics <text>")
            sys.exit(1)
        text = " ".join(args[1:])
        ph = NKoPhonetics()
        ipa = ph.to_ipa(text)
        print(f"Text: {text}")
        print(f"IPA:  {ipa}")
        for ch in text:
            info = ph.get_char_info(ch)
            if info:
                print(f"  {ch} (U+{ord(ch):04X}) → {info.category.value if hasattr(info.category, 'value') else info.category}")

    elif cmd == "morphology":
        from nko.morphology import analyze_word
        if len(args) < 2:
            print("Usage: python -m nko morphology <word>")
            sys.exit(1)
        word = args[1]
        result = analyze_word(word)
        print(f"Word:    {result.word}")
        print(f"Root:    {result.root}")
        print(f"Script:  {result.script}")
        if result.noun_class:
            nc = result.noun_class
            print(f"Noun class: {nc.value if hasattr(nc, 'value') else nc}")
        if result.morphemes:
            print("Morphemes:")
            for m in result.morphemes:
                print(f"  [{m.morpheme_type.value if hasattr(m.morpheme_type, 'value') else m.morpheme_type}] {m.form} — {m.meaning}")

    elif cmd == "proverb":
        from nko.culture import NKoCulture
        culture = NKoCulture()
        subcmd = args[1] if len(args) > 1 else "random"
        if subcmd == "random":
            p = culture.random_proverb()
            nko = p.get("text_nko", "")
            latin = p.get("text_latin", "")
            en = p.get("translation_en", p.get("meaning", ""))
            print(f"  {nko}")
            if latin:
                print(f"  {latin}")
            if en:
                print(f"  — {en}")
        elif subcmd == "search" and len(args) > 2:
            query = " ".join(args[2:])
            results = culture.search_proverbs(query)
            print(f"Found {len(results)} proverb(s) matching '{query}':")
            for p in results:
                print(f"  • {p.get('text_nko', '')} — {p.get('text_latin', '')}")
        else:
            print("Usage: python -m nko proverb [random|search <query>]")

    elif cmd == "culture":
        from nko.cultural_tools import run_cultural_tools
        run_cultural_tools(args[1:])

    elif cmd == "stats":
        from nko.culture import NKoCulture
        culture = NKoCulture()
        st = culture.stats()
        print("N'Ko Cultural Data Statistics:")
        for k, v in st.items():
            print(f"  {k}: {v} entries")

    else:
        print(f"Unknown command: {cmd}")
        print("Run 'python -m nko --help' for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
