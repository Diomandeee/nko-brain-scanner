#!/usr/bin/env python3
"""Seed Graph Kernel with N'Ko lexicon data."""
import json
import time
import sys
import urllib.request
import urllib.error

GK_URL = "http://localhost:8001"
BATCH_DELAY = 0.05  # 50ms between requests

def post_triple(subject, predicate, obj):
    """Post a single triple to the GK."""
    data = json.dumps({"subject": subject, "predicate": predicate, "object": obj}).encode()
    req = urllib.request.Request(
        f"{GK_URL}/api/knowledge",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        resp = urllib.request.urlopen(req, timeout=5)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "message": e.read().decode()[:200]}
    except Exception as e:
        return {"error": str(e)}

def seed_lexicon(lexicon_path, checkpoint_path="data/gk_seed_checkpoint.json"):
    """Seed GK with lexicon entries, with checkpoint support."""
    lexicon = json.load(open(lexicon_path))

    # Load checkpoint
    try:
        checkpoint = json.load(open(checkpoint_path))
        seeded = set(checkpoint.get("seeded_words", []))
        print(f"Resuming from checkpoint: {len(seeded)} already seeded")
    except (FileNotFoundError, json.JSONDecodeError):
        seeded = set()
        checkpoint = {"seeded_words": [], "errors": []}

    total = len(lexicon)
    new_count = 0
    error_count = 0

    for i, entry in enumerate(lexicon):
        word = entry["word"]
        if word in seeded:
            continue

        ipa = entry.get("ipa", "")
        freq = entry.get("freq", 0)

        # Post triples
        results = []
        results.append(post_triple("nko", "has_word", word))
        if ipa:
            results.append(post_triple(word, "ipa", ipa))
        results.append(post_triple(word, "is_valid", "true"))
        results.append(post_triple(word, "frequency", str(freq)))

        # Check for errors
        errors = [r for r in results if "error" in r]
        if errors:
            error_count += 1
            checkpoint["errors"].append({"word": word, "errors": errors})
            if error_count > 50:
                print(f"Too many errors ({error_count}), stopping")
                break
        else:
            seeded.add(word)
            new_count += 1

        # Progress
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}] seeded: {new_count}, errors: {error_count}")
            # Save checkpoint
            checkpoint["seeded_words"] = list(seeded)
            json.dump(checkpoint, open(checkpoint_path, "w"))

        time.sleep(BATCH_DELAY)

    # Final save
    checkpoint["seeded_words"] = list(seeded)
    json.dump(checkpoint, open(checkpoint_path, "w"))
    print(f"Done: {new_count} new words seeded, {error_count} errors, {len(seeded)} total")
    return new_count

def seed_collocations(collocations_path):
    """Seed GK with collocation data."""
    collocations = json.load(open(collocations_path))
    count = 0

    for i, entry in enumerate(collocations):
        w1 = entry["word1"]
        w2 = entry["word2"]

        post_triple(w1, "followed_by", w2)
        post_triple(w1, "pmi_with", f"{w2}:{entry['pmi']}")
        count += 1

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(collocations)}] collocations seeded: {count}")

        time.sleep(BATCH_DELAY)

    print(f"Done: {count} collocations seeded")
    return count

if __name__ == "__main__":
    # Check GK health
    try:
        resp = urllib.request.urlopen(f"{GK_URL}/health", timeout=3)
        health = json.loads(resp.read())
        print(f"GK healthy: {health.get('status')}, {health.get('policy_count')} policies")
    except Exception as e:
        print(f"GK unreachable: {e}")
        sys.exit(1)

    print("\n=== Seeding N'Ko Lexicon ===")
    word_count = seed_lexicon("data/nko_lexicon.json")

    print("\n=== Seeding Collocations ===")
    coll_count = seed_collocations("data/nko_collocations.json")

    print(f"\n=== Summary ===")
    print(f"Words: {word_count}")
    print(f"Collocations: {coll_count}")
    print(f"Total triples added: ~{word_count * 4 + coll_count * 2}")
