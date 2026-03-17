#!/usr/bin/env python3
"""
Evaluate on bam-asr-early test set (1,463 human-labeled samples)
=================================================================
Loads trained char-level model, decodes test features, computes CER.
Also transliterates N'Ko output back to Latin for WER comparison.

Usage (on Vast.ai):
    python3 eval_bam_test.py --model /workspace/human_model/best_char_asr.pt \
                              --features-dir /workspace/human_features \
                              --test-pairs /workspace/bam_test_nko.jsonl
"""

import argparse, json, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path


def build_nko_char_vocab():
    chars = {}
    idx = 0
    for cp in range(0x07C0, 0x07FF + 1):
        chars[chr(cp)] = idx
        idx += 1
    chars[" "] = idx
    idx += 1
    return chars, idx


# Reverse bridge: N'Ko → Latin (for WER comparison)
NKO_TO_LATIN = {
    "\u07ca": "a", "\u07d3": "b", "\u07d7": "c", "\u07d8": "d", "\u07cd": "e",
    "\u07dd": "f", "\u07dc": "g", "\u07e4": "h", "\u07cc": "i", "\u07d6": "j",
    "\u07de": "k", "\u07df": "l", "\u07e1": "m", "\u07e3": "n", "\u07cf": "o",
    "\u07d4": "p", "\u07d9": "r", "\u07db": "s", "\u07d5": "t", "\u07ce": "u",
    "\u07e5": "w", "\u07e6": "y", "\u07d0": "e", "\u07e2": "ny", "\u07d2": "ng",
}


def nko_to_latin(nko_text):
    return "".join(NKO_TO_LATIN.get(c, " " if c == " " else "") for c in nko_text)


def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if a[i - 1] == b[j - 1] else 1))
            prev = temp
    return dp[n]


class CharASR(nn.Module):
    def __init__(self, num_chars, input_dim=1280, hidden_dim=512):
        super().__init__()
        self.num_chars = num_chars
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=3,
                               batch_first=True, bidirectional=True, dropout=0.1)
        self.output_proj = nn.Linear(hidden_dim, num_chars + 1)

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.encoder(x)
        return self.output_proj(x)


def ctc_decode(logits, idx_to_char, num_chars):
    preds = logits.argmax(dim=-1).cpu().tolist()
    decoded = []
    prev = -1
    for idx in preds:
        if idx != num_chars and idx != prev:
            decoded.append(idx_to_char.get(idx, ""))
        prev = idx
    return "".join(decoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--test-pairs", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    char_vocab, num_chars = build_nko_char_vocab()
    idx_to_char = {v: k for k, v in char_vocab.items()}

    model = CharASR(num_chars).to(device)
    model.load_state_dict(torch.load(args.model, weights_only=True))
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    with open(args.test_pairs) as f:
        pairs = [json.loads(l) for l in f if l.strip()]

    # Add feat_id if missing
    for p in pairs:
        if "feat_id" not in p:
            p["feat_id"] = "test_" + str(p["id"]).zfill(6)

    features_dir = Path(args.features_dir)
    available = set(p.stem for p in features_dir.glob("*.pt"))
    matched = [p for p in pairs if p["feat_id"] in available]

    if args.max_samples > 0:
        matched = matched[:args.max_samples]

    print(f"Test pairs: {len(matched)} (of {len(pairs)})")

    total_cer_num, total_cer_den = 0, 0
    total_wer_num, total_wer_den = 0, 0

    for i, p in enumerate(matched):
        feat = torch.load(features_dir / f"{p['feat_id']}.pt", weights_only=True)
        feat = feat[::4]  # downsample 4x
        feat = feat.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(feat)
        pred_nko = ctc_decode(logits[0], idx_to_char, num_chars)

        gold_nko = p.get("nko", "")
        gold_latin = p.get("bam", "")

        # CER on N'Ko
        gc = list(gold_nko.replace(" ", ""))
        pc = list(pred_nko.replace(" ", ""))
        if gc:
            total_cer_num += edit_distance(gc, pc)
            total_cer_den += len(gc)

        # WER on round-trip Latin
        pred_latin = nko_to_latin(pred_nko)
        gw = gold_latin.split()
        pw = pred_latin.split()
        if gw:
            total_wer_num += edit_distance(gw, pw)
            total_wer_den += len(gw)

        if i < 5:
            print(f"\n  Gold Latin: {gold_latin[:60]}")
            print(f"  Gold NKo:   {gold_nko[:60]}")
            print(f"  Pred NKo:   {pred_nko[:60]}")
            print(f"  Pred Latin: {pred_latin[:60]}")

    cer = total_cer_num / max(total_cer_den, 1) * 100
    wer = total_wer_num / max(total_wer_den, 1) * 100

    print(f"\n{'=' * 50}")
    print(f"RESULTS on {len(matched)} test samples")
    print(f"{'=' * 50}")
    print(f"N'Ko CER:        {cer:.1f}%")
    print(f"Round-trip WER:  {wer:.1f}%")
    print(f"MALIBA-AI WER:   45.73% (benchmark)")
    print(f"{'=' * 50}")
    if wer < 45.73:
        print(f">>> BEATS MALIBA-AI by {45.73 - wer:.1f}pp <<<")
    else:
        print(f"Gap to MALIBA-AI: {wer - 45.73:.1f}pp")


if __name__ == "__main__":
    main()
