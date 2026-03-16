#!/usr/bin/env python3
"""Stream + extract Whisper features + transcribe + bridge. Saves features for training."""
import json, os, subprocess, glob, time, torch, sys
from pathlib import Path

TEMP = "/workspace/temp"
FEATURES_DIR = "/workspace/features"
RESULTS = "/workspace/results"
Path(TEMP).mkdir(exist_ok=True)
Path(FEATURES_DIR).mkdir(exist_ok=True)
Path(RESULTS).mkdir(exist_ok=True)

CHANNEL = sys.argv[1] if len(sys.argv) > 1 else "djoko"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 200

CHANNELS = {
    "djoko": "https://www.youtube.com/channel/UCXiIHk2N-qWE-ZFPfPVJ06w/videos",
    "babamamadidiane": "https://www.youtube.com/@babamamadidiane/videos",
}

# Load Whisper for feature extraction
import whisper
print("Loading Whisper large-v3 for feature extraction...")
whisper_model = whisper.load_model("large-v3", device="cuda")
whisper_model.eval()
print("Whisper loaded.")

# Load FarmRadio for transcription
from transformers import pipeline as hf_pipeline
print("Loading FarmRadio ASR...")
asr = hf_pipeline("automatic-speech-recognition",
                   model="FarmRadioInternational/bambara-whisper-asr",
                   device="cuda", return_timestamps=False)
print("FarmRadio loaded.")

# Latin → N'Ko bridge
LATIN_TO_NKO = {
    "a": "\u07ca", "b": "\u07d3", "c": "\u07d7", "d": "\u07d8", "e": "\u07cd",
    "f": "\u07dd", "g": "\u07dc", "h": "\u07e4", "i": "\u07cc", "j": "\u07d6",
    "k": "\u07de", "l": "\u07df", "m": "\u07e1", "n": "\u07e3", "o": "\u07cf",
    "p": "\u07d4", "r": "\u07d9", "s": "\u07db", "t": "\u07d5", "u": "\u07ce",
    "w": "\u07e5", "y": "\u07e6", "z": "\u07d6",
    "\u0254": "\u07cf", "\u025b": "\u07d0", "\u0272": "\u07e2", "\u014b": "\u07d2",
    "v": "\u07dd",
}

def bridge(text):
    return "".join(LATIN_TO_NKO.get(c, " " if c == " " else "") for c in text.lower())

# List videos
print(f"Listing {CHANNEL}...")
r = subprocess.run(
    ["yt-dlp", "--remote-components", "ejs:github", "--flat-playlist", "--print", "id",
     CHANNELS[CHANNEL]],
    capture_output=True, text=True, timeout=300
)
videos = [v.strip() for v in r.stdout.strip().split("\n") if v.strip()][:LIMIT]
print(f"Found {len(videos)} videos")

# Checkpoint
cp_file = f"/workspace/stream_features_{CHANNEL}_checkpoint.json"
if os.path.exists(cp_file):
    with open(cp_file) as f:
        cp = json.load(f)
else:
    cp = {"done": [], "features": 0, "pairs": 0}

done_set = set(cp["done"])
remaining = [v for v in videos if v not in done_set]
print(f"Done: {len(done_set)}, remaining: {len(remaining)}")

pairs_file = open(f"{RESULTS}/feature_pairs_{CHANNEL}.jsonl", "a")
t_start = time.time()

for i, vid in enumerate(remaining):
    print(f"\n[{i+1}/{len(remaining)}] {vid}")

    # Download
    wav = f"{TEMP}/{vid}.wav"
    subprocess.run(
        ["yt-dlp", "--remote-components", "ejs:github", "-x", "--audio-format", "wav",
         "-o", f"{TEMP}/{vid}.%(ext)s", "--no-playlist",
         f"https://www.youtube.com/watch?v={vid}"],
        capture_output=True, timeout=300
    )
    if not os.path.exists(wav):
        print("  skip (download failed)")
        continue

    # Segment
    seg_dir = f"{TEMP}/segments/{vid}"
    os.makedirs(seg_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-i", wav, "-f", "segment", "-segment_time", "30",
         "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
         f"{seg_dir}/{vid}_%04d.wav", "-y", "-loglevel", "error"],
        capture_output=True, timeout=120
    )
    segments = sorted(glob.glob(f"{seg_dir}/*.wav"))
    os.remove(wav)
    print(f"  {len(segments)} segments", end="", flush=True)

    seg_pairs = 0
    for seg in segments:
        seg_name = Path(seg).stem
        feat_path = f"{FEATURES_DIR}/{seg_name}.pt"

        # 1. Extract Whisper features
        try:
            audio = whisper.load_audio(seg)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio, n_mels=128).to("cuda")
            with torch.no_grad():
                features = whisper_model.encoder(mel.unsqueeze(0))
            torch.save(features.squeeze(0).cpu(), feat_path)
            cp["features"] += 1
        except Exception:
            pass

        # 2. Transcribe
        try:
            result = asr(seg)
            latin = result["text"].strip()
        except Exception:
            latin = ""

        # 3. Bridge to N'Ko
        nko = bridge(latin)

        if latin and nko.strip():
            pair = {"feat_id": seg_name, "latin": latin, "nko": nko, "video_id": vid}
            pairs_file.write(json.dumps(pair, ensure_ascii=False) + "\n")
            pairs_file.flush()
            cp["pairs"] += 1
            seg_pairs += 1

        # Delete segment WAV (features are saved)
        os.remove(seg)

    subprocess.run(["rm", "-rf", seg_dir], capture_output=True)
    cp["done"].append(vid)
    with open(cp_file, "w") as f:
        json.dump(cp, f)

    elapsed = time.time() - t_start
    rate = (i + 1) / elapsed * 3600
    print(f" -> {seg_pairs} pairs, {cp['features']} features total ({rate:.0f} vids/hr)")

pairs_file.close()
print(f"\nDone. Features: {cp['features']}, Pairs: {cp['pairs']}")
