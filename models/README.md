---
license: apache-2.0
language:
  - nqo
  - bm
tags:
  - nko
  - asr
  - speech-recognition
  - ctc
  - whisper
  - african-languages
  - low-resource
  - manding
  - bambara
  - west-africa
datasets:
  - RobotsMali/bam-asr-early
pipeline_tag: automatic-speech-recognition
metrics:
  - cer
library_name: pytorch
---

# nko-asr-v1: First Audio-to-N'Ko Speech Recognition System

The world's first ASR system that outputs N'Ko script (U+07C0-U+07FF) directly from audio. No prior system does this. All existing Bambara ASR systems (MALIBA-AI, Meta MMS, Google USM) output Latin transliteration only.

N'Ko is an alphabetic script used by 40+ million Manding-language speakers across West Africa (Mali, Guinea, Ivory Coast, Burkina Faso, Senegal, The Gambia).

## Model Description

A lightweight CTC-based character prediction head trained on top of frozen Whisper large-v3 encoder features.

| Property | Value |
|----------|-------|
| Architecture | Whisper large-v3 (frozen encoder) + BiLSTM CTC head |
| Trainable parameters | 5,420,610 (5.4M) |
| Input | 16kHz audio (via Whisper mel spectrogram) |
| Output | 65 N'Ko character classes (U+07C0-U+07FF + space) + 1 CTC blank |
| Encoder | 3-layer BiLSTM, hidden_dim=512, bidirectional |
| Projection | Linear 1280 -> 512 (Whisper features to BiLSTM) |
| Temporal downsampling | 4x (1500 Whisper frames -> 375) |
| Decoding | CTC greedy decode + optional FSM syllable validation |
| Framework | PyTorch |

### Architecture Diagram

```
Audio (16kHz)
    |
    v
Whisper large-v3 encoder (frozen, 1.5B params)
    |  1280-dim features, 1500 frames per 30s
    v
4x Temporal Downsample (stride=4)
    |  375 frames
    v
Linear Projection (1280 -> 512)
    |
    v
BiLSTM (3 layers, hidden=256 per direction, bidirectional)
    |  512-dim output
    v
Linear Output (512 -> 66)  [65 chars + 1 CTC blank]
    |
    v
CTC Greedy Decode
    |
    v
N'Ko Text (Unicode U+07C0-U+07FF)
```

## Training

### Data

- **Dataset**: [RobotsMali/bam-asr-early](https://huggingface.co/datasets/RobotsMali/bam-asr-early) (CC-BY-4.0)
- **Size**: 37 hours, 37,306 human-labeled Bambara radio speech samples
- **Train/Test split**: 35,843 train / 1,463 test
- **Labels**: Latin Bambara transcriptions, converted to N'Ko via cross-script bridge

### Cross-Script Bridge

Since bam-asr-early provides Latin Bambara transcriptions (not N'Ko), a deterministic transliteration bridge maps Latin graphemes to N'Ko characters:

```
Latin:  "ko muso to soro"
N'Ko:   "ߞߏ ߡߎߛߏ ߕߐ ߛߏߙߏ"
```

This works because N'Ko has near-perfect phoneme-to-character correspondence: each sound maps to exactly one character (unlike English or French orthography).

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW (lr=3e-4, weight_decay=0.01) |
| Scheduler | CosineAnnealingLR |
| Epochs | 200 |
| Batch size | 8 |
| Gradient clipping | 1.0 |
| Loss | CTC loss (zero_infinity=True) |
| Hardware | NVIDIA RTX 4090 (Vast.ai, Sweden) |
| Total cost | ~$3 USD |

## Results

Evaluated on 1,463 held-out test samples from bam-asr-early.

| Metric | Value |
|--------|-------|
| N'Ko CER | 56% |
| Val Loss | 0.143 |
| Test samples | 1,463 |

### Sample Predictions

```
Gold:  ߞߋ ߡߎߛߋ ߕߐ ߛߏߙߏ ߛߋߢߊߟߌ ߟߊ ߞʔߎ ߓʔߊ ߘߍߟߌ
Pred:  ߞߋ ߡߎߛߋ ߕߐ ߛߏߙߏ ߛߋߢߊߟߌ ߟߊ ߞߎ ߓߊ ߘߌߜߌ

Gold:  ߊ ߝߊ ߦߋ ߡߎ ߦߋ
Pred:  ߊ ߝߊ ߦ ߡ ߦ
```

### Context

This is a first-of-its-kind system. There is no prior work on audio-to-N'Ko ASR to compare against. The 56% CER reflects the difficulty of the task: the model must learn to map Bambara speech acoustics to an unseen script (N'Ko) through a transliteration bridge, with only 37 hours of training data.

For comparison, state-of-the-art Bambara ASR in Latin script (MALIBA-AI) achieves 45.73% WER on similar radio data, but outputs Latin characters that most Manding speakers cannot read natively.

## Usage

### Requirements

```bash
pip install torch openai-whisper
```

### Inference

```python
import torch
import torch.nn as nn
import whisper

# ── Define the model architecture ──

class CharASR(nn.Module):
    def __init__(self, num_chars=65, input_dim=1280, hidden_dim=512):
        super().__init__()
        self.num_chars = num_chars
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=3, batch_first=True,
            bidirectional=True, dropout=0.1,
        )
        self.output_proj = nn.Linear(hidden_dim, num_chars + 1)  # +1 for CTC blank

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.encoder(x)
        return self.output_proj(x)

# ── Build N'Ko character vocabulary ──

def build_nko_char_vocab():
    chars = {}
    idx = 0
    for cp in range(0x07C0, 0x07FF + 1):
        chars[chr(cp)] = idx
        idx += 1
    chars[" "] = idx
    idx += 1
    return chars, idx

# ── CTC greedy decode ──

def ctc_decode(logits, idx_to_char, num_chars):
    preds = logits.argmax(dim=-1).cpu().tolist()
    decoded = []
    prev = -1
    for idx in preds:
        if idx != num_chars and idx != prev:
            decoded.append(idx_to_char.get(idx, ""))
        prev = idx
    return "".join(decoded)

# ── Load models ──

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Whisper for feature extraction
whisper_model = whisper.load_model("large-v3", device=device)
whisper_model.eval()

# Load N'Ko ASR head
char_vocab, num_chars = build_nko_char_vocab()
idx_to_char = {v: k for k, v in char_vocab.items()}

model = CharASR(num_chars).to(device)
model.load_state_dict(torch.load("best_char_asr.pt", map_location=device, weights_only=True))
model.eval()

# ── Transcribe audio ──

def transcribe(audio_path):
    # Load and preprocess audio
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(torch.tensor(audio, dtype=torch.float32))
    mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(device)

    # Extract Whisper features
    with torch.no_grad():
        features = whisper_model.encoder(mel.unsqueeze(0)).squeeze(0)

    # Downsample 4x
    features = features[::4].unsqueeze(0)

    # Predict N'Ko characters
    with torch.no_grad():
        logits = model(features)

    # CTC decode
    nko_text = ctc_decode(logits[0], idx_to_char, num_chars)
    return nko_text

# Example
nko = transcribe("bambara_speech.wav")
print(f"N'Ko transcription: {nko}")
```

### Output Format

The model outputs raw N'Ko Unicode characters. You can validate syllable structure using the FSM post-processor from the [nko-brain-scanner](https://github.com/Diomandeee/nko-brain-scanner) repository:

```python
from asr.postprocess import NKoPostProcessor
pp = NKoPostProcessor()
result = pp.process(ctc_logits, idx_to_char, num_chars)
print(result.text)           # Validated N'Ko text
print(result.validity_rate)  # % of valid syllables
```

## Limitations

- **56% CER**: The model makes errors on roughly half the characters. This is a first version trained on limited data (37h). More training data and architectural improvements will reduce this.
- **Cross-script bridge noise**: The Latin-to-N'Ko transliteration is deterministic but imperfect. Tone marks and vowel quality distinctions that exist in N'Ko but not in Latin Bambara orthography are lost in the bridge.
- **No native N'Ko audio labels**: The model has never seen human-written N'Ko transcriptions of speech. All training labels are machine-transliterated. Human-labeled N'Ko audio data would significantly improve quality.
- **30-second limit**: Inherits Whisper's 30-second context window. Longer audio must be chunked.
- **Bambara only**: Trained on Bambara speech. Other Manding languages (Maninka, Dyula, Mandinka) share enough phonology that transfer is plausible but untested.
- **No language model**: Pure CTC output with no language model rescoring. Adding an N'Ko language model would improve results.

## Why This Matters

N'Ko is one of Africa's most successful indigenous scripts, actively used for education, journalism, and commerce across West Africa. Yet it is invisible to modern speech technology. Every existing Bambara ASR system forces speakers to read Latin script, which many N'Ko-literate speakers do not use.

This model is a proof of concept that audio-to-N'Ko ASR is feasible with minimal compute ($3) and existing open-source data. The 56% CER is not production-ready, but it establishes a baseline where none existed before.

## Files

| File | Description |
|------|-------------|
| `best_char_asr.pt` | Best checkpoint (lowest val loss), 21.7 MB |
| `config.json` | Model architecture configuration |
| `inference.py` | Standalone inference script |

## Training Code

Full training pipeline, data preparation, and evaluation code: [github.com/Diomandeee/nko-brain-scanner](https://github.com/Diomandeee/nko-brain-scanner)

## Citation

```bibtex
@misc{diomande2026nkoasr,
  title={First Audio-to-N'Ko ASR: Character-Level CTC on Frozen Whisper Features},
  author={Diomande, Mohamed},
  year={2026},
  url={https://huggingface.co/Diomande/nko-asr-v1}
}
```

## Related

- **Paper**: [The Script That Machines Can't Read: Adapting Large Language Models for N'Ko](https://github.com/Diomandeee/nko-brain-scanner/tree/main/paper)
- **Blog**: [From Dead Circuits to Living Speech](https://github.com/Diomandeee/nko-brain-scanner/blob/main/blog/asr-breakthrough.md)
- **Training data**: [RobotsMali/bam-asr-early](https://huggingface.co/datasets/RobotsMali/bam-asr-early)
- **N'Ko Wikipedia**: [nqo.wikipedia.org](https://nqo.wikipedia.org)

## License

Apache-2.0
