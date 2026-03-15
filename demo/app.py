#!/usr/bin/env python3
"""NKo Brain Scanner Gradio Demo.

Interactive demo for the three-stage fine-tuned N'Ko model.
Tabs: Generate, Brain Scan, Results, About.

Usage:
  python3 demo/app.py  # runs on :7860
"""

import json
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate

MODEL_ID = "mlx-community/Qwen3-8B-8bit"
# V3 paths (preferred), with V1 fallback
FUSED_V3 = os.path.expanduser("~/nko-brain-scanner/fused-v3-nko-qwen3")
FUSED_V1 = os.path.expanduser("~/nko-brain-scanner/fused-nko-qwen3")
ADAPTER_BPE = os.path.expanduser("~/nko-brain-scanner/adapters-bpe")
FUSED_MODEL = FUSED_V3 if os.path.exists(FUSED_V3) else FUSED_V1
RESULTS_FILE = os.path.expanduser("~/nko-brain-scanner/results/profiler_corrected.json")
SCAN_FILE = os.path.expanduser("~/nko-brain-scanner/results/brain_scan_8b.json")

# Load model once at startup
print("Loading fused model...")
if os.path.exists(FUSED_MODEL):
    model, tokenizer = load(FUSED_MODEL)
    print(f"Loaded fused model from {FUSED_MODEL}")
else:
    model, tokenizer = load(MODEL_ID, adapter_path=ADAPTER_BPE)
    print(f"Loaded base + adapter from {ADAPTER_BPE}")


def generate_text(prompt, max_tokens=200, temperature=0.7):
    """Generate text from a prompt."""
    if not prompt.strip():
        return "Please enter a prompt."

    try:
        response = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=int(max_tokens),
            temp=float(temperature),
        )
        return response
    except Exception as e:
        return f"Error: {e}"


def analyze_text(text):
    """Analyze a text sample: compute per-token loss and NKo token count."""
    if not text.strip():
        return "Please enter text to analyze."

    tokens = tokenizer.encode(text)
    if len(tokens) < 4:
        return "Text too short (need at least 4 tokens)."

    tokens = tokens[:256]
    x = mx.array(tokens[:-1])[None]
    y = mx.array(tokens[1:])
    logits = model(x)[0]

    log_probs = nn.log_softmax(logits, axis=-1)
    token_losses = -log_probs[mx.arange(len(y)), y]
    avg_loss = mx.mean(token_losses).item()
    ppl = min(2 ** avg_loss, 99999)

    preds = mx.argmax(logits, axis=-1)
    correct = mx.sum(preds == y).item()
    accuracy = correct / len(y)

    # Count NKo tokens
    nko_count = 0
    for tid in tokens:
        decoded = tokenizer.decode([tid])
        if any(0x07C0 <= ord(c) <= 0x07FF for c in decoded):
            nko_count += 1

    result = f"""**Analysis Results**

| Metric | Value |
|--------|-------|
| Total tokens | {len(tokens)} |
| N'Ko tokens | {nko_count} ({100*nko_count/len(tokens):.1f}%) |
| Average loss | {avg_loss:.4f} |
| Perplexity | {ppl:.2f} |
| Top-1 accuracy | {100*accuracy:.1f}% |
"""
    return result


def get_results_table():
    """Load and format the corrected profiler results."""
    if not os.path.exists(RESULTS_FILE):
        return "Results file not found."

    with open(RESULTS_FILE) as f:
        data = json.load(f)

    table = """## Corrected Evaluation Results (100 English + 100 N'Ko examples)

| Metric | Base (Qwen3-8B) | 2-Stage (CPT+SFT) | 3-Stage (+BPE) |
|--------|:---:|:---:|:---:|
"""
    for lang in ["nko", "english"]:
        lang_label = "N'Ko" if lang == "nko" else "English"
        for metric in ["avg_loss", "perplexity", "top1_accuracy", "nko_token_accuracy"]:
            metric_label = metric.replace("_", " ").title()
            b = data["base"][lang][metric]
            t = data["two_stage"][lang][metric]
            s = data["three_stage_bpe"][lang][metric]
            if isinstance(b, float):
                table += f"| {lang_label} {metric_label} | {b:.4f} | {t:.4f} | **{s:.4f}** |\n"
            else:
                table += f"| {lang_label} {metric_label} | {b} | {t} | **{s}** |\n"

    tax = data.get("translation_tax", {})
    table += f"\n**Translation Tax**: {tax.get('base', 'N/A')}x (base) -> {tax.get('three_stage', 'N/A')}x (fine-tuned) = **{tax.get('reduction', 'N/A')}% reduction**"

    return table


def get_brain_scan():
    """Load and format brain scan results."""
    if not os.path.exists(SCAN_FILE):
        return "Brain scan results not found."

    with open(SCAN_FILE) as f:
        data = json.load(f)

    num_layers = data["metadata"]["num_layers"]
    deltas = data.get("nko_deltas", {})

    text = "## 8B Brain Scan: N'Ko Activation Changes After Fine-Tuning\n\n"
    text += "| Layer | Base L2 | FT L2 | ΔL2 | Zone |\n"
    text += "|-------|---------|-------|-----|------|\n"

    for i in range(num_layers):
        k = str(i)
        b = data["base"]["nko"]["layer_stats"].get(k, {})
        f_ = data["fine_tuned"]["nko"]["layer_stats"].get(k, {})
        d = deltas.get(k, {})
        delta = d.get("l2_delta", 0)

        if i < 28:
            zone = "Frozen"
        elif i < 35:
            zone = "Adaptation"
        else:
            zone = "Output"

        marker = " **" if abs(delta) > 10 else ""
        text += f"| {i} | {b.get('l2_norm',0):.2f} | {f_.get('l2_norm',0):.2f} | {delta:+.2f}{marker} | {zone} |\n"

    text += "\n### Key Findings\n"
    text += "- **Layers 0-27 (Frozen)**: Zero activation change. LoRA only modifies top 8 layers.\n"
    text += "- **Layers 28-34 (Adaptation)**: Reduced L2 norms. More efficient N'Ko encoding.\n"
    text += "- **Layer 35 (Output)**: +573 ΔL2. Sharper, more confident predictions.\n"

    return text


ABOUT_TEXT = """
## The Script That Machines Can't Read

This demo showcases a Qwen3-8B model fine-tuned for N'Ko script processing through a three-stage pipeline.

### What is N'Ko?

N'Ko (ߒߞߏ) is an alphabetic writing system created in 1949 by Solomana Kante for the Manding language family.
It is used by over 40 million speakers across Guinea, Mali, Cote d'Ivoire, and neighboring countries.

**Key properties:**
- 1:1 phoneme-to-character mapping (zero spelling exceptions)
- Explicit tonal diacritics for high, low, and mid tones
- Right-to-left writing direction
- 27 base characters + combining marks
- Unicode block: U+07C0-U+07FF

### Training Pipeline

1. **Continued Pre-Training**: 17,360 examples from N'Ko Wikipedia (3.7M characters)
2. **Supervised Fine-Tuning**: 21,240 combined examples
3. **BPE-Aware Training**: 25,100 examples with 3,860 subword-focused additions

### Key Results (V1 Base Vocab)

- **Translation Tax**: 2.90x -> 0.70x (-76% reduction)
- **N'Ko Token Accuracy**: 23.0% -> 32.8% (+43% relative)
- **English Accuracy Cost**: Only -1.2 percentage points

### V3 Results (Extended Vocab + nicolingua)

- **Training Data**: 92,184 examples (including 32,792 nicolingua parallel segments)
- **Mode Collapse**: Fixed (3/20 degenerate vs V2's 20/20)
- **Unconstrained Syllable Validity**: 99.8%
- **FSM-Constrained Validity**: 100%
- **Total Training Time**: ~6 hours on Apple M4 (16GB)
- **Cloud Cost**: $1.72 (initial brain scan only)

### Links

- [Blog Post](https://diomandeee.github.io/nko-brain-scanner/)
- [GitHub](https://github.com/Diomandeee/nko-brain-scanner)
- [Paper](https://github.com/Diomandeee/nko-brain-scanner/tree/main/paper)
"""


# Build Gradio interface
with gr.Blocks(
    title="N'Ko Brain Scanner",
    theme=gr.themes.Base(
        primary_hue="purple",
        secondary_hue="blue",
    ),
) as demo:
    gr.Markdown("# ߒߞߏ N'Ko Brain Scanner")
    gr.Markdown("*Exploring how language models process the N'Ko script*")

    with gr.Tabs():
        with gr.Tab("Generate"):
            gr.Markdown("### Text Generation")
            gr.Markdown("Enter a prompt to generate text. Try N'Ko or English.")
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="What is N'Ko script? / ߒߞߏ ߦߋ ߡߎ߲ ߘߌ?",
                lines=3,
            )
            with gr.Row():
                max_tokens = gr.Slider(50, 500, value=200, step=50, label="Max Tokens")
                temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
            generate_btn = gr.Button("Generate", variant="primary")
            output = gr.Textbox(label="Generated Text", lines=10)
            generate_btn.click(generate_text, [prompt_input, max_tokens, temperature], output)

        with gr.Tab("Analyze"):
            gr.Markdown("### Text Analysis")
            gr.Markdown("Paste text to analyze its token distribution and model confidence.")
            analyze_input = gr.Textbox(
                label="Text to Analyze",
                placeholder="ߒ ߓߊ߯ߙߊ ߞߊ߲ ߞߊߟߊ߲ ߞߍ",
                lines=5,
            )
            analyze_btn = gr.Button("Analyze", variant="primary")
            analysis_output = gr.Markdown()
            analyze_btn.click(analyze_text, analyze_input, analysis_output)

        with gr.Tab("Brain Scan"):
            gr.Markdown(get_brain_scan())

        with gr.Tab("Results"):
            gr.Markdown(get_results_table())

        with gr.Tab("About"):
            gr.Markdown(ABOUT_TEXT)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
