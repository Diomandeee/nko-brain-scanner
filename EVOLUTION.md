# EVOLUTION.md -- N'Ko Brain Scanner

> Research toolkit for adapting LLMs to N'Ko script: activation profiling, 3-stage LoRA training, BPE tokenizer, constrained decoding, and retrieval-centric ASR.

## Current State
- Activation profiler (scanner/) with per-layer hidden state analysis, heatmap generation, profile comparison
- 3-stage training pipeline (training/): CPT + SFT + BPE-aware LoRA on MLX
- N'Ko BPE tokenizer with 512 merges, 2.75x compression, morpheme-constrained variant
- Vocabulary extension: quantized embedding surgery (151,936 to 152,192 tokens)
- Admissibility-constrained decoding: 4-state FSM encoding CV/CVN syllable structure
- Retrieval-centric ASR (asr/): Whisper audio + SigLIP visual + N'Ko text in d=512 shared embedding space
- V3 training results: val loss 3.275, 3/20 degenerate (down from 20/20), 99.8% unconstrained validity
- Eval suite with round-trip evaluation, OCR comparison, cost estimation
- Paper targeting ACL/EMNLP 2026

## Evolution Tasks

- [ ] `[SWARM]` Build a Gradio or Streamlit demo interface that lets users input N'Ko text and visualize per-layer activation heatmaps interactively
- [ ] `[SWARM]` Implement V4 training with DPO/RLHF alignment using the existing trajectory data to reduce the remaining 3/20 degenerate outputs
- [ ] `[SWARM]` Add automated evaluation CI pipeline that runs the full eval suite on every training checkpoint and logs metrics to Weights & Biases
- [ ] `[SWARM]` Build a live transcription demo combining the ASR pipeline with streaming audio input for real-time N'Ko speech-to-text
- [ ] `[SWARM]` Expand the BPE tokenizer to Bambara Latin script variant and add a unified tokenizer that handles both N'Ko and Latin Bambara
- [ ] `[SWARM]` Implement beam search with n-gram blocking in the constrained decoder to reduce repetitive output patterns
- [ ] `[SWARM]` Add cross-lingual transfer evaluation comparing N'Ko performance against Bambara, Dioula, and Malinke variants
- [ ] `[SWARM]` Build a HuggingFace model card generator that auto-populates metrics, training config, and usage examples from the latest eval run
- [ ] `[SWARM]` Implement the morphological analyzer module that segments N'Ko words into root + suffix chains using the phonotactic FSM
- [ ] `[SWARM]` Add speaker diarization integration into the ASR pipeline so multi-speaker N'Ko audio can be transcribed with speaker labels
- [ ] `[SWARM]` Build a parallel corpus expansion pipeline that scrapes and aligns N'Ko/French/English text from online radio transcripts
- [ ] `[SWARM]` Implement quantized model export (GGUF/CoreML) for on-device inference on iOS and Apple Silicon without Python dependencies
- [ ] `[SWARM]` Add a codebook visualization tool that renders the 3024-entry retrieval codebook as a 2D t-SNE/UMAP embedding map
