# Research Engine Memory - NKO Brain Scanner

## Project State (2026-03-14)
- **Base model**: Qwen3-8B-8bit (mlx-community)
- **Training pipeline**: 3-stage (CPT -> SFT -> BPE-aware SFT)
- **Training data**: 21,240 combined train examples, 2,361 valid
- **Results**: 3-stage BPE achieves PPL 4.57 on NKo (down from 8.57 base), NKo token accuracy 39.38% (from 27.03%)
- **Existing vocab**: custom vocab.json with NKo chars + morphemes from NKo phonetics/morphology libs
- **new_nko_tokens.json**: Contains syllable-level tokens with constituent mappings (for embedding init)
- **Blog post**: "The Script That Machines Can't Read" completed, results from Qwen2-72B on Vast.ai A100

## Key Research References
- arXiv:2406.11477 - "Vocabulary expansion with 0.01GB" (Yamaguchi et al.)
- arXiv:2408.04303 - "Trans-Tokenization" (Remy et al.)
- arXiv:2512.03989 - "Teaching Old Tokenizers New Words"
- Masakhane datasets on HuggingFace (NER, POS for Bambara)
- sudoping01/maliba-llm - First open-source Bambara LLM on HuggingFace
- AfricaNLP 2026 @ EACL (Rabat, March 2026) - deadline passed
- EMNLP 2026 (Budapest, Oct 2026) - ARR deadline May 25, 2026

## Bambara/Manding/NKo Model Landscape (scanned 2026-03-14)
See detailed report: `bambara-manding-nko-models.md`

### Key Models
- **MALIBA-AI org**: 10 models (ASR v1/v2/v3, TTS, LLM, embeddings, bomu/dogon/minianka/songhoy ASR), 4 datasets
- **sudoping01/maliba-llm**: First Bambara LLM (Gemma 3n E2B fine-tune, 1M instruction examples, val loss 0.4952)
- **MALIBA-AI/bambara-asr-v3**: #1 ranked, whisper-large-v3 LoRA, 2B, 45.73% WER (benchmark), 13.23% (normalized internal)
- **RobotsMali/soloni-114m-tdt-ctc-v0**: 114M lightweight ASR, NVIDIA parakeet base
- **AfroLM**: Pretrained MLM for 23 African languages including Bambara
- **UBC-NLP/Simba-M**: Multilingual ASR+TTS for 61+ African languages

### Critical Gap: No N'Ko-script text generation model exists on HuggingFace
- Only N'Ko resource: WMT 2023 nicolingua-0005 (130K parallel segments, 3M+ N'Ko words, 30.83 en-nko chrF++)
- Our nko-brain-scanner project would be the FIRST N'Ko text generation model

### Key Benchmarks
- **AfroBench**: 64 languages, 15 tasks, 22 datasets (Bambara included)
- **MasakhaNER 2.0**: NER for 21 African languages (Bambara included)
- **Bayelemabaga** (NAACL 2025): 46,976 parallel bam-fr, best Bambara MT benchmark
- **MALIBA-AI/bambara-asr-benchmark**: 500 samples, 37 models ranked
- **nicolingua-0005 FLoRes-devtest**: Standardized N'Ko MT evaluation

### Key Datasets for Potential Use
- MALIBA-AI/D4Pret (667K entries) + D4Pret2 (93K) - Bambara pretraining
- djelia/bambara-lm-qa - QA + instruction + ASR correction
- MALIBA-AI/bambara-mt-dataset (186K entries) - translation pairs
- RobotsMali/afvoices (612 hours Bambara speech)
