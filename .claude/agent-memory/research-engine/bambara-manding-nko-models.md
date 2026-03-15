# Bambara / Manding / N'Ko Models on Hugging Face
> Scanned: 2026-03-14

## ASR Models (11 found)

| Model | Org | Base | Size | WER | License |
|-------|-----|------|------|-----|---------|
| MALIBA-AI/bambara-asr-v3 | MALIBA-AI | whisper-large-v3 | 2B LoRA | 45.73% (bench) / 13.23% (norm) | CC-BY-NC-4.0 |
| sudoping01/bambara-asr-v2 | sudoping01 | whisper-large-v2 | ~1.5B | 25.07% (test) | Apache 2.0 |
| sudoping01/bambara-whisper-large-v4 | sudoping01 | whisper-large-v2 | ~1.5B PEFT | Not reported | Apache 2.0 |
| RobotsMali/soloni-114m-tdt-ctc-v0 | RobotsMali | NVIDIA parakeet | 114M | Not reported | Unknown |
| RobotsMali QuartzNet-15x5 | RobotsMali | NVIDIA QuartzNet | 19M | Not reported | Unknown |
| MALIBA-AI/bomu-asr | MALIBA-AI | Unknown | 300M | Not reported | Unknown |
| MALIBA-AI/dogon-asr | MALIBA-AI | Unknown | 300M | Not reported | Unknown |
| MALIBA-AI/minianka-asr | MALIBA-AI | Unknown | 300M | Not reported | Unknown |
| MALIBA-AI/songhoy-asr | MALIBA-AI | Unknown | Unknown | Not reported | Unknown |
| UBC-NLP/Simba-M | UBC-NLP | Whisper-based | Unknown | Not reported for bam | Unknown |
| oza75/whisper-bambara-asr-002 | oza75 | Whisper | Unknown | Not reported | Unknown |

## LLM / Text Generation (1 found)

| Model | Base | Training | Val Loss | Notes |
|-------|------|----------|----------|-------|
| sudoping01/maliba-llm | google/gemma-3n-E2B-it | 1M instruction examples (MALIBA-Instructions) | 0.4952 (from 7.4595) | First Bambara LLM. Supports bam/fr/en code-switching. |

## TTS (2 found)

| Model | Base | Size | Speakers | MOS |
|-------|------|------|----------|-----|
| MALIBA-AI/bambara-tts | Spark-TTS (Qwen2.5) | 500M | 10 native | 4.2/5 |
| MALIBA-AI/malian-tts | Unknown | Unknown | Unknown | Unknown |

## Embeddings (1 found)

| Model | Type | Dims | Vocab |
|-------|------|------|-------|
| MALIBA-AI/bambara-embeddings | FastText skip-gram | 300 | 9,973 words |

## Multilingual Models with Bambara

| Model | Type | Languages | Notes |
|-------|------|-----------|-------|
| AfroLM | Pretrained MLM | 23 African | Outperforms AfriBERTa, mBERT on MasakhaNER |
| NLLB-200 | Translation | 200 | Bambara included |
| AfriMBART / AfriMT5 / AfriM2M100 | Translation | Multiple African | Evaluated in Bayelemabaga |

## N'Ko-Specific Resources

- WMT 2023 nicolingua-0005: 130,850 parallel segments + 3M+ N'Ko words
- FLoRes-200 N'Ko translations (2,009 segments)
- NLLB-Seed N'Ko translations (6,193 segments)
- Best NMT baseline: 30.83 en-nko chrF++ on FLoRes-devtest
- Google Translate added N'Ko support (June 2024)
- NO text generation model for N'Ko exists

## Key Datasets

| Dataset | Size | Type |
|---------|------|------|
| MALIBA-AI/D4Pret | 667K | Pretraining |
| MALIBA-AI/D4Pret2 | 93K | Pretraining v2 |
| MALIBA-AI/bambara-mt-dataset | 186K | Translation pairs |
| MALIBA-AI/bambara-asr-benchmark | 500 samples | ASR eval |
| djelia/bambara-lm-qa | Unknown | QA + instruction |
| djelia/bambara-mt-dataset | Unknown | Translation |
| RobotsMali/afvoices | 612 hours | Speech |
| RobotsMali/jeli-asr | Unknown | Conversational speech |
| RobotsMali/bam-asr-early | 37 hours | Early ASR |
| Bayelemabaga | 46,976 pairs | bam-fr parallel (from 264 texts) |
| oza75/bambara-asr | 2,088+ samples | ASR |
| oza75/bambara-mt | Unknown | MT pairs |

## Key Papers

| Paper | Venue | Relevance |
|-------|-------|-----------|
| Bayelemabaga (NAACL 2025) | aclanthology.org/2025.naacl-long.602 | Largest curated bam-fr dataset, MT baselines |
| WMT 2023 N'Ko MT | aclanthology.org/2023.wmt-1.34 | Only N'Ko NMT baselines |
| AfroLM (SustaiNLP 2022) | aclanthology.org/2022.sustainlp-1.11 | 23-lang African MLM |
| AfroBench (ACL Findings 2025) | aclanthology.org/2025.findings-acl.976 | 64-lang LLM benchmark |
| IrokoBench (NAACL 2025) | aclanthology.org/2025.naacl-long.139 | African LLM eval |
| Bambara ASR Survey (2026) | arxiv.org/html/2602.09785 | State of Bambara ASR |
| Hard Facts Low-Resource (2025) | arxiv.org/pdf/2511.18557 | Challenges in African NLP |
