# arXiv Submission Instructions

## File: `nko-brain-scanner-arxiv.tar.gz` (231KB)

## Steps

1. Go to https://arxiv.org/submit
2. Log in (or create account)
3. Click "Start New Submission"
4. Upload `nko-brain-scanner-arxiv.tar.gz`
5. Fill metadata:

### Metadata

- **Title**: The Script That Machines Can't Read: Adapting Large Language Models for N'Ko
- **Authors**: Mohamed Diomande
- **Abstract**: We present a systematic study of how large language models process N'Ko (U+07C0-U+07FF), an alphabetic script used by over 40 million Manding-language speakers in West Africa. Through activation profiling ("brain scanning") of Qwen3-8B before and after fine-tuning, we demonstrate that: (1) fine-tuning concentrates N'Ko adaptation in the top 8 transformer layers, reducing activation magnitudes in reasoning layers while amplifying output confidence; (2) a three-stage training pipeline (continued pre-training, supervised fine-tuning, and BPE-aware subword training) reduces the N'Ko-to-English perplexity gap ("translation tax") from 2.90x to 0.70x, a 76% reduction; (3) a finite-state machine constrained decoder guarantees 100% valid N'Ko syllable structure with 43% throughput overhead; (4) vocabulary extension with 256 N'Ko-specific BPE tokens resolves mode collapse (3/20 degenerate vs 20/20 for the non-extended model) while achieving 99.8% unconstrained syllable validity. All training runs on consumer hardware (Apple M4, 16GB) at zero cloud cost. We release the model, training pipeline, and a retrieval-centric ASR architecture design for N'Ko speech recognition.
- **Primary Category**: cs.CL (Computation and Language)
- **Secondary Categories**: cs.AI, cs.LG
- **Comments**: 10 pages, 2 figures, 3 tables
- **License**: CC BY 4.0

6. Preview and submit
