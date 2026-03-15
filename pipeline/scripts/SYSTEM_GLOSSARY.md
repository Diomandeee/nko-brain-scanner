# System Glossary

1. Analyzer
1.1 What it is: The primary script that orchestrates download, frame extraction, OCR, world generation, and output writing (e.g., `nko_analyzer.py`).
1.2 What it is not: A long-running service or UI.
1.3 Layer: Architectural.
1.4 Stability: Medium.

2. Pipeline
2.1 What it is: The end-to-end flow from YouTube video input to JSON and optional Supabase output.
2.2 What it is not: A single function or a hosted service.
2.3 Layer: Conceptual.
2.4 Stability: Medium.

3. Frame
3.1 What it is: A single image extracted from a video at a specific index or timestamp.
3.2 What it is not: A full video clip or audio segment.
3.3 Layer: Runtime.
3.4 Stability: High.

4. Detection
4.1 What it is: One OCR finding of N'Ko text within a specific frame.
4.2 What it is not: A translated document or aggregated result.
4.3 Layer: Runtime.
4.4 Stability: Medium.

5. World
5.1 What it is: A contextual style category for generating variants (everyday, formal, storytelling, proverbs, educational).
5.2 What it is not: A database schema or runtime environment.
5.3 Layer: Conceptual.
5.4 Stability: Medium.

6. World Variant
6.1 What it is: A generated text variant for a detection within a specific world.
6.2 What it is not: A new detection or a translation source text.
6.3 Layer: Runtime.
6.4 Stability: Medium.

7. OCR
7.1 What it is: Optical character recognition performed by the Gemini multimodal API.
7.2 What it is not: A local OCR engine or model training process.
7.3 Layer: Architectural.
7.4 Stability: Medium.

8. Gemini API
8.1 What it is: The external API used for OCR and text generation.
8.2 What it is not: A local dependency or a database.
8.3 Layer: Interface.
8.4 Stability: Medium.

9. Supabase Storage
9.1 What it is: Optional persistence of results to a Supabase database.
9.2 What it is not: The primary output path for local runs.
9.3 Layer: Interface.
9.4 Stability: Medium.

10. Analysis Results JSON
10.1 What it is: The structured JSON file containing `summary` and `results` data for a run.
10.2 What it is not: A raw OCR dump or a video archive.
10.3 Layer: Interface.
10.4 Stability: High.

11. Change History
11.1 2026-01-03 v0.1 Initial glossary.
