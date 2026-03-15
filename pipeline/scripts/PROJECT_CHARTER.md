# Project Charter

1. Purpose
1.1 Statement
1.1.1 Must provide a local, script-driven pipeline that, given one or more YouTube video URLs, downloads video data, extracts frames, applies Gemini OCR to detect N'Ko text, optionally generates up to five world variants per detection, and writes a structured JSON report; when Supabase credentials are provided and `--store-supabase` is used, the same results are written to Supabase.
1.1.2 Falsifiability: If a run with valid inputs does not produce the JSON report or expected Supabase writes, the purpose is not met.

2. Non-goals
2.1 Must not provide a hosted web service or end-user UI.
2.1.1 Trace: README.md describes a local, script-driven pipeline.
2.2 Must not train or fine-tune ML models.
2.2.1 Trace: README.md describes using the Gemini API for OCR and generation.
2.3 Must not guarantee availability of YouTube content or bypass access restrictions.
2.3.1 Trace: README.md troubleshooting notes on 403 and IP blocking.

3. Success Criteria
3.1 A run of `nko_analyzer.py --limit 1` with valid `GEMINI_API_KEY` writes a JSON file containing top-level keys `summary` and `results`.
3.1.1 Validation: Inspect output file at the configured path.
3.2 A run with `--store-supabase` and a valid service role key inserts records without RLS policy errors.
3.2.1 Validation: Supabase insert logs show success and no RLS errors.
3.3 A run with `--skip-worlds` writes results that omit world variants.
3.3.1 Validation: Output lacks world variant entries.

4. Direction Constraints
4.1 Must remain compatible with Gemini-based OCR usage as described in README.md.
4.2 Must preserve the JSON output schema described in README.md unless a new schema version is documented.
4.3 Should keep local, script-driven execution without requiring a persistent server.
4.4 Should preserve HLS/m3u8 download preference in download scripts to mitigate 403 errors.

5. Commitment Level
5.1 Draft.

6. Traceability
6.1 Source signals: README.md, existing scripts in this directory.

7. Change History
7.1 2026-01-03 v0.1 Initial draft derived from README.md.
