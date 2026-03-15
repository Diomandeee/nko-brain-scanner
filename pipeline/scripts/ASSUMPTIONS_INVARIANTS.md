# Assumptions and Invariants Ledger

1. Assumptions
1.1 A-1 GEMINI_API_KEY is set and valid.
1.1.1 If false: OCR requests fail and no detections are produced.
1.1.2 Detection: API errors or authentication failures in logs.
1.2 A-2 yt-dlp and ffmpeg are installed and available on PATH.
1.2.1 If false: Downloads or frame extraction fail.
1.2.2 Detection: Command not found or subprocess failures.
1.3 A-3 YouTube access is available or mitigated via HLS/cookies.
1.3.1 If false: Video downloads fail with 403 or similar errors.
1.3.2 Detection: yt-dlp errors and retry exhaustion.
1.4 A-4 Supabase URL and service role key are provided when `--store-supabase` is used.
1.4.1 If false: Storage fails and records are not persisted.
1.4.2 Detection: Supabase insert errors.

2. Invariants
2.1 I-1 Must use Supabase service role key for inserts when `--store-supabase` is enabled.
2.1.1 Why: RLS policies block inserts with anon keys.
2.1.2 If violated: Inserts fail and data is lost.
2.1.3 Canary: RLS policy violation errors.
2.2 I-2 Must write output JSON with top-level keys `summary` and `results`.
2.2.1 Why: Downstream consumers rely on stable schema.
2.2.2 If violated: Consumers cannot parse outputs.
2.2.3 Canary: Missing keys in output validation.
2.3 I-3 Must include `video_id` and `frame_index` for each detection.
2.3.1 Why: Results must remain traceable to source media.
2.3.2 If violated: Detections cannot be audited or deduplicated.
2.3.3 Canary: Missing fields in output validation.
2.4 I-4 Must not generate world variants when `--skip-worlds` is set.
2.4.1 Why: User explicitly opts out to reduce cost and time.
2.4.2 If violated: Cost increases and contract is broken.
2.4.3 Canary: World variants present when `--skip-worlds` is used.

3. Change History
3.1 2026-01-03 v0.1 Initial ledger derived from README.md.
