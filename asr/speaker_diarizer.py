#!/usr/bin/env python3
"""
Speaker Diarization for N'Ko ASR Pipeline.

Segments multi-speaker audio into per-speaker regions using pyannote.audio.
Groups speech turns by speaker identity for downstream ASR processing.

Usage:
    diarizer = SpeakerDiarizer()
    turns = diarizer.diarize("path/to/audio.wav")
    # turns: list of SpeakerTurn(start, end, speaker_id, confidence)
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class SpeakerTurn:
    """A single speaker turn with timing and identity."""
    start: float       # seconds
    end: float         # seconds
    speaker_id: str    # e.g., "SPEAKER_00", "SPEAKER_01"
    confidence: float  # diarization confidence [0, 1]

    @property
    def duration(self):
        return self.end - self.start


class SpeakerDiarizer:
    """
    Speaker diarization using pyannote.audio.

    Segments audio into speaker turns. For Djoko series:
    - Typically 2-4 speakers per episode
    - Bambara language (N'Ko script target)
    - Background noise common (outdoor recordings)
    """

    def __init__(self, auth_token=None, min_duration=1.0, max_speakers=6):
        """
        Args:
            auth_token: HuggingFace token for pyannote access
            min_duration: Minimum turn duration in seconds
            max_speakers: Maximum expected speakers
        """
        self.auth_token = auth_token or os.environ.get("HF_TOKEN")
        self.min_duration = min_duration
        self.max_speakers = max_speakers
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy load pyannote pipeline."""
        if self._pipeline is not None:
            return

        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise ImportError(
                "pyannote.audio required for diarization. "
                "Install: pip install pyannote.audio"
            )

        if not self.auth_token:
            raise ValueError(
                "HuggingFace token required for pyannote. "
                "Set HF_TOKEN env var or pass auth_token."
            )

        print("[diarizer] Loading pyannote pipeline...")
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.auth_token,
        )

        import torch
        if torch.backends.mps.is_available():
            self._pipeline.to(torch.device("mps"))
        print("[diarizer] Pipeline loaded.")

    def diarize(self, audio_path, num_speakers=None):
        """
        Run speaker diarization on an audio file.

        Args:
            audio_path: Path to audio file
            num_speakers: Optional known speaker count (improves accuracy)

        Returns:
            List of SpeakerTurn sorted by start time
        """
        self._load_pipeline()

        params = {}
        if num_speakers is not None:
            params["num_speakers"] = num_speakers
        elif self.max_speakers:
            params["max_speakers"] = self.max_speakers

        diarization = self._pipeline(audio_path, **params)

        turns = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if segment.duration < self.min_duration:
                continue
            turns.append(SpeakerTurn(
                start=segment.start,
                end=segment.end,
                speaker_id=speaker,
                confidence=1.0,  # pyannote 3.x doesn't expose per-segment confidence
            ))

        turns.sort(key=lambda t: t.start)
        return turns

    def diarize_with_vad(self, audio_path, vad_segments):
        """
        Run diarization constrained to VAD-detected speech regions.

        Args:
            audio_path: Path to audio file
            vad_segments: List of (start_sec, end_sec) from VAD

        Returns:
            List of SpeakerTurn with speaker IDs assigned to VAD segments
        """
        full_turns = self.diarize(audio_path)

        # Map VAD segments to speakers by overlap
        assigned_turns = []
        for vad_start, vad_end in vad_segments:
            best_speaker = None
            best_overlap = 0.0

            for turn in full_turns:
                overlap_start = max(vad_start, turn.start)
                overlap_end = min(vad_end, turn.end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = turn.speaker_id

            if best_speaker:
                assigned_turns.append(SpeakerTurn(
                    start=vad_start,
                    end=vad_end,
                    speaker_id=best_speaker,
                    confidence=best_overlap / (vad_end - vad_start),
                ))

        return assigned_turns

    def group_by_speaker(self, turns):
        """
        Group consecutive turns by speaker.

        Merges adjacent turns from the same speaker with gaps < 1s.
        """
        if not turns:
            return []

        grouped = [turns[0]]
        for turn in turns[1:]:
            prev = grouped[-1]
            if (turn.speaker_id == prev.speaker_id and
                    turn.start - prev.end < 1.0):
                # Merge with previous
                grouped[-1] = SpeakerTurn(
                    start=prev.start,
                    end=turn.end,
                    speaker_id=prev.speaker_id,
                    confidence=min(prev.confidence, turn.confidence),
                )
            else:
                grouped.append(turn)

        return grouped

    def speaker_stats(self, turns):
        """Compute per-speaker statistics."""
        stats = {}
        for turn in turns:
            if turn.speaker_id not in stats:
                stats[turn.speaker_id] = {
                    "total_duration": 0.0,
                    "num_turns": 0,
                    "avg_turn_duration": 0.0,
                }
            stats[turn.speaker_id]["total_duration"] += turn.duration
            stats[turn.speaker_id]["num_turns"] += 1

        for speaker, s in stats.items():
            s["avg_turn_duration"] = s["total_duration"] / s["num_turns"]

        return stats


class VADOnlyDiarizer:
    """
    Fallback diarizer using only VAD (no speaker identification).

    Assigns all speech to a single speaker. Useful when pyannote
    is not available or for single-speaker recordings.
    """

    def __init__(self, min_duration=1.0):
        self.min_duration = min_duration

    def diarize(self, audio_path, vad_segments=None):
        """
        Create turns from VAD segments without speaker identification.

        Args:
            audio_path: Path to audio (unused, for API compatibility)
            vad_segments: List of (start_sec, end_sec)
        """
        if vad_segments is None:
            return []

        turns = []
        for start, end in vad_segments:
            if end - start >= self.min_duration:
                turns.append(SpeakerTurn(
                    start=start,
                    end=end,
                    speaker_id="SPEAKER_00",
                    confidence=1.0,
                ))
        return turns


if __name__ == "__main__":
    print("=== Speaker Diarizer Smoke Test ===")

    # Test with VADOnly fallback (no pyannote dependency needed)
    diarizer = VADOnlyDiarizer(min_duration=0.5)

    # Simulate VAD segments
    vad_segments = [
        (0.0, 3.5),
        (4.0, 7.2),
        (8.0, 9.0),
        (10.0, 15.5),
    ]

    turns = diarizer.diarize("dummy.wav", vad_segments=vad_segments)
    print(f"Turns: {len(turns)}")
    for turn in turns:
        print(f"  {turn.start:.1f}-{turn.end:.1f}s: {turn.speaker_id} ({turn.duration:.1f}s)")

    # Test grouping
    full_diarizer = SpeakerDiarizer.__new__(SpeakerDiarizer)
    full_diarizer.min_duration = 0.5
    mixed_turns = [
        SpeakerTurn(0, 3, "SPEAKER_00", 1.0),
        SpeakerTurn(3.5, 6, "SPEAKER_00", 1.0),
        SpeakerTurn(7, 10, "SPEAKER_01", 1.0),
        SpeakerTurn(10.2, 12, "SPEAKER_01", 1.0),
        SpeakerTurn(13, 15, "SPEAKER_00", 1.0),
    ]
    grouped = full_diarizer.group_by_speaker(mixed_turns)
    print(f"\nGrouped turns: {len(grouped)} (from {len(mixed_turns)})")
    for turn in grouped:
        print(f"  {turn.start:.1f}-{turn.end:.1f}s: {turn.speaker_id}")

    print("\nSmoke test passed.")
