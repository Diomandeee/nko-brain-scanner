# NKo Brain Scanner — ASR Module
#
# Retrieval-centric N'Ko ASR pipeline:
#   audio_pipeline     - YouTube download + VAD segmentation
#   audio_encoder      - Whisper encoder (frozen features)
#   speaker_diarizer   - pyannote speaker clustering
#   scene_encoder      - SigLIP visual feature extraction
#   joint_embedding    - Shared embedding space (d=512)
#   syllable_retriever - Codebook retrieval + FSM-constrained beam search
#   round_trip_eval    - Round-trip accuracy evaluation
#   train_asr          - Multi-loss training loop
