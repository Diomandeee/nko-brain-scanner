#!/usr/bin/env python3
"""
Visual Scene Encoder for multimodal N'Ko ASR.

Extracts keyframe visual features from video using a vision-language model.
Visual context helps disambiguate N'Ko speech in context-dependent scenarios
(e.g., market scenes vs. classroom vs. outdoor conversation).

The scene embeddings are projected into the joint embedding space (d=512)
alongside audio embeddings for multimodal syllable retrieval.

Usage:
    encoder = SceneEncoder()
    embeddings = encoder.encode_keyframes("path/to/video.mp4", interval_sec=5.0)
    # embeddings shape: (num_keyframes, embed_dim)
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMBED_DIM = 512


class SceneEncoder:
    """
    Visual scene encoder using a vision model for keyframe captioning
    and feature extraction.

    For the N'Ko ASR pipeline, visual features provide:
    - Scene context (indoor/outdoor, market, school, ceremony)
    - Speaker gesture cues (pointing, emphasis)
    - Object references (disambiguate homophones)
    """

    def __init__(self, model_name="google/siglip-base-patch16-224", device=None):
        """
        Args:
            model_name: HuggingFace vision model for feature extraction
            device: torch device (auto-detects MPS on Apple Silicon)
        """
        import torch
        self.model_name = model_name
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._model = None
        self._processor = None
        self.embed_dim = EMBED_DIM

    def _load_model(self):
        """Lazy load vision model."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoProcessor

        print(f"[scene_encoder] Loading {self.model_name}...")
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()

        for param in self._model.parameters():
            param.requires_grad = False

        # Get the model's native embedding dimension
        # SigLIP base: 768, we project down to 512
        test_dim = self._get_native_dim()
        if test_dim != self.embed_dim:
            self._projector = torch.nn.Linear(test_dim, self.embed_dim).to(self.device)
            # Initialize with scaled random
            torch.nn.init.xavier_uniform_(self._projector.weight)
        else:
            self._projector = None

        print(f"[scene_encoder] Loaded. Native dim: {test_dim}, output dim: {self.embed_dim}")

    def _get_native_dim(self):
        """Determine the model's native embedding dimension."""
        import torch
        from PIL import Image

        dummy = Image.new("RGB", (224, 224))
        inputs = self._processor(images=dummy, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)
        return outputs.shape[-1]

    def extract_keyframes(self, video_path, interval_sec=5.0, max_frames=50):
        """
        Extract keyframes from video at regular intervals using ffmpeg.

        Args:
            video_path: Path to video file
            interval_sec: Seconds between keyframe extractions
            max_frames: Maximum number of keyframes

        Returns:
            List of (timestamp_sec, image_path) tuples
        """
        tmpdir = tempfile.mkdtemp(prefix="nko_keyframes_")

        # Get video duration
        probe_cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            duration = float(result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError):
            duration = max_frames * interval_sec  # fallback

        # Extract frames
        num_frames = min(int(duration / interval_sec), max_frames)
        frame_cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"fps=1/{interval_sec}",
            "-frames:v", str(num_frames),
            "-q:v", "2",
            os.path.join(tmpdir, "frame_%04d.jpg"),
            "-y", "-loglevel", "quiet",
        ]
        subprocess.run(frame_cmd, timeout=300)

        # Collect frames with timestamps
        keyframes = []
        for i, frame_file in enumerate(sorted(Path(tmpdir).glob("frame_*.jpg"))):
            timestamp = i * interval_sec
            keyframes.append((timestamp, str(frame_file)))

        return keyframes

    def encode_image(self, image_path):
        """
        Encode a single image to an embedding vector.

        Args:
            image_path: Path to image file

        Returns:
            embedding: (embed_dim,) numpy array
        """
        import torch
        from PIL import Image

        self._load_model()

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            features = self._model.get_image_features(**inputs)  # (1, native_dim)
            if self._projector is not None:
                features = self._projector(features)
            # L2 normalize
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().numpy()

    def encode_keyframes(self, video_path, interval_sec=5.0, max_frames=50):
        """
        Extract and encode keyframes from video.

        Args:
            video_path: Path to video file
            interval_sec: Seconds between keyframe extractions
            max_frames: Maximum keyframes to extract

        Returns:
            timestamps: list of float (seconds)
            embeddings: (num_frames, embed_dim) numpy array
        """
        keyframes = self.extract_keyframes(video_path, interval_sec, max_frames)

        if not keyframes:
            return [], np.zeros((0, self.embed_dim))

        timestamps = []
        embeddings = []

        for timestamp, frame_path in keyframes:
            emb = self.encode_image(frame_path)
            timestamps.append(timestamp)
            embeddings.append(emb)

            # Clean up temp frame
            try:
                os.unlink(frame_path)
            except OSError:
                pass

        return timestamps, np.stack(embeddings)

    def encode_images_batch(self, image_paths, batch_size=8):
        """
        Batch encode multiple images.

        Args:
            image_paths: List of image paths
            batch_size: Processing batch size

        Returns:
            embeddings: (num_images, embed_dim) numpy array
        """
        import torch
        from PIL import Image

        self._load_model()

        all_embeddings = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self._processor(images=images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                features = self._model.get_image_features(**inputs)
                if self._projector is not None:
                    features = self._projector(features)
                features = features / features.norm(dim=-1, keepdim=True)

            all_embeddings.append(features.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


if __name__ == "__main__":
    print("=== Scene Encoder Smoke Test ===")
    encoder = SceneEncoder()
    print(f"Model: {encoder.model_name}")
    print(f"Output embed dim: {encoder.embed_dim}")
    print(f"Device: {encoder.device}")

    # Test with synthetic image
    from PIL import Image
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        # Create a simple test image
        img = Image.new("RGB", (224, 224), color=(128, 64, 32))
        img.save(f.name)

        embedding = encoder.encode_image(f.name)
        print(f"Output shape: {embedding.shape}")
        print(f"Mean: {embedding.mean():.4f}, Std: {embedding.std():.4f}")
        print(f"L2 norm: {np.linalg.norm(embedding):.4f}")
        os.unlink(f.name)

    print("Smoke test passed.")
