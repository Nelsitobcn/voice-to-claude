"""Whisper.cpp transcription functionality."""

import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

from .config import Config, WHISPER_MODELS


@dataclass
class TranscriptionResult:
    """Result of a transcription."""
    text: str
    duration_seconds: float
    model: str
    success: bool
    error: Optional[str] = None


class Transcriber:
    """Transcribes audio using parakeet-mlx, with whisper.cpp fallback."""

    # NVIDIA Parakeet TDT 0.6b v3 (MLX port). Multilingual incl. Spanish,
    # auto language detection, no silence hallucination. Verified 2026-06-02.
    PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"

    def __init__(self, config: Config):
        self.config = config
        self._parakeet_model = None  # lazy-loaded + cached on first transcribe

    def transcribe(self, audio_path: Path, timeout: int = 120) -> TranscriptionResult:
        """
        Transcribe an audio file.

        Tries parakeet-mlx first (Apple Silicon native, no silence hallucination,
        ~28x realtime on M-series). Falls back to whisper.cpp if parakeet is not
        installed or fails, so dictation never fully breaks.

        Args:
            audio_path: Path to the WAV file to transcribe
            timeout: Maximum time in seconds to wait for transcription

        Returns:
            TranscriptionResult with text and metadata
        """
        # task=translate is whisper-only (parakeet only transcribes). If the user
        # explicitly wants translation, skip parakeet and go straight to whisper.cpp.
        if self.config.task != "translate":
            parakeet = self._transcribe_parakeet(audio_path)
            if parakeet is not None:
                return parakeet
            # parakeet returned None => not installed / hard failure => fall through

        return self._transcribe_whisper_cpp(audio_path, timeout)

    def _transcribe_parakeet(self, audio_path: Path) -> Optional[TranscriptionResult]:
        """
        Transcribe using parakeet-mlx (NVIDIA Parakeet TDT 0.6b v3, MLX port).

        Returns a TranscriptionResult on a clean run (including a successful
        empty result on silence — parakeet does NOT hallucinate filler the way
        whisper does). Returns None to signal "fall back to whisper.cpp"
        (package missing, model load failure, or unexpected exception).
        """
        try:
            from parakeet_mlx import from_pretrained
        except ImportError:
            return None  # not installed -> let whisper.cpp handle it

        start_time = time.time()
        try:
            if not audio_path.exists():
                return TranscriptionResult(
                    text="",
                    duration_seconds=0,
                    model=self.PARAKEET_MODEL,
                    success=False,
                    error=f"Audio file not found: {audio_path}",
                )

            # Lazy-load and cache the model on the instance (~600MB download on
            # first ever run, then cached in ~/.cache/huggingface/hub).
            if self._parakeet_model is None:
                self._parakeet_model = from_pretrained(self.PARAKEET_MODEL)

            # transcribe() accepts the file path directly and returns an
            # AlignedResult dataclass with a `.text` field (verified against
            # parakeet-mlx 0.5.1). Language is auto-detected; no -l flag exists.
            result = self._parakeet_model.transcribe(str(audio_path))
            elapsed = time.time() - start_time

            transcript = " ".join((result.text or "").split())

            if not transcript:
                # Genuine silence/no-speech. parakeet returns empty (no "yyyyy").
                return TranscriptionResult(
                    text="",
                    duration_seconds=elapsed,
                    model=self.PARAKEET_MODEL,
                    success=False,
                    error="No speech detected",
                )

            return TranscriptionResult(
                text=transcript,
                duration_seconds=elapsed,
                model=self.PARAKEET_MODEL,
                success=True,
            )
        except Exception:
            # Any hard failure -> signal fallback to whisper.cpp.
            return None

    def _transcribe_whisper_cpp(self, audio_path: Path, timeout: int) -> TranscriptionResult:
        """Transcribe using whisper.cpp (fallback / translate path)."""
        whisper_cli = self.config.get_whisper_cli()
        model_path = self.config.get_model_path()

        if not whisper_cli or not whisper_cli.exists():
            return TranscriptionResult(
                text="",
                duration_seconds=0,
                model=self.config.model,
                success=False,
                error="whisper-cli not found. Run /voice-to-claude:setup first."
            )

        if not model_path or not model_path.exists():
            return TranscriptionResult(
                text="",
                duration_seconds=0,
                model=self.config.model,
                success=False,
                error=f"Model '{self.config.model}' not found at {model_path}. Run /voice-to-claude:setup first."
            )

        start_time = time.time()

        cmd = [
            str(whisper_cli),
            "-m", str(model_path),
            "-f", str(audio_path),
            "--no-timestamps",
            "-nt",
        ]
        if self.config.language:
            cmd += ["-l", self.config.language]
        if self.config.task == "translate":
            cmd += ["-tr"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            elapsed = time.time() - start_time

            if result.returncode != 0:
                return TranscriptionResult(
                    text="",
                    duration_seconds=elapsed,
                    model=self.config.model,
                    success=False,
                    error=f"Transcription failed: {result.stderr}"
                )

            # Clean up transcript (remove extra whitespace)
            transcript = " ".join(result.stdout.strip().split())

            if not transcript:
                return TranscriptionResult(
                    text="",
                    duration_seconds=elapsed,
                    model=self.config.model,
                    success=False,
                    error="No speech detected"
                )

            return TranscriptionResult(
                text=transcript,
                duration_seconds=elapsed,
                model=self.config.model,
                success=True
            )

        except subprocess.TimeoutExpired:
            return TranscriptionResult(
                text="",
                duration_seconds=timeout,
                model=self.config.model,
                success=False,
                error=f"Transcription timed out after {timeout}s"
            )
        except Exception as e:
            return TranscriptionResult(
                text="",
                duration_seconds=time.time() - start_time,
                model=self.config.model,
                success=False,
                error=f"Transcription error: {e}"
            )

    @staticmethod
    def find_whisper_cli(plugin_root: Path) -> Optional[Path]:
        """Find whisper-cli executable."""
        locations = [
            plugin_root / "whisper.cpp" / "build" / "bin" / "whisper-cli",
            Path.home() / ".local" / "share" / "voice-to-claude" / "whisper.cpp" / "build" / "bin" / "whisper-cli",
        ]

        for loc in locations:
            if loc.exists():
                return loc

        # Check if it's in PATH
        result = subprocess.run(["which", "whisper-cli"], capture_output=True, text=True)
        if result.returncode == 0:
            return Path(result.stdout.strip())

        return None

    @staticmethod
    def find_models_dir(whisper_cli: Path) -> Optional[Path]:
        """Find Whisper models directory relative to whisper-cli."""
        # Models are typically at whisper.cpp/models
        # whisper-cli is at whisper.cpp/build/bin/whisper-cli
        models_dir = whisper_cli.parent.parent.parent / "models"
        if models_dir.exists():
            return models_dir
        return None

    @staticmethod
    def get_available_models(models_dir: Path) -> list:
        """Get list of downloaded models."""
        available = []
        for model_name, model_info in WHISPER_MODELS.items():
            model_path = models_dir / model_info["file"]
            if model_path.exists():
                available.append(model_name)
        return available
