"""
Gradium voice integration — REAL SDK shape.

Per docs at https://docs.gradium.ai/:

    pip install gradium

    import gradium
    client = gradium.client.GradiumClient(api_key="gd_...")

    # Text-to-speech (single-shot)
    result = await client.tts(
        setup={"model_name": "default", "voice_id": "YTpq7expH9539ERJ", "output_format": "wav"},
        text="Hello, world!"
    )
    with open("out.wav", "wb") as f:
        f.write(result.raw_data)

    # Speech-to-text (streaming)
    stream = await client.stt_stream(
        {"model_name": "default", "input_format": "pcm"},
        audio_generator,    # async iterator yielding 1920-sample PCM16 chunks @ 24kHz
    )
    async for msg in stream._stream:
        if msg["type"] == "text":
            print(msg["text"])

Audio spec for STT:
  - PCM16 mono, 24000 Hz, 16-bit signed integer little-endian
  - Chunk size: 1920 samples (80ms) per chunk

Audio spec for TTS output (when output_format="pcm"):
  - PCM16 mono, 48000 Hz, 16-bit signed integer
  - Chunk size: 3840 samples (80ms)
"""
import asyncio
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

from . import config


class GradiumClient:
    """Wraps the gradium Python SDK. Falls back to mocks when USE_MOCK=1 or no API key."""

    def __init__(self):
        self.api_key = config.GRADIUM_API_KEY
        self.mock = config.USE_MOCK or not self.api_key
        self._sdk = None  # lazy-loaded

        # Voice to use for agent speech
        self.voice_id = config.GRADIUM_CUSTOM_VOICE_ID or config.GRADIUM_FALLBACK_VOICE_ID

    def _get_sdk_client(self):
        """Lazy-load the real SDK only when live mode is on."""
        if self._sdk is not None:
            return self._sdk
        try:
            import gradium
        except ImportError as e:
            raise RuntimeError(
                "gradium package not installed. Run: pip install gradium"
            ) from e
        self._sdk = gradium.client.GradiumClient(api_key=self.api_key)
        return self._sdk

    # ============================================================
    # TEXT-TO-SPEECH — agent voice
    # ============================================================
    async def synthesize(self, text: str, output_format: str = "wav") -> bytes:
        """
        Render `text` as speech in the configured voice.
        Returns raw audio bytes.
        """
        if self.mock:
            # Return a tiny silent WAV so the frontend player still works
            return _silent_wav(duration_s=min(5.0, len(text) / 15))

        client = self._get_sdk_client()
        result = await client.tts(
            setup={
                "model_name": "default",
                "voice_id": self.voice_id,
                "output_format": output_format,
            },
            text=text,
        )
        return result.raw_data

    async def synthesize_streaming(self, text: str) -> AsyncIterator[bytes]:
        """Streaming TTS — yields PCM chunks as they are generated."""
        if self.mock:
            # Yield silent chunks
            for _ in range(5):
                yield b"\x00" * 3840 * 2
            return

        client = self._get_sdk_client()
        stream = await client.tts_stream(
            setup={
                "model_name": "default",
                "voice_id": self.voice_id,
                "output_format": "pcm",
            },
            text=text,
        )
        async for chunk in stream.iter_bytes():
            yield chunk

    # ============================================================
    # SPEECH-TO-TEXT — streaming
    # ============================================================
    async def stt_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        on_text: Callable[[str, dict], None],
        on_vad: Optional[Callable[[float], None]] = None,
    ):
        """
        Consume an async iterator of PCM16 @ 24kHz 1920-sample chunks.
        Call `on_text(text, meta)` for each transcription result.
        Call `on_vad(inactivity_prob)` for each VAD step (used for turn-taking).

        This drives the main audio pipeline. Exits when the audio iterator ends.
        """
        if self.mock:
            # In mock mode, we don't consume audio — the browser's Web Speech API
            # provides transcripts directly via /api/session/transcript
            async for _ in audio_chunks:
                pass
            return

        client = self._get_sdk_client()
        stream = await client.stt_stream(
            {"model_name": "default", "input_format": "pcm"},
            audio_chunks,
        )
        async for msg in stream._stream:
            msg_type = msg.get("type")
            if msg_type == "text":
                text = msg.get("text", "")
                if text:
                    on_text(text, msg)
            elif msg_type == "step" and on_vad:
                vad = msg.get("vad", [])
                if len(vad) > 2 and isinstance(vad[2], dict):
                    inactivity = vad[2].get("inactivity_prob")
                    if inactivity is not None:
                        on_vad(inactivity)

    async def close(self):
        """Best-effort cleanup — the real SDK manages its own connections."""
        pass


# ============================================================
# HELPERS
# ============================================================

def _silent_wav(duration_s: float, sample_rate: int = 24000) -> bytes:
    """Minimal valid WAV file with N seconds of silence, for mock mode."""
    import struct
    n_samples = int(duration_s * sample_rate)
    n_bytes = n_samples * 2

    header = b"RIFF"
    header += struct.pack("<I", 36 + n_bytes)
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    header += b"data"
    header += struct.pack("<I", n_bytes)
    return header + (b"\x00" * n_bytes)
