"""
Qwen3-TTS — OpenAI-compatible TTS API Server
Endpoint: POST /v1/audio/speech

Drop-in replacement for OpenAI's TTS API. Compatible with any client
that uses the openai Python SDK or raw HTTP calls.

Usage:
    pip install fastapi uvicorn pydantic
    python tts_server.py

    # or with custom host/port:
    python tts_server.py --host 0.0.0.0 --port 8000
"""

import argparse
import gc
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from contextlib import asynccontextmanager
from typing import Literal, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response, StreamingResponse
    from pydantic import BaseModel, Field, field_validator
    import uvicorn
except ImportError:
    print("Missing dependencies. Run:")
    print("  pip install fastapi uvicorn pydantic")
    sys.exit(1)

try:
    from mlx_audio.tts.utils import load_model
    from mlx_audio.tts.generate import generate_audio
except ImportError:
    print("Error: 'mlx_audio' not found.")
    print("Run: source .venv/bin/activate")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR     = os.getcwd()
MODELS_DIR   = os.path.join(BASE_DIR, "models")
VOICES_DIR   = os.path.join(BASE_DIR, "voices")

# Map OpenAI model names → Qwen3 folder names
# Only the VoiceDesign model is present — all aliases point to it.
MODEL_MAP = {
    "tts-1":            "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
    "tts-1-hd":         "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
    "qwen3-tts-design": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
}

# VoiceDesign uses a text `instruct` string instead of a speaker name.
# Map OpenAI voice names to default style descriptions so standard clients
# still get a distinct-sounding voice per name.
VOICE_INSTRUCT_MAP = {
    "alloy":   "A neutral, balanced voice. Clear and professional.",
    "echo":    "A calm male voice with a slight echo-like resonance.",
    "fable":   "A warm, storytelling voice. Slightly dramatic.",
    "onyx":    "A deep, authoritative male voice. Slow and deliberate.",
    "nova":    "A bright, energetic female voice. Friendly and clear.",
    "shimmer": "A soft, gentle female voice. Soothing and light.",
}

# Supported output formats  (wav is native; others require ffmpeg)
SUPPORTED_FORMATS = {"mp3", "wav", "opus", "aac", "flac", "pcm"}

# Default model loaded at startup (set to None to disable eager loading)
DEFAULT_MODEL_KEY = "tts-1"

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ModelCache:
    """Holds the currently loaded model to avoid reloading on every request."""
    def __init__(self):
        self.model       = None
        self.loaded_key  = None   # the MODEL_MAP key that is currently loaded

    def load(self, model_key: str):
        folder = MODEL_MAP.get(model_key)
        if folder is None:
            raise ValueError(f"Unknown model '{model_key}'. "
                             f"Valid options: {list(MODEL_MAP.keys())}")

        if self.loaded_key == model_key and self.model is not None:
            return  # already loaded

        model_path = _get_model_path(folder)
        if model_path is None:
            raise FileNotFoundError(
                f"Model folder not found: {os.path.join(MODELS_DIR, folder)}"
            )

        print(f"[server] Loading model: {model_key} ({folder}) …")
        if self.model is not None:
            del self.model
            gc.collect()

        self.model      = load_model(model_path)
        self.loaded_key = model_key
        print(f"[server] Model ready: {model_key}")


cache = ModelCache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_model_path(folder_name: str) -> Optional[str]:
    full = os.path.join(MODELS_DIR, folder_name)
    if not os.path.exists(full):
        return None
    snapshots = os.path.join(full, "snapshots")
    if os.path.exists(snapshots):
        subs = [f for f in os.listdir(snapshots) if not f.startswith(".")]
        if subs:
            return os.path.join(snapshots, subs[0])
    return full


def _resolve_instruct(voice: str, override: Optional[str]) -> str:
    """Return the instruct string to use for VoiceDesign generation."""
    if override:
        return override
    return VOICE_INSTRUCT_MAP.get(voice.lower(), f"A clear, natural voice called {voice}.")


def _resolve_ref_audio(voice: str) -> tuple[Optional[str], Optional[str]]:
    """
    If `voice` matches a saved voice file in VOICES_DIR, return
    (wav_path, transcript).  Otherwise return (None, None).
    """
    safe = re.sub(r"[^\w\s-]", "", voice).strip().replace(" ", "_")
    wav  = os.path.join(VOICES_DIR, f"{safe}.wav")
    txt  = os.path.join(VOICES_DIR, f"{safe}.txt")
    if os.path.exists(wav):
        transcript = ""
        if os.path.exists(txt):
            with open(txt, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
        return wav, transcript or "."
    return None, None


def _wav_to_format(wav_path: str, fmt: str) -> bytes:
    """Convert a WAV file to the requested format using ffmpeg (if needed)."""
    if fmt == "wav":
        with open(wav_path, "rb") as f:
            return f.read()

    if fmt == "pcm":
        # Raw 16-bit LE PCM, 24 kHz mono — strip WAV header with ffmpeg
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", wav_path,
            "-f", "s16le", "-ar", "24000", "-ac", "1", "pipe:1",
        ]
    else:
        codec_map = {
            "mp3":  ("libmp3lame", "mp3"),
            "opus": ("libopus",    "ogg"),
            "aac":  ("aac",        "adts"),
            "flac": ("flac",       "flac"),
        }
        codec, fmt_flag = codec_map[fmt]
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", wav_path,
            "-c:a", codec,
            "-f", fmt_flag, "pipe:1",
        ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg conversion to {fmt} failed: {result.stderr.decode()}"
        )
    return result.stdout


MIME_TYPES = {
    "mp3":  "audio/mpeg",
    "wav":  "audio/wav",
    "opus": "audio/ogg",
    "aac":  "audio/aac",
    "flac": "audio/flac",
    "pcm":  "audio/pcm",
}


# ---------------------------------------------------------------------------
# Lifespan — eager model loading
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if DEFAULT_MODEL_KEY:
        try:
            cache.load(DEFAULT_MODEL_KEY)
        except Exception as e:
            print(f"[server] Warning: could not pre-load default model: {e}")
    yield
    # Cleanup on shutdown
    if cache.model is not None:
        del cache.model
        gc.collect()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Qwen3-TTS · OpenAI-compatible API",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    model: str = Field(
        default="tts-1",
        description="'tts-1' (Lite/fast) or 'tts-1-hd' (Pro/quality)",
    )
    input: str = Field(
        ...,
        description="The text to synthesise.",
        max_length=4096,
    )
    voice: str = Field(
        default="alloy",
        description=(
            "OpenAI voices: alloy, echo, fable, onyx, nova, shimmer. "
            "Also accepts Qwen3 speaker names or saved voice-clone names."
        ),
    )
    response_format: Literal["mp3", "wav", "opus", "aac", "flac", "pcm"] = Field(
        default="mp3",
        description="Audio encoding format.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Playback speed multiplier (0.25 – 4.0).",
    )
    # ---- Qwen3-specific extensions (ignored by standard OAI clients) ----
    instruct: Optional[str] = Field(
        default=None,
        description=(
            "[Qwen3 extension] Emotion / style instruction, e.g. "
            "'Excited and fast' or 'A calm deep radio voice'."
        ),
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if v not in MODEL_MAP:
            raise ValueError(
                f"Unknown model '{v}'. Valid: {list(MODEL_MAP.keys())}"
            )
        return v

    @field_validator("response_format")
    @classmethod
    def validate_format(cls, v):
        if v not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format '{v}'.")
        return v


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "Qwen3-TTS OpenAI-compatible server",
        "endpoints": ["/v1/audio/speech", "/v1/models", "/health"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": cache.loaded_key,
    }


@app.get("/v1/models")
def list_models():
    """Returns a minimal OpenAI-style model listing."""
    return {
        "object": "list",
        "data": [
            {"id": k, "object": "model", "owned_by": "qwen3-tts"}
            for k in MODEL_MAP
        ],
    }


@app.post("/v1/audio/speech")
def create_speech(req: SpeechRequest):
    """
    OpenAI-compatible TTS endpoint.

    Compatible with:
        openai.audio.speech.create(model=..., input=..., voice=...)
    """
    # --- Load model (uses cache if same model) ---
    try:
        cache.load(req.model)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load error: {e}")

    model = cache.model

    # --- Resolve voice ---
    # Check for a saved voice-clone first; fall back to VoiceDesign instruct.
    ref_audio, ref_text = _resolve_ref_audio(req.voice)
    instruct = _resolve_instruct(req.voice, req.instruct)

    # --- Generate audio and stream directly to HTTP ---
    tmp = tempfile.mkdtemp()
    try:
        if ref_audio:
            generate_audio(
                model=model,
                text=req.input,
                ref_audio=ref_audio,
                ref_text=ref_text,
                speed=req.speed,
                output_path=tmp,
            )
        else:
            generate_audio(
                model=model,
                text=req.input,
                instruct=instruct,
                speed=req.speed,
                output_path=tmp,
            )
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")

    wav_path = os.path.join(tmp, "audio_000.wav")
    if not os.path.exists(wav_path):
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(status_code=500, detail="TTS engine produced no output file.")

    # For non-wav formats, convert via ffmpeg first (produces full bytes),
    # then stream. For wav, stream the file directly in chunks.
    mime = MIME_TYPES[req.response_format]

    if req.response_format == "wav":
        def wav_stream():
            try:
                with open(wav_path, "rb") as f:
                    while chunk := f.read(65536):
                        yield chunk
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        return StreamingResponse(wav_stream(), media_type=mime)

    else:
        # ffmpeg converts the wav and writes to stdout — pipe directly
        try:
            audio_bytes = _wav_to_format(wav_path, req.response_format)
        except Exception as e:
            shutil.rmtree(tmp, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"Format conversion error: {e}")
        shutil.rmtree(tmp, ignore_errors=True)

        def byte_stream(data: bytes):
            chunk_size = 65536
            for i in range(0, len(data), chunk_size):
                yield data[i:i + chunk_size]

        return StreamingResponse(byte_stream(audio_bytes), media_type=mime)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS OpenAI-compatible server")
    parser.add_argument("--host",    default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port",    type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload",  action="store_true", help="Enable auto-reload (dev mode)")
    parser.add_argument("--model",   default=DEFAULT_MODEL_KEY,
                        help=f"Model to pre-load at startup (default: {DEFAULT_MODEL_KEY})")
    args = parser.parse_args()

    # Allow overriding the startup model via CLI
    DEFAULT_MODEL_KEY = args.model

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )