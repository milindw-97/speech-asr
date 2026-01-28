# voice_agent_rnnt.py
# Real-time voice agent ASR server for RNNT models

import asyncio
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from dataclasses import dataclass, field
from typing import Optional, List, Set
import time
import re
import os
from contextlib import asynccontextmanager

# ============== Language Detection & Filtering ==============

SCRIPT_PATTERNS = {
    "latin": re.compile(r"[a-zA-ZÀ-ÿĀ-žḀ-ỿ]"),
    "devanagari": re.compile(r"[\u0900-\u097F]"),
    "cyrillic": re.compile(r"[\u0400-\u04FF]"),
    "arabic": re.compile(r"[\u0600-\u06FF]"),
    "cjk": re.compile(r"[\u4E00-\u9FFF]"),
    "japanese_kana": re.compile(r"[\u3040-\u30FF]"),
    "korean": re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF]"),
    "thai": re.compile(r"[\u0E00-\u0E7F]"),
    "greek": re.compile(r"[\u0370-\u03FF]"),
    "hebrew": re.compile(r"[\u0590-\u05FF]"),
}

LANGUAGE_SCRIPTS = {
    "en": ["latin"],
    "hi": ["devanagari", "latin"],
    "es": ["latin"],
    "fr": ["latin"],
    "de": ["latin"],
    "it": ["latin"],
    "pt": ["latin"],
    "nl": ["latin"],
    "pl": ["latin"],
    "ru": ["cyrillic"],
    "uk": ["cyrillic"],
    "ar": ["arabic"],
    "zh": ["cjk"],
    "ja": ["cjk", "japanese_kana", "latin"],
    "ko": ["korean", "cjk"],
    "th": ["thai"],
    "el": ["greek"],
    "he": ["hebrew"],
}


def detect_scripts(text: str) -> dict:
    script_counts = {}
    for script_name, pattern in SCRIPT_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            script_counts[script_name] = len(matches)
    return script_counts


def is_text_allowed(text: str, allowed_languages: List[str]) -> bool:
    if not text or not text.strip():
        return True

    detected_scripts = set(detect_scripts(text).keys())
    if not detected_scripts:
        return True

    allowed_scripts = set()
    for lang in allowed_languages:
        if lang in LANGUAGE_SCRIPTS:
            allowed_scripts.update(LANGUAGE_SCRIPTS[lang])

    return bool(detected_scripts & allowed_scripts)


def filter_unwanted_languages(text: str, allowed_languages: List[str]) -> str:
    if is_text_allowed(text, allowed_languages):
        return text
    else:
        detected = detect_scripts(text)
        print(
            f"[Language Filter] Blocked: '{text[:50]}...' (detected scripts: {detected})"
        )
        return ""


# Pattern to match language tags like <en-US>, <hi-IN>, <es-ES>, etc.
LANGUAGE_TAG_PATTERN = re.compile(r"<([a-z]{2}(?:-[A-Z]{2})?)>\s*$")


def extract_language_tag(text: str) -> tuple[str, str | None]:
    """
    Extract and remove language tag from the end of transcript text.

    Args:
        text: Transcript text that may contain a language tag like <en-US> at the end

    Returns:
        Tuple of (cleaned_text, language_tag or None)
        Example: ("Hello world", "en-US") or ("Hello world", None)
    """
    if not text:
        return text, None

    match = LANGUAGE_TAG_PATTERN.search(text)
    if match:
        language_tag = match.group(1)
        cleaned_text = text[:match.start()].rstrip()
        return cleaned_text, language_tag

    return text, None


# ============== Default Configuration ==============
@dataclass
class DefaultConfig:
    """Default configuration values - can be overridden per session"""

    # Model
    model_repo: str = os.getenv("MODEL_REPO", "milind-plivo/parakeet-multilingual-base")
    model_filename: str = os.getenv(
        "MODEL_FILENAME", "parakeet-rnnt-1.1b-multilingual.nemo"
    )
    model_cache_dir: str = os.getenv("MODEL_CACHE_DIR", "/app/models")

    sample_rate: int = 16000

    # Audio chunking
    chunk_ms: int = 100
    min_utterance_ms: int = 250
    max_utterance_ms: int = 15000

    # Endpointing
    endpointing_silence_ms: int = 400

    # VAD
    vad_threshold: float = 0.5

    # RNNT specific
    decoding_strategy: str = "greedy"
    beam_size: int = 4

    # Language
    language: Optional[str] = None
    allowed_languages: List[str] = field(default_factory=lambda: ["en", "hi"])

    # Performance
    transcribe_throttle_ms: int = 150

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


defaults = DefaultConfig()

# ============== Global State ==============
asr_model = None
vad_model = None
has_streaming = False
active_sessions: Set[str] = set()
session_counter = 0


def load_models():
    """Load ASR and VAD models"""
    global asr_model, vad_model, has_streaming

    import nemo.collections.asr as nemo_asr
    from huggingface_hub import hf_hub_download

    # Download model from HuggingFace
    print(f"Downloading model from HuggingFace: {defaults.model_repo}")
    os.makedirs(defaults.model_cache_dir, exist_ok=True)

    model_path = hf_hub_download(
        repo_id=defaults.model_repo,
        filename=defaults.model_filename,
        cache_dir=defaults.model_cache_dir,
        local_dir=defaults.model_cache_dir,
        token=os.getenv("HF_TOKEN", None),
    )
    print(f"Model downloaded to: {model_path}")

    print(f"Loading RNNT model...")
    asr_model = nemo_asr.models.ASRModel.restore_from(model_path)
    asr_model.eval()
    asr_model.to(defaults.device)
    asr_model.freeze()

    print(f"  Model class: {asr_model.__class__.__name__}")
    print(f"  Encoder: {asr_model.encoder.__class__.__name__}")
    if hasattr(asr_model, "decoder"):
        print(f"  Decoder: {asr_model.decoder.__class__.__name__}")
    if hasattr(asr_model, "joint"):
        print(f"  Joint: {asr_model.joint.__class__.__name__}")

    has_streaming = hasattr(asr_model.encoder, "set_default_att_context_size")
    print(f"  Cache-aware streaming support: {has_streaming}")

    if hasattr(asr_model, "change_decoding_strategy"):
        asr_model.change_decoding_strategy(
            decoding_cfg={
                "strategy": defaults.decoding_strategy,
                "beam": {
                    "beam_size": defaults.beam_size,
                    "return_best_hypothesis": True,
                },
                "greedy": {
                    "max_symbols": 10,
                },
            }
        )
        print(f"  Decoding strategy: {defaults.decoding_strategy}")

    print("Loading Silero VAD...")
    vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    vad_model.to(defaults.device)

    print("Models ready!\n")


# ============== Session Configuration ==============
@dataclass
class SessionConfig:
    """Per-session configuration"""

    session_id: str
    language: Optional[str] = None
    allowed_languages: Optional[List[str]] = None
    min_utterance_ms: int = 250
    endpointing_silence_ms: int = 400
    vad_threshold: float = 0.5
    transcribe_throttle_ms: int = 150

    def get_allowed_languages(self) -> Optional[List[str]]:
        """Return allowed languages, falling back to defaults"""
        if self.allowed_languages is not None:
            return self.allowed_languages
        return defaults.allowed_languages


# ============== RNNT Transcription ==============
def transcribe_audio(audio: np.ndarray, session_config: SessionConfig) -> dict:
    start_time = time.time()

    with torch.no_grad():
        result = asr_model.transcribe([audio], batch_size=1, return_hypotheses=True)

    elapsed_ms = (time.time() - start_time) * 1000

    if isinstance(result[0], str):
        text = result[0]
        confidence = None
    elif hasattr(result[0], "text"):
        text = result[0].text
        confidence = getattr(result[0], "score", None)
    else:
        text = str(result[0])
        confidence = None

    text = text.strip()
    filtered = False

    # Extract language tag from transcript (e.g., <en-US>, <hi-IN>)
    text, detected_language = extract_language_tag(text)

    allowed_langs = session_config.get_allowed_languages()
    if allowed_langs:
        original_text = text
        text = filter_unwanted_languages(text, allowed_langs)
        filtered = text != original_text

    return {
        "text": text,
        "confidence": confidence,
        "latency_ms": round(elapsed_ms, 1),
        "audio_duration_ms": round(len(audio) / defaults.sample_rate * 1000, 1),
        "filtered": filtered,
        "language": detected_language,
    }


# ============== VAD ==============
VAD_CHUNK_SAMPLES = 512


def get_speech_probability(audio: np.ndarray) -> float:
    with torch.no_grad():
        max_prob = 0.0

        for i in range(0, len(audio), VAD_CHUNK_SAMPLES):
            chunk = audio[i : i + VAD_CHUNK_SAMPLES]

            if len(chunk) < VAD_CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, VAD_CHUNK_SAMPLES - len(chunk)))

            audio_tensor = torch.tensor(
                chunk, dtype=torch.float32, device=defaults.device
            )
            prob = vad_model(audio_tensor, defaults.sample_rate).item()
            max_prob = max(max_prob, prob)

        return max_prob


# ============== Voice Session ==============
class VoiceSession:
    def __init__(self, config: SessionConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.audio_buffer: List[float] = []
        self.is_speaking = False
        self.silence_ms = 0
        self.last_transcript = ""
        self.last_language = None
        self.last_transcribe_time = 0
        self.utterance_id = 0

    def buffer_duration_ms(self) -> float:
        return len(self.audio_buffer) / defaults.sample_rate * 1000

    def process_chunk(self, audio_chunk: np.ndarray) -> dict:
        chunk_ms = len(audio_chunk) / defaults.sample_rate * 1000
        current_time = time.time() * 1000

        speech_prob = get_speech_probability(audio_chunk)
        has_speech = speech_prob > self.config.vad_threshold

        result = {
            "type": "silence",
            "text": "",
            "is_final": False,
            "speech_prob": round(speech_prob, 2),
            "utterance_id": self.utterance_id,
        }

        if has_speech:
            if not self.is_speaking:
                self.is_speaking = True
                self.utterance_id += 1
                result["utterance_id"] = self.utterance_id

            self.silence_ms = 0
            self.audio_buffer.extend(audio_chunk.tolist())

            buffer_ms = self.buffer_duration_ms()
            time_since_transcribe = current_time - self.last_transcribe_time

            should_transcribe = (
                buffer_ms >= self.config.min_utterance_ms
                and time_since_transcribe >= self.config.transcribe_throttle_ms
            )

            if should_transcribe:
                audio_array = np.array(self.audio_buffer, dtype=np.float32)
                transcription = transcribe_audio(audio_array, self.config)

                self.last_transcript = transcription["text"]
                self.last_language = transcription["language"]
                self.last_transcribe_time = current_time

                result.update(
                    {
                        "type": "partial",
                        "text": transcription["text"],
                        "language": transcription["language"],
                        "confidence": transcription["confidence"],
                        "latency_ms": transcription["latency_ms"],
                        "audio_ms": round(buffer_ms, 1),
                    }
                )
            else:
                result.update(
                    {
                        "type": "speaking",
                        "text": self.last_transcript,
                        "language": self.last_language,
                        "audio_ms": round(buffer_ms, 1),
                    }
                )

        elif self.is_speaking:
            self.silence_ms += chunk_ms
            self.audio_buffer.extend(audio_chunk.tolist())

            if self.silence_ms >= self.config.endpointing_silence_ms:
                audio_array = np.array(self.audio_buffer, dtype=np.float32)
                transcription = transcribe_audio(audio_array, self.config)

                result.update(
                    {
                        "type": "final",
                        "text": transcription["text"],
                        "language": transcription["language"],
                        "is_final": True,
                        "confidence": transcription["confidence"],
                        "latency_ms": transcription["latency_ms"],
                        "audio_ms": round(self.buffer_duration_ms(), 1),
                    }
                )

                self.reset()
            else:
                result.update(
                    {
                        "type": "partial",
                        "text": self.last_transcript,
                        "language": self.last_language,
                        "silence_ms": round(self.silence_ms, 1),
                    }
                )

        return result

    def force_finalize(self) -> Optional[dict]:
        if self.buffer_duration_ms() >= self.config.min_utterance_ms:
            audio_array = np.array(self.audio_buffer, dtype=np.float32)
            transcription = transcribe_audio(audio_array, self.config)
            result = {
                "type": "final",
                "text": transcription["text"],
                "language": transcription["language"],
                "is_final": True,
                "forced": True,
            }
            self.reset()
            return result
        return None


# ============== App Lifespan ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_models()
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(title="Voice Agent ASR", lifespan=lifespan)


# ============== Helper to parse allowed_languages ==============
def parse_allowed_languages(value: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated language codes"""
    if value is None:
        return None
    if value.lower() == "none" or value == "":
        return None
    return [lang.strip() for lang in value.split(",") if lang.strip()]


# ============== WebSocket Endpoint ==============
@app.websocket("/ws/transcribe")
async def websocket_transcribe(
    websocket: WebSocket,
    language: Optional[str] = Query(
        default=None, description="Force specific language"
    ),
    allowed_languages: Optional[str] = Query(
        default=None, description="Comma-separated allowed languages (e.g., 'en,hi')"
    ),
    min_utterance_ms: int = Query(
        default=250, ge=100, le=2000, description="Min audio before transcription"
    ),
    endpointing_silence_ms: int = Query(
        default=400, ge=100, le=2000, description="Silence to trigger endpoint"
    ),
    vad_threshold: float = Query(
        default=0.5, ge=0.1, le=0.9, description="VAD speech probability threshold"
    ),
    transcribe_throttle_ms: int = Query(
        default=150,
        ge=50,
        le=1000,
        description="Min time between partial transcriptions",
    ),
):
    global session_counter

    await websocket.accept()

    # Generate session ID
    session_counter += 1
    session_id = f"session_{session_counter}_{int(time.time())}"

    # Parse allowed_languages
    parsed_allowed_languages = parse_allowed_languages(allowed_languages)

    # Create session config
    session_config = SessionConfig(
        session_id=session_id,
        language=language,
        allowed_languages=parsed_allowed_languages,
        min_utterance_ms=min_utterance_ms,
        endpointing_silence_ms=endpointing_silence_ms,
        vad_threshold=vad_threshold,
        transcribe_throttle_ms=transcribe_throttle_ms,
    )

    session = VoiceSession(session_config)

    # Track session
    active_sessions.add(session_id)
    print(f"[{session_id}] Connected (active sessions: {len(active_sessions)})")

    # Send ready message with session config
    await websocket.send_json(
        {
            "type": "ready",
            "session_id": session_id,
            "config": {
                "sample_rate": defaults.sample_rate,
                "chunk_ms": defaults.chunk_ms,
                "language": session_config.language,
                "allowed_languages": session_config.get_allowed_languages(),
                "min_utterance_ms": session_config.min_utterance_ms,
                "endpointing_silence_ms": session_config.endpointing_silence_ms,
                "vad_threshold": session_config.vad_threshold,
                "transcribe_throttle_ms": session_config.transcribe_throttle_ms,
                "streaming_support": has_streaming,
            },
        }
    )

    try:
        while True:
            data = await websocket.receive_bytes()
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            result = session.process_chunk(audio)
            await websocket.send_json(result)

    except WebSocketDisconnect:
        final = session.force_finalize()
        if final:
            print(f"[{session_id}] Final on disconnect: {final['text'][:50]}...")
    except Exception as e:
        print(f"[{session_id}] Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        active_sessions.discard(session_id)
        print(f"[{session_id}] Disconnected (active sessions: {len(active_sessions)})")


# ============== REST Endpoints ==============
@app.get("/")
def root():
    return {
        "service": "Voice Agent ASR",
        "status": "running",
        "docs": "/docs",
        "websocket": "/ws/transcribe",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": defaults.device,
        "model_repo": defaults.model_repo,
        "streaming_support": has_streaming,
    }


@app.get("/sessions")
def sessions():
    return {
        "active_sessions": len(active_sessions),
        "session_ids": list(active_sessions),
    }


# ============== Run ==============
if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Voice Agent ASR Server")
    print("=" * 60)
    print(f"Model repo: {defaults.model_repo}")
    print(f"Device: {defaults.device}")
    print(f"Default endpointing: {defaults.endpointing_silence_ms}ms")
    print(f"Default allowed languages: {defaults.allowed_languages}")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
