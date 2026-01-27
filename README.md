# Voice Agent ASR

Real-time streaming ASR server for voice agents using NeMo RNNT multilingual model.

## Features

- Real-time WebSocket streaming transcription
- Configurable per-session parameters
- Language filtering (restrict output to specific languages)
- Voice Activity Detection (VAD) with Silero
- Automatic endpointing (utterance detection)
- Session tracking and monitoring
- Docker deployment ready

## Quick Start

### Docker Deployment (Recommended)

```bash
# Build and run
docker compose up -d

# Check logs
docker compose logs -f

# Check health
curl http://localhost:8000/health

# Check active sessions
curl http://localhost:8000/sessions
```

### Manual Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python voice_agent_rnnt.py
```

## API Reference

### WebSocket: `/ws/transcribe`

Connect with optional query parameters to customize session behavior:

```
ws://localhost:8000/ws/transcribe?allowed_languages=en,hi&endpointing_silence_ms=300
```

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language` | string | null | Force specific language |
| `allowed_languages` | string | "en,hi" | Comma-separated allowed languages |
| `min_utterance_ms` | int | 250 | Min audio before transcription (100-2000) |
| `endpointing_silence_ms` | int | 400 | Silence to trigger endpoint (100-2000) |
| `vad_threshold` | float | 0.5 | VAD speech probability threshold (0.1-0.9) |
| `transcribe_throttle_ms` | int | 150 | Min time between partial transcriptions (50-1000) |

#### Input

Send raw PCM audio bytes: **16-bit signed integer, 16kHz, mono**

#### Output Messages

```json
// Ready (on connect)
{
  "type": "ready",
  "session_id": "session_1_1706000000",
  "config": { ... }
}

// Partial result (while speaking)
{
  "type": "partial",
  "text": "hello world",
  "is_final": false,
  "latency_ms": 85,
  "audio_ms": 1200
}

// Final result (after silence)
{
  "type": "final",
  "text": "hello world how are you",
  "is_final": true,
  "latency_ms": 120,
  "audio_ms": 2400
}

// Silence (no speech)
{
  "type": "silence",
  "speech_prob": 0.1
}
```

### REST Endpoints

#### `GET /health`
```json
{
  "status": "ok",
  "device": "cuda",
  "model_repo": "milind-plivo/parakeet-multilingual-base",
  "streaming_support": false
}
```

#### `GET /sessions`
```json
{
  "active_sessions": 3,
  "session_ids": ["session_1_1706000000", "session_2_1706000001", ...]
}
```

## Test Client

```bash
# Basic usage
python test_client.py

# Custom server
python test_client.py -s ws://my-server:8000/ws/transcribe

# Override settings
python test_client.py --allowed-languages "en,es" --endpointing 300

# Test with audio file
python test_client.py -f test.wav
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_REPO` | milind-plivo/parakeet-multilingual-base | HuggingFace model repository |
| `MODEL_FILENAME` | parakeet-rnnt-1.1b-multilingual.nemo | Model filename in repo |
| `MODEL_CACHE_DIR` | /app/models | Directory to cache downloaded model |
| `HF_TOKEN` | - | HuggingFace token (if repo is private) |

### Default Session Settings

Edit `voice_agent_rnnt.py` to change defaults:

```python
@dataclass
class DefaultConfig:
    # Endpointing
    endpointing_silence_ms: int = 400

    # VAD
    vad_threshold: float = 0.5

    # Language filtering
    allowed_languages: List[str] = field(default_factory=lambda: ["en", "hi"])
```

## Language Filtering

The server filters transcriptions to only allow specific language scripts:

| Code | Language | Script |
|------|----------|--------|
| `en` | English | Latin |
| `hi` | Hindi | Devanagari + Latin |
| `es` | Spanish | Latin |
| `ru` | Russian | Cyrillic |
| `zh` | Chinese | CJK |
| `ar` | Arabic | Arabic |

Pass `allowed_languages=none` to disable filtering.

## Architecture

```
Client (Browser/Python)
    │
    │ WebSocket (16-bit PCM @ 16kHz)
    ▼
┌──────────────────────────────────────┐
│  FastAPI Server                      │
│  ├── Session Management              │
│  ├── Silero VAD (filter silence)     │
│  ├── Audio Buffer (accumulate)       │
│  ├── NeMo RNNT Model (transcribe)    │
│  ├── Language Filter                 │
│  └── Endpointer (detect utterance)   │
└──────────────────────────────────────┘
    │
    │ JSON (transcription results)
    ▼
Client
```

## Docker Build

```bash
# Build image
docker build -t voice-agent-asr .

# Run with GPU
docker run --gpus all -p 8000:8000 voice-agent-asr

# Run with custom model
docker run --gpus all -p 8000:8000 \
  -e MODEL_REPO=your-org/your-model \
  -e MODEL_FILENAME=model.nemo \
  voice-agent-asr
```

## Monitoring

```bash
# Check active sessions
curl http://localhost:8000/sessions

# Health check
curl http://localhost:8000/health

# View logs
docker compose logs -f voice-agent-asr
```

## Troubleshooting

### CUDA out of memory
- Use a smaller model or reduce concurrent sessions

### High latency
- Reduce `transcribe_throttle_ms`
- Use greedy decoding (default)
- Check GPU utilization with `nvidia-smi`

### Wrong language output
- Adjust `allowed_languages` parameter
- Lower `vad_threshold` if speech is being missed

### Model download fails
- Check `HF_TOKEN` if repository is private
- Verify `MODEL_REPO` and `MODEL_FILENAME` are correct
