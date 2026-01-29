# test_client.py
# Python client to test voice agent ASR server

import asyncio
import websockets
import numpy as np
import json
import argparse
from urllib.parse import urlencode

try:
    import sounddevice as sd

    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    print("Warning: sounddevice not installed. Install with: pip install sounddevice")


def build_ws_url(base_url: str, **params) -> str:
    """Build WebSocket URL with query parameters"""
    # Filter out None values
    params = {k: v for k, v in params.items() if v is not None}
    if params:
        return f"{base_url}?{urlencode(params)}"
    return base_url


async def voice_client(
    server_url: str,
    language: str = None,
    allowed_languages: str = None,
    min_utterance_ms: int = None,
    endpointing_silence_ms: int = None,
    vad_threshold: float = None,
    lookback_ms: int = None,
):
    """Connect to voice agent and stream microphone audio"""

    if not HAS_SOUNDDEVICE:
        print("Error: sounddevice is required for microphone input")
        print("Install with: pip install sounddevice")
        return

    # Build URL with parameters
    url = build_ws_url(
        server_url,
        language=language,
        allowed_languages=allowed_languages,
        min_utterance_ms=min_utterance_ms,
        endpointing_silence_ms=endpointing_silence_ms,
        vad_threshold=vad_threshold,
        lookback_ms=lookback_ms,
    )

    print(f"Connecting to {url}...")

    try:
        async with websockets.connect(url) as ws:
            config_msg = json.loads(await ws.recv())

            if config_msg["type"] != "ready":
                print(f"Unexpected message: {config_msg}")
                return

            server_config = config_msg["config"]
            session_id = config_msg.get("session_id", "unknown")

            print(f"Connected! Session: {session_id}")
            print(f"  Sample rate: {server_config['sample_rate']} Hz")
            print(f"  Chunk size: {server_config['chunk_ms']} ms")
            print(f"  Allowed languages: {server_config.get('allowed_languages')}")
            print(f"  Endpointing: {server_config.get('endpointing_silence_ms')} ms")
            print(f"  VAD threshold: {server_config.get('vad_threshold')}")
            print(f"  Lookback: {server_config.get('lookback_ms')} ms")

            sample_rate = server_config["sample_rate"]
            chunk_samples = int(sample_rate * server_config["chunk_ms"] / 1000)

            audio_queue = asyncio.Queue()

            def audio_callback(indata, frames, time, status):
                if status:
                    print(f"Audio status: {status}")
                audio_queue.put_nowait(indata.copy())

            stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype=np.int16,
                blocksize=chunk_samples,
                callback=audio_callback,
            )

            async def send_audio():
                with stream:
                    print("\n" + "=" * 60)
                    print("🎙️  LISTENING - Speak now! (Ctrl+C to stop)")
                    print("=" * 60 + "\n")

                    while True:
                        audio = await audio_queue.get()
                        await ws.send(audio.tobytes())

            async def receive_results():
                while True:
                    msg = json.loads(await ws.recv())
                    msg_type = msg.get("type", "")

                    if msg_type == "partial":
                        text = msg.get("text", "")
                        print(f"\r🎤 {text:<70}", end="", flush=True)

                    elif msg_type == "speaking":
                        text = msg.get("text", "...")
                        audio_ms = msg.get("audio_ms", 0)
                        print(f"\r🎤 {text:<60} [{audio_ms:.0f}ms]", end="", flush=True)

                    elif msg_type == "final":
                        text = msg.get("text", "")
                        latency = msg.get("latency_ms", 0)
                        audio_ms = msg.get("audio_ms", 0)

                        if text:
                            print(f"\r✅ {text:<70}")
                            print(
                                f"   [Latency: {latency:.0f}ms | Audio: {audio_ms:.0f}ms]"
                            )
                            print("-" * 60)

                    elif msg_type == "silence":
                        pass

            await asyncio.gather(send_audio(), receive_results())

    except websockets.exceptions.ConnectionClosed as e:
        print(f"\nConnection closed: {e}")
    except ConnectionRefusedError:
        print(f"\nError: Could not connect to {server_url}")
        print("Make sure the server is running: python voice_agent_rnnt.py")
    except Exception as e:
        print(f"\nError: {e}")


async def test_file(server_url: str, audio_file: str, language: str = None):
    """Test with an audio file instead of microphone"""
    import wave

    url = build_ws_url(server_url, language=language)

    print(f"Testing with file: {audio_file}")
    print(f"Connecting to {url}...")

    with wave.open(audio_file, "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)

    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Channels: {channels}")
    print(f"  Duration: {n_frames / sample_rate:.2f}s")

    audio = np.frombuffer(audio_data, dtype=np.int16)
    if channels == 2:
        audio = audio[::2]

    async with websockets.connect(url) as ws:
        config_msg = json.loads(await ws.recv())
        server_config = config_msg["config"]

        chunk_samples = int(
            server_config["sample_rate"] * server_config["chunk_ms"] / 1000
        )

        if sample_rate != server_config["sample_rate"]:
            print(
                f"Warning: Audio sample rate ({sample_rate}) doesn't match server ({server_config['sample_rate']})"
            )

        print("\nStreaming audio to server...")

        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

            await ws.send(chunk.tobytes())

            msg = json.loads(await ws.recv())
            msg_type = msg.get("type", "")

            if msg_type == "partial":
                print(f"\r  Partial: {msg.get('text', ''):<60}", end="", flush=True)
            elif msg_type == "final":
                print(f"\r  Final: {msg.get('text', ''):<60}")

            await asyncio.sleep(server_config["chunk_ms"] / 1000)

        print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Voice Agent ASR Test Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python test_client.py

  # Custom server URL
  python test_client.py -s ws://my-server:8000/ws/transcribe

  # Restrict to specific languages
  python test_client.py --allowed-languages "en,es"

  # Faster endpointing (more responsive)
  python test_client.py --endpointing 300

  # Test with audio file
  python test_client.py -f test.wav
        """,
    )
    parser.add_argument(
        "--server",
        "-s",
        default="ws://20.85.213.137:8080/ws/transcribe",
        help="WebSocket server URL",
    )
    parser.add_argument(
        "--language",
        "-l",
        default=None,
        help="Force specific language code (en, hi, es, etc.)",
    )
    parser.add_argument(
        "--allowed-languages",
        "-a",
        default=None,
        help="Comma-separated allowed languages (e.g., 'en,hi')",
    )
    parser.add_argument(
        "--min-utterance",
        type=int,
        default=None,
        help="Min audio (ms) before transcription (default: 250)",
    )
    parser.add_argument(
        "--endpointing",
        type=int,
        default=None,
        help="Silence (ms) to trigger endpoint (default: 400)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=None,
        help="VAD speech probability threshold 0.1-0.9 (default: 0.5)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="Audio lookback buffer in ms to capture speech onset (default: 200)",
    )
    parser.add_argument(
        "--file",
        "-f",
        default=None,
        help="Audio file to test (WAV format, 16kHz mono recommended)",
    )

    args = parser.parse_args()

    try:
        if args.file:
            asyncio.run(test_file(args.server, args.file, args.language))
        else:
            asyncio.run(
                voice_client(
                    args.server,
                    language=args.language,
                    allowed_languages=args.allowed_languages,
                    min_utterance_ms=args.min_utterance,
                    endpointing_silence_ms=args.endpointing,
                    vad_threshold=args.vad_threshold,
                    lookback_ms=args.lookback,
                )
            )
    except KeyboardInterrupt:
        print("\n\nStopped by user.")


if __name__ == "__main__":
    main()
