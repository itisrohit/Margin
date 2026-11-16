import asyncio
import json
import os
import time
import wave
from pathlib import Path
from uuid import uuid4

import httpx
import websockets

KEY = os.environ["NVIDIA_API_KEY"]
API = os.getenv("MARGIN_API_URL", "http://127.0.0.1:8000")
WS = API.replace("http://", "ws://").replace("https://", "wss://")


async def check():
    with wave.open(os.getenv("MARGIN_BENCHMARK_AUDIO", "/tmp/margin-question.wav"), "rb") as wav:
        audio = wav.readframes(wav.getnframes())
        if (
            wav.getframerate() != 16000
            or wav.getnchannels() != 1
            or wav.getsampwidth() != 2
            or not audio
        ):
            raise ValueError("Use nonempty 16kHz mono 16-bit PCM WAV audio.")
    released = 0.0
    partials = 0
    async with websockets.connect(f"{WS}/speech/stream") as ws:
        await ws.send(
            json.dumps(
                {
                    "apiKey": KEY,
                    "modelId": "cache-aware-parakeet-rnnt-en-US-asr-streaming-sortformer",
                }
            )
        )

        async def send():
            nonlocal released
            for i in range(0, len(audio), 3200):
                await ws.send(audio[i : i + 3200])
                await asyncio.sleep(0.1)
            released = time.perf_counter()
            await ws.send("stop")

        sender = asyncio.create_task(send())
        transcript = ""
        async for message in ws:
            event = json.loads(message)
            if event["type"] == "transcript":
                partials += 1
                transcript = event["text"]
            if event["type"] == "error":
                raise RuntimeError(event["detail"])
            if event["type"] == "done":
                transcript = event["text"] or transcript
                break
        await sender
        final = time.perf_counter()
        print("voice_result", repr(transcript), "partials", partials, flush=True)
    async with httpx.AsyncClient(timeout=60) as c:
        body = {
            "clientId": str(uuid4()),
            "question": transcript,
            "articleUrl": "margin:voice-benchmark",
            "title": "Voice benchmark",
            "paragraphIndex": 0,
            "contextStart": 0,
            "context": [
                "Attention can be deliberately directed toward a goal, "
                "but sudden sights and sounds can also capture it. These influences interact."
            ],
            "scope": "passage",
        }
        first = None
        answer = ""
        metrics = {}
        async with c.stream(
            "POST", f"{API}/companion/ask", headers={"Authorization": f"Bearer {KEY}"}, json=body
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event["type"] == "delta":
                    if first is None:
                        first = time.perf_counter()
                    answer += event["text"]
                if event["type"] == "error":
                    raise RuntimeError(event["detail"])
                if event["type"] == "done":
                    metrics = event["metrics"]
        result = {
            "audio": "macOS synthesized speech, streamed at real-time pace; not a microphone test",
            "transcript": transcript,
            "partialEvents": partials,
            "releaseToFinalTranscriptMs": round((final - released) * 1000, 1),
            "releaseToFirstAnswerMs": round((first - released) * 1000, 1) if first else None,
            "answer": answer,
            "metrics": metrics,
        }
        Path("/tmp/margin-voice-benchmark.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


asyncio.run(check())
