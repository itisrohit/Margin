import asyncio
import io
import queue
import wave
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any, TypedDict

import grpc
import riva.client
from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from articles import router as articles_router
from chats import router as chats_router
from companion import close_client
from companion import router as companion_router
from memory import router as memory_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_client()


app = FastAPI(title="Margin API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(memory_router)
app.include_router(articles_router)
app.include_router(companion_router)
app.include_router(chats_router)


class SpeechModel(TypedDict):
    function_id: str
    streaming: bool


SPEECH_MODELS: dict[str, SpeechModel] = {
    "cache-aware-parakeet-rnnt-en-US-asr-streaming-sortformer": {
        "function_id": "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa",
        "streaming": True,
    },
    "parakeet-rnnt-1.1b-unified-ml-cs-universal-multi-asr-streaming-silero-vad-sortformer": {
        "function_id": "71203149-d3b7-4460-8231-1be2543a1fca",
        "streaming": True,
    },
    "parakeet-rnnt-1.1b-unified-ml-cs-universal-multi-asr-offline-silero-vad-sortformer": {
        "function_id": "71203149-d3b7-4460-8231-1be2543a1fca",
        "streaming": False,
    },
    "parakeet-tdt-0.6b-en-US-asr-offline": {
        "function_id": "d3fe9151-442b-4204-a70d-5fcc597fd610",
        "streaming": False,
    },
    "canary-1b-flash-multi-asr-offline": {
        "function_id": "b0e8b4a5-217c-40b7-9b96-17d84e666317",
        "streaming": False,
    },
}
MAX_AUDIO_BYTES = 6_000_000


def serialize_result(result: Any) -> dict[str, object]:
    alternatives = list(
        dict.fromkeys(
            alternative.transcript.strip()
            for alternative in result.alternatives
            if alternative.transcript.strip()
        )
    )
    top_alternative = result.alternatives[0]
    return {
        "text": top_alternative.transcript.strip(),
        "alternatives": alternatives,
        "words": [
            {
                "text": word.word,
                "start": word.start_time,
                "end": word.end_time,
                "confidence": word.confidence,
            }
            for word in top_alternative.words
        ],
    }


def transcribe_pcm(
    audio: bytes, api_key: str, model_id: str, correction_mode: bool = False
) -> dict[str, object]:
    model = SPEECH_MODELS[model_id]
    auth = riva.client.Auth(
        use_ssl=True,
        uri="grpc.nvcf.nvidia.com:443",
        metadata_args=[
            ["function-id", model["function_id"]],
            ["authorization", f"Bearer {api_key}"],
        ],
    )
    service = riva.client.ASRService(auth)
    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=16_000,
        audio_channel_count=1,
        language_code="en-US",
        model=model_id,
        max_alternatives=3,
        enable_automatic_punctuation=not correction_mode,
        enable_word_time_offsets=True,
    )

    try:
        if model["streaming"]:
            streaming_config = riva.client.StreamingRecognitionConfig(
                config=config,
                interim_results=False,
            )
            audio_chunks: Iterator[bytes] = (
                audio[start : start + 3_200] for start in range(0, len(audio), 3_200)
            )
            responses = service.streaming_response_generator(
                audio_chunks=audio_chunks,
                streaming_config=streaming_config,
            )
            results = [
                result
                for response in responses
                for result in getattr(response, "results", [])
                if result.is_final and result.alternatives
            ]
        else:
            response = service.offline_recognize(audio, config)
            results = [result for result in getattr(response, "results", []) if result.alternatives]
        segments = [serialize_result(result) for result in results]
        transcript = " ".join(str(segment["text"]) for segment in segments).strip()
        return {"text": transcript, "segments": segments}
    finally:
        auth.channel.close()


def stream_transcription(
    audio_queue: queue.Queue[bytes | None],
    api_key: str,
    model_id: str,
    loop: asyncio.AbstractEventLoop,
    websocket: WebSocket,
    correction_mode: bool,
) -> None:
    """Forward live PCM to Riva and stream cumulative transcript updates to the browser."""
    model = SPEECH_MODELS[model_id]
    auth = riva.client.Auth(
        use_ssl=True,
        uri="grpc.nvcf.nvidia.com:443",
        metadata_args=[
            ["function-id", model["function_id"]],
            ["authorization", f"Bearer {api_key}"],
        ],
    )
    service = riva.client.ASRService(auth)
    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=16_000,
        audio_channel_count=1,
        language_code="en-US",
        model=model_id,
        max_alternatives=3,
        enable_automatic_punctuation=not correction_mode,
        enable_word_time_offsets=True,
    )
    streaming_config = riva.client.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    def audio_chunks() -> Iterator[bytes]:
        while (chunk := audio_queue.get()) is not None:
            yield chunk

    final_segments: list[dict[str, object]] = []
    try:
        responses = service.streaming_response_generator(
            audio_chunks=audio_chunks(),
            streaming_config=streaming_config,
        )
        for response in responses:
            interim = ""
            for result in response.results:
                if not result.alternatives:
                    continue
                text = result.alternatives[0].transcript.strip()
                if result.is_final:
                    if text:
                        final_segments.append(serialize_result(result))
                else:
                    interim = text
            final_parts = [str(segment["text"]) for segment in final_segments]
            cumulative = " ".join(final_parts + ([interim] if interim else [])).strip()
            if cumulative:
                asyncio.run_coroutine_threadsafe(
                    websocket.send_json({"type": "transcript", "text": cumulative}),
                    loop,
                ).result()
        final_text = " ".join(str(segment["text"]) for segment in final_segments).strip()
        asyncio.run_coroutine_threadsafe(
            websocket.send_json({"type": "done", "text": final_text, "segments": final_segments}),
            loop,
        ).result()
    except grpc.RpcError as error:
        detail = (
            "NVIDIA denied access to this model. Check the API key and model access."
            if getattr(error, "code", lambda: None)() == grpc.StatusCode.PERMISSION_DENIED
            else "NVIDIA transcription failed."
        )
        asyncio.run_coroutine_threadsafe(
            websocket.send_json({"type": "error", "detail": detail}), loop
        ).result()
    finally:
        auth.channel.close()


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/speech/transcribe", tags=["speech"])
async def transcribe_speech(
    request: Request,
    authorization: str | None = Header(default=None),
    x_speech_model: str | None = Header(default=None),
    x_speech_correction: bool = Header(default=False),
) -> dict[str, object]:
    api_key = authorization.removeprefix("Bearer ").strip() if authorization else ""
    if not api_key:
        raise HTTPException(status_code=401, detail="Add an NVIDIA NIM API key in Settings.")
    if x_speech_model not in SPEECH_MODELS:
        raise HTTPException(status_code=422, detail="Choose a supported speech model in Settings.")

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Recording is too long.")
    body = await request.body()
    if len(body) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Recording is too long.")

    try:
        with wave.open(io.BytesIO(body), "rb") as recording:
            if (
                recording.getnchannels() != 1
                or recording.getsampwidth() != 2
                or recording.getframerate() != 16_000
            ):
                raise HTTPException(
                    status_code=422,
                    detail="Audio must be mono 16 kHz WAV.",
                )
            audio = recording.readframes(recording.getnframes())
    except (wave.Error, EOFError) as error:
        raise HTTPException(
            status_code=422, detail="Could not read the microphone recording."
        ) from error

    if not audio:
        raise HTTPException(status_code=422, detail="The recording was empty.")

    try:
        transcription = await run_in_threadpool(
            transcribe_pcm, audio, api_key, x_speech_model, x_speech_correction
        )
    except grpc.RpcError as error:
        if getattr(error, "code", lambda: None)() == grpc.StatusCode.PERMISSION_DENIED:
            raise HTTPException(
                status_code=502,
                detail="NVIDIA denied access to this model. Check the API key and model access.",
            ) from error
        raise HTTPException(status_code=502, detail="NVIDIA transcription failed.") from error

    return {**transcription, "model_id": x_speech_model}


@app.websocket("/speech/stream")
async def stream_speech(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        credentials = await websocket.receive_json()
        api_key = credentials.get("apiKey", "").strip()
        model_id = credentials.get("modelId", "")
        correction_mode = credentials.get("correctionMode") is True
        if not api_key or model_id not in SPEECH_MODELS or not SPEECH_MODELS[model_id]["streaming"]:
            await websocket.send_json(
                {
                    "type": "error",
                    "detail": "Choose a supported streaming model in Settings.",
                }
            )
            await websocket.close(code=1008)
            return

        audio_queue: queue.Queue[bytes | None] = queue.Queue()
        loop = asyncio.get_running_loop()
        worker = asyncio.create_task(
            run_in_threadpool(
                stream_transcription,
                audio_queue,
                api_key,
                model_id,
                loop,
                websocket,
                correction_mode,
            )
        )
        try:
            while True:
                message = await websocket.receive()
                if message.get("bytes") is not None:
                    chunk = message["bytes"]
                    if len(chunk) > 64_000:
                        await websocket.send_json(
                            {"type": "error", "detail": "Audio chunk is too large."}
                        )
                        break
                    audio_queue.put(chunk)
                elif message.get("text") == "stop":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            audio_queue.put(None)
        await worker
    except WebSocketDisconnect:
        return
