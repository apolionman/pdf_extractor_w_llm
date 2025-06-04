from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query, Request
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Dict, Optional
from openai import OpenAI
from app.services.llm_extract import *
from app.services.graph_query import *
import tempfile, os, httpx, asyncio, subprocess, whisper

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL")
stt = whisper.load_model("large")

router = APIRouter()
client = OpenAI()

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.post("/extract-pdf")
async def extract_pdf(
    files: List[UploadFile] = File(..., description="Upload your PDF files here."),
    prompt: Optional[str] = Form(None, description="Custom extraction prompt in JSON string format.")
):
    summaries = []

    try:
        user_prompt_data = json.loads(prompt) if prompt else None
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid prompt JSON format.")

    for file in files:
        file_ext = os.path.splitext(file.filename)[-1].lower()

        if file_ext != ".pdf":
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file '{file.filename}'. Only .pdf is allowed."
            )

        tmpdir = tempfile.mkdtemp()
        file_path = os.path.join(tmpdir, file.filename)

        with open(file_path, "wb") as out_file:
            shutil.copyfileobj(file.file, out_file)

        try:
            # Modify summarization to accept dynamic prompt
            genetiq_summarized = await summarization(file_path, user_prompt_data)
            summaries.append({
                "filename": file.filename,
                "summary": genetiq_summarized
            })
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return {"summaries": summaries}

@router.post("/generate")
async def generate(request: Request):
    print("Generate endpoint hit")
    payload = await request.json()
    print(f"Connecting to Ollama at: {OLLAMA_URL}")
    print(f"Payload: {payload}")

    async def stream_response():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json=payload) as r:
                    print(f"Ollama responded with status: {r.status_code}")
                    r.raise_for_status()  # raise for HTTP errors (4xx, 5xx)

                    async for chunk in r.aiter_text():
                        yield chunk
        except httpx.RequestError as e:
            print(f"Request error connecting to Ollama: {e}")
            yield '{"error": "Failed to connect to Ollama."}'
        except httpx.HTTPStatusError as e:
            print(f"HTTP error from Ollama: {e}")
            yield f'{{"error": "Ollama returned HTTP {e.response.status_code}"}}'

    return StreamingResponse(stream_response(), media_type="application/json")

@router.post("/query-graph")
async def query_kgs(kg_conn: str, query=str):
    response = await query_graph(kg_conn, query)
    return JSONResponse(content=response, media_type="application/json")

ALLOWED_TYPES = {
    "audio/mpeg", "audio/webm", "video/mp4",
    "audio/mp4", "video/webm", "audio/x-m4a", 
    "audio/m4a",  "audio/ogg"
}

MIME_EXTENSION_MAP = {
    "audio/mpeg": ".mp3",
    "audio/webm": ".webm",
    "video/webm": ".webm",
    "video/mp4": ".mp4",
    "audio/mp4": ".mp4",
    "audio/x-m4a": ".m4a",
    "audio/m4a": ".m4a",
    "audio/ogg": ".ogg"
}

@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
):
    """
    Transcribe uploaded audio/video using Whisper locally.
    Supports: audio/mpeg, audio/webm, video/mp4, audio/mp4, video/webm, audio/x-m4a
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Must be one of {ALLOWED_TYPES}",
        )

    suffix = MIME_EXTENSION_MAP.get(file.content_type, ".tmp")
    input_tmp = None
    wav_path = None

    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as input_tmp:
            input_tmp.write(await file.read())
            input_path = input_tmp.name

        # Convert to mono WAV at 24kHz for Whisper
        wav_path = input_path.replace(suffix, ".wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "24000", "-ac", "1", wav_path
        ], check=True)

        # Transcribe using whisper
        result = stt.transcribe(wav_path, fp16=False)
        text = result["text"].strip()
        return JSONResponse(content={"transcription": text})

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in [input_tmp.name if input_tmp else None, wav_path]:
            if path and os.path.exists(path):
                os.remove(path)