from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query, Request
from fastapi.responses import StreamingResponse
from typing import List
from app.services.llm_extract import *
import tempfile, os, httpx, asyncio

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL")

router = APIRouter()

@router.post("/extract-pdf")
async def extract_pdf(
    files: List[UploadFile] = File(..., description="Upload your PDF files here.")
):
    summaries = []

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
            genetiq_summarized = await summarization(file_path)
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

    async def stream_response():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", OLLAMA_URL, json=payload) as r:
                async for chunk in r.aiter_text():
                    yield chunk

    return StreamingResponse(stream_response(), media_type="application/json")
