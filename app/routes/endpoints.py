from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query
from typing import List
from app.services.llm_extract import *
import tempfile, os

router = APIRouter()

@router.post("/extract-pdf")
async def extract_pdf(
    files: List[UploadFile] = File(..., description="Upload your pdf file here.")
):
    for file in files:
        tmpdir = tempfile.mkdtemp()
        file_ext = os.path.splitext(file.filename)[-1].lower()
        print("[DEBUG] This is the content =>",file.content_type)
        if file_ext == ".pdf":
            file_path = os.path.join(tmpdir, file.filename)
            genetiq_summarized = await summarization(file_path)
            print(genetiq_summarized) 
            
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file '{file.filename}'. Only PDF files are allowed."
            )   

        if file_ext != ".pdf":
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file '{file.filename}'. Only .pdf is allowed."
            )
        
        shutil.rmtree(tmpdir, ignore_errors=True)
