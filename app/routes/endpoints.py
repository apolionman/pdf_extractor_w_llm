from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query
from typing import List
import tempfile, os

router = APIRouter()

@router.post("/extract-pdf")
async def extract_pdf(
        files: List[UploadFile] = File(
        ...,
        description="Upload your pdf file here."
    ),
        ):
    
    tmpdir = tempfile.mkdtemp()
    print(tmpdir)

    for file in files:
        file_ext = os.path.splitext(file.filename)[-1].lower()
        if file.content_type != ".pdf":
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file '{file.filename}'. Only .pdf is allowed."
            )
        print("[DEBUG] checking path => ", file_ext)

    # try:
    #     output_dir = os.path.join(tmpdir, "output")
    #     os.makedirs(output_dir, exist_ok=True)
    #     genetiq_summarized = await summarization(pdf_local_path)        
        
    # except Exception as e:
    #     print(f"failed: {e}")
    # finally:
    #     shutil.rmtree(tmpdir, ignore_errors=True)