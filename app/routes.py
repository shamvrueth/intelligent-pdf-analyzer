from fastapi import APIRouter, UploadFile, File, Form
from typing import List
import tempfile
import shutil
from pathlib import Path
from model import model_call

router = APIRouter()

@router.post("/analyze")
async def analyze_documents(
    persona: str = Form(...),
    job_to_be_done: str = Form(...),
    files: List[UploadFile] = File(...)):

    temp_dir = tempfile.mkdtemp() # create temporary folder on disk
    pdf_path = []

    try:
        for f in files:
            file_path = Path(temp_dir) / f.filename
            with open(file_path, "wb") as buffer: 
                shutil.copyfileobj(f.file, buffer)
            pdf_path.append(str(file_path))
        output = model_call(pdf_path, persona, job_to_be_done)
        return output
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    

    