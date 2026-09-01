from fastapi import FastAPI, UploadFile, File
from indic_asr_onnx import IndicTranscriber
import shutil
import os

app = FastAPI()

# Global transcriber instance
transcriber = None

@app.on_event("startup")
def load_model():
    global transcriber
    # Uses INT8 quantized ONNX checkpoints (~180MB - 400MB RAM)
    transcriber = IndicTranscriber()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), lang: str = "hi"):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Transcribe using CTC decoding for fast CPU inference
        text = transcriber.transcribe_ctc(temp_path, language=lang)
        return {"text": text}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)