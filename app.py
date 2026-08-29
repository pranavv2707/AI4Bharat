import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse

from asr_client import IndicASRClient
from tts_client import IndicTTSClient

app = FastAPI()

asr = IndicASRClient()
tts_cache = {}

TMP_DIR = "tmp_audio"
os.makedirs(TMP_DIR, exist_ok=True)


def get_tts(language):
    if language not in tts_cache:
        tts_cache[language] = IndicTTSClient(language=language)
    return tts_cache[language]


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), language: str = Form("hi")):
    in_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}.wav")
    with open(in_path, "wb") as f:
        f.write(await audio.read())

    try:
        text = asr.transcribe(in_path, language)
    finally:
        os.remove(in_path)

    return JSONResponse({"text": text})


@app.post("/synthesize")
async def synthesize(text: str = Form(...), language: str = Form("hi")):
    client = get_tts(language)
    out_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}.mp3")
    client.synthesize(text, out_path=out_path)
    return FileResponse(out_path, media_type="audio/mpeg", filename="response.mp3")