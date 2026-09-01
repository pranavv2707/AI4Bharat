import os
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"


class IndicASRClient:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Read the Hugging Face Token set in Render's Environment Variables
        hf_token = os.getenv("HF_TOKEN")
        
        self.model = AutoModel.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True,
            token=hf_token
        ).to(self.device)
        self.model.eval()

    def transcribe(self, audio_path, language="hi", decoding="ctc"):
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        target_sr = 16000
        if sr != target_sr:
            num_samples = int(len(audio) * target_sr / sr)
            audio = np.interp(
                np.linspace(0, len(audio), num_samples, endpoint=False),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)

        wav = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        with torch.no_grad():
            transcription = self.model(wav, language, decoding)
        return transcription