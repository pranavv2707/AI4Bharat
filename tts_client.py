import soundfile as sf
from transformers import pipeline

LANGUAGE_MODEL_IDS = {
    "hi": "facebook/mms-tts-hin",
    "ta": "facebook/mms-tts-tam",
    "te": "facebook/mms-tts-tel",
    "kn": "facebook/mms-tts-kan",
    "bn": "facebook/mms-tts-ben",
    "mr": "facebook/mms-tts-mar",
}


class IndicTTSClient:
    def __init__(self, language="hi", device=None):
        if language not in LANGUAGE_MODEL_IDS:
            raise ValueError(f"No MMS-TTS model mapped for language '{language}'")

        self.pipe = pipeline("text-to-speech", model=LANGUAGE_MODEL_IDS[language])

    def synthesize(self, text, out_path="output.wav"):
        result = self.pipe(text)
        print("PIPELINE OUTPUT KEYS:", result.keys())
        print("AUDIO SHAPE:", result["audio"].shape)
        sf.write(out_path, result["audio"].squeeze(), result["sampling_rate"])
        return out_path