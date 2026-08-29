from asr_client import IndicASRClient
from tts_client import IndicTTSClient

LANGUAGE = "hi"


def run_demo(audio_path):
    asr = IndicASRClient()
    text = asr.transcribe(audio_path, LANGUAGE)
    print(f"Beneficiary said: {text}")

    response_text = "धन्यवाद। आपकी जानकारी के आधार पर, सिलाई प्रशिक्षण कार्यक्रम की सिफारिश की जाती है।"

    tts = IndicTTSClient()
    out_path = tts.synthesize(response_text, out_path="assistant_response.wav")
    print(f"Assistant response saved to: {out_path}")


if __name__ == "__main__":
    run_demo("sample_input.wav")
