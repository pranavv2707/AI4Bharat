# Speech module — ASR/TTS + robustness benchmark (Hugging Face / AI4Bharat)

No Bhashini account needed. Uses AI4Bharat's open models directly from
Hugging Face: IndicConformer for ASR, Indic Parler-TTS for TTS.

## 1. Log in to Hugging Face (needed once, for the gated ASR model)

1. Create a free account at https://huggingface.co if you don't have one.
2. Go to https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
   and click "Agree and access repository" (just requires accepting terms,
   no approval wait).
3. Run `huggingface-cli login` and paste a token from
   https://huggingface.co/settings/tokens (read access is enough).

## 2. Install dependencies

```
pip install -r requirements.txt --break-system-packages
```

First run downloads both models (a few GB total) — do this ahead of any
demo/presentation, not live.

## 3. Run the end-to-end demo

Put a short `sample_input.wav` (any sample rate, mono or stereo — it's
resampled automatically) in this folder, then:

```
python demo.py
```

Transcribes the audio, prints the recognized text, and generates a spoken
response saved to `assistant_response.wav`.

## 4. Run the WER robustness benchmark

This is the evidence for your feasibility slide's technical risk claim —
it measures how much accuracy drops under simulated phone-call conditions.

1. Create `test_set.csv` with columns: `audio_path,reference_text,language`
   — a handful of short clips with correct transcriptions is enough for a
   demo-scale result. Language codes: hi, ta, te, kn, bn, mr, etc.
2. Add a short ambient noise clip at `noise_samples/ambient.wav` (fan,
   traffic, crowd — any real background noise).
3. Run:

```
python wer_benchmark.py
```

Prints clean-audio WER vs. degraded-audio WER — the real gap you're
addressing, not a claim it's already solved.

## Files

- `asr_client.py` — IndicConformer ASR wrapper (audio in, text out)
- `tts_client.py` — Indic Parler-TTS wrapper (text in, audio out, with a
  warm/conversational voice description you can edit)
- `audio_augment.py` — simulates telephony conditions (8kHz, bandpass
  filtered to 300-3400Hz, added background noise at a target SNR)
- `wer_benchmark.py` — compares ASR accuracy on clean vs. degraded audio
- `demo.py` — minimal end-to-end ASR to TTS flow
- `bhashini_client.py` — kept as an alternate path if Bhashini access
  gets sorted out later; not used by demo.py or wer_benchmark.py

## Notes

- No GPU required, but a demo run is much faster with one (Colab's free
  GPU tier works fine if your machine doesn't have one).
- `IndicASRClient.transcribe(path, language, decoding)` — `decoding` can
  be `"ctc"` (faster) or `"rnnt"` (usually more accurate).
- Edit `DEFAULT_DESCRIPTION` in `tts_client.py` to change the voice tone,
  gender, accent, or pace — it's a plain text description, no retraining
  needed.
