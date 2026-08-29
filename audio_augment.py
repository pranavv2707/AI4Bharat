import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter


def bandpass_filter(audio, sr, low=300, high=3400, order=4):
    nyq = 0.5 * sr
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return lfilter(b, a, audio)


def add_noise(audio, noise, snr_db):
    audio_power = np.mean(audio ** 2)
    noise = np.resize(noise, audio.shape)
    noise_power = np.mean(noise ** 2)
    target_noise_power = audio_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / (noise_power + 1e-10))
    return audio + noise * scale


def simulate_telephony(in_path, out_path, target_sr=8000, snr_db=10, noise_path=None):
    audio, sr = sf.read(in_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        num_samples = int(len(audio) * target_sr / sr)
        audio = np.interp(
            np.linspace(0, len(audio), num_samples, endpoint=False),
            np.arange(len(audio)),
            audio,
        )
        sr = target_sr

    audio = bandpass_filter(audio, sr)

    if noise_path:
        noise, noise_sr = sf.read(noise_path)
        if noise.ndim > 1:
            noise = noise.mean(axis=1)
        if noise_sr != sr:
            num_samples = int(len(noise) * sr / noise_sr)
            noise = np.interp(
                np.linspace(0, len(noise), num_samples, endpoint=False),
                np.arange(len(noise)),
                noise,
            )
        audio = add_noise(audio, noise, snr_db)

    audio = np.clip(audio, -1.0, 1.0)
    sf.write(out_path, audio, sr, subtype="PCM_16")
    return out_path
