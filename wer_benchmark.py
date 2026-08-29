import csv
import jiwer
from asr_client import IndicASRClient
from audio_augment import simulate_telephony

TEST_SET_CSV = "test_set.csv"
NOISE_FILE = "noise_samples/ambient.wav"


def run_benchmark():
    asr = IndicASRClient()
    clean_refs, clean_hyps = [], []
    degraded_refs, degraded_hyps = [], []

    with open(TEST_SET_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = row["audio_path"]
            reference = row["reference_text"]
            language = row["language"]

            clean_hyp = asr.transcribe(audio_path, language)
            clean_refs.append(reference)
            clean_hyps.append(clean_hyp)

            degraded_path = audio_path.replace(".wav", "_degraded.wav")
            simulate_telephony(audio_path, degraded_path, noise_path=NOISE_FILE)
            degraded_hyp = asr.transcribe(degraded_path, language)
            degraded_refs.append(reference)
            degraded_hyps.append(degraded_hyp)

    clean_wer = jiwer.wer(clean_refs, clean_hyps)
    degraded_wer = jiwer.wer(degraded_refs, degraded_hyps)

    print(f"Clean audio WER:    {clean_wer:.2%}")
    print(f"Degraded audio WER: {degraded_wer:.2%}")
    print(f"WER increase:       {degraded_wer - clean_wer:.2%}")


if __name__ == "__main__":
    run_benchmark()
