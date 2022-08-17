import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"] # corpus_path: "../dataset/LJSpeech-1.1/"
    out_dir = config["path"]["raw_path"] # raw_path: "./raw_data/LJSpeech/"
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"] # sampling_rate: 22050
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"] # sampling_rate: 22050
    cleaners = config["preprocessing"]["text"]["text_cleaners"] # text_cleaners: ["english_cleaners"]
    speaker = "LJSpeech"
    with open(os.path.join(in_dir, "metadata.csv"), encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split("|")
            base_name = parts[0]
            text = parts[2]
            text = _clean_text(text, cleaners)

            wav_path = os.path.join(in_dir, "wavs", "{}.wav".format(base_name))
            if os.path.exists(wav_path):
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wav, _ = librosa.load(wav_path, sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(
                    os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                with open(
                    os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                    "w",
                ) as f1:
                    f1.write(text)