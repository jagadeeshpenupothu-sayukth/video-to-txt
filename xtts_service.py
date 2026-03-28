import os
import threading

import torch

# Force full checkpoint load for trusted XTTS models on PyTorch 2.6+.
original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)


torch.load = patched_torch_load

from TTS.api import TTS


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPEAKER_PATH = os.path.join(BASE_DIR, "sample.wav")

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
print("XTTS loaded using patched torch.load (weights_only=False)")
tts_lock = threading.Lock()
