import os
import threading

from TTS.api import TTS


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPEAKER_PATH = os.path.join(BASE_DIR, "sample.wav")

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts_lock = threading.Lock()
