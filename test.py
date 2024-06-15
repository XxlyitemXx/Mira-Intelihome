# Load model directly

import pyaudio
from pythaiasr import ASR

from transformers import AutoProcessor, AutoModelForCTC

processor = AutoProcessor.from_pretrained("vosk_model_th_w2v2_20210816")
model = AutoModelForCTC.from_pretrained("vosk_model_th_w2v2_20210816")

# Initialize ASR
asr = ASR(model='vosk_model_th_w2v2_20210816')  

# PyAudio setup
p = pyaudio.PyAudio()
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Speak now...")

while True:
    data = stream.read(CHUNK)
    text = asr.transcribe_stream(data)

    if text:
        print("You said:", text)

    # (Optional) Add a stopping condition (e.g., press a key, say a keyword)
    # if some_stopping_condition:
    #     break
