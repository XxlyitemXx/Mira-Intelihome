from transformers import pipeline
import pyaudio  # For audio input
import time     # For optional timing 
import numpy as np
from transformers.pipelines.audio_utils import ffmpeg_read
import subprocess

import wave
# Audio Configuration (Optimized for Whisper)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

with wave.open("debug_audio.wav", "wb") as wave_file:
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(buffer)

def ffmpeg_read(inputs, sampling_rate: int = 16000):
    """
    Reads an audio file or URL using FFmpeg and returns it as a NumPy array.
    """
    ffmpeg_command = ["ffmpeg", "-i", inputs, "-ar", str(sampling_rate), "-ac", "1", "-f", "f32le", "-"]
    try:
        with subprocess.Popen(
            ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as process:
            output_stream = process.stdout
            output_bytes, _ = output_stream.read(), process.stderr.read()
            if process.wait() != 0:
                raise ValueError(f"FFmpeg error: {_.decode('utf-8')}")
            return np.frombuffer(output_bytes, np.float32)

    except Exception as e:
        raise ValueError(f"Error reading audio file: {e}")

# Hugging Face Pipeline (Using a pre-trained Whisper model)
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")


# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

print("** Speech-to-Text Transcription is Active **")
print("Press Ctrl+C to stop.")

transcription = ""
prev_text = ""
buffer = b""  # Audio buffer for better Whisper performance

while True:
    try:
        # Record Audio Chunk and accumulate into the buffer. 
        data = stream.read(CHUNK)
        buffer += data 
        # Process Audio in buffer when a certain size is reached
        if len(buffer) >= CHUNK * 4:
            # Transcribe with Whisper (Hugging Face Pipeline)
            start_time = time.time()  # Optional timing
            text = transcriber(buffer)["text"]
            end_time = time.time()  # Optional timing
            transcription_time = end_time - start_time  # Optional timing
            buffer = b""
            # Display Output
            if text != prev_text and text != "":
                transcription += text
                print(text, end="", flush=True) 
                prev_text = text 

    except KeyboardInterrupt:
        print("\nTranscription stopped.")
        break

stream.stop_stream()
stream.close()
audio.terminate()

print("\nFinal Transcription:")
print(transcription)
