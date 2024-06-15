import tensorflow as tf
import numpy as np
import sounddevice as sd
import collections
import os
import playsound
import requests
import json
import sqlite3
import sys
import time
import pytz
import cv2
from gtts import gTTS
from datetime import datetime


def load_config():
    try:
        with open('config.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("Configuration file not found. Please ensure 'config.json' exists.")
        exit(1)
    except json.JSONDecodeError:
        print("Error reading 'config.json'. Please ensure it contains valid JSON.")
        exit(1)
        
config = load_config()
api_key = config.get("api_key")

if not api_key:
    print("API key not found in configuration file.")
    exit(1)
    
def get_time_in_timezone(timezone_name):
  """Displays the current time in a specified timezone."""
  try:
    tz = pytz.timezone(timezone_name)
    now = datetime.now(tz)  # Get time aware of the specified timezone
    formatted_time = now.strftime("%A, %B %d, %Y - %H:%M:%S %Z (%z)")
    print(f"Current time in {timezone_name}: {formatted_time}")

  except pytz.UnknownTimeZoneError:
    print(f"Invalid timezone: {timezone_name}")


model = tf.saved_model.load("")
SAMPLE_RATE = 16000
CHUNK_DURATION = 1
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
buffer = collections.deque(maxlen=CHUNK_SIZE)


def text_to_speech(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    playsound.playsound("output.mp3")

def count_fingers(hand_landmark):
    finger_tips = [4,8,12,16,20]
    finger = []
    if hand_landmark.landmark[finger_tips[0]].x <hand_landmark.landmark[finger_tips[0] - 1].x:
        finger.append(1)
    else:
        finger.append(0)
    for tip in finger_tips[1:]:
        if hand_landmark.landmark[tip].y < hand_landmark.landmark[tip - 2].y:
            finger.append(1)
        else:
            finger.append(0)

def sagen():
    cap = cv2.VideoCapture(1)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    



def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    buffer.extend(indata[:, 0])  # assuming mono channel
    if len(buffer) == CHUNK_SIZE:
        audio_chunk = np.array(buffer).reshape(1, -1)
        predictions = model(audio_chunk)
        predicted_label = np.argmax(predictions, axis=1)
        print(f"Predicted Label: {predicted_label}")
        if predicted_label:
            
            base_url = "http://api.openweathermap.org/data/2.5/weather?"
            complete_url = f"{base_url}q={cityname}&appid={api_key}&units=metric"
            
            try:
                response = requests.get(complete_url)
                response.raise_for_status()  # Raise an HTTPError for bad responses
                data = response.json()

                if data["cod"] != 200:
                    print(f"Error fetching data: {data['message']}")
                    return None
                
                main = data.get("main", {})
                wind = data.get("wind", {})
                weather = data.get("weather", [{}])[0]
                rain = data.get("rain", {}).get("1h", 0)
            except requests.RequestException as e:
                print(f"Error making request to OpenWeatherMap API: {e}")
                
            while True:
                if predicted_label == "temps":
                    print("The Temps is " + int() + "Celsius")
                    text_to_speech(
                        "The Temps is "
                        + int()
                        + "Celsius And Humidity Is"
                        + int()
                        + "Percentage"
                    )
                elif predicted_label == "rain":
                    print()
                    text_to_speech("The Raining Rate is" + int() + "mm/hours")
                elif predicted_label == "windspeed":
                    print()
                    text_to_speech("The Wind Speed Is" +int() + "m/s")
                else:
                    break


if __name__ == "__main__":
    with sd.InputStream(
        callback=audio_callback,
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=int(SAMPLE_RATE * CHUNK_DURATION / 2),
    ):
        print("Listening...")
    text = "Starting"
    text_to_speech(text)
    text_to_speech("The service is now Online")
    cityname = str(input("Your City? \n >>>"))