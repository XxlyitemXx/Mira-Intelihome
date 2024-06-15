import cv2
import collections
import os
import playsound
import requests
import json
import sqlite3
import sys
import time
import pytz
import numpy as np
import sounddevice as sd
import tensorflow as tf
import mediapipe as mp
import speech_recognition as sr
from gtts import gTTS
from datetime import datetime


### Load The config from config.json ###
def load_config():
    try:
        with open("config.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print("Configuration file not found. Please ensure 'config.json' exists.")
        exit(1)
    except json.JSONDecodeError:
        print("Error reading 'config.json'. Please ensure it contains valid JSON.")
        exit(1)


config = load_config()
api_key = config.get("api_key")
city = config.get("city")
### Error Handler ###
if not api_key:
    print("API key not found in configuration file.")
    exit(1)
if not city:
    print("City name not found in configuration file.")


### Timezone ###
def get_time_in_timezone(timezone_name):
    """Displays the current time in a specified timezone."""
    try:
        tz = pytz.timezone(timezone_name)
        now = datetime.now(tz)  # Get time aware of the specified timezone
        formatted_time = now.strftime("%A, %B %d, %Y - %H:%M:%S %Z (%z)")
        print(f"Current time in {timezone_name}: {formatted_time}")

        return formatted_time

    except pytz.UnknownTimeZoneError:
        print(f"Invalid timezone: {timezone_name}")


### Loading The Trained Teachable machine model ###

model = tf.saved_model.load(
    "/Users/pongkunsriaroon/Documents/Projekt/Idk/group5/model.json"
)

### Setting ###

SAMPLE_RATE = 16000
CHUNK_DURATION = 1
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
buffer = collections.deque(maxlen=CHUNK_SIZE)

### TTS ###


def text_to_speech(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    playsound.playsound("output.mp3")


### detector ###


def count_fingers(hand_landmarks):
    # Define the tips of the fingers
    finger_tips = [4, 8, 12, 16, 20]

    # List to keep track of which fingers are up
    fingers = []

    # Thumb
    if (
        hand_landmarks.landmark[finger_tips[0]].x
        < hand_landmarks.landmark[finger_tips[0] - 1].x
    ):
        fingers.append(1)
    else:
        fingers.append(0)

    # Other 4 fingers
    for tip in finger_tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # Return the total number of fingers up
    return fingers.count(1)


def main(request):
    cap = cv2.VideoCapture(1)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(0.75)
    mp_draw = mp.solutions.drawing_utils

    c = 0
    h = 0
    prev_time = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Hand detection and finger counting
        hand_results = hands.process(rgb_img)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Count fingers
                num_fingers = count_fingers(hand_landmarks)
                cv2.putText(
                    img,
                    f"Fingers: {num_fingers}",
                    (10, 130),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 255, 0),
                    3,
                )
                c = num_fingers
        # Face detection
        face_results = face_detection.process(rgb_img)
        if face_results.detections:
            human_count = len(face_results.detections)
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                x1 = int(bboxC.xmin * iw)
                y1 = int(bboxC.ymin * ih)
                x2 = int(bboxC.width * iw)
                y2 = int(bboxC.height * ih)
                bbox = (x1, y1, x1 + x2, y1 + y2)
                cv2.rectangle(
                    img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2
                )
                cv2.putText(
                    img,
                    f"{int(detection.score[0]* 100)}%",
                    (bbox[0], bbox[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 255),
                    2,
                )
                h = human_count
        # Display the number of people detected
        cv2.putText(
            img,
            f"People count: {human_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("group5things", img)
        cv2.waitKey(1)
        time.sleep(2)
        cap.release()
        cv2.destroyAllWindows()

    if request == "finger":
        return c
    elif request == "human":
        return h


### Main ###


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
            complete_url = f"{base_url}q={city}&appid={api_key}&units=metric"

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
                    text_to_speech("The Wind Speed Is" + int() + "m/s")
                elif predicted_label == "FD":
                    pass
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
