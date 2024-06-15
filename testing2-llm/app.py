import google.generativeai as genai
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

# Set your API key
genai.configure(api_key='AIzaSyDBuhqABFVgEmat3cgsjJ8L6Gt-cFNUkkU')
model = genai.GenerativeModel('gemini-1.5-flash-latest')  


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


def detector(request):
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


# Text-to-speech function
def text_to_speech(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    playsound.playsound("output.mp3")

while True:
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        # Speech-to-text using Google's recognizer
        text = recognizer.recognize_google(audio, language='en-EN')
        print("You said:", text)
        if text:
            if text in ["What time is it", "what time is it", "time"]:
                now = datetime.now()
                current_time = now.strftime("%I:%M %p")
                text_to_speech(f"Right now is {current_time}")
                text = ""
            elif text in ["Summerize", "summerize this", "summarizes", "summarize"]:
                k = input("Paste it here \n >>>")
                p = f"You're mia and you're on the task to summerize this: {k} text only no section"
                r = model.generate_content(contents=p)
                assistant_text = r.parts[0].text
                print("Mia Summerize:", assistant_text) 
                text_to_speech(assistant_text)
                 ###### Dear Gemini, I'll add more later on theres will be finger counting, face recognition, face detection!
            else:

                k = str(text)
                t = f"You're Mia from Group 5 And you got asked to do this task: {k} from them For a User In grade 9 Make them understand as possible. Text only And Short as possible no section In group 5 they are working on AI called mia a model from gemini flash aka you I hope you understand them! goodluck"
                response = model.generate_content(contents=t)

    # Extract and print the assistant's response (extract first candidate)
                assistant_text = response.parts[0].text
                print("Mia:", assistant_text) 
                text_to_speech(assistant_text)

    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e)) 
    except Exception as e:  # Catch any other unexpected errors
        print("An error occurred:", e) 