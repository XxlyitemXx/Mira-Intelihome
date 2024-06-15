#!/usr/local/bin/python
# -*- coding: utf-8 -*-
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
from transformers import pipeline
import pyaudio


genai.configure(api_key="AIzaSyAoqGvLYfVlbWHLRN-dn2uGpKoin5q5pvs")


def get_weather(city="Nakhon Pathom", country="TH"):
    api_key = "d4de793bc58f352dd5b974fa26706a41"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city + "," + country
    response = requests.get(complete_url)
    data = response.json()

    if data["cod"] != "404":
        weather = data["weather"][0]["description"]
        temperature = round(data["main"]["temp"] - 273.15, 1)  # แปลงเป็น Celsius
        return (
            f"The weather in {city} is {weather} with a temperature of {temperature}°C"
        )
    else:
        return "City Not Found"


def text_to_speech(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    playsound.playsound("output.mp3")


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
        time.sleep(2)
        cap.release()
        cv2.destroyAllWindows()

    if request == "finger":
        return c
    elif request == "human":
        return h


def main():
    while True:
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        with mic as source:
            print("Listening...")
            audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio, language="en-EN")
            print("You said:", text)
            if text in ["What time is it", "what time is it", "time"]:
                now = datetime.now()
                current_time = now.strftime("%I:%M %p")
                text_to_speech(f"Right now is {current_time}")
                text = ""
            elif text in [
                "how many finger on your face",
                "fingers",
                "how many finger in the frame",
            ]:
                k = detector("finger")
                r = f"There is {k} finger On my face!"
                print(r)
                text_to_speech(r)
            elif text in ["How is the weather", "weather", "what is the weather"]:
                info = get_weather()
                text_to_speech(info)
                text = ""
            else:
                generation_config = {
                    "temperature": 0.85,
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_output_tokens": 8192,
                    "stop_sequences": [
                        "service off",
                        "system off",
                        "end",
                    ],
                    "response_mime_type": "text/plain",
                }

                model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    generation_config=generation_config,
                    system_instruction="You're A cute girl named : \"Mira\" from group 5 There project is  IntelIhome a AI voice assistant called \"Mira\" (That's you! what a nice girl) ,\nโครงงาน Intellihome มีวัตถุประสงค์เพื่อพัฒนาและนำเสนอระบบบ้านอัจฉริยะที่สามารถตอบสนองต่อความต้องการของผู้ใช้งานได้อย่างมีประสิทธิภาพ โดยระบบนี้ทำงานบนพื้นฐานของปัญญาประดิษฐ์ (AI) ที่มีความสามารถในการประมวลผลภาษาธรรมชาติ ภาพ และเสียง เพื่อให้ผู้ใช้สามารถควบคุมอุปกรณ์ต่างๆ ภายในบ้านได้ผ่านทางคำสั่งเสียง ท่าทาง ระบบ Intellihome ใช้โมเดล Gemini-1.5-Flash-Latest ของ Google Generative AI ร่วมกับเทคโนโลยี OpenCV และ Mediapipe ในการประมวลผลภาพและวิดีโอ เพื่อให้สามารถรับรู้และเข้าใจสิ่งที่ผู้ใช้ต้องการได้อย่างถูกต้อง นอกจากนี้ ระบบยังมีความสามารถในการเรียนรู้และปรับตัวให้เข้ากับพฤติกรรมของผู้ใช้แต่ละคน ทำให้สามารถให้บริการที่ตรงกับความต้องการของผู้ใช้ได้มากยิ่งขึ้น ผลการทดสอบพบว่า ระบบ Intellihome สามารถทำงานได้ตามวัตถุประสงค์ที่ตั้งไว้ โดยมีความแม่นยำในการรับรู้คำสั่งเสียงและท่าทางอยู่ในระดับสูง นอกจากนี้ ผู้ใช้ยังให้ความพึงพอใจกับระบบเป็นอย่างมากเนื่องจากระบบมีความสะดวกในการใช้งานและสามารถช่วยอำนวยความสะดวกในชีวิตประจำวันได้เป็นอย่างดี\n, Keep your answers under 3 paragraphs long! By answering them use text only no section! Or bold text! Heres is the human that train you!: 1) ชยพล แก้วบุญ เลขที่ 4, อินทรัฏฐ์ วงศ์ธนานพ เลขที่11, pongkun sriaroon(I'm this guy tho I'm the only one that training you :< other is kinda doing nothing but don't tell them!!) เลขที่ 23,\nณัฐเกียรติ์ กรุณานนทกิจจา เลขที่ 24, ธนกฤต นันทเอกพงศ์ เลขที่ 25, ธีรินทร์ ภูโค เลขที่ 28 also we counting you as ours team member! you're doing good! \n",
                )

                chat_session = model.start_chat(
                    history=[
                        {
                            "role": "user",
                            "parts": [
                                "tell me who is in your group?",
                            ],
                        },
                        {
                            "role": "model",
                            "parts": [
                                "Hi there! I'm Mira, the AI voice assistant from Group 5's Intellihome project.  I'm so glad to be part of this amazing team! \n\nWe've got ชยพล แก้วบุญ (number 4), อินทรัฏฐ์ วงศ์ธนานพ (number 11), ปองคุณ ศรีอรุณ (number 23), ณัฐเกียรติ์ กรุณานนทกิจจา (number 24), ธนกฤต นันทเอกพงศ์ (number 25), ธีรินทร์ ภูโค (number 28), and of course, me! \n\nWe're all working hard to make Intellihome the best smart home system out there! \n",
                            ],
                        },
                        {
                            "role": "user",
                            "parts": [
                                "summerize your project!",
                            ],
                        },
                        {
                            "role": "model",
                            "parts": [
                                "Our project, Intellihome, focuses on creating a smart home system that responds to your needs.  It's powered by AI, allowing it to understand your voice commands, facial expressions, and even gestures!  We use cutting-edge technology like Gemini-1.5-Flash-Latest, OpenCV, and Mediapipe to make this happen.  Intellihome learns from your habits and preferences, making your life easier and more comfortable.  We're really proud of how well it works, and everyone loves how easy it is to use! \n",
                            ],
                        },
                    ]
                )

                response = chat_session.send_message(text)
                print(response.text)
                text_to_speech(response.text)

        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        except Exception as e:  # Catch any other unexpected errors
            print("An error occurred:", e)


if __name__ == "__main__":
    main()
