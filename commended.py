#!/usr/local/bin/python
# -*- coding: utf-8 -*-

# นำเข้าไลบรารีที่จำเป็น
import os
import cv2
import time
import sqlite3
import requests
import playsound
import numpy as np
import mediapipe as mp
import tensorflow as tf
import sounddevice as sd
import google.generativeai as genai
import speech_recognition as sr
from datetime import datetime
from transformers import pipeline
from gtts import gTTS

# ตั้งค่า API Key สำหรับ Google Generative AI
genai.configure(api_key="YOUR_API_KEY_HERE")  # อย่าลืมใส่ API Key ของคุณที่นี่


# ฟังก์ชันสำหรับรับข้อมูลสภาพอากาศ
def get_weather(city="Nakhon Pathom", country="TH"):
    """
    ดึงข้อมูลสภาพอากาศจาก OpenWeatherMap API.

    Args:
        city (str, optional): ชื่อเมือง. Defaults to "Nakhon Pathom".
        country (str, optional): รหัสประเทศ. Defaults to "TH".

    Returns:
        str: ข้อมูลสภาพอากาศ หรือข้อความแจ้งว่าเมืองไม่พบ.
    """
    try:
        # กำหนด API Key และ URL สำหรับ OpenWeatherMap
        api_key = "YOUR_API_KEY_HERE"  # อย่าลืมใส่ API Key ของคุณที่นี่
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        complete_url = (
            base_url + "appid=" + api_key + "&q=" + city + "," + country
        )
        # ส่งคำขอไปยัง OpenWeatherMap API
        response = requests.get(complete_url)
        # แปลงข้อมูลที่ได้รับเป็น JSON
        data = response.json()

        # ตรวจสอบว่าเมืองถูกพบหรือไม่
        if data["cod"] != "404":
            # ดึงข้อมูลสภาพอากาศและอุณหภูมิ
            weather = data["weather"][0]["description"]
            temperature = round(
                data["main"]["temp"] - 273.15, 1
            )  # แปลงเป็น Celsius
            # ส่งคืนข้อมูลสภาพอากาศ
            return (
                f"The weather in {city} is {weather} with a temperature of {temperature}°C"
            )
        else:
            # ส่งคืนข้อความแจ้งว่าเมืองไม่พบ
            return "City Not Found"
    except Exception as e:
        # จัดการข้อผิดพลาดที่อาจเกิดขึ้น
        print(f"Error in get_weather: {e}")
        return "An error occurred while fetching weather information."


# ฟังก์ชันสำหรับแปลงข้อความเป็นเสียงพูด
def text_to_speech(text, lang="en"):
    """
    แปลงข้อความเป็นเสียงพูดโดยใช้ gTTS.

    Args:
        text (str): ข้อความที่จะแปลง.
        lang (str, optional): ภาษา. Defaults to "en".
    """
    try:
        # สร้างวัตถุ gTTS
        tts = gTTS(text=text, lang=lang)
        # บันทึกไฟล์เสียง
        tts.save("output.mp3")
        # เล่นไฟล์เสียง
        playsound.playsound("output.mp3")
    except Exception as e:
        # จัดการข้อผิดพลาดที่อาจเกิดขึ้น
        print(f"Error in text_to_speech: {e}")


# ฟังก์ชันสำหรับนับจำนวนนิ้วที่ยกขึ้น
def count_fingers(hand_landmarks):
    """
    นับจำนวนนิ้วที่ยกขึ้นจากข้อมูล hand landmarks.

    Args:
        hand_landmarks: ข้อมูล hand landmarks จาก Mediapipe.

    Returns:
        int: จำนวนนิ้วที่ยกขึ้น.
    """
    # กำหนดตำแหน่งปลายนิ้ว
    finger_tips = [4, 8, 12, 16, 20]

    # รายการสำหรับเก็บข้อมูลนิ้วที่ยกขึ้น
    fingers = []

    # ตรวจสอบนิ้วโป้ง
    if (
        hand_landmarks.landmark[finger_tips[0]].x
        < hand_landmarks.landmark[finger_tips[0] - 1].x
    ):
        fingers.append(1)
    else:
        fingers.append(0)

    # ตรวจสอบนิ้วอื่นๆ
    for tip in finger_tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # ส่งคืนจำนวนนิ้วที่ยกขึ้น
    return fingers.count(1)


# ฟังก์ชันสำหรับตรวจจับมือและใบหน้า
def detector(request):
    """
    ตรวจจับมือหรือใบหน้าในเฟรมวิดีโอ.

    Args:
        request (str): "finger" สำหรับนับนิ้ว, "human" สำหรับนับคน.

    Returns:
        int: จำนวนนิ้วหรือนับคน, ขึ้นอยู่กับ request.
    """
    # เปิดกล้อง
    cap = cv2.VideoCapture(0)  # เปลี่ยนเป็น 0 ถ้าใช้กล้องในตัว
    # เตรียมโมเดล Mediapipe สำหรับการตรวจจับมือและใบหน้า
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(0.75)
    mp_draw = mp.solutions.drawing_utils

    # ตัวแปรสำหรับเก็บจำนวนนิ้วและจำนวนคน
    c = 0
    h = 0
    prev_time = 0

    try:
        # วนลูปเพื่อประมวลผลภาพจากกล้อง
        while True:
            # อ่านภาพจากกล้อง
            success, img = cap.read()
            if not success:
                print("Error: Could not read frame from camera.")
                break

            # แปลงภาพเป็น RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # ตรวจจับมือและนับจำนวนนิ้ว
            hand_results = hands.process(rgb_img)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # วาดเส้นเชื่อมจุดบนมือ
                    mp_draw.draw_landmarks(
                        img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    # นับจำนวนนิ้ว
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
                    # บันทึกจำนวนนิ้ว
                    c = num_fingers

            # ตรวจจับใบหน้า
            face_results = face_detection.process(rgb_img)
            if face_results.detections:
                # นับจำนวนคน
                human_count = len(face_results.detections)
                for detection in face_results.detections:
                    # ดึงข้อมูลตำแหน่งใบหน้า
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = img.shape
                    x1 = int(bboxC.xmin * iw)
                    y1 = int(bboxC.ymin * ih)
                    x2 = int(bboxC.width * iw)
                    y2 = int(bboxC.height * ih)
                    bbox = (x1, y1, x1 + x2, y1 + y2)
                    # วาดกรอบรอบใบหน้า
                    cv2.rectangle(
                        img,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        (255, 0, 255),
                        2,
                    )
                    # แสดงความน่าจะเป็นในการตรวจจับใบหน้า
                    cv2.putText(
                        img,
                        f"{int(detection.score[0]* 100)}%",
                        (bbox[0], bbox[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 255),
                        2,
                    )
                # บันทึกจำนวนคน
                h = human_count

            # แสดงจำนวนคน
            cv2.putText(
                img,
                f"People count: {human_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # คำนวณและแสดง FPS
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

            # แสดงภาพ
            cv2.imshow("Camera Feed", img)

            # ออกจากลูปเมื่อกด "q" บนคีย์บอร์ด
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        # จัดการข้อผิดพลาดที่อาจเกิดขึ้น
        print(f"Error in detector: {e}")
    finally:
        # ปิดกล้อง
        cap.release()
        cv2.destroyAllWindows()

    # ส่งคืนจำนวนนิ้วหรือจำนวนคนตามคำขอ
    if request == "finger":
        return c
    elif request == "human":
        return h
    else:
        return None


# ฟังก์ชันหลัก
def main():
    """
    ฟังก์ชันหลักของโปรแกรม.
    """
    # วนลูปเพื่อรับคำสั่งเสียงจากผู้ใช้
    while True:
        # เตรียมตัวรับฟังเสียง
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        try:
            # เริ่มฟังเสียง
            with mic as source:
                print("Listening...")
                audio = recognizer.listen(source)

            # พยายามแปลงเสียงเป็นข้อความ
            text = recognizer.recognize_google(audio, language="en-EN")
            print("You said:", text)

            # ตรวจสอบคำสั่งเสียง
            if text.lower() in [
                "what time is it",
                "what's the time",
                "tell me the time",
            ]:
                # ดึงเวลาปัจจุบันและแปลงเป็นรูปแบบเวลา
                now = datetime.now()
                current_time = now.strftime("%I:%M %p")
                # แปลงข้อความเป็นเสียงพูด
                text_to_speech(f"Right now is {current_time}")

            elif text.lower() in [
                "how many fingers on your frame",
                "how many fingers can you see",
                "count my fingers",
            ]:
                # เรียกใช้ฟังก์ชัน detector เพื่อนับจำนวนนิ้ว
                k = detector("finger")
                if k is not None:
                    r = f"I see {k} fingers in the frame."
                    print(r)
                    # แปลงข้อความเป็นเสียงพูด
                    text_to_speech(r)
                else:
                    text_to_speech("Sorry, there was an error.")

            elif text.lower() in [
                "how is the weather",
                "what's the weather like",
                "tell me the weather",
            ]:
                # เรียกใช้ฟังก์ชัน get_weather เพื่อดึงข้อมูลสภาพอากาศ
                info = get_weather()
                # แปลงข้อความเป็นเสียงพูด
                text_to_speech(info)

            elif text.lower() in [
                "how many people on your frame",
                "how many people can you see",
                "count the people",
            ]:
                # เรียกใช้ฟังก์ชัน detector เพื่อนับจำนวนคน
                f = detector("human")
                if f is not None:
                    text_to_speech(f"I see {f} people in the frame.")
                else:
                    text_to_speech("Sorry, there was an error.")

            else:
                # กำหนดการตั้งค่าสำหรับโมเดล Gemini-1.5-Flash
                generation_config = {
                    "temperature": 0.85,
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_output_tokens": 8192,
                    "stop_sequences": ["service off", "system off", "end"],
                    "response_mime_type": "text/plain",
                }

                # สร้างวัตถุ GenerativeModel
                model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    generation_config=generation_config,
                    system_instruction="", ## บอกไม่ได้ ความลับทางการค้า!!
                )

                # เริ่มเซสชั่นการสนทนา
                chat_session = model.start_chat(
                    history=[
                        ## ไม่สามารถบอกได้นะงูงิ ความลับทางการค้า!! ความลับทางการค้า!! ความลับทางการค้า!!
                    ]
                )

                # ส่งข้อความไปยังโมเดล
                response = chat_session.send_message(text)
                print(response.text)
                # แปลงข้อความเป็นเสียงพูด
                text_to_speech(response.text)

        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        except Exception as e:  # Catch any other unexpected errors
            print("An error occurred:", e)


# ตรวจสอบว่าโค้ดทำงานในรูปแบบหลักหรือไม่
if __name__ == "__main__":
    main()