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
import random
import speech_recognition as sr
from gtts import gTTS
from datetime import datetime

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
def countdown_timer(seconds):
    deac = seconds
    while seconds > 0:
        mins, secs = divmod(seconds, 60)
        time.sleep(1)
        seconds -= 1
        progress_width = 30  # Adjust as needed
        blocks = int(round(progress_width * seconds / deac)) 
        progress_bar =  "[" + "#" * blocks + " " * (progress_width - blocks) + "]"
        sys.stdout.write(f'\rTime Remaining: {mins:02d}:{secs:02d} {progress_bar}')
    os.system("clear")
    print("\033[92mTimer Complete!\033[0m")
    
def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening...")
        audio = recognizer.listen(source)

    languages = ['en-US', 'th-TH']
    for language in languages:
        try:
            text = recognizer.recognize_google(audio, language=language)
            print(f"You said ({language}): {text}")
            return text
        except sr.UnknownValueError:
            continue  # Try the next language
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service")
            return None

    print("Google Speech Recognition could not understand audio in any language")
    return None

def text_to_speech(text, lang="th"):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    playsound.playsound("output.mp3")
    os.remove("output.mp3")

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
def weather_get(type):
    
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
        
        if type == "main":
            return main
        elif type == "wind":
            return wind
        elif type == "weather":
            return weather
        elif type == "rain":
            return rain
        
        
def main():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    text = ""

    with mic as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language='th-TH')
        print("You said: " + text)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service")
    if text:
       while True: 
        if text in ["hello", "สวัสดี"]:
            responses = ["สวัสดีค่ะ", "ว่าไง", "มีอะไรให้ช่วยไหมค่ะ"]
            text_to_speech(random.choice(responses))
            text = ""
        elif text in ["Good Morning", "สวัสดีตอนเช้า"]:
            res = ["สวัสดีตอนเช้าค่ะ", "สวัสดีตอนเช้าค่ะมีอะไรให้ช่วยไหม"]
            text_to_speech(random._choice(res))
            text = ""
        elif text in ["Good Night", "ราตรีสวัสดิ์"]:
            text_to_speech("ราตรีสวัสดิ์ค่ะ")
            text = ""
        elif text in ["how are you", "คุณสบายดีไหม", "เป็นไง"]:
            text_to_speech("กูสบายดีถามหาพ่อง")
            text = ""
        elif text in ["what is your name", "คุณชื่ออะไร"]:
            text_to_speech("ฉันชื่อ Mira ค่ะ")
            text = ""
        elif text in ["Where are you from", "คุณมาจากไหน"]:
            text_to_speech("nothings")
            text = ""
        elif text in ["thank", "thank you", "ขอบคุณ", "ขอบใจ"]:
            text_to_speech("จ้า")
            text = ""
        elif text in ["sorry", "ขอโทษ"]:
            text_to_speech("มึงขอโทษไร")
            text = ""
        elif text in ["please", "กรุณา"]:
            text_to_speech("เค")
            text = ""
        elif text in ["yes", "ใช่"]:
            text_to_speech("ใช่อะไรมึง")
            text = ""
        elif text in ["no", "ไม่ใช่"]:
            text_to_speech("ห่ะ")
            text = ""
        elif text in ["maybe", "อาจจะ"]:
            text_to_speech("อาจจะอะไรไอสัส")
            text = ""
        elif text in ["excuse me", "ขอโทษ"]:
            text_to_speech("ไม่เป็นไรจ้า")
            text = ""
        elif text in ["i need help", "ฉันต้องการความช่วยเหลือ"]:
            text_to_speech("เคให้กูทำไร")
            text = ""
        elif text in ["where is the bathroom", "ห้องน้ำอยู่ที่ไหน"]:
            text_to_speech("มึงถามกูทำไม")
            text = ""
        elif text in ["goodbye", "ลาก่อน", "ไปไกลไกลเลย"]:
            text_to_speech("ลาก่อนค่ะ ขอให้ไม่ได้้เจอกันอีกเลย")
            exit()
        elif text in ["see you later", "เจอกันใหม่"]:
            text_to_speech("เจอกันใหม่ค่ะ")
            text = ""
        elif text in ["วันนี้อากาศเป็นยังไงบ้าง", "ตอนนี้อากาศเป็นยังไง", "พรุ่งนี้อากาศจะเป็นยังไง?", 
                    "พยากรณ์อากาศวันนี้ว่าไงบ้าง?", "วันนี้อากาศร้อนไหม", "วันนี้ฝนจะตกไหม,", 
                    "อุณหภูมิวันนี้เท่าไหร่", "วันนี้อากาศดีไหม", "ช่วงนี้อากาศเป็นยังไงบ้าง?", 
                    "พยากรณ์อากาศพรุ่งนี้ว่าไง"]:
            l = weather_get("weather")
            text_to_speech(f"ข้อมูลจาก openweathermap มีดังนี้ {l} เคป่ะขีั้เกลียดเลยดึงมาแม่งหมดเลย")
            text = ""
        elif text in ["ลมพัดเร็วเท่าไหร่", "ลมพัดแรงแค่ไหน", "ความเร็วลมเป็นเท่าไหร่", "วันนี้ลมแรงไหม", "ลมพัดที่ความเร็วเท่าไหร่", "ความเร็วลมเท่าไร"]:
            l = weather_get("wind")
            text_to_speech(f"ข้อมูลจาก openweathermap มีดังนี้ {l} เคป่ะขีั้เกลียดเลยดึงมาแม่งหมดเลย")
            text = ""
        elif text in ["Internet", "อินเทอร์เน็ต"]:
            text_to_speech("อินเทอร์เน็ตคือเครือข่ายคอมพิวเตอร์ทั่วโลกที่เชื่อมต่อกัน")
            text = ""
        elif text in ["Computer", "คอมพิวเตอร์"]:
            text_to_speech("คอมพิวเตอร์คือเครื่องมืออิเล็กทรอนิกส์ที่ใช้ในการคำนวณและประมวลผลข้อมูล")
            text = ""
        elif text in ["Software", "ซอฟต์แวร์"]:
            text_to_speech("ซอฟต์แวร์คือชุดของคำสั่งที่ใช้ในการควบคุมการทำงานของคอมพิวเตอร์")
            text = ""
        elif text in ["hardware", "ฮาร์ดแวร์"]:
            text_to_speech("ฮาร์ดแวร์คือส่วนประกอบที่จับต้องได้ของคอมพิวเตอร์ เช่น หน่วยประมวลผลกลางและหน่วยความจำ")
            text = ""
        elif text in ["network", "เครือข่าย"]:
            text_to_speech("เครือข่ายคือระบบที่เชื่อมต่ออุปกรณ์หลายๆ ตัวเข้าด้วยกันเพื่อแชร์ข้อมูลและทรัพยากร")
            text = ""
        elif text in ["database", "ฐานข้อมูล"]:
            text_to_speech("ฐานข้อมูลคือการจัดเก็บข้อมูลอย่างมีระบบระเบียบเพื่อการค้นหาและใช้งานได้ง่าย")
            text = ""
        elif text in ["programming", "การเขียนโปรแกรม"]:
            text_to_speech("การเขียนโปรแกรมคือกระบวนการสร้างชุดคำสั่งเพื่อควบคุมการทำงานของคอมพิวเตอร์")
            text = ""
        elif text in ["algorithm", "อัลกอริทึม"]:
            text_to_speech("อัลกอริทึมคือขั้นตอนหรือกระบวนการในการแก้ปัญหาหรือทำงานบางอย่าง")
            text = ""
        elif text in ["Machine Learning", "การเรียนรู้ของเครื่อง"]:
            text_to_speech("การเรียนรู้ของเครื่องคือการที่คอมพิวเตอร์สามารถเรียนรู้และพัฒนาจากข้อมูลได้เอง")
            text = ""
        elif text in ["artificial intelligence", "ปัญญาประดิษฐ์", "AI", "เอไอ"]:
            text_to_speech("ปัญญาประดิษฐ์คือการพัฒนาระบบคอมพิวเตอร์ให้สามารถทำงานที่ต้องใช้ความคิดเหมือนมนุษย์")
            text = ""
        elif text in ["cloud computing", "การประมวลผลแบบคลาวด์"]:
            text_to_speech("การประมวลผลแบบคลาวด์คือการใช้เซิร์ฟเวอร์ระยะไกลผ่านอินเทอร์เน็ตในการจัดเก็บ จัดการ และประมวลผลข้อมูล")
            text = ""
        elif text in ["cybersecurity", "ความปลอดภัยทางไซเบอร์"]:
            text_to_speech("ความปลอดภัยทางไซเบอร์คือการป้องกันระบบเครือข่าย คอมพิวเตอร์ และข้อมูลจากการถูกโจมตีหรือเข้าถึงโดยไม่ได้รับอนุญาต")
            text = ""
        elif text in ["What time is it", "กี่โมงแล้ว"]:
            k = get_time_in_timezone("Asia/Bangkok")
            text_to_speech(f"ตอนนี้เวลา {k} จ่ะ เอาแบบนี้ละขี้เกลียด")
            text = ""
        elif text in ["Countdown", "นับถอยหลัง"]:
            text_to_speech("ไส่จำนวนจ่ะไอสัส เป็นวิน่ะ แม่งไม่เสียงมาให้กูเทรน เลยมึงก็ต้องใช้แบบนี้ละ")
            k = int(input(' >>> '))            
            countdown_timer(k) 
            text = ""
        
        else:
            main()

if __name__ == "__main__":
    main()