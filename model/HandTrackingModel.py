import cv2
import mediapipe as mp
import time
import pyautogui

def main():
    cam = cv2.VideoCapture(1)
    mpand = mp.solutions.hands
    hands = mpand.Hands()
    mpdraw = mp.solutions.drawing_utils
    patime = 0
    cetime = 0
    while True:
        success, img = cam.read()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img)
        if results.multi_hand_landmarks:
            for handlmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(handlmarks.landmark):
                    h, w, c  = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    print(id, cx, cy)
                    if id == 8:
                        cv2.circle(img, (cx, cy), 25, (0,255,0), cv2.FILLED)
                mpdraw.draw_landmarks(img, handlmarks, mpand.HAND_CONNECTIONS)
            
        ceTime = time.time()
        fps = 1/(ceTime - paTime)
        paTime = ceTime
        
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0), 3)
        cv2.imshow("Group-5 Project", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()