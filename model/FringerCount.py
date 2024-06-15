import cv2
import mediapipe as mp
import time

def count_fingers(hand_landmarks, img):
    # Define the tips of the fingers
    finger_tips = [4, 8, 12, 16, 20]
    
    # List to keep track of which fingers are up
    fingers = []

    # Thumb
    if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_tips[0] - 1].x:
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

def main():
    cam = cv2.VideoCapture(1)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils
    pa_time = 0
    ce_time = 0

    while True:
        success, img = cam.read()
        if not success:
            break
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Count fingers
                num_fingers = count_fingers(hand_landmarks, img)
                cv2.putText(img, f'Fingers: {num_fingers}', (10, 130), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

        ce_time = time.time()
        fps = 1 / (ce_time - pa_time)
        pa_time = ce_time
        
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
        cv2.imshow("Finger Count", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
