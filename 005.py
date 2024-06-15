import cv2
import mediapipe as mp
import time

def count_fingers(hand_landmarks):
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
    cap = cv2.VideoCapture(1)  # Use the default camera (usually camera 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set the resolution to 640x480 for better performance
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.75)
    mp_draw = mp.solutions.drawing_utils

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
                cv2.putText(img, f'Fingers: {num_fingers}', (10, 130), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # Face detection
        face_results = face_detection.process(rgb_img)
        human_count = 0
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
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2)
                cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        # Display the number of people detected
        cv2.putText(img, f'People count: {human_count}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(img, f'FPS: {int(fps)}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Finger and Face Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
