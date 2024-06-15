# detection.py
import cv2
import mediapipe as mp
import time
from utils import count_fingers

class FingerFaceDetector:
    def __init__(self, hand_max_num=1, hand_min_detection_confidence=0.5, hand_min_tracking_confidence=0.5, face_min_detection_confidence=0.75):
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=hand_max_num,
                                         min_detection_confidence=hand_min_detection_confidence,
                                         min_tracking_confidence=hand_min_tracking_confidence)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=face_min_detection_confidence)
        self.mp_draw = mp.solutions.drawing_utils

        self.prev_time = 0

    def process_frame(self):
        success, img = self.cap.read()
        if not success:
            return None, None

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Hand detection and finger counting
        hand_results = self.hands.process(rgb_img)
        num_fingers = 0
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                num_fingers = count_fingers(hand_landmarks)
                cv2.putText(img, f'Fingers: {num_fingers}', (10, 130), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # Face detection
        face_results = self.face_detection.process(rgb_img)
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
        fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        cv2.putText(img, f'FPS: {int(fps)}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img, (num_fingers, human_count)

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
