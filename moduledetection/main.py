# main.py
from detection import FingerFaceDetector
import cv2

def kmain():
    detector = FingerFaceDetector()

    while True:
        img, counts = detector.process_frame()
        if img is None:
            break

        cv2.imshow("Finger and Face Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.release()

if __name__ == "__main__":
    kmain()
