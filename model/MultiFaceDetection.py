import cv2
import time
import mediapipe as mp

def main():
    cap = cv2.VideoCapture(1)
    mpface = mp.solutions.face_detection
    mpdraw = mp.solutions.drawing_utils
    facedetection = mpface.FaceDetection(0.75)
    petime = time.time()

    while True:
        success, img = cap.read()
        if not success:
            break
        
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = facedetection.process(imgrgb)
        human_count = 0
        
        if results.detections:
            human_count = len(results.detections)
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                x1 = int(bboxC.xmin * iw)
                y1 = int(bboxC.ymin * ih)
                x2 = int(bboxC.width * iw)
                y2 = int(bboxC.height * ih)
                bbox = (x1, y1, x1 + x2, y1 + y2)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2)
                cv2.putText(img, f'{int(detection.score[0]* 100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
        
        # Display the number of people detected
        cv2.putText(img, f'People count: {human_count}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Calculate and display FPS
        cetime = time.time()
        fps = 1 / (cetime - petime)
        petime = cetime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Group-5 Project", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
