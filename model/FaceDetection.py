import cv2
import time
import mediapipe as mp

def main():
    cap = cv2.VideoCapture(1)
    mpface = mp.solutions.face_detection
    mpdraw = mp.solutions.drawing_utils
    facedetection = mpface.FaceDetection(0.75)
    petime = 0
    cetime = 0
    human = 0
    while True:
        success, img = cap.read()
        
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = facedetection.process(imgrgb)
        if results.detections:
            for id, detection in enumerate(results.detections):
                
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                x1 = int(bboxC.xmin * iw)
                y1 = int(bboxC.ymin * ih)
                x2 = int(bboxC.width * iw)
                y2 = int(bboxC.height * ih)
                bbox = (x1, y1, x1 + x2, y1 + y2)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2)
                
                
                
                
                cv2.putText(img, f'{int(detection.score[0]* 100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
        
        
        petime = time.time()
        fps = 1/(cetime - petime )
        petime = cetime
        
        
        cv2.imshow("Group-5 Project", img)
        cv2.waitKey(1)
    
if __name__ == "__main__":
    main()