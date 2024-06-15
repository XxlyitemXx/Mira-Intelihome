import cv2
import time
import mediapipe as mp


def count_fingers(hand_landmarks):
    """
    Counts the number of raised fingers from hand landmarks data.

    Args:
        hand_landmarks: Hand landmarks data from Mediapipe.

    Returns:
        int: Number of raised fingers.
    """
    finger_tips = [4, 8, 12, 16, 20]
    fingers = []

    if (
        hand_landmarks.landmark[finger_tips[0]].x
        < hand_landmarks.landmark[finger_tips[0] - 1].x
    ):
        fingers.append(1)
    else:
        fingers.append(0)

    for tip in finger_tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)


def detector(request):
    """
    Detects hands or faces in a video frame.

    Args:
        request (str): "finger" to count fingers, "human" to count people.

    Returns:
        int: Number of fingers or people count, based on the request.
    """
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(0.75)
    mp_draw = mp.solutions.drawing_utils

    c = 0
    h = 0
    prev_time = 0

    try:
        while True:
            success, img = cap.read()
            if not success:
                print("Error: Could not read frame from camera.")
                break

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            hand_results = hands.process(rgb_img)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
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
                        img,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        (255, 0, 255),
                        2,
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

            cv2.putText(
                img,
                f"People count: {human_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

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

            cv2.imshow("Camera Feed", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"Error in detector: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if request == "finger":
        return c
    elif request == "human":
        return h
    else:
        return None