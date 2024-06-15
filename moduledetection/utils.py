# utils.py
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
