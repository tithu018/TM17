import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Finger tip landmarks (index: 8, middle: 12, ring: 16, pinky: 20)
finger_tips = [8, 12, 16, 20]

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert to RGB for MediaPipe processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    finger_count = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get all landmark positions
            landmarks = hand_landmarks.landmark

            # Thumb logic (works for right hand)
            if landmarks[4].x < landmarks[3].x:
                finger_count += 1

            # Other 4 fingers
            for tip in finger_tips:
                if landmarks[tip].y < landmarks[tip - 2].y:
                    finger_count += 1

            # Display the number of fingers
            cv2.putText(img, f'Fingers: {finger_count}', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("Finger Counter", img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
