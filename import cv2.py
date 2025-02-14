import cv2
import mediapipe as mp
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    h, w, c = image.shape

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        
        dx = index_tip.x - thumb_tip.x
        dy = index_tip.y - thumb_tip.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        print(f"Distance between index tip and thumb tip: {distance:.3f}")
        
        if distance < 0.05:
            cv2.putText(
                        image,
                        "Pinch gesture detected!",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        for i, landmark in enumerate(hand_landmarks.landmark):
          px = int(landmark.x * w)
          py = int(landmark.y * h)

          text = f"{i}:({landmark.x:.2f},{landmark.y:.2f},{landmark.z:.2f})"
          cv2.putText(
              image, 
              text, 
              (px, py - 10),
              cv2.FONT_HERSHEY_SIMPLEX, 
              0.4, 
              (255, 0, 0), 1, 
              cv2.LINE_AA
          )
    
    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()