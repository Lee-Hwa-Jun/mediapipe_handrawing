import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def cal_ratio(x1,y1,x2,y2):
    x = x2-x1
    y = y2-y1
    return math.sqrt(x ** 2 + y ** 2) * 100

width = 1920
height = 1080

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

empty_canvas = np.zeros((height,width,3),np.uint8)

std_ratio = 1
t_ratio = 1
ratio = 1
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        canvas = empty_canvas.copy()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:


                std_ratio = cal_ratio(hand_landmarks.landmark[0].x,hand_landmarks.landmark[0].y,
                                  hand_landmarks.landmark[5].x,hand_landmarks.landmark[5].y)

                t_ratio = cal_ratio(hand_landmarks.landmark[8].x,hand_landmarks.landmark[8].y,
                                  hand_landmarks.landmark[4].x,hand_landmarks.landmark[4].y)

                ratio = std_ratio / t_ratio

                x = int(hand_landmarks.landmark[4].x*width)
                y = int(hand_landmarks.landmark[4].y*height)

                isdraw = False#손가락 모양 인식 판단용
                if(ratio > 3):
                    cv2.circle(canvas,(x,y),10,[255,0,0],-1)
                mp_drawing.draw_landmarks(
                    canvas,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        cv2.imshow('MediaPipe Hands', cv2.flip(canvas,1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()