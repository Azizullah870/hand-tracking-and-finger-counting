import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_Hands = mp.solutions.hands
hand = mp_Hands.Hands()
tipIds = [4, 8, 12, 16, 20]

with mp_Hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    try:
        while True:
            success, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame=cv2.resize(frame,(640,480))
            result = hand.process(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            lmList = []

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    my_hands = result.multi_hand_landmarks[0]
                    for id, lm in enumerate(my_hands.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_Hands.HAND_CONNECTIONS)

            fingers = []

            if len(lmList) != 0:
                if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                for id in range(1, 5):
                    if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                    total = fingers.count(1)
                    # cnt.led(total)
                    if total == 0:
                        cv2.rectangle(frame, (20, 300), (270, 425), (0, 255, 0),-1)
                        cv2.putText(frame, "f:0", (45, 375), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 250),5)
                        
                    elif total==1 :
                        
                        cv2.rectangle(frame, (20, 300), (270, 425), (0, 255, 0),-1)
                        cv2.putText(frame, "f:1", (45, 375), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 250),5)

                    elif total == 2:
                        cv2.rectangle(frame, (20, 300), (270, 425), (0, 255, 0),-1)
                        cv2.putText(frame, "f:2", (45, 375), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 250),5)
                    elif total==3:
                        cv2.rectangle(frame, (20, 300), (270, 425), (0, 255, 0),-1)
                        cv2.putText(frame, "f:3", (45, 375), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 250),5)
                    elif total==4:
                        cv2.rectangle(frame, (20, 300), (270, 425), (0, 255, 0),-1)
                        cv2.putText(frame, "f:4", (45, 375), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 250),5)
                    elif total==5:
                        cv2.rectangle(frame, (20, 300), (270, 425), (0, 255, 0),-1)
                        cv2.putText(frame, "f:5", (45, 375), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 250),5)
            cv2.imshow("Hand tracking", frame)
            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
