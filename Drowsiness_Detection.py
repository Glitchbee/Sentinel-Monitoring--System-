from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt


mixer.init()
mixer.music.load("Alarm.wav")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])#1st vertical distance ----> upper eye lash
    B = distance.euclidean(eye[2], eye[4])#2nd vertical distance ----> lower eye lash
    C = distance.euclidean(eye[0], eye[3])# horizontal distance
    #ear====>eye aspect ratio...........
    ear = (A + B) / (2.0 * C)#when our eyes open, ear will be around 0.35 to 0.9 
    return ear

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distances = abs(top_mean[1] - low_mean[1])
    return distances

thresh = 0.25
frame_check = 20
YAWN_THRESH = 20
flag = 0
alert = False
alert_frames = 0

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
#used to generate graph
ear_val = []
frame_count = 0
alerts = []  # List to store the frames with alerts

# Lists to store the last 5 EAR values and corresponding frame numbers
last_5_ear_values = []
last_5_frame_numbers = []

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        distances = lip_distance(shape)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                if not alert:
                    alert = True
                    alert_frames = 0
                else:
                    alert_frames += 1
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
        else:
            flag = 0
            if alert:
                alert_frames += 1
                if alert_frames >= frame_check:
                    alert = False
                    alerts.append(frame_count)

        if (distances > YAWN_THRESH):
            cv2.putText(frame, "Yawn Alert", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            mixer.music.play()
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distances), (300, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Store EAR values and frame numbers after EAR falls below the threshold
        if alert:
            ear_val.append(ear)
            frame_count += 1
            

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()

# Plot the graph for frames and EAR values after alert
plt.figure()

ear_alert = []
frame_alert = []

for i in range(frame_count):
    ear_alert.append(ear_val[i])
    frame_alert.append(i + 1)

plt.plot(frame_alert, ear_alert, marker='o', linestyle='-', label='EAR Value')
plt.axhline(y=thresh, color='r', linestyle='--', label='Threshold')

# Highlight alert frames with green dashed lines
for i in range(len(frame_alert)):
    if i in alerts:
        plt.axvline(x=frame_alert[i], color='g', linestyle='--', label='Alert')

# Indicate alarm after EAR < Threshold
for i in range(len(frame_alert)):
    if frame_alert[i] in alerts:
        plt.annotate("ALARM", (frame_alert[i], ear_alert[i]), textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=8, color='red')

plt.title("EAR vs. Threshold (Frames with EAR < Threshold)")
plt.ylabel("EAR Value")
plt.xlabel("Frames")
plt.legend()
plt.grid(True)
plt.show()
cv2.destroyAllWindows()
cap.release()



