import sys
import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime
from pyzbar.pyzbar import decode


# -----------------------------
# PATH HANDLING (WORKS FOR EXE)
# -----------------------------
if getattr(sys, 'frozen', False):
    BASE_PATH = os.path.dirname(sys.executable)
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_PATH, "dataset")
ATTENDANCE_FILE = os.path.join(BASE_PATH, "attendance.csv")


# -----------------------------
# LOAD DATASET
# -----------------------------
images = []
classNames = []

for file in os.listdir(DATASET_PATH):

    img_path = os.path.join(DATASET_PATH, file)

    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    images.append(img)
    classNames.append(os.path.splitext(file)[0])

print("Loaded images:", classNames)


# -----------------------------
# FACE ENCODING
# -----------------------------
def findEncodings(images):

    encodeList = []

    for img in images:

        encodes = face_recognition.face_encodings(img)

        if len(encodes) > 0:
            encodeList.append(encodes[0])

    return encodeList


encodeListKnown = findEncodings(images)

print("Encoding Complete")


# -----------------------------
# ATTENDANCE MEMORY
# -----------------------------
markedNames = set()


def markAttendance(name):

    if name in markedNames:
        return

    now = datetime.now()

    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    with open(ATTENDANCE_FILE, "a") as f:
        f.write(f"\n{name},{date},{time}")

    markedNames.add(name)

    print("Attendance marked:", name)


# -----------------------------
# CAMERA START
# -----------------------------
cap = cv2.VideoCapture(0)

qrVerified = False


while True:

    success, img = cap.read()

    if not success:
        print("Camera error")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -------------------------
    # STEP 1: QR SCAN
    # -------------------------
    if not qrVerified:

        for barcode in decode(gray):

            qrData = barcode.data.decode("utf-8")

            print("QR detected:", qrData)

            qrVerified = True

    # -------------------------
    # STEP 2: FACE RECOGNITION
    # -------------------------
    if qrVerified:

        imgSmall = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgSmall)
        encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):

            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)

            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:

                name = classNames[matchIndex].upper()

                markAttendance(name)

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*2, x2*2, y2*2, x1*2

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    img,
                    name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Smart AI Attendance System", img)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()