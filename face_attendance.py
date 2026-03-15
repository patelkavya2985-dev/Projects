import face_recognition
import cv2
import os
import numpy as np
from datetime import datetime

# dataset folder
path = "dataset"

images = []
classNames = []

# load dataset images
for file in os.listdir(path):

    img_path = os.path.join(path, file)

    img = cv2.imread(img_path)

    if img is None:
        print("Image could not load:", file)
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    images.append(img)
    classNames.append(os.path.splitext(file)[0])

print("Loaded images:", classNames)


# encode faces
def findEncodings(images):

    encodeList = []

    for img in images:

        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except:
            print("Face not detected in dataset image")

    return encodeList


encodeListKnown = findEncodings(images)

print("Encoding Complete")


# start webcam
cap = cv2.VideoCapture(0)

# to prevent duplicate attendance
markedNames = set()


# attendance function
def markAttendance(name):

    global markedNames

    if name in markedNames:
        return

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    file_path = os.path.join(os.getcwd(), "attendance.csv")

    with open(file_path, "a") as f:
        f.write(f"\n{name},{date},{time}")

    markedNames.add(name)

    print("Attendance marked for:", name)

while True:

    success, img = cap.read()

    imgSmall = cv2.resize(img, (0,0), None, 0.5, 0.5)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgSmall, model="hog")
    encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):

        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:

            name = classNames[matchIndex].upper()

            markAttendance(name)

            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*2,x2*2,y2*2,x1*2

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.putText(img,name,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,(0,255,0),2)

    cv2.imshow("AI Attendance System", img)

    # press ESC to exit
    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()