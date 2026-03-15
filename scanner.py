import cv2
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    for barcode in decode(img):
        qr_data = barcode.data.decode('utf-8')
        print("QR Detected:", qr_data)

    cv2.imshow("QR Scanner", img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()