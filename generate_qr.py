import qrcode

data = "AI_ATTENDANCE_SYSTEM"

qr = qrcode.make(data)

qr.save("attendance_qr.png")

print("QR Code Generated Successfully")