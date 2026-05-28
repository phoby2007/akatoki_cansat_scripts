import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

print("opened:", cap.isOpened())

while True:
    ret, frame = cap.read()

    print(ret, frame is None)

    if ret:
        print(frame.shape)

    key = cv2.waitKey(1)
    if key == 27:
        break