import cv2
import numpy as np

resize_rate = 0.10
detect_threshold = 0.008  # コーン検出の閾値（要調整）
lower_hue = 150  # 下限
upper_hue = 179  # 上限

def cap_to_fog(src, ratio = 0.1):   
    resized = cv2.resize(src, None, fx = ratio, fy = ratio, interpolation = cv2.INTER_NEAREST)
    resized = cv2.resize(resized, src.shape[:2][::-1], 1, 1, cv2.INTER_NEAREST)
    return resized


def corn_detection(cap):

    camstatus, capimg = cap.read()

    capimg = cv2.cvtColor(capimg, cv2.COLOR_BGR2HSV) #RGBtoHSV


    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))# hisutogram equalization
    h, s, v = cv2.split(capimg)
    v = clahe.apply(v)
    capimg = cv2.merge((h, s, v))

    #capimg = cap_to_fog(capimg, resize_rate) # fog effect 必要に応じて
    capimg = cv2.inRange(capimg, (lower_hue, 100, 0), (upper_hue, 255, 255)) #赤色フィルタ
    shapeh, shapew = capimg.shape

    M = cv2.moments(capimg)

    if M["m00"] / 255 > shapeh * shapew * detect_threshold: # コーン検出閾値設定 - 要調整
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = -1, -1 

    return cx, cy

if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    while True:
        x, y = corn_detection(capture)
        print(f"Cone detected at: ({x}, {y})")
        if cv2.waitKey(1) != -1:
            break
    capture.release()
    cv2.destroyAllWindows()