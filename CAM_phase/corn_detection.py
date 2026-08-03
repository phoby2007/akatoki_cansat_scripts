#インスタンス生成は関数呼び出し前に行うこと

import cv2
import numpy as np
from picamera2 import Picamera2

resize_rate = 0.10
detect_threshold = 0.008  # コーン検出の閾値（要調整）
lower_hue = 50  # 下限
upper_hue = 150  # 上限

def cap_to_fog(src, ratio = 0.1):   
    resized = cv2.resize(src, None, fx = ratio, fy = ratio, interpolation = cv2.INTER_NEAREST)
    resized = cv2.resize(resized, src.shape[:2][::-1], 1, 1, cv2.INTER_NEAREST)
    return resized


def corn_detection(cap):

    if cap is None:
        capstatus = False
    else:
        capstatus = True
    capimg = cap.copy()

    capimg = cv2.cvtColor(capimg, cv2.COLOR_BGR2HSV) #RGBtoHSV


    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))# hisutogram equalization
    h, s, v = cv2.split(capimg)
    v = clahe.apply(v)
    capimg = cv2.merge((h, s, v))

    #capimg = cap_to_fog(capimg, resize_rate) # fog effect 必要に応じて
    mask1 = cv2.inRange(capimg, (0, 100, 50), (lower_hue, 255, 255)) #赤色フィルタ
    mask2 = cv2.inRange(capimg, (upper_hue, 100, 50), (180, 255, 255)) #赤色フィルタ
    capimg = cv2.bitwise_or(mask1, mask2)
    shapeh, shapew = capimg.shape

    M = cv2.moments(capimg)
    print(M["m00"]/255, shapeh * shapew * detect_threshold) # デバッグ用 - コーン検出の閾値設定に役立てる

    if M["m00"] / 255 > shapeh * shapew * detect_threshold: # コーン検出閾値設定 - 要調整
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = -1, -1 

    return cx, cy, capstatus

if __name__ == "__main__":
    camera = Picamera2()
    camera.start()

    while True:
        x, y, status = corn_detection(camera.capture_array()) # カメラから画像を取得してコーン検出
        print(f"Cone detected at: ({x}, {y})")
        key = cv2.waitKey(1) #lp stop
        if key == 27:
            break
    camera.stop()
    cv2.destroyAllWindows()