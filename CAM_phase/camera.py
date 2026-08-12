import time
import cv2
import numpy as np
from picamera2 import Picamera2

class Camera:
    def __init__(self):
        try:
            self.picam2 = Picamera2()
            self.picam2.configure(self.picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.picam2 = None
        self.hsv_min1 = np.array([0, 180, 180])
        self.hsv_max1 = np.array([11, 255, 255])
        self.hsv_min2 = np.array([169, 180, 180])
        self.hsv_max2 = np.array([179, 255, 255])
    def start(self):
        if self.picam2 is not None:
            self.picam2.start()
        else:
            print("Camera not initialized. Cannot start.")
    def capture_image(self):
        if self.picam2 is not None:
            return self.picam2.capture_array()
        else:
            print("Camera not initialized. Cannot capture image.")
            return None

    def release(self):
        if self.picam2 is not None:
            self.picam2.stop()
    def detect_cone(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(image, self.hsv_min1, self.hsv_max1)
        mask2 = cv2.inRange(image, self.hsv_min2, self.hsv_max2)
        image = cv2.bitwise_or(mask1, mask2)
        return image
    def histogram_equalization(self, image):
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        h, s, v = cv2.split(image)
        v = clahe.apply(v)
        return cv2.merge((h, s, v))
    def get_cone_position(self, image):
        #モルフォロジー変換
        kernel = np.ones((5,5), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        #輪郭抽出
        counts, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if counts:
            largest_contour = max(counts, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            print(f"Largest contour area: {area}")  # デバッグ用 - 面積を表示
            if area < 500:
                return None, None, image
            else:
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return cx, cy, image
        return None, None, image
if __name__ == "__main__":
    camera = Camera()
    camera.start()
    while True:
        try:
            cap = camera.capture_image()
            capcp = cap.copy()
            if cap is None:
                print("Failed to capture image. Exiting.")
                break
            cap = camera.histogram_equalization(cap)
            cap = camera.detect_cone(cap)
            cx, cy, cap = camera.get_cone_position(cap)
            if cx is not None and cy is not None:
                capcp = cv2.circle(capcp, (cx, cy), 10, (0, 255, 0), -1)
            cv2.imshow("Detected Cone", capcp)
            try:
                cv2.imshow("Mask", cap)
            except cv2.error as e:
                print(f"Error displaying image: {e}")
            cv2.waitKey(1)  # Adjust the wait time as needed
            print(cx, cy)
        except Exception as e:
            print(f"Error during processing: {e}")
    camera.release()
    cv2.destroyAllWindows()