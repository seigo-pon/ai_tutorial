import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while True:
  _, frame = cap.read()

  frame = cv2.resize(frame, (500, 300))

  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  # hsvを分割
  h = hsv[:, :, 0]
  s = hsv[:, :, 1]
  v = hsv[:, :, 2]
  # 赤を抽出
  img = np.zeros(h.shape, dtype=np.uint8)
  img[((h < 50) | (h > 200)) & (s > 100)] = 255

  cv2.imshow('camera', img)

  k = cv2.waitKey(1)
  if k == 27 or k == 13:
    break

cap.release()
cv2.destroyAllWindows()
