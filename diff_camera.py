import cv2
import numpy as np

img_last = None
cap = cv2.VideoCapture(0)
while True:
  _, frame = cap.read()

  frame = cv2.resize(frame, (500, 300))

  # 二値化
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  gray = cv2.GaussianBlur(gray, (9, 9), 0)
  img_b = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]

  if img_last is None:
    img_last = img_b
    continue

  # 差分
  frame_diff = cv2.absdiff(img_last, img_b)
  cnts = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

  # 差分を描画
  for pt in cnts:
    x, y, w, h = cv2.boundingRect(pt)
    if w < 30:
      continue
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
  img_last = img_b

  cv2.imshow('camera', frame)

  k = cv2.waitKey(1)
  if k == 27 or k == 13:
    break

cap.release()
cv2.destroyAllWindows()
