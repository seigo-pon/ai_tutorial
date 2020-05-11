import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fps = 20.0
size = (640, 360)
writer = cv2.VideoWriter('test.m4v', fmt, fps, size)

while True:
  _, frame = cap.read()

  frame = cv2.resize(frame, size)
  writer.write(frame)
  cv2.imshow('camera', frame)

  k = cv2.waitKey(1)
  if k == 27 or k == 13:
    break

writer.release()
cap.release()
cv2.destroyAllWindows()
