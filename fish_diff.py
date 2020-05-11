import cv2
import os
import shutil

img_last = None
idx = 0

save_dir = './fish'
if os.path.exists(save_dir):
  shutil.rmtree(save_dir)
os.makedirs(save_dir)

cap = cv2.VideoCapture('./fish.mp4')
while True:
  success, frame = cap.read()
  if not success:
    break

  frame = cv2.resize(frame, (640, 360))

  # 二値化
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  gray = cv2.GaussianBlur(gray, (15, 15), 0)
  img_b = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

  if not img_last is None:
    # 差分
    frame_diff = cv2.absdiff(img_last, img_b)
    cnts = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # 差分を保存
    for pt in cnts:
      x, y, w, h = cv2.boundingRect(pt)
      if w < 100 or w > 500:
        continue

      img = frame[y:y+h, x:x+w]
      cv2.imwrite(os.path.join(save_dir, f'{idx}.jpg'), img)
      idx += 1

  img_last = img_b

cap.release()
