import cv2
import os
import copy
import shutil
from sklearn.externals import joblib

# モデル
clf = joblib.load('./fish.pkl')

img_last = None
th = 3
count = 0
frame_count = 0

save_dir = './best_fish'
if os.path.exists(save_dir):
  shutil.rmtree(save_dir)
os.makedirs(save_dir)

cap = cv2.VideoCapture('./fish.mp4')
while True:
  success, frame = cap.read()
  if not success:
    break

  frame = cv2.resize(frame, (640, 360))
  frame2 = copy.copy(frame)
  frame_count += 1

  # 二値化
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  gray = cv2.GaussianBlur(gray, (15, 15), 0)
  img_b = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

  if not img_last is None:
    # 差分
    frame_diff = cv2.absdiff(img_last, img_b)
    cnts = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # 予測
    fish_count = 0
    for pt in cnts:
      x, y, w, h = cv2.boundingRect(pt)
      if w < 100 or w > 500:
        continue

      img = frame[y:y+h, x:x+w]
      img2 = cv2.resize(img, (64, 32))
      img_data = img2.reshape(-1, )
      y_pred = clf.predict([img_data])
      if y_pred[0] == 1:
        fish_count += 1
        cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

      if fish_count > th:
        cv2.imwrite(os.path.join(save_dir, f'{count}.jpg'), frame)
        count += 1

  cv2.imshow('fish', frame2)
  if cv2.waitKey(1) == 13:
    break

  img_last = img_b

cap.release()
cv2.destroyAllWindows()