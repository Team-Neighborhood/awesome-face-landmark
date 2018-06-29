from face_alignment import api
from face_alignment.api import LandmarksType

import cv2
import numpy as np

fa = api.FaceAlignment(LandmarksType._2D, enable_cuda=True, flip_input=False)

vc = cv2.VideoCapture('./data/tkwoo.mp4')

cv2.namedWindow('show', 0)

while True:
    bgr_img = vc.read()[1]
    if bgr_img is None:
        break
    start = cv2.getTickCount()
    faces, preds = fa.get_landmarks(bgr_img)
    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    print ('elapsed time: %.2fms'%time)
    
    for face in faces:
        l,t,r,b,confidence = face
        cv2.rectangle(bgr_img, (l,t), (r,b), (0,255,0), 2)
        text = "face: %.2f" % confidence
        text_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y = t #- 1 if t - 1 > 1 else t + 1
        cv2.rectangle(bgr_img, (l,y-text_size[1]),(l+text_size[0], y+base_line), (0,255,0), -1)
        cv2.putText(bgr_img, text, (l, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    for pred in preds:
        for point in pred:
            cv2.circle(bgr_img, (point[0], point[1]), 2, (0,255,255), -1)
    
    cv2.imshow('show', bgr_img)
    if cv2.waitKey(1) == 27:
        break