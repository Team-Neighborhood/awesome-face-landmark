from __future__ import print_function
import numpy as np
import cv2
# import dlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--with_draw', help='do draw?', default='True')
args = parser.parse_args()

landmarknet = cv2.dnn.readNetFromCaffe('./models/vanilla_deploy.prototxt', './models/_iter_150000.caffemodel')
net = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt', './models/res10_300x300_ssd_iter_140000.caffemodel')

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def preprocess(img):
    ### analysis
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(3):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray_img.mean() < 130:
            img = adjust_gamma(img, 1.5)
        else:
            break
    return img

vc = cv2.VideoCapture('./data/tkwoo.mp4')

cv2.namedWindow('show', 0)

idx = 0
while True:
    bgr_img = vc.read()[1]
    if bgr_img is None:
        break

    start = cv2.getTickCount()
    
    bgr_img = preprocess(bgr_img)

    ### detection
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    (h, w) = bgr_img.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(bgr_img, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    ### bbox
    list_bboxes = []
    list_confidence = []
    # list_dlib_rect = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.6:
                continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (l, t, r, b) = box.astype("int") # l t r b
        
        original_vertical_length = b-t
        t = int(t + (original_vertical_length)*0.15)
        b = int(b - (original_vertical_length)*0.05)

        margin = ((b-t) - (r-l))//2
        l = l - margin if (b-t-r+l)%2 == 0 else l - margin - 1
        r = r + margin
        refined_box = [l, t, r, b]
        list_bboxes.append(refined_box)
        list_confidence.append(confidence)

    ### landmark
    LM_caffe_param = 40
    list_CLM = [] # caffe landmark list
    for bbox in list_bboxes:
        l,t,r,b = bbox
        roi = bgr_img[t:b+1, l:r+1]
        res = cv2.resize(roi, (LM_caffe_param, LM_caffe_param)).astype(np.float32)
        
        normalized_roi = res/127.5-1.0
        
        blob = cv2.dnn.blobFromImage(normalized_roi, 1.0, (LM_caffe_param, LM_caffe_param), None)
        landmarknet.setInput(blob)
        caffe_landmark = landmarknet.forward()
        
        for landmark in caffe_landmark:
            LM = []
            for i in range(len(landmark)//2):
                x = landmark[2*i] * (r-l) + l
                y = landmark[2*i+1] * (b-t) + t
                LM.append((int(x),int(y)))
            list_CLM.append(LM)

    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    print ('elapsed time: %.2fms'%time)

    ### draw rectangle bbox
    if args.with_draw == 'True':
        for bbox, confidence in zip(list_bboxes, list_confidence):
            l, t, r, b = bbox
            
            cv2.rectangle(bgr_img, (l, t), (r, b),
                (0, 255, 0), 2)
            text = "face: %.2f" % confidence
            text_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y = t #- 1 if t - 1 > 1 else t + 1
            cv2.rectangle(bgr_img, (l,y-text_size[1]),(l+text_size[0], y+base_line), (0,255,0), -1)
            cv2.putText(bgr_img, text, (l, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        for landmark in list_CLM:
            for idx, point in enumerate(landmark):
                cv2.circle(bgr_img, point, 2, (0, 255, 255), -1)

        
        cv2.imshow('show', bgr_img)
        
        if cv2.waitKey(1) == 27:
            break
        idx += 1