import jetson.inference
import jetson.utils
import cv2
import os
import numpy as np

net = jetson.inference.detectNet('ssd-mobilenet-v2',['--model=/home/boikalab/jetson-inference/python/training/detection/ssd/models/amoebaModel_mb2/mb2-ssd-lite.onnx','--input_blob=input_0','--output_cvg=scores','--output_bbox=boxes','--labels=/home/boikalab/jetson-inference/python/training/detection/ssd/models/amoebaModel_mb2/labels.txt','--threshold=0.5'])

fileList = os.listdir('./testDataset/')
for i in fileList:
	img = cv2.imread("./testDataset/" + str(i))
	img = jetson.utils.cudaFromNumpy(img)
	detections=net.Detect(img)
	img = np.ascontiguousarray(img)
	ImgName = str(i)
	cv2.imwrite(ImgName,img)

