import jetson.inference
import jetson.utils
import cv2
import matplotlib.pyplot as plt
import numpy as np

#net =jetson.inference.detectNet('ssd-mobilenet-v2',threshold=0.5)
net = jetson.inference.detectNet('ssd-mobilenet',['--model=/home/boikalab/jetson-inference/python/training/detection/ssd/models/amoebaModel_20210208mb2/mb2-ssd-lite.onnx','--input_blob=input_0','--output_cvg=scores','--output_bbox=boxes','--labels=/home/boikalab/jetson-inference/python/training/detection/ssd/models/amoebaModel_0207/labels.txt'])
font=cv2.FONT_HERSHEY_SIMPLEX

for i in range(1,40):
	img = cv2.imread("./c1/" + str(i) + ".jpg")
	img = jetson.utils.cudaFromNumpy(img)
	detections=net.Detect(img)
	#for detect in detections:
		#if detect.Confidence < 0.6:
			#del detect
		#	img = cv2.imread("./c2/" + str(i) + ".jpg")
		#	img = jetson.utils.cudaFromNumpy(img)
	img = np.ascontiguousarray(img)

	ImgName = str(i) + ".jpg"
	cv2.imwrite(ImgName,img)

