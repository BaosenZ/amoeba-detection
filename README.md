<p align="center">
  <img src="ML.png" alt="banner" width="800" />
</p>
# Workshop for Computer Vision Using Machine learning: Classification, Detection and Analysis of amoeba

The datasets and codes that are used for the paper "Amoeba proteus identification and concentration prediction using deeping learning" are here. And students will know how to setup environment for computer vision, load images dataset, data preparation, train custom model, evaluate the model and run prediction on new images with trained model. 

Topics included are: 
## Image classification
The "level2_imageClassificationWithML" focus on teaching students the basic image classification and workflow of computer vision using machine learning. Students will learn to train the machine learning model to classify different clothes pictures from Fashion_MNIST dataset in the example notebook. Students will know how to use the trained model to distinguish if one image contains amoeba or not from the dataset we collected in our lab in the assignment and solution notebook. 

## Object detection With Mask R-CNN
The "level3_objectDetectionWithMaskRCNN" presents one network that can run object detection for the image. At first, one example from source code (https://github.com/matterport/Mask_RCNN)  is introduced to students to let them know how Mask R-CNN works in the example notebook. Then, the amoeba images are used to train Mask R-CNN model and trained model is used to distinguish the postion, number of amoeba in one image. We collected continued images in one water drop and the amoeba in the water drop can be analyed by the trained Mask R-CNN model. 

## Object detection using Jetson Nano
The "level4_objectDetectionWithJetsonNano" mainly introduce students to run object detection with Jetson Nano. Linux command line is the basic technique that students need to know. Then students are encouraged to follow the tutorial in the supporting information (SI) and source code of Jetson Nano (https://github.com/dusty-nv/jetson-inference) to install necessary packages and run object detection. 



Other useful information:
## Datasets
Images were collected with a microscope camera (QImaging Go-3, Teledyne Photometrics, USA) . The images need to convert from .TIF format to .jpg format. And the python code that can perform this job is provided in "tools" folder. For object detection, the images are labeled and converted into the Pascal Visual Object Challenge (VOC) data type.  The amoebae in the images were labeled with the open source tool LabelImg in GitHub (Tzutalin, LabelImg, Git code (2015), https://github.com/tzutalin/labelImg). 

## Implementation 
For Mask R-CNN, the requirements are Python 3, Tensorflow(1.15.0) and Keras(2.1.6). All the training and inference were performed on Google Colab, using their GPU. MobileNetV2-SSD-Lite was implemented in a NVIDIA Jetson Nano device. Installed on the device was NVIDIA JetPack SDK (version JetPack 4.4), which includes the latest Linux Driver Package (L4T) with Linux operating system, CUDA-X accelerated libraries, and APIs for deep learning. The GitHub tool Jetson Inference (https://github.com/dusty-nv/jetson-inference) provided tools for the training of the MobileNetV2-SSD-Lite model. 

## Arduino
Arduino Uno stepper motor is used to control the movement of table to image the whole drops. The code is contained in the folder `arduino/uno_steppermotor`. 

