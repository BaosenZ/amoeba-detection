# Amoeba proteus identification and concentration prediction using deep learning

The datasets and codes that are used for the paper "Amoeba proteus identification and concentration prediction using deeping learning" are here. 

## Datasets
Images were collected with a microscope camera (QImaging Go-3, Teledyne Photometrics, USA) and converted into the Pascal Visual Object Challenge (VOC) data type.  196 images were collected for training and validation purposes, and 20 separate images were used to evaluate the trained models. The amoebae in the images were labeled with the open source tool LabelImg in GitHub (Tzutalin, LabelImg, Git code (2015), https://github.com/tzutalin/labelImg). The training and test datasets for Mask R-CNN are in the folder `amoebaDataset/maskrcnn-dataset`. The datasets for MobileNetV2-SSD-Lite are in the folder `amoebaDataset/mb2-dataset`. All of these images in the datasets can be visulized by `dataset_visulization.ipynb`. The inference for test dataset and the different drops are in the `training-results` and `drop-inference` folder. 

## Implementation 
For Mask R-CNN, the requirements are Python 3, Tensorflow(1.15.0) and Keras(2.1.6). All the training and inference were performed on Google Colab, using their GPU. The code and results are in `trainModel.ipynb`. MobileNetV2-SSD-Lite was implemented in a NVIDIA Jetson Nano device. Installed on the device was NVIDIA JetPack SDK (version JetPack 4.4), which includes the latest Linux Driver Package (L4T) with Linux operating system, CUDA-X accelerated libraries, and APIs for deep learning. The GitHub tool Jetson Inference (https://github.com/dusty-nv/jetson-inference) provided tools for the training of the MobileNetV2-SSD-Lite model. The results can be reproduced by following their tutorial. 

## Arduino

Arduino Uno stepper motor is used to control the movement of table to image the whole drops. The code is contained in the folder `arduino/uno_steppermotor`. 

