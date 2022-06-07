<p align="center">
  <img src="ML.png" alt="banner" width="800" />
</p>
  

# Workshop for Computer Vision Using Machine learning: Classification, Detection and Analysis of amoeba

This is a workshop for computer vision using machine learning. Students will learn to classify, detect and analyze microorganisms in optical microscopic images with machine learning model. Students will know how to setup environment for computer vision, load images dataset, data preparation, train custom model, evaluate the model and run prediction on new images with trained model. We set up 3 levels, including image classification using machine learning (level 1), object detection in Google Colab using Mask R-CNN (level 2), object detection in Jetson Nano (level 3). It is better for students to have python basics and data analysis with python. 

## Students prerequisites
The workshop is designed to teach analytical chemistry senior or master students to learn how to classify and detect microorganisms, such as amoeba, in microscopic images. This requires the fundamentals of python basics and data analysis with python. A great workshop for this topic is published in Journal of Chemical Education by Lafuente etc. Students are encouraged to check their tutorial in Github (https://github.com/ML4chemArg/Intro-to-Machine-Learning-in-Chemistry) Some basic knowledge of statistics, analytical chemistry and biology is also required to know the optical microscopic images and analyze the concentration of amoeba. 

## Workshop contents 
### Level 1: Image classification with machine learning
  The `level1_imageClassificationWithML` folder focuses on teaching students the basic image classification and workflow of computer vision using machine learning. Students will learn to train the machine learning model to classify different clothes pictures from Fashion_MNIST dataset in the example notebook. Students will know how to use the trained model to distinguish if one image contains amoeba or not from the dataset we collected in our lab in the assignment and solution notebook. 
  We use Keras as the start of this topic, since Keras is simple, flexible and powerful to start for students. Keras is a deep learning API written in Python, running on top of the machine learning platform Tensorflow. More information about Keras can refer to the doc: https://keras.io/about/. 

### Level 2: Object detection with Mask R-CNN
  The `level2_objectDetectionWithMaskRCNN` folder presents one network that can run object detection for the image. At first, one example from source code (https://github.com/matterport/Mask_RCNN)  is introduced to students to let them know how Mask R-CNN works in the example notebook. Then, the amoeba images are used to train Mask R-CNN model and trained model is used to distinguish the position, number of amoeba in one image. We collected continued images in one water drop and the amoeba in the water drop can be analyzed by the trained Mask R-CNN model. 
  The inference images are used to analyze the amoeba in the water drop and they are provided in the `results` folder. 

  We choose Mask R-CNN as second level is because the API and platform it uses is the same with level 1. Mask R-CNN is implemented on Python 3, Keras, Tensorflow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone. 

### Level 3: Object detection using Jetson Nano
  The `level3_objectDetectionWithJetsonNano` folder mainly introduces students to run object detection in Jetson Nano. Linux command line is the basic technique that students need to know. Then students are encouraged to follow the tutorial in the supporting information (SI) and source code of Jetson Nano (https://github.com/dusty-nv/jetson-inference) to install necessary packages and run object detection. 

<br/>

## Other information:
### Datasets: 
  Images were collected with a microscope camera (QImaging Go-3, Teledyne Photometrics, USA) . The images need to convert from .TIF format to .jpg format. And the python code that can perform this job is provided in `tools` folder. For object detection, the images are labeled and converted into the Pascal Visual Object Challenge (VOC) data type.  The amoebae in the images were labeled with the open source tool LabelImg in GitHub (Tzutalin, LabelImg, Git code (2015), https://github.com/tzutalin/labelImg). We can use code in `tools` folder to visualize the labeled images. All datasets that are used for three levels are provided in `dataset-level1`, `dataset-level2`, `dataset-level3` folder. The Google Colab notebook will load the specific dataset automatically. 

### Arduino: 
  Arduino Uno stepper motor is used to control the movement of table to image the whole drops. The code is contained in the folder `arduino/uno_steppermotor`. For setting up the Arduino, the tutorials are recommended: https://www.youtube.com/playlist?list=PLGs0VKk2DiYw-L-RibttcvK-WBZm8WLEP. 

