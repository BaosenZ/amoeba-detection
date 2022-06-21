**Jetson Nano Tutorials**



**Object Detection in Jetson Nano Introduction**

This document is the detailed tutorials for Level 3 (Object detection in Jetson Nano). At first, students need to learn Linux command line basics and know how to run the command line in Jetson Nano device. Then if students get a new Jetson Nano device, they will take time to install Jetson Inference project on it. After building up the project, students need to download the project in github (https://github.com/BaosenZ/amoeba-detection ) and save the Jetson Nano dataset to correct folder, and start to train by command. After training, we can evaluate the performance and run inference for the drop or test dataset by command. The results are shown in the Figure S6-10. For the bonus part, students follow the same procedure to difference two types of amoeba with the command line with the dataset of Proteus and Fowleri that we provided.



**The Linux command line basics**

One of the early operating systems is called Unix. At that time, there was no mouse, no fancy graphics, not even any choice of color, but programmer produce text as output and accept text as an input via Unix mainframe. Linux is a sort-of-descendent of Unix. The core part of Linux is designed to behave similarly to a Unix System. The Linux command line, like Unix mainframe, is a text interface to your computer. Although running Windows graphical programs is more common in the recent days, learning text interface of Linux command line is still meaningful, since it can let us understand more underlying logic of the computer. This will also increase our efficiency dealing with computer programs after being familiar with Linux command line. 

The command can finish tasks like changing directories (cd), listing the contents (ls), moving files (mv), and so on. Here, we provide the common used command line as a cheat sheet for students to have a quick reference (Table 1). The Figure 1 is the example of Linux command line terminal. More Linux command line learning resources can check the video in YouTube (https://www.youtube.com/watch?v=MfpvdC-QrgY&list=PLGs0VKk2DiYxP-ElZ7-QXIERFFPkOuP4_&index=2&t=2117s&ab_channel=PaulMcWhorter.), the Ubuntu command line tutorials (https://ubuntu.com/tutorials/command-line-for-beginners#9-conclusion.), and Linux command line books (http://linuxcommand.org/tlcl.php.). 

 

Table 1. Common used Linux command line: 

| Linux command line | Explanation                             |
| ------------------ | --------------------------------------- |
| $ pwd              | Show current directory                  |
| $ mkdir dir        | Make directory called dir               |
| $ rm –r dir        | Delete directory called dir             |
| $ cd dir           | Change directory to dir                 |
| $ cd ..            | Go up a directory                       |
| $ ls               | List files                              |
| $ touch file1      | Create a file called file1              |
| $ file file1       | Get type of file1                       |
| $ cp file1 file2   | Copy file1 to file2                     |
| $ mv file1 file2   | Move file1 to file2                     |
| $ rm file1         | Delete file1                            |
| $ ctrl-c           | Stop current command                    |
| $ clear            | Clear all command line                  |
| Tab                | Fill out the command line automatically |

 

![image-20220621133535601](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220621133535601.png)



**Example: Amoeba Detection**

The example for level 3 is amoeba detection. Students can go to the Github website (https://github.com/BaosenZ/amoeba-detection ) to download the zip of folders, then drag the dataset to the correct directory. Before training, we need to do mounting swap to free extra memory (https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-transfer-learning.md#mounting-swap ). After installation of the Jetson Inference project, students can run the command line in Box 2 to train and evaluate the model. For the train_ssd.py, it involves many arguments, type $ python3 train_ssd.py help to see all arguments. Box 3 is the python code to run the inference. 



```
$ git clone https://github.com/BaosenZ/amoeba-detection.git
```





```
$ cd jetson-inference/python/training/detection/ssd

 

\# Start to train the model 

$ python3 train_ssd.py --data=data/amoebaDataset/trainingDataset --model-dir=models/amoebaModel_mb2 --dataset-type=voc --net=mb2-ssd-lite --pretrained-ssd=models/mb2-ssd-lite-mp-0_686.pth --epochs=60



\# Convert the model to ONNX 

$ python3 onnx_export.py --model-dir=models/amoebaModel_mb2 --net=mb2-ssd-lite

 

\# Evaluate the model 

$ python3 eval_ssd.py –trained_model=models/amoebaModel_mb2/<.pth> --net=mb2-ssd-lite --eval_dir=evals_amoebaEval --label_file=models/amoebaModel/label.txt --dataset=data/amoebaDataset/testDataset
```



```import jetson.inference
import jetson.utils

import cv2

import matplotlib.pyplot as plt

import numpy as np

 

net = jetson.inference.detectNet('ssd-mobilenet',['--model=/home/boikalab/jetson-inference/python/training/detection/ssd/models/amoebaModel_mb2/mb2-ssd-lite.onnx','--input_blob=input_0','--output_cvg=scores','--output_bbox=boxes','--labels=/home/boikalab/jetson-inference/python/training/detection/ssd/models/amoebaModel_mb2/labels.txt', threshold=0.5])



for i in range(1,21): 

​    \# Put the original images to ‘c1’ folder. 

​    img = cv2.imread("./c1/" + str(i) + ".jpg")

​    img = jetson.utils.cudaFromNumpy(img)

​    detections=net.Detect(img)

​    img = np.ascontiguousarray(img)

​    ImgName = str(i) + ".jpg"

​    cv2.imwrite(ImgName,img)


```



**Learning resources about Jetson Nano:** 

  This is the original source of Jetson inference. They provide detailed tutorials of how to use the Jetson Nano or other NVIDIA product to train or do inference: [**https://github.com/dusty-nv/jetson-inference**](https://github.com/dusty-nv/jetson-inference)

  This is a Jetson Nano tutorial. It is friendly to the beginner: https://www.youtube.com/playlist?list=PLGs0VKk2DiYxP-ElZ7-QXIERFFPkOuP4_&ab_channel=PaulMcWhorter 

  Ubuntu is one kind of Linux system and they provide command line tutorial for beginers: https://ubuntu.com/tutorials/command-line-for-beginners#9-conclusion 

  Command line books: http://linuxcommand.org/tlcl.php 