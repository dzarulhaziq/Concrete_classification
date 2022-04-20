# Identifying Cracks on Concrete with Image Classification using Convolutional Neural Network

## 1. Summary
- The project aims to create a convolutional neural network model that can identify cracks on concrete with high accuracy.
- The problem is modelled as a binary classification problem (no cracks/negative and cracks/positive).
- The dataset contains concrete images having cracks. The data is collected from various METU Campus Buildings.The dataset is divided into two as negative and positive crack images for image classification. Each class has 20000 images with a total of 40000 images with 227 x 227 pixels with RGB channels. The dataset is generated from 458 high-resolution images (4032x3024 pixel) with the method proposed by Zhang et al (2016). High-resolution images have variance in terms of surface finish and illumination conditions. No data augmentation in terms of random rotation or flipping is applied. 
- The data set can be obtain [here.](https://data.mendeley.com/datasets/5y9wdsg2zt/2)

## 2.IDE and Framework
- The project is completed with Jupyter Notebook as the main IDE. 
- The main frameworks used in this project are :
```bash
import numpy as np
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import datetime
import os
```

## 3.Methodology

The documentation on the official TensorFlow website served as a model for this project's methods. You can find a link to it [here.](https://www.tensorflow.org/tutorials/images/transfer_learning)

### 3.1 Data Pipeline

The image data is loaded, along with the labels associated. The data is first divided into a train-validation set with a 70:30 ratio. The validation data is then split into two portions, using an 80:20 ratio, to obtain some test data. The overall split ratio for train-validation-test is 70:24:6. Because the data amount and variance are already sufficient, no data augmentation is used.

### 3.2 Model Pipeline
- The input layer is set up to accept coloured images with a size of 160x160 pixels. The final form will be (160,160,3).
- The deep learning model for this project is built via transfer learning. To begin, a preprocessing layer is established, which modifies the pixel values of incoming images to a range of -1 to 1. This layer acts as a feature scaler and is necessary for the transfer learning model to produce the correct signals.
- A MobileNet v2 pretrained model is employed for the feature extractor. The model is included in the TensorFlow Keras package and is pre-trained with ImageNet parameters. It's also locked, so it won't update while the model is being trained.
- To generate softmax signals, a global average pooling and dense layer classifier is used. To determine the anticipated class, softmax signals are used.

The model is depicted in simplified form in the diagram below.

![image](https://user-images.githubusercontent.com/103733709/164209435-e14be77d-b607-4250-89e7-32494bf2598a.png)




