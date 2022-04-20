# Identifying Cracks on Concrete with Image Classification using Convolutional Neural Network

## 1. Summary
- The project aims to create a convolutional neural network model that can identify cracks on concrete with high accuracy.
- The problem is modelled as a binary classification problem (no cracks/negative and cracks/positive).
- The dataset contains concrete images having cracks. The data is collected from various METU Campus Buildings.The dataset is divided into two as negative and positive crack images for image classification. Each class has 20000 images with a total of 40000 images with 227 x 227 pixels with RGB channels. The dataset is generated from 458 high-resolution images (4032x3024 pixel) with the method proposed by Zhang et al (2016). High-resolution images have variance in terms of surface finish and illumination conditions. No data augmentation in terms of random rotation or flipping is applied. 
- The data set can be obtain [here.](https://data.mendeley.com/datasets/5y9wdsg2zt/2)

## 2.IDE and Framework
- The project is completed with Spyder as the main IDE. 
- The main frameworks used in this project are :
```bash
import numpy as np
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import datetime
import os
```

