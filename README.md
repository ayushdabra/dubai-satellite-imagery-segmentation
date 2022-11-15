<p>
<a href="https://colab.research.google.com/drive/1xaxxrE8qzTWsHfdOO09NGjrEu4WHQPLy#scrollTo=E8vL3FQbl1_5">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> 

<a href="https://mybinder.org/v2/gh/ayushdabra/dubai-satellite-imagery-segmentation/HEAD">
<img src="https://mybinder.org/badge_logo.svg" alt="launch binder"/>
</a>
</p>

# Dubai Satellite Imagery Semantic Segmentation Using Deep Learning

## Abstract

<p align="justify">
Semantic segmentation is the task of clustering parts of an image together which belong to the same object class. It is a form of pixel-level prediction because each pixel in an image is classified according to a category. In this project, I have performed semantic segmentation on <a href="https://humansintheloop.org/resources/datasets/semantic-segmentation-dataset/">Dubai's Satellite Imagery Dataset</a> by using transfer learning on a InceptionResNetV2 encoder based UNet CNN model. In order to artificially increase the amount of data and avoid overfitting, I preferred using data augmentation on the training set. The model has achieved ~81% dice coefficient and ~86% accuracy on the validation set.
</p>

<!-- ## Libraries Used

- NumPy
- Pandas
- Matplotlib
- IPython
- Open-CV
- Albumentations
- Tensorflow
- Keras
- Keract

The Jupyter Notebook can be accessed from <a href="./dubai-satellite-imagery-segmentation.ipynb">here</a>. -->

## Tech Stack

|<a href="https://www.python.org/"><p align="center"><img width = "auto" height= "auto" src="./readme_images/tech_stack/python.png" /></p></a>|<a href="https://jupyter.org/"><p align="center"><img width = "auto" height= "auto" src="./readme_images/tech_stack/jupyter.png" /></p></a>|<a href="https://ipython.org/"><p align="center"><img width = "auto" height= "auto" src="./readme_images/tech_stack/IPython.png" /></p></a>|<a href="https://numpy.org/"><p align="center"><img width = "auto" height= "auto" src="./readme_images/tech_stack/numpy.png" /></p></a>|<a href="https://pandas.pydata.org/"><p align="center"><img width = "auto" height= "auto" src="./readme_images/tech_stack/pandas.png" /></p></a>|
|---|---|---|---|---|

|<a href="https://matplotlib.org/"><p align="center"><img width = "auto" height= "auto" src="./readme_images/tech_stack/matplotlib.png" /></p></a>|<a href="https://opencv.org/"><p align="center"><img width = "auto" height= "auto" src="./readme_images/tech_stack/opencv.png" /></p></a>|<a href="https://albumentations.ai/"><p align="center"><img width = "auto" height= "auto" src="./readme_images/tech_stack/albumentations.png" /></p></a>|<a href="https://keras.io/"><p align="center"><img width = "auto" height= "auto" src="./readme_images/tech_stack/keras.png" /></p></a>|<a href="https://www.tensorflow.org/"><p align="center"><img width = "auto" height= "auto" src="./readme_images/tech_stack/tensorflow.png" /></p></a>|<a href="https://github.com/philipperemy/keract"><p align="center"><img width = "auto" height= "auto" src="./readme_images/tech_stack/keract.png" /></p></a>|
|---|---|---|---|---|---|

The Jupyter Notebook can be accessed from <a href="https://www.kaggle.com/code/ayushdabra/inceptionresnetv2-unet-81-dice-coeff-86-acc/notebook">here</a>.

The pre-trained model weights can be accessed from <a href="https://www.kaggle.com/code/ayushdabra/inceptionresnetv2-unet-81-dice-coeff-86-acc/data?select=InceptionResNetV2-UNet.h5">here</a>.

## Dataset

<p align="justify">
<a href="https://humansintheloop.org/">Humans in the Loop</a> has published an open access dataset annotated for a joint project with the <a href="https://www.mbrsc.ae/">Mohammed Bin Rashid Space Center</a> in Dubai, the UAE. The dataset consists of aerial imagery of Dubai obtained by MBRSC satellites and annotated with pixel-wise semantic segmentation in 6 classes. The images were segmented by the trainees of the Roia Foundation in Syria.
</p>

<p align="center"><img src="./readme_images/MBRSC-Logo.png" /></p>

### Semantic Annotation

The images are densely labeled and contain the following 6 classes:

| Name       | R   | G   | B   | Color                                                                                              |
| ---------- | --- | --- | --- | -------------------------------------------------------------------------------------------------- |
| Building   | 60  | 16  | 152 | <p align="center"><img width = "30" height= "20" src="./readme_images/label_building.png" /></p>   |
| Land       | 132 | 41  | 246 | <p align="center"><img width = "30" height= "20" src="./readme_images/label_land.png" /></p>       |
| Road       | 110 | 193 | 228 | <p align="center"><img width = "30" height= "20" src="./readme_images/label_road.png" /></p>       |
| Vegetation | 254 | 221 | 58  | <p align="center"><img width = "30" height= "20" src="./readme_images/label_vegetation.png" /></p> |
| Water      | 226 | 169 | 41  | <p align="center"><img width = "30" height= "20" src="./readme_images/label_water.png" /></p>      |
| Unlabeled  | 155 | 155 | 155 | <p align="center"><img width = "30" height= "20" src="./readme_images/label_unlabeled.png" /></p>  |


### Sample Images & Masks

<p align="center"><img width = "95%" height= "auto" src="./readme_images/sample_image_t8_004.jpg" /></p>

<p align="center"><img width = "95%" height= "auto" src="./readme_images/sample_image_t8_003.jpg" /></p>

<p align="center"><img width = "95%" height= "auto" src="./readme_images/sample_image_t4_001.jpg" /></p>

<p align="center"><img width = "95%" height= "auto" src="./readme_images/sample_image_t6_002.jpg" /></p>

## Technical Approach

### Data Augmentation using Albumentations Library

<p align="justify">
<a href="https://albumentations.ai/">Albumentations</a>  is a Python library for fast and flexible image augmentations. Albumentations efficiently implements a rich variety of image transform operations that are optimized for performance, and does so while providing a concise, yet powerful image augmentation interface for different computer vision tasks, including object classification, segmentation, and detection.
</p>

<p align="justify">
There are only 72 images (having different resolutions) in the dataset, out of which I have used 56 images (~78%) for training set and remaining 16 images (~22%) for validation set. It is a very small amount of data, in order to artificially increase the amount of data and avoid overfitting, I preferred using data augmentation. By doing so I have increased the training data upto 9 times. So, the total number of images in the training set is 504 (56+448), and 16 (original) images in the validation set, after data augmentation.
</p>

Data augmentation is done by the following techniques:

- Random Cropping
- Horizontal Flipping
- Vertical Flipping
- Rotation
- Random Brightness & Contrast
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Grid Distortion
- Optical Distortion

Here are some sample augmented images and masks from the dataset:

<p align="center"><img width = "auto" height= "auto" src="./readme_images/aug_image_image_t5_004.jpg" /></p>

<p align="center"><img width = "auto" height= "auto" src="./readme_images/aug_image_image_t3_002.jpg" /></p>

<p align="center"><img width = "auto" height= "auto" src="./readme_images/aug_image_image_t6_007.jpg" /></p>

## InceptionResNetV2 Encoder based UNet Model

### InceptionResNetV2 Architecture

<p align="center"><img width = "90%" height= "auto" src="./readme_images/InceptionResNetV2.jpeg" /></p>

<p align="center">Source: <a href="https://arxiv.org/pdf/1602.07261v2.pdf">https://arxiv.org/pdf/1602.07261v2.pdf</a></p>

### UNet Architecture

<p align="center"><img width = "80%" height= "auto" src="./readme_images/UNet.png" /></p>

<p align="center">Source: <a href="https://arxiv.org/pdf/1505.04597.pdf">https://arxiv.org/pdf/1505.04597.pdf</a></p>

### InceptionResNetV2-UNet Architecture

- InceptionResNetV2 model pre-trained on the ImageNet dataset has been used as an encoder network.

- A decoder network has been extended from the last layer of the pre-trained model, and it is concatenated to the consecutive layers.

A detailed layout of the model is available [here](./readme_images/model.png).

### Hyper-Parameters

1. Batch Size = 16.0
2. Steps per Epoch = 32.0
3. Validation Steps = 4.0
4. Input Shape = (512, 512, 3)
5. Initial Learning Rate = 0.0001 (with Exponential Decay LearningRateScheduler callback)
6. Number of Epochs = 45 (with ModelCheckpoint & EarlyStopping callback)

## Results

### Training Results

|         Model          |               Epochs               | Train Dice Coefficient | Train Accuracy | Train Loss | Val Dice Coefficient | Val Accuracy | Val Loss |
| :--------------------: | :--------------------------------: | :--------------------: | :------------: | :--------: | :------------------: | :----------: | :------: |
| InceptionResNetV2-UNet | 45 (best at 34<sup>th</sup> epoch) |         0.8525         |     0.9152     |   0.2561   |        0.8112        |    0.8573    |  0.4268  |

<p align="center"><img width = "auto" height= "auto" src="./readme_images/model_metrics_plot.png" /></p>

The <a href="./model_training.csv">`model_training.csv`</a> file contain epoch wise training details of the model.

### Visual Results

Predictions on Validation Set Images:

<p align="center"><img width = "95%" height= "auto" src="./predictions/prediction_3.jpg" /></p>
<p align="center"><img width = "95%" height= "auto" src="./predictions/prediction_6.jpg" /></p>
<p align="center"><img width = "95%" height= "auto" src="./predictions/prediction_9.jpg" /></p>
<p align="center"><img width = "95%" height= "auto" src="./predictions/prediction_12.jpg" /></p>
<p align="center"><img width = "95%" height= "auto" src="./predictions/prediction_5.jpg" /></p>
<p align="center"><img width = "95%" height= "auto" src="./predictions/prediction_13.jpg" /></p>
<p align="center"><img width = "95%" height= "auto" src="./predictions/prediction_11.jpg" /></p>

All predictions on the validation set are available in the <a href="./predictions">`predictions`</a> directory.

## Activations (Outputs) Visualization

Activations/Outputs of some layers of the model:

| <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/1_conv2d.png" /><b>conv2d</b></p>         | <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/14_conv2d_4.png" /><b>conv2d_4</b></p>    | <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/18_conv2d_8.png" /><b>conv2d_8</b></p>    | <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/30_conv2d_10.png" /><b>conv2d_10</b></p>  |
| ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/67_conv2d_22.png" /><b>conv2d_22</b></p>  | <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/89_conv2d_28.png" /><b>conv2d_28</b></p>  | <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/96_conv2d_29.png" /><b>conv2d_29</b></p>  | <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/111_conv2d_34.png" /><b>conv2d_34</b></p> |
| <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/118_conv2d_35.png" /><b>conv2d_35</b></p> | <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/133_conv2d_40.png" /><b>conv2d_40</b></p> | <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/220_conv2d_61.png" /><b>conv2d_61</b></p> | <p align="center"><img width = "auto" height= "auto" src="./activations/compressed/243_conv2d_70.png" /><b>conv2d_70</b></p> |

Some more activation maps are available in the <a href="./activations">`activations`</a> directory.

Code for visualizing activations is in the <a href="./get_activations.py">`get_activations.py`</a> file.


## References

1. Dataset- <a href="https://humansintheloop.org/resources/datasets/semantic-segmentation-dataset/">https://humansintheloop.org/resources/datasets/semantic-segmentation-dataset/</a>
2. C. Szegedy, S. Ioffe, V. Vanhoucke, and A. Alemi, “Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,” arXiv.org, 23-Aug-2016. [Online]. Available: https://arxiv.org/abs/1602.07261.
3. O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” arXiv.org, 18-May-2015. [Online]. Available: https://arxiv.org/abs/1505.04597.
