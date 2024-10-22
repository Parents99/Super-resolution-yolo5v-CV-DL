# Object Detection with Super-Resolution for Small Objects

## Introduction

In the field of object detection, accurately identifying small objects remains a significant challenge. This difficulty arises from the limited and distorted data present in small regions of interest (ROIs). Our project explores a solution to overcome these limitations by utilizing a Super-Resolution (SR) technique to improve the detection of small objects, specifically focusing on weapon detection.

Our approach aims to double the size of feature maps, enabling detection models to better identify smaller objects. The final objective is to develop an optimized detection model that will be deployed on highly compact edge devices, such as the Jetson Nano. These devices are known for their limited computational resources, requiring real-time object detection, which is why we chose YOLOv5 as our model. We focused on detecting two types of weapons—knives and pistols—from surveillance camera footage.

The framework and scripts developed for this project are available in this GitHub repository.

## State of the Art

The need for lightweight frameworks on edge devices for real-time applications, such as weapon detection in video surveillance, has been well-documented in various research papers. One such work, *A Deep-Learning Framework Running on Edge Devices for Handgun and Knife Detection*, highlights the necessity of using efficient object detection models like YOLO for edge devices, which can operate at 3-5 frames per second (FPS).

In our research, we drew inspiration from two main articles:
- *Better to Follow, Follow to Be Better: Towards Precise Supervision of Feature Super-Resolution for Small Object Detection* [2], which presents an innovative approach for feature extraction and amplification using a Generative Adversarial Network (GAN).
- *SuperYOLO: Super Resolution Assisted Object Detection in Multimodal Remote Sensing Imagery* [3], which modifies the YOLOv5 architecture to integrate SR techniques between its Backbone and Head.

We customized these approaches to suit our specific requirements and developed a unique solution for small weapon detection on edge devices.

## Materials and Methods

Our project is built on the YOLOv5 architecture, which consists of three main components:
- **Backbone**: Extracts features from the input image.
- **Neck**: Aggregates features from different stages.
- **Head**: Outputs the final detection predictions.

We integrated a DCGAN (Deep Convolutional GAN) into the YOLOv5 architecture to perform Super-Resolution on feature maps extracted from the Backbone. The SR technique helps amplify small features and improve the model’s ability to detect small weapons such as knives and pistols.

### Training
We used a custom dataset that includes labeled images of people handling knives and pistols. The dataset contains images in different resolutions: 1280x720 and 416x416 (focused on people). We trained YOLOv5 with images resized to 224x224, 320x320, 416x416, and 512x512.

For each training session, we evaluated the model’s performance on key metrics: Precision, Recall, mAP50, mAP50-95, and inference time. The best results were obtained using 416x416 images, but 512x512 yielded the highest mAP scores.

### GAN Architecture
We employed a modified DCGAN to perform Super-Resolution on YOLOv5’s feature maps. The architecture consists of a generator and discriminator, each specifically designed to handle feature maps of varying sizes.

- **Generator**: Amplifies the feature maps extracted by YOLOv5.
- **Discriminator**: Compares the SR feature maps with high-resolution ground-truth features to guide the generator.

Both components were trained using features extracted from images of different sizes. However, the GAN training encountered convergence issues, which will be addressed in future iterations.

## Results

After training YOLOv5 with various image sizes, we observed that larger input images resulted in better mAP scores, albeit at the cost of slightly increased inference time. The best results were achieved with 416x416 images for weapon detection, with a mAP50 of 0.739 for all objects and 0.616 for knives.

For future work, we plan to improve the GAN architecture to achieve better convergence during training and explore the use of alternative Super-Resolution techniques such as Sub-pixel CNNs.

### Key Metrics:
| Image Size | Precision | Recall | mAP50 | mAP50-95 | Inference Time |
|------------|-----------|--------|-------|----------|----------------|
| 224x224    | 0.751     | 0.54   | 0.59  | 0.248    | 0.5 ms         |
| 320x320    | 0.809     | 0.58   | 0.678 | 0.309    | 1.1 ms         |
| 416x416    | 0.8       | 0.67   | 0.739 | 0.339    | 1.5 ms         |
| 512x512    | 0.835     | 0.65   | 0.767 | 0.369    | 2.4 ms         |

## References
1. *A deep-learning framework running on edge devices for handgun and knife detection from indoor video-surveillance cameras*.
2. *Better to Follow, Follow to Be Better: Towards Precise Supervision of Feature Super-Resolution for Small Object Detection*.
3. *SuperYOLO: Super Resolution Assisted Object Detection in Multimodal Remote Sensing Imagery*.

