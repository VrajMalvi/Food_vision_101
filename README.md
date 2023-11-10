# Food Vision 101 Project

## Overview

Welcome to the Food Vision 101 project, where we aim to build an advanced image classification model capable of recognizing various food items (101 classes). This project is inspired by the DeepFood paper, and we will leverage the Food101 dataset for our training.

In addition to achieving high accuracy, we will explore two key methods to significantly enhance the training speed: Prefetching and Mixed Precision Training.

## Project Structure

- **Project Planning:**
  - Using TensorFlow Datasets to download and explore data
  - Creating preprocessing functions for our data
  - Batching & preparing datasets for modeling (making our datasets run fast)
  - Creating modeling callbacks
  - Setting up mixed precision training
  - Building a feature extraction model
  - Fine-tuning the feature extraction model

- **Helper Functions:**
  - We have included a set of helper functions in the `helper_functions.py` file, available in this repository.

## Data Exploration

We leverage TensorFlow Datasets to download and explore the Food101 dataset. The dataset consists of various food images, and we delve into a sample to understand its structure.

## Preprocessing

We implement preprocessing functions to convert images to the desired format, resize them, and cast them to float32. We preprocess both the training and test datasets and organize them into batches for efficient training.

## Model Development

### Feature Extraction Model

We constructed a feature extraction model utilizing `EfficientNetV2B0` as the base model. The model was compiled with mixed precision settings to enhance its performance. Following the training process, we saved the model and assessed its performance on the test dataset.

Post feature extraction, we attained an accuracy of **72.12%** on the training data and **72.82%** accuracy on the test data.

### Fine-Tuning

We fine-tune the pre-trained model on all layers, adjusting the learning rate dynamically. Early stopping and model checkpoint callbacks are employed to monitor and save the best model during training.

## Model Comparison

We repeat the process with `EfficientNetB4` to compare the performance of different base models. The results, including accuracy and loss metrics, are reported for both models.

## Conclusion

The Food Vision 101 project exemplifies the implementation of advanced techniques in image classification, emphasizing the optimization of speed and accuracy. The utilization of pre-trained models, coupled with fine-tuning strategies, results in robust solutions for food image recognition tasks.

In this project, two base models, namely `EfficientNetB0` from `efficientnet_v2` and `EfficientNetB4` from `efficientnet`, were employed for transfer learning. After fine-tuning the `EfficientNetB0` model, training accuracy of **99.77%** and a test accuracy of **81.10%** were achieved. Similarly, the `EfficientNetB4` model demonstrated atraining accuracy of **99.24%** and a test accuracy of **83.53%**.

These outcomes underscore the effectiveness of leveraging powerful pre-trained models, enabling the development of highly accurate classifiers for food images. The fine-tuning process further enhances the models, tailoring them to the specifics of the Food101 dataset. The saved models are poised for future applications, offering a reliable solution for various food image recognition tasks.
