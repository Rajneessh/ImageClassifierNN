# CIFAR-10 Image Classification with CNN

This repository contains a Convolutional Neural Network (CNN) model implemented in TensorFlow/Keras for classifying images in the CIFAR-10 dataset.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Results](#results)
- [References](#references)
- [License](#license)

## Introduction

This project demonstrates how to build, train, and evaluate a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The model is capable of classifying images into one of the following categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

## Model Architecture

The CNN model is built using the following layers:
- Two convolutional layers with ReLU activation and max pooling
- Dropout layers for regularization
- Flatten layer to convert the 2D matrix to a 1D vector
- Two dense (fully connected) layers with ReLU and softmax activation

## Installation

To get started, clone the repository and install the required dependencies:

```sh
git clone https://github.com/your-username/cifar10-cnn.git
cd cifar10-cnn
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Results

The trained model achieves an accuracy of approximately 70% on the CIFAR-10 test dataset. Detailed classification reports is available in the preview section of the code. It is a basic CNN model for learning purposes so thats why it has accuracy on lower end.

## References

### Dataset
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
  Official page for the CIFAR-10 dataset, providing details on its structure and usage.

### TensorFlow/Keras Documentation
- [TensorFlow Documentation](https://www.tensorflow.org/)
  Official documentation for TensorFlow, the deep learning framework used in this project.
- [Keras Documentation](https://keras.io/)
  Official documentation for Keras, the high-level neural networks API used as part of TensorFlow.

### Additional Resources
- [Convolutional Neural Networks (CNNs)](https://en.wikipedia.org/wiki/Convolutional_neural_network)
  Wikipedia article on Convolutional Neural Networks (CNNs), explaining their architecture and applications.
- [Machine Learning Mastery](https://machinelearningmastery.com/)
  Provides tutorials and resources on machine learning and deep learning topics.
- [Stack Overflow](https://stackoverflow.com/)
  Community-driven platform for programming and technical questions, useful for troubleshooting and learning.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
