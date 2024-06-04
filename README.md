# Convolutional Neural Network Experiments for Fashion Image Classification

## Introduction
Convolutional Neural Networks (CNNs) have shown remarkable success in several computer vision tasks such as image classification, object detection, and image segmentation. The Fashion MNIST dataset is a frequently used benchmark for evaluating CNN models in fashion item classification. In this study, we conducted a series of experiments with different model configurations to investigate the performance of CNNs on the Fashion MNIST dataset.

## Methodology
1. **Data Loading and Preprocessing**
   - We load the Fashion MNIST dataset using the torchvision library in PyTorch.
   - Data preprocessing includes normalization and transformation operations to prepare it for the CNN model.

2. **Model Definition**
   - We define a function called `build_cnn()` that constructs CNN models with customizable parameters.
   - Parameters include input channels, hidden layers, activation functions, pooling methods, optimizers, learning rates, and dropout probabilities.

3. **Training and Evaluation**
   - We implement a training and evaluation pipeline using the `train_and_evaluate()` function.
   - Techniques such as stochastic gradient descent and backpropagation are used for model optimization.

### Experimental Setup
- We define six different configurations for the CNN models, varying activation functions, pooling methods, optimizers, learning rates, dropout probabilities, and data augmentation settings.
- Each configuration represents a unique combination of these parameters.

### Results
- We evaluate the trained CNN models on the test dataset and measure their accuracy in classifying fashion items.
- Test accuracy serves as the primary metric for comparing different model configurations.
- Example results:
  - Sigmoid activation with Adam optimizer, max pooling, LR=0.001, dropout=0.5, and augmentation achieved 88% accuracy.
  - Tanh activation with Adam optimizer, max pooling, LR=0.001, dropout=0.5 achieved 85% accuracy.
  - ReLU activation with SGD optimizer, max pooling, LR=0.01, dropout=0 achieved 10% accuracy.

## Conclusion
Our study provides insights into model architecture and hyperparameter tuning for achieving high accuracy in image classification. Researchers and practitioners can use this understanding to develop more efficient CNN models for fashion recognition and similar applications.

# By: Nour Raafat – Nada Abd-alftah
