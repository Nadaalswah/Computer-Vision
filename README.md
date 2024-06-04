# Convolutional Neural Network Experiments for Fashion Image Classification
## Introduction
Convolutional Neural Networks (CNNs) have shown remarkable success in several computer vision tasks such as image classification, object detection, and image segmentation. Fashion MNIST is a frequently used dataset to benchmark CNN models in fashion item classification. We conducted a series of experiments with different model configurations to investigate the performance of CNNs on the Fashion MNIST dataset.
## Methodology
### 1.	Data Loading and Preprocessing
We begin by loading the Fashion MNIST dataset using the torchvision library in PyTorch. The dataset consists of grayscale images of fashion items categorized into 10 classes. We preprocess the data by applying normalization and transformation operations to ensure compatibility with the CNN model.
### 2.	Model Definition
We define a function build_cnn() to construct CNN models with customizable parameters, including input channels, hidden layers, activation functions, pooling methods, optimizers, learning rates, and dropout probabilities. This function allows us to create diverse CNN architectures tailored to specific requirements.
### 3.	Training and Evaluation
We implement a training and evaluation pipeline using the train_and_evaluate() function. This function trains the CNN model on the training dataset for multiple epochs and evaluates its performance on the test dataset. During training, we utilize techniques such as stochastic gradient descent and backpropagation to optimize the model parameters.
3.1. Experimental Setup
Configuration Parameters
We define six different configurations for the CNN models, varying activation functions, pooling methods, optimizers, learning rates, dropout probabilities, and data augmentation settings. Each configuration represents a unique combination of these parameters, allowing us to explore their effects on model performance.
Training Process
For each configuration, we build a CNN model using the build_cnn() function and train it on the training dataset. We conduct experiments over multiple epochs to ensure convergence and monitor training progress.
3.2. Results
Performance Evaluation
We evaluate the trained CNN models on the test dataset and measure their accuracy in classifying fashion items. The test accuracy serves as the primary metric for comparing the performance of different model configurations. We analyze the results to identify trends and patterns in how varying model parameters impact classification accuracy. 
Activation	Optimizer	Pooling	LR	Drop-out	Augmentation	Accuracy
Sigmoid	Adam	Max	0.001	0.5	✅	88%
Tanh	Adam	Max	0.001	0.5	❎	85%
ReLU	Adam	Avg	0.01	0	✅	10%
ReLU	Adam	Avg	0.01	0.5	❎	10%
ReLU	SGD	Max	0.01	0	✅	92%
ReLU	SGD	Max	0.01	0.5	✅	89%
Sigmoid	SGD	Max	0.05	0.5	✅	87%
Sigmoid	SGD	Max	0.05	0	✅	90%
### 4.	Conclusion

Our study evaluated various configurations of CNNs for classifying fashion items on the Fashion MNIST dataset. The results provide insights into model architecture and hyperparameter tuning for achieving high accuracy in image classification. This understanding can help researchers and practitioners develop more efficient CNN models for fashion recognition and similar applications.
By: Nour Raafat – Nada Abd-alftah
