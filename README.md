# Federated Learning
This repository contains the implementation of Federated Learning for vision models. 

Federated Learning is a decentralized machine learning approach where models are trained collaboratively across multiple devices while keeping data localized. This ensures privacy and security of data while leveraging the computational power of edge devices.

# Features
- Decentralized Training: Train models across multiple devices without sharing data.
- Privacy Preserving: Data remains on the local device, enhancing privacy and security.
- Scalable: Easily scale the training process to a large number of devices.
- Custom Vision Models: Support for various vision model architectures including CNNs, ResNet, etc.
- Extensive Logging and Monitoring: Keep track of training progress and performance metrics.

# Dataset
- MNIST

# Types of training 
- FedAvg:
  - Each client train a local model on its private data and then sending the model updates (not the data) to a central server. The server averages these updates to produce a global model, which is then sent back to the clients for further training. This process is repeated for multiple rounds until the model converges.
- FedProx:
  - An extension of FedAvg that adds a proximal term to the local objective functions to tackle the heterogeneity in federated learning. This method helps in stabilizing the training process by reducing the impact of non-iid (non-independent and identically distributed) data across clients.
- FedSGD:
  - Clients compute the gradients on their local data and send these gradients to the central server. The server then aggregates these gradients to update the global model. This method requires more frequent communication compared to FedAvg, as gradients are sent in every iteration.

# Installations 
Prerequisites
Python 3.7+
