
GPU Accelerated EuroSAT Image Classification Pipeline
Overview

This project implements an end-to-end GPU accelerated remote sensing image classification pipeline using PyTorch, CUDA, and custom GPU kernels.

The objective was to build a production-style ML training system that combines:

GPU accelerated data preprocessing
CNN based image classification
Performance benchmarking
Model evaluation
GPU profiling and observability

The pipeline classifies satellite imagery from the EuroSAT dataset into 10 land-cover categories.

System Architecture
                 EuroSAT Dataset
                       |
                       |
                PyTorch DataLoader
                       |
                       |
              CPU → GPU Data Transfer
                       |
                       |
          Custom CUDA Image Processing Kernel
                       |
                       |
                CNN Feature Extractor
                       |
                       |
              Classification Head
                       |
                       |
             Prediction + Evaluation
Project Flow
1. Data Pipeline

The system begins by loading the EuroSAT remote sensing dataset.

Dataset characteristics:

27,000 satellite images
RGB imagery
64x64 resolution
10 land-cover classes

Images are split into:

70% Training

15% Validation

15% Testing

The training split is used for optimization, validation tracks model generalization, and the test set provides final unbiased evaluation.

2. GPU Accelerated Preprocessing

A custom C++/CUDA extension accelerates image normalization.

Traditional pipeline:

CPU preprocessing

       |

GPU training

Optimized pipeline:

GPU memory

       |

CUDA kernel execution

       |

Normalized tensors

       |

Model training

The CUDA implementation is benchmarked against an OpenMP CPU implementation to measure preprocessing speedup.

Metrics collected:

CPU preprocessing latency
CUDA preprocessing latency
GPU acceleration factor
3. Neural Network Architecture

The model uses a lightweight ResNet-inspired CNN.

Architecture:

Input Image
     |
Conv Layers
     |
Residual Blocks
     |
Adaptive Average Pooling
     |
Fully Connected Layer
     |
10 Class Prediction

Design choices:

Residual Connections

Improve gradient flow and allow deeper feature extraction.

Batch Normalization

Improves training stability.

Adaptive Pooling

Allows spatial feature aggregation independent of image size.

Dropout

Reduces overfitting.

4. Training Pipeline

The training system includes:

AdamW optimizer
Cosine learning-rate scheduler
Automatic Mixed Precision (AMP)
CUDA accelerated preprocessing
Gradient backpropagation
Checkpoint saving

Training produces:

models/

    best_model.pth

    last_checkpoint.pth
5. Performance Monitoring

The pipeline integrates:

TensorBoard

Tracks:

Training loss
Validation loss
Accuracy
GPU memory usage
Gradient statistics

Run:

tensorboard --logdir runs
PyTorch Profiler

Captures:

CUDA kernel execution
CPU operations
Memory usage
Operator latency

Profiler traces can be inspected directly through TensorBoard.

6. Evaluation Pipeline

After training, the model is evaluated on unseen test data.

Metrics:

Accuracy
Precision
Recall
Macro F1 score
Confusion matrix

Example:

Forest:

95% correctly classified


River:

88% correctly classified


Highway:

84% correctly classified

The evaluation pipeline also exports:

metrics/

evaluation_metrics.json
7. Engineering Challenges
Challenge: CPU Bottleneck

Problem:

Image preprocessing can limit GPU utilization.

Solution:

Implemented CUDA preprocessing kernels to move compute-heavy operations closer to GPU execution.

Challenge: Measuring GPU Performance

Problem:

CUDA execution is asynchronous.

Solution:

Used CUDA events and synchronization barriers for accurate kernel timing.

Challenge: Reproducibility

Problem:

Training results vary between runs.

Solution:

Implemented deterministic seeds, checkpoint restoration, and fixed dataset splits.

Interview Summary

"I built a GPU accelerated satellite image classification pipeline where I integrated a custom CUDA preprocessing kernel into a PyTorch training workflow. I benchmarked CPU OpenMP preprocessing against CUDA execution, trained a ResNet-inspired CNN on EuroSAT imagery, and built a complete evaluation and profiling system using TensorBoard and PyTorch Profiler. The goal was not only model accuracy but understanding the full ML systems pipeline from data movement, GPU execution, training optimization, and production observability."
