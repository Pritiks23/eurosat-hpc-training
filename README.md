# GPU Accelerated EuroSAT Image Classification Pipeline

## Overview

This project implements an end-to-end GPU accelerated remote sensing image classification pipeline using **PyTorch, CUDA, and custom GPU kernels**.

The system trains a convolutional neural network on the **EuroSAT satellite imagery dataset** while integrating GPU-accelerated preprocessing, mixed precision training, checkpointing, profiling, and evaluation workflows.

The goal of this project was to build a production-style ML training pipeline that demonstrates the complete lifecycle of a GPU-based deep learning system:

- Data ingestion and preprocessing
- CUDA accelerated computation
- GPU optimized training
- Model evaluation
- Performance profiling
- Experiment tracking


---

# Project Highlights

## Results

| Metric | Value |
|---|---:|
| Dataset | EuroSAT |
| Images | 27,000 satellite images |
| Classes | 10 land-cover categories |
| Image Size | 64x64 RGB |
| GPU | NVIDIA RTX 3090 |
| Framework | PyTorch |
| Mixed Precision | AMP FP16 |
| Best Validation Accuracy | **94.1%** |
| Training Epochs | 20 |


---

# System Architecture

## Project Pipeline

```text
                          [ EuroSAT Dataset ]
                                 |
                           [ dataset.py ]
                                 |
               [ Train / Validation / Test Split ]
                                 |
                     [ PyTorch DataLoader ]
                                 |
         +-----------------------+-----------------------+
         |                                               |
         v                                               v
  [ benchmark.py ]                               [ train.py ]
         |                                               |
         |                                +--------------+--------------+
         |                                | GPU Training Pipeline       |
+--------+-----------------------+         |----------------------------|
| CPU OpenMP Benchmark           |         | CNN Forward Pass           |
| CUDA Accelerated Processing    |         | Loss Calculation           |
+--------------------------------+         | Backpropagation + AMP      |
                                           | Model Checkpointing        |
                                           +--------------+-------------+
                                                          |
                                          +---------------+---------------+
                                          |                               |
                                          v                               v
                                   [ evaluate.py ]                [ visualize.py ]
                                          |                               |
                          +---------------+---------------+   +-----------+-----------+
                          | Accuracy / F1 Metrics         |   | Prediction Visualization |
                          | Confusion Matrix              |   | Confidence Scores         |
                          +-------------------------------+   +---------------------------+
```

---

# Dataset Pipeline

The pipeline uses the **EuroSAT remote sensing dataset**, containing satellite imagery across 10 land-cover classes.

Dataset characteristics:

- 27,000 satellite images
- RGB imagery
- 64x64 resolution
- 10 classification categories


Dataset split:
Training:
70%

Validation:
15%

Testing:
15%


The training split is used for optimization, validation monitors generalization, and the test set provides final model evaluation.

---

# GPU Accelerated Preprocessing

A custom C++/CUDA extension was implemented to accelerate image preprocessing operations.

The preprocessing pipeline compares:

## CPU Path
Image Tensor

  |

OpenMP CPU Normalization

  |

Training Pipeline



## GPU Path


Image Tensor

  |

CUDA Kernel Execution

  |

GPU Normalized Tensor

  |

Model Training



The benchmark measures:

- CPU preprocessing latency
- CUDA preprocessing latency
- GPU acceleration factor


Example RTX 3090 benchmark:
CPU preprocessing:
0.0584 seconds

GPU preprocessing:
0.0374 seconds

GPU Speedup:
1.56x

---

# Neural Network Architecture

The model is a lightweight CNN optimized for satellite image classification.

Architecture:
Input Image
|
|
Convolution Layer
|
Batch Normalization
|
ReLU Activation
|
Max Pooling
|
Convolution Layer
|
Batch Normalization
|
ReLU Activation
|
Adaptive Average Pooling
|
Fully Connected Classifier
|
10 Class Prediction



Design choices:

## Batch Normalization

Improves training stability and convergence.

## Adaptive Average Pooling

Creates fixed-size feature representations independent of spatial dimensions.

## Dropout

Reduces overfitting during training.

---

# Training Pipeline

The training system includes:

- PyTorch CUDA execution
- Automatic Mixed Precision (AMP)
- AdamW optimizer
- Cosine learning-rate scheduler
- GPU accelerated preprocessing
- Validation monitoring
- Model checkpointing


Training workflow:

Load Dataset

  |

Transfer Batch To GPU

  |

CUDA Accelerated Processing

  |

Forward Pass

  |

Loss Calculation

  |

AMP Backpropagation

  |

Optimizer Update

  |

Checkpoint Save



Generated artifacts:


best_model.pth

final_model_weights.pth

training_metrics.json

checkpoints/


---

# Training Performance

The model was trained on an NVIDIA RTX 3090 GPU.

Example training progression:


Epoch 1:
Validation Accuracy: 73.6%

Epoch 10:
Validation Accuracy: 90.8%

Epoch 20:
Validation Accuracy: 94.1%



Final result:


Best Validation Accuracy:

94.12%


---

# Performance Monitoring

The project integrates TensorBoard and PyTorch profiling.

## TensorBoard

Tracks:

- Training loss
- Validation accuracy
- Gradient statistics
- Training metrics


Launch:

```bash
tensorboard --logdir runs
PyTorch Profiler

Captures:

CUDA kernel execution
CPU operations
Memory utilization
Operator latency

Profiler traces can be visualized through TensorBoard.

Evaluation Pipeline

After training, the model is evaluated on unseen test data.

Evaluation includes:

Accuracy
Precision
Recall
Macro F1 score
Confusion matrix

The evaluation pipeline validates:

Overall classification performance
Per-class accuracy
Model confidence
Prediction behavior

Example prediction workflow:

Satellite Image

      |

Trained CNN

      |

Class Prediction

      |

Confidence Score
Engineering Challenges
Challenge 1: GPU Utilization Bottlenecks
Problem

Data preprocessing can prevent GPUs from reaching full utilization.

Solution

Implemented custom CUDA preprocessing kernels to move computational work closer to GPU execution.

Challenge 2: Accurate GPU Benchmarking
Problem

CUDA operations execute asynchronously, making naive timing inaccurate.

Solution

Used GPU synchronization and timing measurements to correctly compare CPU and CUDA execution paths.

Challenge 3: Training Stability
Problem

Large GPU training workloads can experience unstable convergence.

Solution

Implemented:

AdamW optimization
Learning-rate scheduling
Batch normalization
AMP mixed precision
Validation monitoring


How To Run

Install dependencies:

pip install -r requirements.txt

Train:

python train.py

Evaluate:

python evaluate.py

Visualize predictions:

python visualize.py



I built a GPU accelerated satellite image classification pipeline using PyTorch, CUDA, and custom GPU kernels. The system processes the EuroSAT remote sensing dataset through a complete ML workflow including CUDA preprocessing, mixed precision training, checkpointing, profiling, and evaluation. I trained a CNN on 27,000 satellite images using an NVIDIA RTX 3090 and achieved 94.1% validation accuracy. The project focused not only on model performance, but also on understanding the complete ML systems stack: data movement, GPU execution, training optimization, and observability.
