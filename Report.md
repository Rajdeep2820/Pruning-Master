# Project Report: Self-Pruning Neural Network

## 1. Overview
This project implements a **Self-Pruning Neural Network** that learns to architect its own efficiency during training. By replacing standard linear layers with a gated mechanism, the network identifies and "zeros out" redundant connections.

## 2. Methodology
- **Gated Weights:** Every weight is controlled by a learnable `gate_score`.
- **Sigmoid Transformation:** Gate scores are mapped to a 0.0–1.0 range to act as a mask.
- **L1 Regularization:** An L1 penalty is applied to the sum of all gates, creating "pressure" to prune connections.
- **Thresholding:** Connections with gate values < 0.01 are considered pruned.

## 3. Why an L1 penalty on the sigmoid gates encourages sparsity?
    The L1 penalty encourages sparsity by applying a steady, constant pressure that pushes gate values all the way to zero. Unlike other methods that get weaker as values shrink, L1 forces the model to treat every connection as an expense or "rent". If a weight isn’t vital for accuracy, the optimizer kills it off to save on the penalty cost, driving the gate below your $1 \times 10^{-2}$ threshold. This effectively turns your training into an automated search for the leanest, most efficient version of the network.

## 4. Results Summary (CIFAR-10)

RESULT - 1
Lambda     | Test Accuracy   | Sparsity Level 
--------------------------------------------------
0.001      | 51.37%          | 94.50%         
0.01       | 43.46%          | 99.81%         
0.05       | 34.46%          | 99.96%         
==================================================

RESULT - 2
==================================================
Lambda     | Test Accuracy   | Sparsity Level 
--------------------------------------------------
0.005      | 45.95%          | 99.56%         
0.015      | 40.64%          | 99.88%         
0.025      | 37.30%          | 99.93%         
==================================================

## 5. Analysis
The experiments confirm that CIFAR-10 models are highly over-parameterized. The network successfully maintained over **51% accuracy** while removing **94.58%** of its connections. The distribution of gate values shifted toward zero as the sparsity penalty (λ) increased, effectively demonstrating "automatic" model compression.


## 6. Visual Evidence
The generated `gate_distribution.png` confirms a massive spike at 0.0, representing the pruned parameters, and a small peak of active, essential parameters.

---
**Developer:** Rajdeep Singh Panwar
**Date:** April 2026