## README: Self-Pruning Neural Network (CIFAR-10)

This repository contains a dynamic, self-architecting neural network that learns to prune its own redundant connections during training. By integrating a gated weight mechanism, the model identifies the most critical paths for image classification, allowing for massive model compression without the need for manual post-training pruning.

---

### **Project Overview**
The core objective of this project is to implement a **Self-Pruning Linear Layer** that uses learnable parameters to decide which weights are essential for performance. The network is trained on the **CIFAR-10** dataset—a benchmark of 60,000 $32 \times 32$ color images—to demonstrate the efficiency-accuracy trade-off.

### **Key Features**
* **Custom `PrunableLinear` Layer:** Replaces standard dense layers with a gated version that applies a Sigmoid-transformed "gate score" to every weight.
* **Dynamic Gating:** In the forward pass, weights are multiplied element-wise by their respective gates. Gates near $1.0$ stay active, while those driven toward $0.0$ are effectively deleted.
* **L1 Sparsity Regularization:** A custom loss function penalizes the sum of active gates, creating a "cost of living" for every connection.
* **Automated Compression:** Achieves over **94% sparsity** while maintaining functional classification accuracy.

---

### **How the Self-Pruning Works**
The L1 penalty encourages sparsity by applying a steady, constant pressure that pushes gate values all the way toward zero. Unlike other regularization methods that get weaker as values shrink, L1 forces the model to treat every connection as an expense. If a weight isn’t vital for accuracy, the optimizer "kills it off" to save on the penalty cost, driving the gate below the **$1 \times 10^{-2}$** threshold. This effectively turns your training into an automated search for the leanest, most efficient version of the network.



### **Results Summary**
The model was tested across varying Sparsity Penalty ($\lambda$) values to find the "Efficiency Frontier":

| Lambda ($\lambda$) | Test Accuracy | Sparsity Level (%) |
| :--- | :--- | :--- |
| 0.001 | 51.41% | 94.58% |
| 0.010 | 44.00% | 99.81% |
| 0.025 | 37.30% | 99.93% |

---

### **Getting Started**

#### **Prerequisites**
* Python 3.x
* PyTorch & Torchvision
* Matplotlib & NumPy

#### **Installation**
1. Clone the repository and navigate to the project folder.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install torch torchvision matplotlib numpy
   ```

#### **Usage**
Run the training and evaluation script:
```bash
python3 self_pruning_nn.py
```
Upon completion, the script will output a results table and generate `gate_distribution.png`, showing the bimodal distribution of active vs. pruned weights.
