import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        # 1. Standard weight and bias parameters [cite: 71]
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # 2. Gate scores: same shape as weights, registered as model parameter [cite: 72, 73, 74]
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialization
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.zeros_(self.bias)
        # Initialize gate_scores so gates start near 1.0 (active)
        nn.init.constant_(self.gate_scores, 1.0) 

    def forward(self, x):
        # 3. Transform gate_scores to gates between 0 and 1 using Sigmoid [cite: 77, 78]
        gates = torch.sigmoid(self.gate_scores)
        
        # 4. Calculate pruned weights via element-wise multiplication [cite: 79]
        pruned_weights = self.weight * gates
        
        # 5. Standard linear operation using pruned_weights and bias [cite: 80, 81]
        return F.linear(x, pruned_weights, self.bias)
    
# Define how to process the images (Step 1 requirement)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# This command triggers the actual download 
trainset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True, 
    transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True, 
    transform=transform
)

print("Dataset downloaded and verified successfully!")