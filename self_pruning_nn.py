import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Define how to process the images (Step 1 requirement)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Set up DataLoaders [cite: 95]
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Load already downloaded data [cite: 107]
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

# This command triggers the actual download 
# trainset = torchvision.datasets.CIFAR10(
    #root='./data', 
    #train=True,
    #download=True, 
    #transform=transform
#)
# print("Dataset downloaded and verified successfully!")#
# DataLoaders are required for the training loop 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

print("Data loaders initialized from local storage.")

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        # 1. Standard weight and bias parameters 
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # 2. Gate scores: same shape as weights, registered as model parameter
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialization
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.zeros_(self.bias)
        # Initialize gate_scores so gates start near 0.0 (active)
        nn.init.constant_(self.gate_scores, 0.0) 

    def forward(self, x):
        # 3. Transform gate_scores to gates between 0 and 1 using Sigmoid
        gates = torch.sigmoid(self.gate_scores)
        
        # 4. Calculate pruned weights via element-wise multiplication 
        pruned_weights = self.weight * gates
        
        # 5. Standard linear operation using pruned_weights and bias 
        return F.linear(x, pruned_weights, self.bias)
        
class SelfPruningNet(nn.Module):
    def __init__(self):
        super(SelfPruningNet, self).__init__()
        # Defining three prunable layers 
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        # Flatten: (Batch, 3, 32, 32) -> (Batch, 3072)
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Final output layer (no activation here, CrossEntropy handles it)
        x = self.fc3(x)
        return x

# --- PART 2: Sparsity Regularization Loss
def get_sparsity_loss(model):
    """Calculates the L1 norm of all gate values across all PrunableLinear layers."""
    sparsity_loss = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            # The case study requires L1 norm of the sigmoid gates [cite: 89, 90]
            gates = torch.sigmoid(module.gate_scores)
            sparsity_loss += torch.sum(gates)
    return sparsity_loss

# --- PART 3: Training and Evaluation [cite: 94] ---

def train_and_evaluate(lambd, epochs=5):
    # Device selection (handles Apple Silicon/MPS, CUDA, or CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # Standard optimizer
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining with Lambda: {lambd}")
    
    for epoch in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            class_loss = criterion(outputs, labels)
            
            # Total Loss = ClassificationLoss + Lambda * SparsityLoss
            s_loss = get_sparsity_loss(model)
            total_loss = class_loss + lambd * s_loss
            
            total_loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    all_gates = []
    
    with torch.no_grad():
        # Calculate Accuracy
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate Sparsity Level (%) 
        total_weights = 0
        pruned_weights = 0
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores).cpu().numpy()
                all_gates.extend(gates.flatten())
                total_weights += gates.size
                # Threshold of 1e-2 as per specification 
                pruned_weights += np.sum(gates < 1e-2)
                
    accuracy = 100 * correct / total
    sparsity_level = 100 * pruned_weights / total_weights
    
    return accuracy, sparsity_level, np.array(all_gates)

# --- RUN EXPERIMENTS & REPORTING 
lambdas = [0.001, 0.01, 0.05] # Increased from your previous run
epochs = 10 # Recommended to allow sparsity to settle
results = []
best_gates = None
best_acc = 0

for l in lambdas:
    accuracy, sparsity, gates = train_and_evaluate(l, epochs=epochs)
    results.append({
        "Lambda": l,
        "Test Accuracy": f"{accuracy:.2f}%",
        "Sparsity Level": f"{sparsity:.2f}%"
    })

    # CRITICAL: Capture the gates from the best model for the plot
    if accuracy > best_acc:
        best_acc = accuracy
        best_gates = gates

# Print Summary Table [cite: 116]
print("\n" + "="*50)
print(f"{'Lambda':<10} | {'Test Accuracy':<15} | {'Sparsity Level':<15}")
print("-" * 50)
for res in results:
    print(f"{res['Lambda']:<10} | {res['Test Accuracy']:<15} | {res['Sparsity Level']:<15}")
print("="*50)

# Generate Distribution Plot for the best model [cite: 117, 118]
if best_gates is not None:
    plt.figure(figsize=(10, 6))
    plt.hist(best_gates, bins=50, color='skyblue', edgecolor='black') # Added best_gates here
    plt.title(f"Gate Value Distribution (Best Model Accuracy: {best_acc:.2f}%)")
    plt.xlabel("Gate Value (Sigmoid Output)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plt.savefig("gate_distribution.png")
