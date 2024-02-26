import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from utils import *
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
#device="cpu"
# Define the neural network with 3 hidden layers
class SoftLInfinityNormLoss(nn.Module):
    def __init__(self, beta=10):
        """
        Initializes the SoftLInfinityNormLoss module.
        
        Args:
        - beta (float): Controls the smoothness of the approximation. Higher values make the function more like max.
        """
        super(SoftLInfinityNormLoss, self).__init__()
        self.beta = beta

    def forward(self, predictions, targets):
        """
        Forward pass of the loss function.
        
        Args:
        - predictions (torch.Tensor): The predicted values.
        - targets (torch.Tensor): The ground truth values.
        
        Returns:
        - torch.Tensor: The computed soft L-infinity norm loss.
        """
        # Compute the absolute difference
        errors = torch.abs(predictions - targets)
        
        # Compute the soft L-infinity norm using Log-Sum-Exp for a smooth approximation of max
        soft_linf = (1 / self.beta) * torch.log(torch.mean(torch.exp(self.beta * errors)))
        
        return soft_linf


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class dccNN(nn.Module):
    def __init__(self):
        super(dccNN, self).__init__()
        self.layer1 = nn.Linear(in_features=3, out_features=8*64)  # Input layer adapted for 3 input features
        self.layer2 = nn.Linear(in_features=8*64, out_features=8*64)
        self.layer3 = nn.Linear(in_features=8*64, out_features=8*32)
        self.layer4 = nn.Linear(in_features=8*32, out_features=8*16)
        self.output_layer = nn.Linear(in_features=8*16, out_features=1)  # Output layer adapted for 1 output
        self.swish = Swish()
        self.elu=nn.ELU()
    
    def forward(self, x):
        x = self.elu(self.layer1(x))
        x = self.elu(self.layer2(x))
        x = self.elu(self.layer3(x))
        x = self.swish(self.layer4(x))
        x = self.output_layer(x)  # No softmax needed since output is 1 (assuming regression or binary classification)
        return x

def get_new_data():
    s_values = torch.rand(5*10**6,device=device,dtype=torch.float64)
    v_values = (2*torch.pi)*(2*torch.rand(5*10**6,device=device,dtype=torch.float64)-1)    
    theta_values = 2*torch.pi*torch.rand(5*10**6,device=device,dtype=torch.float64)  # 45 degrees in radians
    r_values = 10*torch.rand(5*10**6,device=device,dtype=torch.float64) # r should be positive
    spherical_coords = torch.stack([s_values, v_values, theta_values, r_values],dim=1)
    cartesian_coords = spherical_to_cartesian(spherical_coords)
    labels=(spherical_coords [:,0]*spherical_coords [:,3]).unsqueeze(-1)
    return cartesian_coords, labels

X, y = get_new_data()  # Load your initial dataset


train_dataset = TensorDataset(X, y)
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


model = dccNN().to(dtype=torch.float64).to(device)
#model=torch.jit.load('dccNN.pth',map_location=device).to(dtype=torch.float64)
# Define the loss function
criterion = SoftLInfinityNormLoss()

# Define the optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99998)

# Training loop
epochs =10

for epoch in range(epochs):
   
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Forward pass: Compute predicted outputs by passing inputs to the model
        outputs = model(data)
    # Calculate the loss
        loss = criterion(outputs, targets)
    
        # Clear the gradients of all optimized variables
        optimizer.zero_grad()
    
        # Backward pass: Compute gradient of the loss with respect to model parameters
        loss.backward()
    
        # Perform a single optimization step (parameter update)
        optimizer.step()
    print(loss.item())
    X, y = get_new_data()  # Load your initial dataset
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

example =data[0]
traced_model = torch.jit.trace(model, example)
torch.jit.save(traced_model, "dccNN.pth")
print('Entire model saved after 10 epochs.')   