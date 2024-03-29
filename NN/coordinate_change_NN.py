import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
type=torch.float64


class ChCoorNN(nn.Module):
    def __init__(self):
        super(ChCoorNN, self).__init__()
        self.layer1 = nn.Linear(in_features=3, out_features=640)  # Input layer adapted for 3 input features
        self.layer2 = nn.Linear(in_features=640, out_features=640)
        self.layer3 = nn.Linear(in_features=640, out_features=640)
        self.layer4 = nn.Linear(in_features=640, out_features=640)
        self.layer5 = nn.Linear(in_features=640, out_features=640)
        self.layer6 = nn.Linear(in_features=640, out_features=640)
        self.layer7 = nn.Linear(in_features=640, out_features=640)
        self.output_layer = nn.Linear(in_features=640, out_features=3)  # Output layer adapted for 1 output
        self.activation1 = nn.PReLU()
        self.activation2 = nn.PReLU()
        self.activation3 = nn.PReLU()
        self.activation4 = nn.PReLU()
        self.activation5 = nn.PReLU()
        self.activation6 = nn.PReLU()
        self.activation7 = nn.PReLU()
        
    def forward(self, x):
        x = self.activation1(self.layer1(x))
        x = self.activation2(self.layer2(x))+x
        x = self.activation3(self.layer3(x))+x
        x = self.activation4(self.layer4(x))+x
        x = self.activation5(self.layer5(x))+x
        x = self.activation6(self.layer6(x))+x
        x = self.activation7(self.layer7(x))+x
        x = self.output_layer(x) # No softmax needed since output is 1 (assuming regression or binary classification)
        return x

epsilon=0.01
def get_new_data(epsilon):
    s_values = torch.ones(5*10**6,device=device,dtype=type)
    v_values = (2*torch.pi-epsilon)*(2*torch.rand(5*10**6,device=device,dtype=type)-1)    
    theta_values = (1-2*epsilon)*2*torch.pi*torch.rand(5*10**6,device=device,dtype=type)  # 45 degrees in radians
    r_values = 5*torch.rand(5*10**6,device=device,dtype=type)+epsilon # r should be positive
    spherical_coords = torch.stack([s_values, v_values, theta_values, r_values],dim=1)
    spherical_coords2= torch.stack([v_values, theta_values, r_values],dim=1)
    cartesian_coords = spherical_to_cartesian(spherical_coords)
    labels=spherical_coords2
    return cartesian_coords, labels

X, Y = get_new_data(epsilon)  # Load your initial dataset


train_dataset = TensorDataset(X, Y)
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Define the loss function
criterion = nn.MSELoss()

ler=1e-4
# Define the optimizer
model = ChCoorNN().to(dtype=type).to(device)
optimizer = optim.AdamW(model.parameters(), lr=ler)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1-1e-6)

# Training loop
epochs = 500
for epoch in range(epochs):
    a=-1
    for batch_idx, (data, targets) in enumerate(train_loader):
        ler=ler*(1-1e-6)
        # Forward pass: Compute predicted outputs by passing inputs to the model
        outputs = model(data)
    # Calculate the loss
        loss = criterion(outputs, targets)
    
        # Clear the gradients of all optimized variables
        optimizer.zero_grad()
    
        # Backward pass: Compute gradient of the loss with respect to model parameters
        loss.backward()
        