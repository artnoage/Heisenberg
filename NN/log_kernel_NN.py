import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
type=torch.float64


class KernelNN(nn.Module):
    def __init__(self):
        super(KernelNN, self).__init__()
        self.layer1 = nn.Linear(in_features=3, out_features=640)  # Input layer adapted for 3 input features
        self.layer2 = nn.Linear(in_features=640, out_features=640)
        self.layer3 = nn.Linear(in_features=640, out_features=640)
        self.layer4 = nn.Linear(in_features=640, out_features=640)
        self.layer5 = nn.Linear(in_features=640, out_features=640)
        self.layer6 = nn.Linear(in_features=640, out_features=640)
        self.layer7 = nn.Linear(in_features=640, out_features=640)
        self.output_layer = nn.Linear(in_features=640, out_features=1)  # Output layer adapted for 1 output
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

# Initialize the network
model = KernelNN().to(dtype=torch.float64).to(device)
# Define the loss function
criterion = nn.MSELoss()
lr=1e-4
# Define the optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9999)
# Example dataset
a=torch.load("kerdata.pt",map_location=device)
X=a[:,0:3]  # Load your initial dataset
Y=a[:,3].unsqueeze(1)
train_dataset = TensorDataset(X, Y)
batch_size = 2048
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Training loop

for epoch in range(80):
    b=-1
    for batch_idx, (data, targets) in enumerate(train_loader):
        lr=lr*0.9999
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
        b=max(b,loss.item())
    print(b)
    print("learning rate is", lr)
example=X[0]
traced_model = torch.jit.trace(model, example)
torch.jit.save(traced_model, "log_kernel_NN.pth")
print('Entire model saved.')
