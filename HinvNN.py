import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# Define the neural network with 3 hidden layers
class DenseNN(nn.Module):
    def __init__(self):
        super(DenseNN, self).__init__()
        self.fc1 = nn.Linear(1,128)  # Input layer to first hidden layer
        self.fc2 = nn.Linear(128, 256) # First to second hidden layer
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128) # Second to third hidden layer
        self.fc5 = nn.Linear(128, 1)   # Third hidden layer to output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Initialize the network
model = DenseNN().to(dtype=torch.float64).to(device)

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Example dataset
s_values = torch.rand(10000).to(dtype=torch.float64).to(device)
s_values= 0.999995*s_values/s_values.max()
values=H(s_values)
x_train = values.unsqueeze(-1)
y_train = s_values.unsqueeze(-1)
# Training loop
epochs = 1000000
for epoch in range(epochs):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_train)

    # Compute and print loss
    loss = criterion(y_pred, y_train)
    if epoch % 100 == 0:  # Print the loss every 100 epochs
        print(f'Epoch {epoch} | Loss: {loss.item()}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save the entire model after 10,000 epochs
    if epoch == epochs - 1:
        example =x_train[0].float()
        model.float()
        traced_model = torch.jit.trace(model, example)
    # Save the traced model
        torch.jit.save(traced_model, "H_inv.pth")
        print('Entire model saved after 10000 epochs.')

# Note: The saved model can be loaded with `torch.load('dense_nn_model_complete.pth')`.
# Remember that when loading the model in this way, you do not need to define the model class first.
# However, this approach requires that the code be run where PyTorch is installed.

# Adjust learning rate, the architecture (e.g., number of neurons in hidden layers), or other parameters as needed.
