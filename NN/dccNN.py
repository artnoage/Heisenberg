import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device="cpu"
# Define the neural network with 3 hidden layers

class dccNN(nn.Module):
    def __init__(self):
        super(dccNN, self).__init__()
        # Define the first hidden layer
        self.hidden1 = nn.Linear(3, 256)
        # Define the second hidden layer
        self.hidden2 = nn.Linear(256,512)
        # Define the third hidden layer
        self.hidden3 = nn.Linear(512, 256)
        # Define the fourth hidden layer
        self.hidden4 = nn.Linear(256, 256)
        # Define the output layer
        self.output = nn.Linear(256, 1)
        # Define the PReLU activation function with learnable parameters
        self.activation1 = nn.PReLU()
        self.activation2 = nn.PReLU()
        self.activation3 = nn.PReLU()
        self.activation4 = nn.PReLU()

    def forward(self, x):
        # Pass the input through the first hidden layer, then activation
        x = self.activation1(self.hidden1(x))
        # Pass through the second hidden layer, then activation
        x = self.activation2(self.hidden2(x))
        # Pass through the third hidden layer, then activation
        x = self.activation3(self.hidden3(x))
        # Pass through the fourth hidden layer, then activation
        x = self.activation4(self.hidden4(x))
        # Pass through the output layer
        x = self.output(x)
        return x

# Initialize the network
model = dccNN().to(dtype=torch.float64).to(device)

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.001,weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9995)
# Example dataset

# Training loop
epochs =5000000

for epoch in range(epochs):
    s_values = torch.rand(200,device=device,dtype=torch.float64)
    v_values = (2*torch.pi)*(2*torch.rand(200,device=device,dtype=torch.float64)-1)    
    theta_values = 2*torch.pi*torch.rand(200,device=device,dtype=torch.float64)  # 45 degrees in radians
    r_values = 10*torch.rand(200,device=device,dtype=torch.float64) # r should be positive
    spherical_coords = torch.stack([s_values, v_values, theta_values, r_values],dim=1)
    cartesian_coords = spherical_to_cartesian(spherical_coords)

    x_train = cartesian_coords
    y_train = (spherical_coords [:,0]*spherical_coords [:,3]).unsqueeze(-1)
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_train)

    # Compute and print loss
    loss = criterion(y_pred, y_train)
    
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:  # Print the loss every 100 epochs
        print(f'Epoch {epoch} | Loss: {loss.item()}')
        learning_rate = optimizer.param_groups[0]['lr']
        scheduler.step()
        print(f'Current learning rate: {learning_rate}')
    # Save the entire model after 10,000 epochs
    if (epoch+1)  % 100000==0:
        example =x_train[0]
        traced_model = torch.jit.trace(model, example)
    # Save the traced model
        torch.jit.save(traced_model, "dccNN.pth")
        print('Entire model saved after 10000 epochs.')

# Note: The saved model can be loaded with `torch.load('dense_nn_model_complete.pth')`.
# Remember that when loading the model in this way, you do not need to define the model class first.
# However, this approach requires that the code be run where PyTorch is installed.

# Adjust learning rate, the architecture (e.g., number of neurons in hidden layers), or other parameters as needed.