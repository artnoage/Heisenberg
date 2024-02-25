import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Define the neural network with 3 hidden layers
class DenseNN(nn.Module):
    def __init__(self):
        super(DenseNN, self).__init__()
        # Define the first hidden layer
        self.hidden1 = nn.Linear(1, 256)
        # Define the second hidden layer
        self.hidden2 = nn.Linear(256, 256)
        # Define the third hidden layer
        self.hidden3 = nn.Linear(256, 256)
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
model = DenseNN().to(dtype=torch.float64).to(device)

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.01,weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)
# Example dataset
epsilon=0.01
s_values=torch.rand(1000,device=device,dtype=torch.float64)
s_values=s_values/max(s_values)
s_values = 2*s_values-1
s_values= torch.clamp(s_values, min= -1 +epsilon, max= 1- epsilon )
values=H(s_values)
# Training loop
epochs =1000000

for epoch in range(epochs):
    x_train = values.unsqueeze(-1)
    y_train = s_values.unsqueeze(-1)
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_train)

    # Compute and print loss
    loss = criterion(y_pred, y_train)
    
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:  # Print the loss every 100 epochs
        epsilon=max(epsilon*0.999,1e-05)
        s_values=torch.rand(5000,device=device,dtype=torch.float64)
        s_values=s_values/max(s_values)
        s_values = 2*s_values-1
        s_values= torch.clamp(s_values, min= -1 +epsilon, max= 1- epsilon )
        s_values=torch.concat([s_values,-s_values])
        values=H(s_values)
        print("epsilon is ", epsilon, f'Epoch {epoch} | Loss: {loss.item()}')
        scheduler.step()
    # Save the entire model after 10,000 epochs
    if (epoch+1)  % 500000==0:
        example =x_train[0]
        traced_model = torch.jit.trace(model, example)
    # Save the traced model
        torch.jit.save(traced_model, "HNN.pth")
        print('Entire model saved after 10000 epochs.')

# Note: The saved model can be loaded with `torch.load('dense_nn_model_complete.pth')`.
# Remember that when loading the model in this way, you do not need to define the model class first.
# However, this approach requires that the code be run where PyTorch is installed.

# Adjust learning rate, the architecture (e.g., number of neurons in hidden layers), or other parameters as needed.
