import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
#device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
device="cpu"

# Define the neural network with 3 hidden layers
class KernelNN(nn.Module):
    def __init__(self):
        super(KernelNN, self).__init__()
        # Define the first hidden layer
        self.hidden1 = nn.Linear(3, 256)
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
model = KernelNN().to(dtype=torch.float64).to(device)

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.001,weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)
# Example dataset
epsilon=0.01


# Training loop
epochs =100000

for epoch in range(epochs):
    h_value = (1-epsilon)*torch.rand(3).to(device).to(dtype=torch.float64)+epsilon
    r_value =  5*torch.rand(3).to(device).to(dtype=torch.float64)  # r should be positive
    t_value = 10*(torch.rand(3).to(device).to(dtype=torch.float64)-1/2) # t should be positive
    point=torch.cartesian_prod(h_value,r_value,t_value)
    x_train =point
    y_train = Kernel(point)
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_train)

    # Compute and print loss
    loss = criterion(y_pred, y_train)
    
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:  # Print the loss every 100 epochs
        print("epsilon is ", epsilon, f'Epoch {epoch} | Loss: {loss.item()}')
        scheduler.step()
    # Save the entire model after 10,000 epochs
    if (epoch+1)  % 100000==0:
        example =x_train[0]
        traced_model = torch.jit.trace(model, example)
    # Save the traced model
        torch.jit.save(traced_model, "KernelNN.pth")
        print('Entire model saved every 100000 epochs.')

