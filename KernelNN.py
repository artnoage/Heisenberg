import torch
import torch.nn as nn
import torch.optim as optim
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Kernel_unintegrated(input_tensor):
    # Assuming the last dimension of the input_tensor is 4, in the order: tau, y, t, r
    h = input_tensor[..., 0]  # Extracts h
    rsquare = input_tensor[..., 1]    # Extracts R^2=\xi^2+\eta^2
    t = input_tensor[..., 2]    # Extracts t
    y = input_tensor[..., 3]   # Extracts y
   
    # Compute the expression
    part1  = (1 / (4 * torch.pi * h)) ** 2
    part2 = torch.where(y == 0, torch.tensor(1.0), (2 * y) / torch.sinh(2 * y))
    part3  = torch.cos((t * y) / 2*h)
    part4a = torch.where(y == 0, torch.tensor(1.0), (2 * y)/torch.tanh(2 * y))
    part4  = torch.exp(-((rsquare) / (4 * h)) * (part4a))

    result = part1 * part2 * part3 * part4
    # Ensure the last dimension is 1 by summing or averaging if needed
    # Here, the last dimension is already 1 due to the operations, so we can return the result directly
    return result

def Kernel(input_tensor,precision=250000,int=8):
    original_tuples_expanded=input_tensor.unsqueeze(1)
    y_values = torch.linspace(0,int, int*precision,dtype=input_tensor.dtype, device=input_tensor.device)
    new_points_expanded = y_values.unsqueeze(0).unsqueeze(2)
    combined_tensor = torch.cat((original_tuples_expanded.expand(-1, int*precision, -1), new_points_expanded.expand(input_tensor.shape[0], -1, -1)), dim=2)
    A=Kernel_unintegrated(combined_tensor)
    A=2*torch.sum(A,dim=1)/precision      
    return A


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
optimizer = optim.AdamW(model.parameters(), lr=0.01,weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)
# Example dataset
epsilon=0.001


# Training loop
epochs =100000
start=time.time()
for epoch in range(epochs):
    
    h_value = (1-epsilon)*torch.rand(50,device=device,dtype=torch.float64)+epsilon
    r_value = 10*torch.rand(50,device=device,dtype=torch.float64)  # r should be positive
    t_value = 10*(torch.rand(50,device=device,dtype=torch.float64)-1/2) # t should be positive
    point=torch.stack([h_value,r_value,t_value],dim=1)
    x_train =point
    y_train = Kernel(point).unsqueeze(1)
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_train)
    # Compute and print loss
    loss = criterion(y_pred, y_train)
    
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:  # Print the loss every 100 epochs
        print("epsilon is ", epsilon, f'Epoch {epoch} | Loss: {loss.item()}')
        #print("time is", time.time()-start)
        scheduler.step()
    # Save the entire model after 10,000 epochs
    if (epoch+1)  % 10000==0:
        example =x_train[0]
        traced_model = torch.jit.trace(model, example)
    # Save the traced model
        torch.jit.save(traced_model, "KernelNN.pth")
        print('Entire model saved every 100000 epochs.')
