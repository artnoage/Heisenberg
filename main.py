import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *

# Set a fixed value for v, theta, r
v = torch.tensor(2.0)
theta = torch.tensor(torch.pi / 4)  # 45 degrees in radians
r = torch.tensor(1.0)  # r should be positive

# Randomly select 100 points for time (s)
s_values = torch.rand(100) * 10  # 100 random points for s, scaled for a wider range

# Create a tensor of shape [100, 4] for spherical coordinates with the same v, theta, r
spherical_coords = torch.stack((s_values, torch.full_like(s_values, v), 
                                torch.full_like(s_values, theta), torch.full_like(s_values, r)), dim=1)

# Convert to Cartesian coordinates
cartesian_coords = spherical_to_cartesian_higher_dim(spherical_coords)

# Extract x, y, z coordinates for plotting
x = cartesian_coords[:, 0].numpy()
y = cartesian_coords[:, 1].numpy()
z = cartesian_coords[:, 2].numpy()

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for the 3D points
ax.scatter(x, y, z)

# Set labels
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')

# Set title
ax.set_title('3D Scatter Plot of Cartesian Coordinates')

# Show plot
plt.show()