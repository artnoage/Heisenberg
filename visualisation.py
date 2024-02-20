import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_geod_3d(cartesian_coords):
# Extract x, y, z coordinates for plotting
    x = cartesian_coords[:, 0].numpy()
    y = cartesian_coords[:, 1].numpy()
    z = cartesian_coords[:, 2]
    z1= torch.zeros_like(z)
    z=z.numpy() 
    z1=z1.numpy()
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for the 3D points
    ax.scatter(x, y, z, color='r', label='Curve')
    ax.scatter(x, y, z1, color='b', label='2d Projection')

    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

# Set labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

# Set title
    ax.set_title('3D Scatter Plot of Cartesian Coordinates')

# Show plot
    plt.show()

def draw_geod_2d(cartesian_coords):
    x = cartesian_coords[:, 0].numpy()
    y = cartesian_coords[:, 1].numpy()
    fig, ax = plt.subplots()

# Plotting the two sets of data points with different colors
    ax.scatter(x, y, color='r', label='Set 1')


# Setting the axes to be centered at (0, 0)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Moving ticks to the bottom and left side
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Labeling the axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    # Adding a legend
    ax.legend()

    plt.show()