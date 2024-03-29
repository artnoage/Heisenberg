import torch

def load_and_process_data():
    # List to hold all tensors
    tensors = []

    # Load each tensor and append to the list
    for i in range(1, 6):  # 1 to 10 inclusive
        tensor = torch.load(f'data{i}.pt',map_location="cpu")
        tensors.append(tensor)

    # Concatenate all tensors along the first dimension
    concatenated_tensor = torch.cat(tensors, dim=0)

    # Filter out rows where the last entry is not zero
    filtered_tensor = concatenated_tensor[concatenated_tensor[:, -1] != 0]

    return filtered_tensor

# Call the function and get the processed tensor
processed_tensor = load_and_process_data()
torch.save(processed_tensor,"kerdata.pt")
print(processed_tensor.shape)
