import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}...")

# Ensure the dataset directory exists
dataset_dir = 'C:/Users/DELL/Desktop/neuroimaging/DATASET'
if not os.path.exists(dataset_dir):
    raise ValueError(f"Dataset directory {dataset_dir} does not exist.")

# Data transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor(),         # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the images
])

# Load the dataset
dataset = datasets.ImageFolder(dataset_dir, transform=transform)

# Check that classes are loaded correctly (yes = tumor, no = no tumor)
print(f"Classes: {dataset.class_to_idx}")

# Split the dataset into training (80%) and validation (20%) sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # First convolutional layer
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 1)  # Output layer for binary classification

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Apply convolution and pooling
        x = x.view(-1, 32 * 32 * 32)  # Flatten the output
        x = torch.relu(self.fc1(x))  # First fully connected layer
        x = torch.sigmoid(self.fc2(x))  # Sigmoid output for binary classification
        return x

# Initialize the model and move it to the device (GPU/CPU)
model = CNNModel().to(device)

# Binary cross-entropy loss and Adam optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        
        labels = labels.float().unsqueeze(1)  # Reshape labels to match output dimensions
        
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item()  # Accumulate loss

    # Print average loss for the epoch
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient tracking
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            labels = labels.float().unsqueeze(1)  # Reshape labels for loss calculation
            loss = criterion(outputs, labels)
            val_loss += loss.item()  # Accumulate validation loss

    # Print average validation loss for the epoch
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

# Save the trained model
model_path = 'C:/Users/DELL/Desktop/neuroimaging/brain_tumor_model.pth'
torch.save(model.state_dict(), model_path)  # Save the model weights
print("Model saved successfully!")
