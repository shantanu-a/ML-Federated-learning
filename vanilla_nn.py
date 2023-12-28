import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_ITR=100
EPOCH=5
LR=0.001
NUM_CLASSES=3

# Define your neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=5, device='cuda'):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")


input_size = 64 * 64 * 3  
num_classes = NUM_CLASSES  
learning_rate = LR

# Replace 'path/to/dataset' with the actual path to your dataset
dataset_path = r'data\raw-img'

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Load the dataset
train_dataset = datasets.ImageFolder(dataset_path, transform=transform)

# Create a DataLoader for training
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize your model
model = SimpleNN(input_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=EPOCH, device='cpu')

# Save the trained model
torch.save(model.state_dict(), r'data\\animal_classifier_model.pth')
