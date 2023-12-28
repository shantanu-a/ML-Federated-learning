import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import copy
import os
from tqdm import tqdm
import random

NUM_ITR=5
EPOCH=5
LR=0.001
NUM_CLASSES=3

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

# Function to train the model on a dataset and save parameters internally
def train_and_save_parameters(dataset_folder, model, optimizer, device, saved_parameters):
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    dataset = ImageFolder(dataset_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)  # Decreased batch size

    for epoch in range(5):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Clear unnecessary variables to save memory
            del inputs, labels, outputs, loss

    saved_parameters.append(copy.deepcopy(model.state_dict()))


# Function to average parameters and run the model
def average_parameters_and_run(model, saved_parameters, dataset_folders, device, num_iterations, save_path):
    original_model = copy.deepcopy(model)
    
    for iteration in tqdm(range(num_iterations)):
        train_list = random.sample(dataset_folders, 10)
        for dataset_folder in train_list:
            # Train on each dataset and save parameters internally
            saved_parameters.clear()
            train_and_save_parameters(dataset_folder, model, optim.Adam(model.parameters(), lr=0.001), device, saved_parameters)

            # Average parameters
            for param, original_param in zip(model.parameters(), original_model.parameters()):
                param.data += original_param.data / len(train_list)

    # Save the final averaged parameters to a file
    torch.save(model.state_dict(), save_path)

    # Run the model with averaged parameters
    


# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN(input_size=128*128*3, num_classes=3).to(device)

saved_parameters = []

save_path = r'data\\saved_model.pth'  # Replace with the desired save path

folders=os.listdir('data/split-img')

dataset_folders=[]


for i in folders:
    i = "data\\split-img\\" + i

    dataset_folders.append(i)

num_iterations = 5  # Replace with the desired number of iterations

average_parameters_and_run(model, saved_parameters, dataset_folders, device, num_iterations, save_path)
