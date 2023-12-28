import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Your original neural network class
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

def test_model(model, test_folder, device):
    # Load the test dataset
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    test_dataset = ImageFolder(test_folder, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Set the model to evaluation mode
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace 'path/to/your/model.pth' with the actual path to your saved model
model_path = r'data\animal_classifier_model.pth'
# model_path = r'data\saved_model.pth'
model = SimpleNN(input_size=64*64*3, num_classes=3).to(device) #there are 3 classes
model.load_state_dict(torch.load(model_path))
model.eval()

# Replace 'path/to/test/dataset' with the actual path to your test dataset
test_dataset_folder = r'data\raw-img-test'
train_dataset_folder=r'data\raw-img'

predictions, true_labels = test_model(model, test_dataset_folder, device)

predictions_train, true_labels_train = test_model(model, train_dataset_folder, device)

# Print or use the predictions and true labels as needed
# print("Predictions:", predictions)
# print("True Labels:", true_labels)

count=0
train_count=0

for i in range(len(predictions)):
    if predictions[i] == true_labels[i]:
        count+=1
    else:
        pass

for i in range(len(predictions_train)):
    if predictions_train[i] == true_labels_train[i]:
        train_count+=1
    else:
        pass

print("Accuracy(train): ", 100*train_count/len(predictions_train))
print("Accuracy(test): ", 100*count/len(predictions))
