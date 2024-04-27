import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18  # Example backbone
from torchvision.models import resnet50  # Example backbone
from torchvision.models import resnet101  # Example backbone
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np

# Define the larger model with a backbone and a head
class LargerModel(nn.Module):
    def __init__(self, backbone):
        super(LargerModel, self).__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.fc.in_features, num_classes)  # Assuming classification task
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return self.softmax(x)

# Define the smaller CNN model
class SmallerModel(nn.Module):
    def __init__(self):
        super(SmallerModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * 7 * 7, num_classes)  # Assuming input size 224x224

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 128 * 7 * 7)
        return nn.functional.softmax(self.fc(x), dim=1)

# Define your dataset class
class CustomDataset(Dataset):
    def __init__(self):
        # Initialize your dataset here
        pass

    def __getitem__(self, index):
        # Return a sample from your dataset
        pass

    def __len__(self):
        # Return the total number of samples in your dataset
        pass

# Hyperparameters
num_classes = 10
in_channels = 3
batch_size = 32
temperature = 5.0
lr = 0.001
num_epochs = 10

# Initialize larger model (backbone can be any pre-trained backbone)
backbone = resnet18(pretrained=True)  # Example backbone
larger_model = LargerModel(backbone)

# Initialize smaller model
smaller_model = SmallerModel()

# Define loss function (Knowledge Distillation Loss)
def kd_loss(outputs, labels, teacher_outputs, temperature):
    soft_targets = nn.functional.softmax(teacher_outputs / temperature, dim=1)
    soft_outputs = nn.functional.log_softmax(outputs / temperature, dim=1)
    return nn.KLDivLoss()(soft_outputs, soft_targets) * temperature * temperature

# Define optimizer
optimizer = optim.Adam(smaller_model.parameters(), lr=lr)

# Initialize dataloaders
train_dataset = CustomDataset()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass through larger model (teacher)
        teacher_outputs = larger_model(inputs)

        # Forward pass through smaller model (student)
        outputs = smaller_model(inputs)

        # Calculate knowledge distillation loss
        loss = kd_loss(outputs, labels, teacher_outputs, temperature)

        # Backpropagation
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Optionally, save the trained smaller model
torch.save(smaller_model.state_dict(), 'smaller_model.pth')
