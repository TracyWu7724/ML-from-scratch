import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Simple CNN with 3 convolutional layers, 2 fully connected layers, and dropout for regularization.
    Input → [Conv → ReLU → Pool] → [Conv → ReLU → Pool] → [Conv → ReLU → Pool] → Flatten → FC → ReLU → Dropout → FC → Output
    input: 32x32x3 images
    output: class scores for 10 classes
    -------------
    Formulas for output dimensions:
    
    Convolutional Layer:
    H_out = (H_in + 2*padding - kernel_size) / stride + 1
    W_out = (W_in + 2*padding - kernel_size) / stride + 1
    C_out = number of filter = out_channels

    Pooling Layer:
    H_out = (H_in - kernel_size) / stride + 1
    W_out = (W_in - kernel_size) / stride + 1
    C_out = C_in

    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Assuming input images are 32x32x3 (e.g., CIFAR-10)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1) # 32x32x3 -> 32x32x16
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) # 16x16x16 -> 16x16x32
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # 8x8x32 -> 8x8x64
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # Reduces each dimension by a factor of 2
         # After 3 pooling layers, the image size is reduced from 32x32 to 4x4
         # Therefore, the input to the first fully connected layer is 64 * 4 * 4 = 1024
        self.fc1 = nn.Linear(64 * 4 * 4, 512)  
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 16x16x16
        x = self.pool(F.relu(self.conv2(x)))  # 32x8x8
        x = self.pool(F.relu(self.conv3(x)))  # 64x4x4
        x = x.view(-1, 64 * 4 * 4)            # Flatten the tensor 
        x = F.relu(self.fc1(x)) # fc1: Linear(input_dim = 1024, output_dim = 512)
        x = self.dropout(x) 
        x = self.fc2(x) # fc2: Linear(input_dim = 512, output_dim = num_classes)
        return x # Output logits for each class

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, train_loader, num_epochs):
        self.model.to(self.device)
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')