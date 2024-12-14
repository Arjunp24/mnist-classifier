import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# Set random seed for reproducibility
torch.manual_seed(42)

# Define data augmentation and transforms
train_transform = transforms.Compose([
    transforms.RandomRotation((-5, 5), fill=(1,)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 10, kernel_size=1, padding=0)

        self.conv4 = nn.Conv2d(10, 16, kernel_size=3, padding=0)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv5 = nn.Conv2d(16, 32, kernel_size=3, padding=0)
        self.bn4 = nn.BatchNorm2d(32)

        self.avgpool = nn.AvgPool2d(6)

        self.conv6 = nn.Conv2d(32, 16, kernel_size=3, padding=0)
        self.bn5 = nn.BatchNorm2d(16)

        self.conv7 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(16)

        self.conv8 = nn.Conv2d(16, 10, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(F.relu(self.bn2(self.conv2(x))))
        x = F.max_pool2d(self.conv3(x), 2)
        x = self.dropout(F.relu(self.bn3(self.conv4(x))))
        x = self.dropout(F.relu(self.bn4(self.conv5(x))))
        x = self.dropout(F.relu(self.bn5(self.conv6(x))))
        x = self.dropout(F.relu(self.bn6(self.conv7(x))))
        x = self.conv8(self.avgpool(x))
        x = x.view(-1, 10)
        return x

# Initialize model, optimizer, and loss function
model = SmallCNN()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=2e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max',
    factor=0.1,
    patience=2,
    verbose=True,
    min_lr=1e-5
)
criterion = nn.CrossEntropyLoss()

# Print model summary and parameter count
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {count_parameters(model)}")

# Training loop
epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
final_accuracy = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    # Training
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        progress_bar.set_postfix({
            'loss': f'{train_loss/(batch_idx+1):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    # Testing
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Update learning rate
    scheduler.step(accuracy)
    
    # Update final accuracy
    final_accuracy = accuracy

# Save final model and accuracy
torch.save(model.state_dict(), 'final_model.pth')
with open('test_accuracy.txt', 'w') as f:
    f.write(f"{final_accuracy}")

print("\nTraining completed!") 