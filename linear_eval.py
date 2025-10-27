import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import model_1
import torch.nn.functional as F


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)



image_size = 32
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


full_model = model_1()
full_model.to(device)
encoder = torch.nn.Sequential(*list(full_model.resnet.children())[:-1])
encoder.load_state_dict(torch.load("simclr_encoder.pth", map_location=device))
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False
print("Encoder loaded and frozen.")

class LinearClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=10, hidden_dim1=256, hidden_dim2=128, dropout=0.3):
        super().__init__()
        
        # Hidden layers
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim2, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc_out(x)
        return x


classifier = LinearClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-2)

num_epochs = 7 # quick test
for epoch in range(num_epochs):
    classifier.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            embeddings = encoder(imgs)
            embeddings = torch.flatten(embeddings, 1)

        outputs = classifier(embeddings)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Acc: {100*correct/total:.2f}%")


# test set

classifier.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        embeddings = encoder(imgs)
        embeddings = torch.flatten(embeddings, 1)
        outputs = classifier(embeddings)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print(f"Test Accuracy: {100*correct/total:.2f}%")
