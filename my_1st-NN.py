import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# Load and preprocess the data
transform = transforms.Compose([
    transforms.ToTensor(),                      
    transforms.Normalize((0.5,), (0.5,))
])

train_data = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()          
        self.fc1 = nn.Linear(28*28, 128)      
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)       

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x                              

# Set device, model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)

criterion = nn.CrossEntropyLoss()             # For classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()                             # Training mode
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()                 
        outputs = model(imgs)                 
        loss = criterion(outputs, labels)     
        loss.backward()                       
        optimizer.step()                      

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Accuracy: {train_acc:.2f}%")

# Testing (Evaluation)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Single image prediction
img_label = int(input("Enter image index (0-9999) from test dataset: "))
def predict(img_label):
    model.eval()
    with torch.no_grad():
        img_tensor = test_data[img_label][0].unsqueeze(0).to(device)
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = probs.max(dim=1)
        return pred.item(), conf.item()
    

sample_img, label = test_data[img_label]
pred, conf = predict(img_label)
print(f"True Label: {label}, Predicted: {pred}, Confidence: {conf:.2f}")