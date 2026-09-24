from mlp import MLP
from torch.utils.data import DataLoader
import os
import time
from torchvision import datasets, transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

train_dataset = datasets.MNIST(root=DATA_DIR, train = True, download=True, transform=transform)
test_dataset = datasets.MNIST(root = DATA_DIR, train = False, download = True, transform=transform)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

import torch
import torch.nn as nn

start_time = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def train_model(activation, num_epochs=5):
    model = MLP(784, [128,64], 10, activation).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    loss_history = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss/len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        loss_history.append(avg_loss)

    def evaluate(model, loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return correct/total


    train_acc = evaluate(model, train_loader)
    test_acc = evaluate(model, test_loader)
    print(f"Train acc: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    return model, loss_history, test_acc


activations = ['relu','sigmoid','tanh']
results = {}
test_accs = {}

for act in activations:
    model, history, test_acc = train_model(act, num_epochs=5)
    results[act] = history
    test_accs[act] = test_acc

end_time = time.time()

elapsed = end_time - start_time

print(f"Total training time for all 3 models: {elapsed:.2f} seconds")

print("\nFinal Test Accuracy by Activation:")
for act, acc in test_accs.items():
    print(f"  {act}: {acc:.4f}")

import matplotlib.pyplot as plt


for act, history in results.items():
    plt.plot(history, label = act)
plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.title('Training Loss by Activation Function')
plt.legend()
plt.savefig('activation_comparison.png')
plt.show()





