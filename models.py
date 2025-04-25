import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class MulticlassClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.network(x)

class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.network(x)

class RNNModel(nn.Module):
    def __init__(self, input_size=128*3, hidden_size=256, num_layers=2, num_classes=10):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3).contiguous().view(x.size(0), 128, 128*3)
        h0 = torch.zeros(2, x.size(0), 256).to(x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])

class LSTMModel(nn.Module):
    def __init__(self, input_size=128*3, hidden_size=256, num_layers=2, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3).contiguous().view(x.size(0), 128, 128*3)
        h0 = torch.zeros(2, x.size(0), 256).to(x.device)
        c0 = torch.zeros(2, x.size(0), 256).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def get_resnet(num_classes=10):
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in resnet.parameters():
        param.requires_grad = True
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet

def get_vgg16(num_classes=10):
    vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    for param in vgg16.features[:-5].parameters():
        param.requires_grad = False
    vgg16.classifier[6] = nn.Linear(4096, num_classes)
    return vgg16

def get_alexnet(num_classes=10):
    alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    alexnet.classifier[6] = nn.Linear(4096, num_classes)
    return alexnet

def get_rnn(num_classes=10):
    return RNNModel(input_size=128*3, hidden_size=256, num_layers=2, num_classes=num_classes)

def get_lstm(num_classes=10):
    return LSTMModel(input_size=128*3, hidden_size=256, num_layers=2, num_classes=num_classes)

def get_autoencoder():
    return Autoencoder()

import copy

def compare_optimizers(model, train_loader, criterion, epochs=3):
    optimizers = {
        "SGD": optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        "Adam": optim.Adam(model.parameters(), lr=0.001)
    }
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, optimizer in optimizers.items():
        print(f"Training with {name}...")
        cloned_model = copy.deepcopy(model)
        cloned_model.to(device)
        cloned_model.train()
        loss_history = []

        for epoch in range(epochs):
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = cloned_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            loss_history.append(total_loss / len(train_loader))
        results[name] = loss_history
    return results
