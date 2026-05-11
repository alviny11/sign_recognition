import torch.nn as nn
import torch.nn.functional as F


class TrafficSignNet(nn.Module):
    def __init__(self, num_classes=43):
        super(TrafficSignNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Раскомментировали

        self.pool = nn.MaxPool2d(2, 2)

        # Обновили размер входа на основе математического расчета: 64 канала * 4 * 4
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)  # Чуть усилили дропаут для более глубокой сети

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))  # Добавили проход через третий слой

        x = x.view(-1, 64 * 4 * 4)  # Обновили размер для вектора

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x