import torch.nn as nn
import torch.nn.functional as F


class TrafficSignNet(nn.Module):
    def __init__(self, num_classes=43):
        super(TrafficSignNet, self).__init__()

        # Слой 1: на входе 3 канала (RGB), на выходе 16 "фишек", которые мы нашли
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # Слой 2: из 16 делаем 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Слой пулинга (уменьшает размер картинки в 2 раза, сохраняя главное)
        self.pool = nn.MaxPool2d(2, 2)

        # Полносвязные слои (классификатор)
        # После двух пулингов картинка 32x32 станет 8x8. 32 канала * 8 * 8 = 2048
        self.fc1 = nn.Linear(32 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(0.2)  # Чтобы модель не зазубривала данные

    def forward(self, x):
        # Прогоняем через свертки с активацией ReLU
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        # x = self.pool(F.relu(self.conv3(x)))


        # "Выпрямляем" в один длинный вектор
        x = x.view(-1, 32 * 8 * 8)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)  # На выходе логиты для каждого из 43 классов
        return x