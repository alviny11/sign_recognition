import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import os
import json
import matplotlib.pyplot as plt

# Импортируем наши модули из папки src
from src.dataset import TrafficSignDataset
from src.model import TrafficSignNet
from src.train import train_model
import torchvision.models as models

class ApplyTransform(Dataset):
    """Класс-обертка для применения разных трансформаций к Subset после random_split"""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)


def main():
    # --- 1. Настройки ---
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001

    IMG_DIR = 'data/images'
    ANNOT_DIR = 'data/annotations'
    WEIGHTS_DIR = 'weights'

    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)

    # --- 2. Устройство (Apple Silicon / GPU / CPU) ---
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Используемое устройство: {device}")

    # --- 3. Подготовка данных ---

    # Аугментация для тренировки
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Возвращаем оригинальный размер для ResNet
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        # Используем константы нормализации ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Чистая трансформация для валидации
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Возвращаем оригинальный размер
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("📦 Загрузка датасета...")
    # Загружаем данные без первичной трансформации, чтобы применить разные позже
    full_dataset = TrafficSignDataset(IMG_DIR, ANNOT_DIR, transform=None)

    if len(full_dataset) == 0:
        print("❌ Ошибка: Данные не найдены. Проверь пути к изображениям и XML.")
        return

    with open(os.path.join(WEIGHTS_DIR, 'classes.json'), 'w') as f:
        json.dump(full_dataset.label_map, f)

    # Разделение на Train (80%) и Validation (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # Оборачиваем подвыборки в трансформы
    train_dataset = ApplyTransform(train_subset, transform=train_transform)
    val_dataset = ApplyTransform(val_subset, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(full_dataset.label_map)
    print(f"✅ Найдено классов: {num_classes}")
    print(f"✅ Всего фото: {len(full_dataset)} (Train: {train_size}, Val: {val_size})")

    # --- 4. Инициализация модели ---
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)
    # --- 5. Обучение ---
    print("\n--- Старт обучения ---")
    trained_model, loss_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,  # ДОБАВИЛИ ЭТУ СТРОЧКУ
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )

    # --- 6. Оценка модели (Скор / Accuracy) ---
    print("\n--- Оценка на валидационной выборке ---")
    trained_model.eval()

    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = trained_model(images)
            _, predicted = torch.max(outputs, 1)

            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = (correct_predictions / total_samples) * 100
    print(f"🏆 Итоговый скор (Accuracy): {accuracy:.2f}% ({correct_predictions} из {total_samples})")

    # --- 7. Сохранение и Визуализация ---
    model_path = os.path.join(WEIGHTS_DIR, 'traffic_sign_model.pth')
    torch.save(trained_model.state_dict(), model_path)
    print(f"\n✅ Модель сохранена в {model_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), loss_history, marker='o', linestyle='-', color='royalblue', label='Training Loss')
    plt.title('История обучения (Loss Curve)')
    plt.xlabel('Эпоха')
    plt.ylabel('Значение ошибки (Loss)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    graph_path = os.path.join(WEIGHTS_DIR, 'loss_plot.png')
    plt.savefig(graph_path)
    print(f"📊 График сохранен: {graph_path}")
    plt.show()


if __name__ == '__main__':
    main()
