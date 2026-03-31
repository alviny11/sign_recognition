import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import json
import matplotlib.pyplot as plt

# Импортируем наши модули из папки src
from src.dataset import TrafficSignDataset
from src.model import TrafficSignNet
from src.train import train_model


def main():
    # --- 1. Настройки ---
    BATCH_SIZE = 32
    EPOCHS = 25  # Увеличим до 15, чтобы график был нагляднее
    LEARNING_RATE = 0.001

    IMG_DIR = 'data/images'
    ANNOT_DIR = 'data/annotations'
    WEIGHTS_DIR = 'weights'

    # Создаем папку для весов, если её нет
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)

    # --- 2. Устройство (Apple Silicon / GPU / CPU) ---
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Используемое устройство: {device}")

    # --- 3. Подготовка данных ---
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("📦 Загрузка датасета...")
    full_dataset = TrafficSignDataset(IMG_DIR, ANNOT_DIR, transform=transform)

    if len(full_dataset) == 0:
        print("❌ Ошибка: Данные не найдены. Проверь пути к изображениям и XML.")
        return

    # Сохраняем словарь классов для predict.py
    with open(os.path.join(WEIGHTS_DIR, 'classes.json'), 'w') as f:
        json.dump(full_dataset.label_map, f)

    # Разделение на Train (80%) и Validation (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(full_dataset.label_map)
    print(f"✅ Найдено классов: {num_classes}")
    print(f"✅ Всего фото: {len(full_dataset)} (Train: {train_size}, Val: {val_size})")

    # --- 4. Инициализация модели ---
    model = TrafficSignNet(num_classes=num_classes).to(device)

    # --- 5. Обучение ---
    print("\n--- Старт обучения ---")
    # Обновленная функция train_model теперь должна возвращать историю потерь
    trained_model, loss_history = train_model(
        model=model,
        train_loader=train_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )

    # --- 6. Сохранение и Визуализация ---
    # Сохраняем веса
    model_path = os.path.join(WEIGHTS_DIR, 'traffic_sign_model.pth')
    torch.save(trained_model.state_dict(), model_path)
    print(f"\n✅ Модель сохранена в {model_path}")

    # Строим график потерь (Loss)
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

    plt.show()  # Откроет окно с графиком


if __name__ == '__main__':
    main()