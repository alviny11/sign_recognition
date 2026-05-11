import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader, epochs=10, learning_rate=0.001):
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # 1. Сначала создаем обычный оптимизатор с нашим числовым learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 2. Создаем планировщик (Scheduler) и "подключаем" к нему оптимизатор
    # mode='min' - реагируем на падение Loss
    # factor=0.5 - уменьшаем шаг в 2 раза, если нет прогресса
    # patience=3 - ждем 3 эпохи стагнации перед уменьшением
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Список для хранения среднего Loss каждой эпохи
    loss_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        epoch_loss = running_loss / batch_count
        loss_history.append(epoch_loss)

        # 3. Делаем шаг планировщика, передавая ему текущий Loss
        scheduler.step(epoch_loss)

        # Получаем текущий шаг обучения, чтобы вывести его на экран (для наглядности)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Эпоха [{epoch + 1}/{epochs}] - Средний Loss: {epoch_loss:.4f} | Шаг (LR): {current_lr}")

    return model, loss_history