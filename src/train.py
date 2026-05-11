import torch
import torch.nn as nn
import torch.optim as optim
import os


# Добавили val_loader в аргументы
def train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    loss_history = []
    best_val_acc = 0.0  # Переменная для хранения лучшего скора
    weights_dir = 'weights'

    for epoch in range(epochs):
        # --- ТРЕНИРОВКА ---
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
        scheduler.step(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # --- ВАЛИДАЦИЯ В КОНЦЕ ЭПОХИ ---
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                _, predicted = torch.max(val_outputs, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()

        val_acc = (correct / total) * 100

        print(f"Эпоха [{epoch + 1}/{epochs}] | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr}")

        # --- СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(weights_dir, 'traffic_sign_model_best.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"🌟 Новый рекорд! Модель сохранена ({best_val_acc:.2f}%)")

    print(f"\n🏆 Обучение завершено. Лучший скор на валидации: {best_val_acc:.2f}%")

    # Возвращаем модель с ЛУЧШИМИ весами, а не с весами последней (переобученной) эпохи
    model.load_state_dict(torch.load(os.path.join(weights_dir, 'traffic_sign_model_best.pth'), weights_only=True))
    return model, loss_history