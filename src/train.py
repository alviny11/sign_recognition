import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader, epochs=10, learning_rate=0.001):
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
        print(f"Эпоха [{epoch + 1}/{epochs}] - Средний Loss: {epoch_loss:.4f}")

    return model, loss_history  # Возвращаем и модель, и историю