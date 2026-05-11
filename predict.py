import torch
import json
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn


def predict(image_path):
    # 1. Подготовка устройства
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 2. ЗАГРУЖАЕМ КЛАССЫ ИЗ ФАЙЛА (Чтобы не было ошибки Size Mismatch)
    with open('weights/classes.json', 'r') as f:
        class_map = json.load(f)

    # Создаем обратный словарь, чтобы по индексу получить название
    inv_map = {v: k for k, v in class_map.items()}
    num_classes = len(class_map)

    # 3. Загрузка модели (строим структуру ResNet18)
    # Нам не нужны веса из интернета (weights=None), мы загрузим свои локальные
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Теперь безопасно "надеваем" обученные веса на правильный каркас
    model.load_state_dict(torch.load('weights/traffic_sign_model.pth', map_location=device))
    model.to(device)
    model.eval()

    # 4. Трансформация картинки (ОБЯЗАТЕЛЬНО СОВПАДАЕТ С val_transform)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # 5. Предсказание
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        index = predicted.item()

    return inv_map[index]


if __name__ == '__main__':
    # Просто укажи путь к картинке, которую хочешь проверить
    # Убедись, что файл 'test_image.jpg' лежит в папке проекта
    try:
        for i in range(1, 8):
            result = predict(f'test_{i}.png')
            print(f"Нейросеть считает, что это знак: {result}")
    except FileNotFoundError:
        print("Файл не найден")