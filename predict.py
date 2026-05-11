import torch
import json
from torchvision import transforms
from PIL import Image
from src.model import TrafficSignNet


def predict(image_path):
    # 1. Подготовка устройства
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 2. ЗАГРУЖАЕМ КЛАССЫ ИЗ ФАЙЛА (Чтобы не было ошибки Size Mismatch)
    with open('weights/classes.json', 'r') as f:
        class_map = json.load(f)

    # Создаем обратный словарь, чтобы по индексу получить название
    inv_map = {v: k for k, v in class_map.items()}
    num_classes = len(class_map)

    # 3. Загрузка модели с правильным числом классов
    model = TrafficSignNet(num_classes=num_classes)
    model.load_state_dict(torch.load('weights/traffic_sign_model.pth', map_location=device))
    model.to(device)
    model.eval()

    # 4. Трансформация картинки
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
        for i in range(1, 5):
            result = predict(f'test_{i}.png')
            print(f"Нейросеть считает, что это знак: {result}")
    except FileNotFoundError:
        print("Файл не найден. Положи его в папку с проектом!")