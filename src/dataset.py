import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset


class TrafficSignDataset(Dataset):
    def __init__(self, img_dir, annot_dir, transform=None):
        self.img_dir = img_dir
        self.annot_dir = annot_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Словарь для превращения имен (например, 'stop') в числа (0, 1...)
        self.label_map = {}
        next_label_id = 0

        # Проходим по всем XML файлам
        for annot_file in os.listdir(annot_dir):
            if not annot_file.endswith('.xml'):
                continue

            # 1. Парсим XML
            tree = ET.parse(os.path.join(annot_dir, annot_file))
            root = tree.getroot()

            # 2. Ищем имя файла картинки и название знака
            # Обычно в одном файле один знак, берем первый попавшийся <object>
            obj = root.find('object')
            if obj is not None:
                label_name = obj.find('name').text
                file_name = root.find('filename').text

                img_path = os.path.join(img_dir, file_name)

                if os.path.exists(img_path):
                    # 3. Наполняем мапу классов
                    if label_name not in self.label_map:
                        self.label_map[label_name] = next_label_id
                        next_label_id += 1

                    self.images.append(img_path)
                    self.labels.append(self.label_map[label_name])

        print(f"✅ Успешно! Найдено картинок: {len(self.images)}")
        print(f"✅ Уникальных классов знаков: {len(self.label_map)}")
        print(f"Список классов: {list(self.label_map.keys())}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label