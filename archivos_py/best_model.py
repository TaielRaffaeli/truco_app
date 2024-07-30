
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import cv2
import os

# Ruta a los pesos del modelo entrenado
model_path = 'runs/detect/aumentado_2/weights/best.pt'

# Cargar el modelo entrenado
model = YOLO(model_path)


# Directorio con las imágenes de prueba
test_images_dir = 'img_test'

# Obtener la lista de imágenes en el directorio
test_images = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir) if img.endswith('.jpg')]

# Realizar predicciones y mostrar los resultados
for i, image_path in enumerate(test_images):
    results = model(image_path)
    img = results[0].plot()
    plt.imshow(img)
    plt.title(f'Predicción {i + 1}')
    plt.show(block=True)