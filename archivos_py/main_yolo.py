import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt

def main():
    # Verificar si CUDA está disponible y cuál GPU se está utilizando
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Nombre de la GPU: {torch.cuda.get_device_name(0)}")

    # Directorios de imágenes para entrenamiento y validación
    train_images_dir = "C:/Users/raffa/OneDrive/Escritorio/truco_app/data_yolo_aumentado_(10mil)/images/train"
    val_images_dir = "C:/Users/raffa/OneDrive/Escritorio/truco_app/data_yolo_aumentado_(10mil)/images/val"

    # Ruta al archivo data.yaml
    data_yaml_path = "C:/Users/raffa/OneDrive/Escritorio/truco_app/data_yolo_aumentado_(10mil)/data.yaml"

    # Crear y entrenar el modelo YOLOv8
    model = YOLO('yolov8n.pt')  
    model.train(data=data_yaml_path, epochs=2, batch=32, imgsz=640, name='prueba', device=0)

    print("Entrenamiento completado.")


if __name__ == '__main__':
    main()
