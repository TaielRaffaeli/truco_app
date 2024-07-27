import torch
"""
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Nombre de la GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA no está disponible. Asegúrate de que tu GPU es compatible y que tienes los controladores correctos instalados.")

"""

# Verificar si CUDA está disponible
cuda_available = torch.cuda.is_available()

# Obtener la versión de CUDA utilizada por PyTorch
cuda_version = torch.version.cuda

# Obtener la versión de PyTorch
torch_version = torch.__version__

# Obtener el nombre de la GPU (si CUDA está disponible)
gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"

print(f"CUDA Disponible: {cuda_available}")
print(f"Versión de CUDA: {cuda_version}")
print(f"Versión de PyTorch: {torch_version}")
print(f"Nombre de la GPU: {gpu_name}")

# Comprobar las versiones de otros paquetes necesarios para YOLOv8
import ultralytics
print(f"Versión de Ultralytics YOLO: {ultralytics.__version__}")
