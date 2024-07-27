import os
import glob
import cv2 as cv
import numpy as np
import os
import glob
import cv2 as cv
import numpy as np

# Ruta a la carpeta que contiene las imágenes y archivos .txt
DATA_PATH = r"./fotos_textos_data"

def is_yolo_format(line):
    parts = line.strip().split()
    if len(parts) == 5:
        try:
            int(parts[0])
            [float(coord) for coord in parts[1:]]
            return True
        except ValueError:
            return False
    return False

def convert_to_yolo_format(annotation_lines, image_w, image_h):
    yolo_lines = []
    for line in annotation_lines:
        parts = line.strip().split()
        if len(parts) > 5:
            class_id = parts[0]
            points = [float(p) for p in parts[1:]]
            points = np.array(points).reshape(-1, 2)
            
            x_min = np.min(points[:, 0]) * image_w
            y_min = np.min(points[:, 1]) * image_h
            x_max = np.max(points[:, 0]) * image_w
            y_max = np.max(points[:, 1]) * image_h

            x_center = (x_min + x_max) / 2 / image_w
            y_center = (y_min + y_max) / 2 / image_h
            width = (x_max - x_min) / image_w
            height = (y_max - y_min) / image_h

            yolo_line = f"{class_id} {x_center} {y_center} {width} {height}\n"
        else:
            yolo_line = line

        yolo_lines.append(yolo_line)
    return yolo_lines

def process_annotations(data_path):
    label_files = glob.glob(f"{data_path}/*.txt")

    for label_file in label_files:
        base_name = os.path.basename(label_file).split(".")[0]
        image_path = None

        for ext in ["jpg", "png", "jpeg"]:
            possible_path = os.path.join(data_path, f"{base_name}.{ext}")
            if os.path.exists(possible_path):
                image_path = possible_path
                break

        if not image_path:
            print(f"Imagen no encontrada para {label_file}, eliminando {label_file}")
            os.remove(label_file)
            continue

        image = cv.imread(image_path)
        if image is None:
            print(f"No se pudo leer la imagen: {image_path}, eliminando {label_file}")
            os.remove(label_file)
            continue

        image_h, image_w, _ = image.shape

        with open(label_file, "r") as file:
            lines = file.readlines()

        yolo_lines = []
        for line in lines:
            if is_yolo_format(line):
                yolo_lines.append(line)
            else:
                yolo_lines.extend(convert_to_yolo_format([line], image_w, image_h))

        with open(label_file, "w") as file:
            file.writelines(yolo_lines)

        print(f"Procesado {label_file} - Transformación aplicada correctamente")

def validate_images_and_annotations(data_path):
    image_files = []
    for ext in ["jpg", "png", "jpeg"]:
        image_files.extend(glob.glob(f"{data_path}/*.{ext}"))

    label_files = glob.glob(f"{data_path}/*.txt")

    image_basenames = {os.path.splitext(os.path.basename(f))[0] for f in image_files}
    label_basenames = {os.path.splitext(os.path.basename(f))[0] for f in label_files}

    images_without_labels = image_basenames - label_basenames
    labels_without_images = label_basenames - image_basenames

    for basename in images_without_labels:
        for ext in ["jpg", "png", "jpeg"]:
            image_path = os.path.join(data_path, f"{basename}.{ext}")
            if os.path.exists(image_path):
                print(f"Eliminando imagen sin etiqueta: {image_path}")
                os.remove(image_path)

    for basename in labels_without_images:
        label_path = os.path.join(data_path, f"{basename}.txt")
        if os.path.exists(label_path):
            print(f"Eliminando etiqueta sin imagen: {label_path}")
            os.remove(label_path)

# Primero validar imágenes y anotaciones
validate_images_and_annotations(DATA_PATH)

# Luego procesar anotaciones para convertirlas al formato YOLO si es necesario
process_annotations(DATA_PATH)
