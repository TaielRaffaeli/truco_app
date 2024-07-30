import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

def draw_bounding_boxes(image, label_file, original_size, resized_size):
    with open(label_file, "r") as file:
        lines = file.readlines()

    orig_h, orig_w = original_size
    new_h, new_w = resized_size

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            class_id, x_center, y_center, width, height = map(float, parts)
            x_center, y_center, width, height = x_center * orig_w, y_center * orig_h, width * orig_w, height * orig_h
            x_min = int((x_center - width / 2) * new_w / orig_w)
            y_min = int((y_center - height / 2) * new_h / orig_h)
            x_max = int((x_center + width / 2) * new_w / orig_w)
            y_max = int((y_center + height / 2) * new_h / orig_h)
            
            cv.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
            cv.putText(image, str(class_id), (x_min, y_min - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

def create_collage(images, collage_shape=(5, 6), image_size=(256, 256)):
    collage_h = collage_shape[0] * image_size[0]
    collage_w = collage_shape[1] * image_size[1]
    collage = np.zeros((collage_h, collage_w, 3), dtype=np.uint8)

    for idx, image in enumerate(images):
        row = idx // collage_shape[1]
        col = idx % collage_shape[1]
        y_start = row * image_size[0]
        y_end = y_start + image_size[0]
        x_start = col * image_size[1]
        x_end = x_start + image_size[1]

        collage[y_start:y_end, x_start:x_end] = image

    return collage

def visualize_images_with_bboxes(data_path, num_images=30):
    image_files = []
    for ext in ["jpg", "png", "jpeg"]:
        image_files.extend(glob.glob(f"{data_path}/*.{ext}"))

    total_images = len(image_files)
    num_batches = (total_images + num_images - 1) // num_images  # Calcula el número de lotes

    for batch in range(num_batches):
        start_idx = batch * num_images
        end_idx = min(start_idx + num_images, total_images)
        batch_files = image_files[start_idx:end_idx]

        images = []
        for image_file in batch_files:
            base_name = os.path.splitext(os.path.basename(image_file))[0]
            label_file = os.path.join(data_path, f"{base_name}.txt")

            if os.path.exists(label_file):
                image = cv.imread(image_file)
                if image is None:
                    print(f"No se pudo leer la imagen: {image_file}")
                    continue
                
                orig_h, orig_w, _ = image.shape
                resized_image = cv.resize(image, (256, 256))
                draw_bounding_boxes(resized_image, label_file, (orig_h, orig_w), (256, 256))
                images.append(resized_image)

        collage = create_collage(images)

        plt.figure(figsize=(20, 15))
        plt.imshow(cv.cvtColor(collage, cv.COLOR_BGR2RGB))  # Convierte BGR a RGB para Matplotlib
        plt.axis('off')
        plt.show()

# Ruta a la carpeta que contiene las imágenes y archivos .txt
DATA_PATH = r"fotos_textos_data"
visualize_images_with_bboxes(DATA_PATH)


