import os
import numpy as np
from PIL import Image
import albumentations as A
import yaml
import cv2

def augment_dataset(num_augmentations=5):

    input_path = "C:/Users/raffa/OneDrive/Escritorio/truco_app/data_yolo/images/train"
    labels_path = 'C:/Users/raffa/OneDrive/Escritorio/truco_app/data_yolo/labels/train'
    output_path_images = "C:/Users/raffa/OneDrive/Escritorio/truco_app/data_yolo/images/train"
    output_path_labels = "C:/Users/raffa/OneDrive/Escritorio/truco_app/data_yolo/labels/train"
 
    os.makedirs(output_path_images, exist_ok=True)
    os.makedirs(output_path_labels, exist_ok=True)

    # Define augmentations
    transform = A.Compose([
        A.RandomBrightnessContrast(p=1),
        A.RandomRotate90(p=1),
        A.Blur(p=1, blur_limit=(3, 7)),
        A.RandomShadow(p=1),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=1)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    skipped_images = []

    for filename in os.listdir(input_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(input_path, filename)
            txt_path = os.path.join(labels_path, os.path.splitext(filename)[0] + '.txt')

            if os.path.exists(txt_path):
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Unable to read image: {img_path}")
                    skipped_images.append(filename)
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                with open(txt_path, 'r') as f:
                    lines = f.readlines()

                bboxes = []
                class_labels = []
                valid_annotations = True
                for line in lines:
                    values = list(map(float, line.strip().split()))
                    if len(values) != 5:
                        print(f"Invalid annotation format in {txt_path}")
                        valid_annotations = False
                        break
                    class_labels.append(int(values[0]))
                    bbox = values[1:]
                    if not all(0 <= coord <= 1 for coord in bbox):
                        print(f"Invalid bounding box coordinates in {txt_path}")
                        valid_annotations = False
                        break
                    bboxes.append(bbox)

                if not valid_annotations:
                    skipped_images.append(filename)
                    continue

                try:
                    for i in range(num_augmentations):
                        augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                        aug_img = Image.fromarray(augmented['image'])
                        aug_img.save(os.path.join(output_path_images, f"{os.path.splitext(filename)[0]}_aug{i}.jpg"))

                        with open(os.path.join(output_path_labels, f"{os.path.splitext(filename)[0]}_aug{i}.txt"), 'w') as f:
                            for bbox, label in zip(augmented['bboxes'], augmented['class_labels']):
                                f.write(f"{label} {' '.join(map(str, bbox))}\n")
                except ValueError as e:
                    print(f"Error processing {filename}: {str(e)}")
                    skipped_images.append(filename)

    print(f"Augmented dataset saved to {output_path_images} and {output_path_labels}")
    print(f"Skipped images: {skipped_images}")


augment_dataset(5)