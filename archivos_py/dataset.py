import os
import glob
import shutil
import random
import yaml

# Ruta a las imágenes y anotaciones
DATA_PATH = "fotos_textos_data_copy"
DST_DATASET = "data_yolo_aumentado"

# Configuración del dataset YOLO
yolo_ds_config = {
    "train": "./images/train/",
    "val": "./images/val/",
    "nc": 51,
    "names": [
        "1O", "1C", "1E", "1B", "2O", "2C", "2E", "2B", "3O", "3C", "3E", "3B", "4O", "4C", "4E", "4B", "5O", "5C", "5E", "5B",
        "6O", "6C", "6E", "6B", "7O", "7C", "7E", "7B", "8O", "8C", "8E", "8B", "9O", "9C", "9E", "9B", "10O", "10C", "10E", "10B",
        "11O", "11C", "11E", "11B", "12O", "12C", "12E", "12B", "J", "SKIP", "SSKIP"
    ]
}

# Crear carpetas para el dataset YOLO
yolo_ds_dirs = {
    "img_train": os.path.join(DST_DATASET, "images/train/"),
    "img_val": os.path.join(DST_DATASET, "images/val/"),
    "lbl_train": os.path.join(DST_DATASET, "labels/train/"),
    "lbl_val": os.path.join(DST_DATASET, "labels/val/")
}

for dir_path in yolo_ds_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Guardar la configuración en un archivo yaml
with open(os.path.join(DST_DATASET, 'data.yaml'), 'w') as outfile:
    yaml.dump(yolo_ds_config, outfile, default_flow_style=False)

# Guardar los nombres de las clases en un archivo txt
with open(os.path.join(DST_DATASET, 'classes.txt'), 'w') as outfile:
    outfile.writelines("\n".join(yolo_ds_config["names"]))

# Recoger todos los archivos de anotaciones
label_files = glob.glob(os.path.join(DATA_PATH, "*.txt"))

# Dividir en conjuntos de entrenamiento y validación
# val_files = set(random.sample(label_files, k=int(len(label_files) * 0.20)))
val_files = set(random.sample(label_files, k=int(len(label_files) * 0.30)))

for file in val_files:
    shutil.move(file, yolo_ds_dirs["lbl_val"])

train_files = [f for f in label_files if f not in val_files]

for file in train_files:
    shutil.move(file, yolo_ds_dirs["lbl_train"])

# Mover las imágenes correspondientes a cada conjunto
def move_images(label_dir, img_dir):
    for label_file in glob.glob(os.path.join(label_dir, "*.txt")):
        base_name = os.path.basename(label_file).split(".")[0]
        img_file = None
        for ext in ["jpg", "jpeg", "png"]:
            img_path = os.path.join(DATA_PATH, f"{base_name}.{ext}")
            if os.path.exists(img_path):
                img_file = img_path
                break
        if img_file:
            shutil.move(img_file, img_dir)

move_images(yolo_ds_dirs["lbl_val"], yolo_ds_dirs["img_val"])
move_images(yolo_ds_dirs["lbl_train"], yolo_ds_dirs["img_train"])

print("Dataset YOLO creado exitosamente.")
