### Descripción del Proyecto
Este proyecto contiene el código y notebooks necesarios para el procesamiento de datos, creación de datasets, aumentación de imágenes, y entrenamiento de modelos YOLOv8 para detección de objetos. Específicamente, se ha entrenado un modelo YOLOv8n para el reconocimiento de cartas españolas, identificando sus respectivos valores y clases (Oros, Copas, Espadas y Bastos). Además, se han utilizado scripts adicionales para calcular el puntaje de envido en el juego argentino de cartas Truco.

---

### Estructura del Proyecto

1. **Notebooks de Google Colab y Archivos Python**:
   - `procesamiento.ipynb`: contiene `duplicados_out.py` y `procesamiento.py`
   - `dataset.ipynb`: contiene `dataset.py` y `img_augment.py`
   - `main_yolo.ipynb`: contiene `main_yolo.py` y `best_model.py`
   - `00.run_envido.ipynb`: procesamiento inicial

2. **Descripción de Archivos Principales**:
   - **`duplicados_out.py`**: Elimina archivos duplicados calculando y comparando hashes MD5.
   - **`procesamiento.py`**: Procesa archivos de anotaciones e imágenes, verifica y convierte al formato YOLO.
   - **`dataset.py`**: Crea y estructura un dataset para YOLO, divide en conjuntos de entrenamiento y validación.
   - **`img_augment.py`**: Realiza aumentación de datos utilizando la librería Albumentations.
   - **`main_yolo.py`**: Configura y entrena un modelo YOLOv8 utilizando PyTorch y Ultralytics.
   - **`best_model.py`**: Carga un modelo YOLOv8 entrenado y realiza predicciones en imágenes de prueba.
   - **`cuda.py`**: Verifica la disponibilidad de CUDA y muestra información relevante de la GPU.
   - **`verfotos.py`**: Dibuja bounding boxes en imágenes y crea collages para visualización.

---

### Instrucciones para Correr el Código

#### Instalación de Librerías de Forma Local

1. **Clona el Repositorio**:
   ```sh
   git clone https://github.com/TaielRaffaeli/truco_app
   cd truco_app
   ```

2. **Crear y Activar un Entorno Virtual**:
   ```sh
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Instalar Dependencias**:
   ```sh
   pip install -r requirements.txt
   ```

4. **Ejecutar el Código**:
   Ejecuta los scripts necesarios de forma local.

#### Instalación de Librerías en Google Colab

1. **Abrir Google Colab**:
   Ve a [Google Colab](https://colab.research.google.com/) y abre una nueva notebook.

2. **Subir `requirements.txt`**:
   Sube el archivo `requirements.txt` al entorno de Colab.

3. **Instalar Dependencias**:
   En la primera celda de la notebook, ejecuta:
   ```python
   !pip install -r requirements.txt
   ```

4. **Ejecutar el Código**:
   Ejecuta el código en la notebook de Colab.

### Información de Contacto

Taiel Raffaeli  
Email: raffaelitaiel@gmail.com