### Descripción del Proyecto

Este proyecto contiene el código y notebooks necesarios para el procesamiento de datos, creación de datasets, aumentación de imágenes, y entrenamiento de modelos YOLOv8 para detección de objetos. Específicamente, se ha entrenado un modelo YOLOv8n para el reconocimiento de cartas españolas, identificando sus respectivos valores y clases (Oros, Copas, Espadas y Bastos). Además, se han utilizado scripts adicionales para calcular el puntaje de envido en el juego argentino de cartas Truco.

### Estructura del Proyecto

1. **Notebooks de Google Colab y Archivos Python**:
   - `procesamiento.ipynb`: contiene `duplicados_out.py` y `procesamiento.py`
   - `dataset.ipynb`: contiene `dataset.py` y `img_augment.py`
   - `main_yolo.ipynb`: contiene `main_yolo.py` y `best_model.py`
   - `run_envido.ipynb`: inferencia

2. **Descripción de Archivos Principales**:
   - **`duplicados_out.py`**: Elimina archivos duplicados calculando y comparando hashes MD5.
   - **`procesamiento.py`**: Procesa archivos de anotaciones e imágenes, verifica y convierte al formato YOLO.
   - **`dataset.py`**: Crea y estructura un dataset para YOLO, divide en conjuntos de entrenamiento y validación.
   - **`img_augment.py`**: Realiza aumentación de datos utilizando la librería Albumentations.
   - **`main_yolo.py`**: Configura y entrena un modelo YOLOv8 utilizando PyTorch y Ultralytics.
   - **`best_model.py`**: Carga un modelo YOLOv8 entrenado y realiza predicciones en imágenes de prueba.
   - **`cuda.py`**: Verifica la disponibilidad de CUDA y muestra información relevante de la GPU.
   - **`verfotos.py`**: Dibuja bounding boxes en imágenes y crea collages para visualización.

### Funcionalidades Clave

1. **Entrenamiento del Modelo YOLOv8n para Reconocimiento de Cartas**:
   - El modelo YOLOv8n ha sido entrenado para reconocer cartas españolas, identificando su valor y clase (Oros, Copas, Espadas, Bastos).
   - Utiliza PyTorch y la librería Ultralytics para el entrenamiento y evaluación del modelo.

2. **Cálculo del Puntaje de Envido en Truco**:
   - Scripts específicos calculan el puntaje de envido, una combinación de cartas en el juego argentino de cartas Truco.
   - El puntaje de envido se calcula basado en las cartas reconocidas y sus valores correspondientes.

### Utilización

1. **Generación de Detecciones y Archivo JSON**:
   - El notebook `run_envido.ipynb` genera una carpeta donde se guardan las detecciones en archivos "txt" de las imágenes presentes en una carpeta (por ejemplo, `img_test`).
   - Además, crea un archivo JSON que contiene todas estas detecciones con la siguiente estructura:
     ```json
     {
         "nombre_img.jpg": {
             "total_cards": TOTAL_DE_CARTAS,
             "cards": {
                 "E": [ VALORES ],
                 "C": [ VALORES ],
                 "B": [ VALORES ],
                 "O": [ VALORES ],
                 "J": [ VALORES ]
             },
             "points": PUNTOS_DE_ENVIDO,
             "figure": CLASE_DEL_ENVIDO
         },
         ...
     }
     ```

2. **Inferencia en Tiempo Real con Cámara Web**:
   - El archivo `camara.py` permite utilizar la cámara web del PC para realizar la inferencia en tiempo real, utilizando el modelo YOLOv8 entrenado.

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

#### Autor: Taiel Raffaeli  
#### Email: raffaelitaiel@gmail.com