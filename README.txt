
## Instrucciones para Correr el Código

### Instalación de Librerías de Forma Local

1. **Clona el Repositorio (si aplica)**:
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
   Ahora puedes ejecutar el código de forma local.


### Instalación de Librerías en Google Colab

1. **Abrir Google Colab**:
   Ve a [Google Colab](https://colab.research.google.com/) y abre una nueva notebook.

2. **Subir el Archivo `requirements.txt`**:
   Sube el archivo `requirements.txt` a tu entorno de Colab usando el icono de carpeta en la barra lateral izquierda para abrir el navegador de archivos y el botón "Subir" para cargar el archivo.

3. **Instalar Dependencias**:
   En la primera celda de la notebook, instala las dependencias:
   ```python
   !pip install -r requirements.txt
   ```

4. **Ejecutar el Código**:
   Ahora puedes ejecutar tu código en la notebook.