import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import Counter

def clases_cartas(detections):
    name_ids = {
        0: "1O", 1: "1C", 2: "1E", 3: "1B",
        4: "2O", 5: "2C", 6: "2E", 7: "2B",
        8: "3O", 9: "3C", 10: "3E", 11: "3B",
        12: "4O", 13: "4C", 14: "4E", 15: "4B",
        16: "5O", 17: "5C", 18: "5E", 19: "5B",
        20: "6O", 21: "6C", 22: "6E", 23: "6B",
        24: "7O", 25: "7C", 26: "7E", 27: "7B",
        28: "8O", 29: "8C", 30: "8E", 31: "8B",
        32: "9O", 33: "9C", 34: "9E", 35: "9B",
        36: "10O", 37: "10C", 38: "10E", 39: "10B",
        40: "11O", 41: "11C", 42: "11E", 43: "11B",
        44: "12O", 45: "12C", 46: "12E", 47: "12B",
        48: "J", 49: "SKIP", 50: "SSKIP"
    }

    clases = [name_ids.get(int(detection.cls)) for detection in detections]
    return clases

def calcular_envido(clases):
    def determinar_clase_mayoritaria(clases):
        if not clases:
            return "N/A"
        clases_nombres = [clase[-1] for clase in clases]
        conteo_clases = Counter(clases_nombres)
        return conteo_clases.most_common(1)[0][0]

    if len(clases) != 3:
        return 0, determinar_clase_mayoritaria(clases)

    # Extraer valores y palos de las cartas
    valores = []
    palos = []
    for clase in clases:
        palo = clase[-1]
        if palo == "J":
            valor = 0
        else:
            valor = int(clase[:-1])
            
        # Asignar valor 0 a las cartas 10, 11 y 12
        if valor in [10, 11, 12]:
            valor = 0
        elif valor in [8, 9] or clase == 'J':
            return 0, determinar_clase_mayoritaria(clases)

        valores.append(valor)
        palos.append(palo)

    max_puntos = 0
    for i in range(3):
        for j in range(i + 1, 3):
            if palos[i] == palos[j]:
                puntos = 20 + valores[i] + valores[j]
                if puntos > max_puntos:
                    max_puntos = puntos

    if max_puntos == 0:
        return 0, determinar_clase_mayoritaria(clases)

    return max_puntos, palos[0]

# Cargar el modelo YOLOv8 entrenado
model = YOLO('runs/detect/aumentado_2/weights/best.pt')

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)  # 0 es el ID de la cámara por defecto

plt.ion()  # Habilitar modo interactivo
fig, ax = plt.subplots()

# Variable para controlar el cierre de la ventana
window_closed = False

def on_close(event):
    global window_closed
    window_closed = True

# Conectar el evento de cierre de la ventana
fig.canvas.mpl_connect('close_event', on_close)

while cap.isOpened() and not window_closed:
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar detección de objetos en el frame capturado
    results = model(frame)

    # Obtener los resultados y dibujar los bounding boxes
    for result in results:
        detections = result.boxes
        clases = clases_cartas(detections)
        envido, palo = calcular_envido(clases)

        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf.item()
            cls = int(box.cls.item())
            label = f'{model.names[cls]} {confidence:.2f}'

            # Dibujar el bounding box y la etiqueta en el frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Mostrar los puntos de envido en la imagen
        cv2.putText(frame, f'Envido: {envido} {palo}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Mostrar el frame con los bounding boxes usando matplotlib
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.draw()
    plt.pause(0.001)
    ax.clear()

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar las ventanas
cap.release()
plt.ioff()
plt.show()