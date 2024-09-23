import torch
import cv2

# Cargar el modelo YOLOv5 preentrenado
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Definir las clases de interés para detectar vehículos y personas
VEHICLE_CLASSES = ['car', 'truck', 'bus']
PERSON_CLASS = ['person', 'face', 'mask','helmet', 'balaclava']

# Variables para seguimiento de estados
vehicle_detected = False
person_detected = False
vehicle_position = None
stopped_vehicle = False

def is_vehicle(label):
    """Verifica si el objeto detectado es un vehículo."""
    return label in VEHICLE_CLASSES

def is_person(label):
    """Verifica si el objeto detectado es una persona."""
    return label in PERSON_CLASS

def is_vehicle_in_front(x1, x2, frame_width):
    """Verifica si el vehículo está justo en frente de la cámara."""
    box_center = (x1 + x2) / 2
    return abs(box_center - frame_width / 2) < frame_width * 0.4  # Ajusta el umbral si es necesario

# Capturar video desde la cámara o un archivo de video
cap = cv2.VideoCapture('video3.mp4')  # Usa 0 para la cámara web, o 'ruta/a/tu/video.mp4' para un video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]

     # Convertir la imagen a escala de grises y aumentar el contraste (opcional para videos de baja calidad)
    enhanced_frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)

    # Realizar la detección en el frame actual
    results = model(enhanced_frame)

    # Reiniciar los estados por cada frame
    vehicle_detected = False
    person_detected = False
    current_vehicle_position = None

    # Procesar las detecciones
    for det in results.pred[0]:
        x1, y1, x2, y2, conf, cls = det
        label = model.names[int(cls)]

        # Comprobar si es un vehículo
        if is_vehicle(label):
            vehicle_detected = True
            current_vehicle_position = ((x1 + x2) / 2, y2)  # Posición del vehículo

            # Verificar si el vehículo está en frente
            if is_vehicle_in_front(x1, x2, frame_width):
                stopped_vehicle = True

            # Dibujar la caja delimitadora y el texto del vehículo
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Comprobar si es una persona
        if is_person(label):
            person_detected = True

            # Dibujar la caja delimitadora y el texto de la persona
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Verificar si el vehículo se ha detenido basado en la posición entre frames
    if vehicle_position and current_vehicle_position:
        movement_threshold = 10  # Umbral para considerar que el vehículo se detuvo
        stopped_vehicle = abs(vehicle_position[1] - current_vehicle_position[1]) < movement_threshold

    # Actualizar la posición del vehículo para el siguiente frame
    vehicle_position = current_vehicle_position

    # Mostrar alertas basadas en las condiciones
    if vehicle_detected and person_detected:
        cv2.putText(frame, 'Posible Amenaza', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)  # Mensaje en amarillo

    if vehicle_detected and stopped_vehicle and person_detected:
        cv2.putText(frame, 'ALERTA DE PELIGRO', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)  # Mensaje en rojo

    # Mostrar el frame procesado
    cv2.imshow('Detección de Movimientos Sospechosos con YOLOv5', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
