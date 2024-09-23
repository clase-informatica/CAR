import cv2
import numpy as np

# Cargar los nombres de las clases
with open('coco.names', 'r') as f:
    class_names = f.read().strip().split('\n')

# Cargar el modelo YOLOv3
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Inicializar captura de video (0 para webcam o 'video.mp4' para un archivo)
cap = cv2.VideoCapture('video.mp4')  # Cambia 'video.mp4' por 0 para la webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width = frame.shape[:2]

    # Crear un blob a partir del fotograma
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Obtener los nombres de las capas de salida
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # Realizar la detección
    detections = net.forward(output_layers)

    # Listas para almacenar las cajas, confidencias y clases detectadas
    boxes = []
    confidences = []
    class_ids = []

    # Iterar sobre las detecciones
    for detection in detections:
        for object_detection in detection:
            scores = object_detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_names[class_id] == "car":  # Filtrar solo autos
                # Obtener las coordenadas de la caja
                box = object_detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, box_width, box_height) = box.astype('int')

                x = int(center_x - (box_width / 2))
                y = int(center_y - (box_height / 2))

                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar Non-Maximum Suppression para eliminar cajas duplicadas
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Verifica si `indices` no está vacío y si contiene elementos
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar el fotograma
    cv2.imshow('Detección de Automóviles', frame)

    # Salir del bucle al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
