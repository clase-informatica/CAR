import cv2
import numpy as np

# Cargar el modelo YOLOv3
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Cargar los nombres de las clases del archivo coco.names
with open('coco.names', 'r') as f:
    class_names = f.read().strip().split('\n')

# Captura de video desde un archivo
cap = cv2.VideoCapture('video.mp4')

# Establecer los FPS deseados (por ejemplo, 30 FPS)
fps_deseado = 30
cap.set(cv2.CAP_PROP_FPS, fps_deseado)

# Obtener los FPS actuales para verificación
fps_actual = cap.get(cv2.CAP_PROP_FPS)
print(f'FPS actual: {fps_actual}')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Preprocesar la imagen para YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    detections = net.forward(output_layers)

    # Procesar las detecciones de YOLOv3
    for detection in detections:
        for object_detection in detection:
            scores = object_detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filtrar solo detecciones de autos con confianza suficiente
            if confidence > 0.5 and class_names[class_id] == 'car':
                box = object_detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, box_width, box_height) = box.astype('int')

                x = int(center_x - (box_width / 2))
                y = int(center_y - (box_height / 2))
                w = int(box_width)
                h = int(box_height)

                # Dibujar el rectángulo alrededor del auto detectado
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f'Car: {confidence:.2f}'
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar el fotograma con los autos detectados
    cv2.imshow('Detección de Carros', frame)

    # Controlar la velocidad de reproducción para que coincida con los FPS deseados
    if cv2.waitKey(int(1000 / fps_deseado)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
