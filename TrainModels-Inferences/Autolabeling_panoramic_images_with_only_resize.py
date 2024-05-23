import random
from ultralytics import YOLO
import os
import shutil

# Obtener el número de la última carpeta creada
last_folder_num = 0
for folder_name in os.listdir('.'):
    if folder_name.startswith('resultados_inferencia_OBB_') and os.path.isdir(folder_name):
        try:
            num = int(folder_name.split('_')[-1])
            last_folder_num = max(last_folder_num, num)
        except ValueError:
            continue

# Incrementar el número para la nueva carpeta
new_folder_num = last_folder_num + 1
output_dir = f"resultados_inferencia_OBB_{new_folder_num}"

# Directorio de salida
output_dir = os.path.join(output_dir, 'inferencia_OBB')
threshold_confidence = 0.1

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

#carpeta de entrada de las imagenes
image_dir = '/home/pf2024/Escritorio/PF/imagenes panoramicas'

# Obtener la lista de imágenes
image_files = os.listdir(image_dir)


# Load a model
model = YOLO('/home/pf2024/Escritorio/PF/runs/detect/train5/weights/best.pt')  # pretrained YOLOv8n model

#Inicializar el contador de imágenes procesadas
image_counter = 1

# Run batched inference on a list of images
results = model.predict(image_dir, save=True, conf=threshold_confidence, imgsz=2048, pretrained=True)  # return a list of Results objects

# Procesar lista de resultados
for result in results:
    # Obtener el nombre de la imagen original
    image_name = os.path.basename(result.path)

    # Filtrar las detecciones basadas en el umbral de confianza
    filtered_boxes = result.boxes[result.boxes.conf >= threshold_confidence]

    # Si no hay detecciones por encima del umbral de confianza, continuar con la siguiente imagen
    if len(filtered_boxes) == 0:
        #print(f"No se encontraron detecciones por encima del umbral de confianza en la imagen {image_name}.")
        continue

    # Guardar la imagen de inferencia
    image_save_path = os.path.join(output_dir, 'images', f"{image_counter}.jpg")
    result.save(filename=image_save_path)

    # Guardar las cajas delimitadoras en un archivo de texto con el mismo nombre de la imagen
    probs_save_path = os.path.join(output_dir, 'labels', f"{image_counter}.txt")

    with open(probs_save_path, 'w') as file:
        for box in filtered_boxes.xyxy:
            # Escribir las coordenadas formateadas en el archivo de texto
            class_id = 0
            x_center = (box[0] + box[2]) / 2 / result.orig_shape[1]
            y_center = (box[1] + box[3]) / 2 / result.orig_shape[0]
            width = (box[2] - box[0]) / result.orig_shape[1]
            height = (box[3] - box[1]) / result.orig_shape[0]

            # Escribir las etiquetas en el archivo de texto
            file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    #Incrementar el contador de imágenes procesadas
    image_counter += 1

print(f"Los resultados se han guardado en la carpeta '{output_dir}'.")