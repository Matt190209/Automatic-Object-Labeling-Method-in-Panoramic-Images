import os
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Función para recortar la parte superior de la imagen
def recortar_parte_superior(imagen, porcentaje_eliminar):
    ancho, alto = imagen.size
    altura_recortada = int(alto * (1 - porcentaje_eliminar))
    return imagen.crop((0, alto - altura_recortada, ancho, alto))

# Función para dividir la imagen en ventanas con superposición
def dividir_imagen_con_superposicion(imagen, ancho_ventana, superposicion):
    ancho, alto = imagen.size
    ventanas = []
    paso = ancho_ventana - superposicion

    for x in range(0, ancho, paso):
        if x + ancho_ventana > ancho:
            x = ancho - ancho_ventana
        ventana = imagen.crop((x, 0, x + ancho_ventana, alto))
        ventanas.append((ventana, x, 0))

    return ventanas

# Función para ampliar el tamaño de las ventanas
def ampliar_ventanas(ventanas, factor_ampliacion):
    ventanas_ampliadas = []
    for ventana, x, y in ventanas:
        ancho, alto = ventana.size
        nuevo_ancho = int(ancho * factor_ampliacion)
        nuevo_alto = int(alto * factor_ampliacion)
        nueva_ventana = ventana.resize((nuevo_ancho, nuevo_alto), Image.BICUBIC)
        ventanas_ampliadas.append((nueva_ventana, x, y))
    return ventanas_ampliadas

# Función para redimensionar las ventanas procesadas a su tamaño original
def redimensionar_a_original(ventanas_procesadas, ancho_original, alto_original):
    ventanas_redimensionadas = []
    for ventana, x, y in ventanas_procesadas:
        nueva_ventana = ventana.resize((ancho_original, alto_original), Image.BICUBIC)
        ventanas_redimensionadas.append((nueva_ventana, x, y))
    return ventanas_redimensionadas

# Función para procesar las ventanas con YOLO y guardar las inferencias
def procesar_con_yolo(ventanas, model, threshold_confidence, division_dir):
    ventanas_procesadas = []

    for idx, (ventana, x, y) in enumerate(ventanas):
        ventana_array = np.array(ventana)
        result = model.predict(ventana_array, conf=threshold_confidence, imgsz=640, save=False)[0]
        draw = Image.fromarray(result.plot())

        # Guardar cada ventana procesada en la carpeta de división
        ventana_save_path = os.path.join(division_dir, f"ventana_{idx+1}.jpg")
        draw.save(ventana_save_path)

        ventanas_procesadas.append((draw, x, y))

    return ventanas_procesadas

# Función para recombinar las ventanas procesadas en una imagen completa
def recombinar_imagen(ventanas, tamano_original):
    ancho_total, alto_total = tamano_original
    imagen_completa = Image.new('RGB', (ancho_total, alto_total))

    for ventana, x, y in ventanas:
        imagen_completa.paste(ventana, (x, y))

    return imagen_completa

# Directorio de entrada de las imágenes
image_dir = '/path/to/panoramic/images'

# Directorio de salida
output_dir = '/path/to/output/dir'

# Crear directorios de salida
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

# Cargar el modelo YOLO
model = YOLO('/path/to/pretrained/weights')  # Ruta a los pesos preentrenados de YOLOv8n

# Parámetros para el recorte y la división
porcentaje_eliminar = 0.25 # Porcentaje de la parte superior a eliminar (ajusta según necesites)
ancho_ventana = 2048  # Ancho de cada ventana
superposicion = 512  # Superposición entre ventanas
factor_ampliacion = 1  # Ajusta según necesites
threshold_confidence = 0.2  # Umbral de confianza para YOLO

# Obtener la lista de imágenes
image_files = os.listdir(image_dir)

# Inicializar el contador de imágenes procesadas
image_counter = 1

# Procesar cada imagen
for image_name in image_files:
    ruta_imagen = os.path.join(image_dir, image_name)

    try:
        imagen = Image.open(ruta_imagen)
    except Exception as e:
        print(f"Error al abrir la imagen {image_name}: {e}")
        continue

    # Recortar la parte superior de la imagen
    imagen_recortada = recortar_parte_superior(imagen, porcentaje_eliminar)

    # Dividir la imagen en ventanas con superposición
    ventanas = dividir_imagen_con_superposicion(imagen_recortada, ancho_ventana, superposicion)

    # Ampliar el tamaño de las ventanas
    ventanas_ampliadas = ampliar_ventanas(ventanas, factor_ampliacion)

    # Procesar cada ventana con YOLO
    ventanas_procesadas = procesar_con_yolo(ventanas_ampliadas, model, threshold_confidence, output_dir)

    # Redimensionar las ventanas procesadas a su tamaño original
    ancho_original, alto_original = imagen_recortada.size
    ventanas_redimensionadas = redimensionar_a_original(ventanas_procesadas, ancho_ventana, alto_original)

    # Recomponer la imagen original a partir de las ventanas procesadas
    imagen_recombinada = recombinar_imagen(ventanas_redimensionadas, imagen_recortada.size)
    image_save_path = os.path.join(output_dir, 'images', f"{image_counter}.jpg")
    imagen_recombinada.save(image_save_path)

    # Guardar las cajas delimitadoras en un archivo de texto
    probs_save_path = os.path.join(output_dir, 'labels', f"{image_counter}.txt")
    with open(probs_save_path, 'w') as file:
        for ventana, x, y in ventanas_procesadas:
            # Escribe tus operaciones para guardar las cajas delimitadoras en el formato deseado
            pass

    # Incrementar el contador de imágenes procesadas
    image_counter += 1

print(f"Los resultados se han guardado en la carpeta '{output_dir}'.")