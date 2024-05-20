# Automatic-Object-Labeling-Method-in-Panoramic-Images
Nuestro método de etiquetado automático de objetos en imágenes panorámicas utiliza técnicas avanzadas de visión por computadora para clasificar objetos en entornos urbanos y de carreteras, minimizando la intervención humana y agilizando el análisis de imágenes.

![image](https://github.com/Matt190209/Automatic-Object-Labeling-Method-in-Panoramic-Images/assets/143963923/bd8685db-3a83-4338-94d6-e9882fe6ca88)

![image](https://github.com/Matt190209/Automatic-Object-Labeling-Method-in-Panoramic-Images/assets/143963923/0af60eb7-4c78-4b57-9795-a8aff55ffb78)

## Funcionalidades Principales

- **Minimización de la Intervención Humana:** Automatiza el etiquetado de objetos en imágenes panorámicas, reduciendo la necesidad de intervención manual.
  
- **Técnicas Avanzadas de Visión por Computadora:** Utiliza técnicas como transfer learning y fine tuning para clasificar objetos con precisión.
  
- **Eficiencia y Precisión:** Busca optimizar la eficiencia y precisión en la identificación de objetos en entornos urbanos y de tráfico.
  
- **Proceso de Entrenamiento:** Utiliza modelos reconocidos como YOLOv8 y Fast R-CNN para el entrenamiento y ajuste de los algoritmos.

## Diagrama de Bloques del Proyecto

![Diagrama de Bloques PF (2)](https://github.com/Matt190209/Automatic-Object-Labeling-Method-in-Panoramic-Images/assets/143963923/7a9c882e-7ba8-4d84-b6f4-b5f2086a92f6)

## Resultados del Método con YOLOv8

![Captura de pantalla YOLOv8](https://github.com/Matt190209/Automatic-Object-Labeling-Method-in-Panoramic-Images/assets/143963923/892a85c8-7401-432e-b507-a6997018904b)

Aquí puedes agregar una descripción de los resultados obtenidos utilizando YOLOv8.

## Resultados del Método con Fast R-CNN

![Captura de pantalla Fast R-CNN](https://github.com/Matt190209/Automatic-Object-Labeling-Method-in-Panoramic-Images/assets/143963923/892a85c8-7401-432e-b507-a6997018904b)

Aquí puedes agregar una descripción de los resultados obtenidos utilizando Fast R-CNN.

## Estructura del Proyecto

El proyecto está estructurado de la siguiente manera:

- **Dataset:** Contiene un conjunto de datos de imágenes previamente etiquetadas, diseñado para el reentrenamiento de los modelos ya preentrenados. Además, proporciona la posibilidad de ingresar datos revisados, lo que permite continuar mejorando el aprendizaje de los modelos y aumentar su precisión con el tiempo.
- **Data:** Es la carpeta destino donde deberan agregar las imagenes panoramicas a usar  
- **TrainModels:** Contiene los dos diferentes métodos de Machine Learning para el etiquetado de imágenes panorámicas.
- **Requierements_fast.txt:** Archivo de requerimientos con las dependencias del codigo de fast r-cnn
- **Requierements_yolo.txt:** Archivo de requerimientos con las dependencias del codigo de Yolo_V8 

## Instalación y Uso

1. Clona este repositorio:
git clone https://github.com/Matt190209/Automatic-Object-Labeling-Method-in-Panoramic-Images.git

3. Instala las dependencias del proyecto:
pip install -r requirements_fast/yolo.txt

4. Ejecuta el código:
python algo.py

5. Revise las etiquetas generadas y de ser necesario vuelva a ejecutar el código modificando los hiperparametros

## Contribución
Si deseas contribuir a este proyecto, sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama para tu funcionalidad: git checkout -b nueva-funcionalidad.
3. Realiza tus cambios y haz commit: git commit -m "Agrega nueva funcionalidad".
4. Haz push a la rama: git push origin nueva-funcionalidad.
5. Envía un pull request.

## Créditos
Este proyecto fue desarrollado por Robinson Luis Campo Charris, Matthieu Navarro Chamucero y Luis Alejandro Vallejo Morales.

 



