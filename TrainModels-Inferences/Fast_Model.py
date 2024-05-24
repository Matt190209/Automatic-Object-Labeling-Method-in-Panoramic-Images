import os
import xml.etree.ElementTree as ET
import cv2
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
from sklearn.preprocessing import label_binarize
from collections.abc import Iterable
import numpy as np

#FUNCIONES DE CARGA DE DATOS Y ENTRENAMIENTO:
def load_data(image_dir, annotation_dir=None):
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"El directorio de imágenes '{image_dir}' no existe.")
    
    image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.jpg')])
    
    if annotation_dir is not None:
        if not os.path.exists(annotation_dir):
            raise FileNotFoundError(f"El directorio de anotaciones '{annotation_dir}' no existe.")
        
        annotations = sorted([os.path.join(annotation_dir, ann) for ann in os.listdir(annotation_dir) if ann.endswith('.xml')])
        
        if len(image_paths) != len(annotations):
            raise ValueError("El número de imágenes y anotaciones no coincide.")
    else:
        annotations = [None] * len(image_paths)

    dataset = []
    for image_path, annotation_path in zip(image_paths, annotations):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error al leer la imagen {image_path}.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (800, 800))  # Asegúrate de que todas las imágenes tengan el mismo tamaño
        
        if annotation_path is not None:
            try:
                target = parse_xml(annotation_path)
            except ET.ParseError:
                raise ValueError(f"Error al analizar el archivo XML {annotation_path}.")
        else:
            target = None

        if target is not None or annotation_dir is None:
            dataset.append((image_path, F.to_tensor(image), target))
    return dataset

def load_images_without_annotations(image_dir):
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"El directorio de imágenes '{image_dir}' no existe.")

    image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.jpg')])

    dataset = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error al leer la imagen {image_path}.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (800, 800))  # Asegúrate de que todas las imágenes tengan el mismo tamaño
        dataset.append((image_path, F.to_tensor(image)))
    return dataset

def parse_xml(xml_file, max_objects=101):
    try:
        tree = ET.parse(xml_file)
    except ET.ParseError as e:
        warnings.warn(f"Error al analizar el archivo XML '{xml_file}': {e}")
        return None

    root = tree.getroot()
    if root is None:
        warnings.warn(f"El archivo XML '{xml_file}' no tiene raíz.")
        return None
    
    boxes = []
    labels = []
    objects = root.findall('object')
    
    if not objects:
        warnings.warn(f"No se encontraron objetos en el archivo {xml_file}.")
        return None
    
    for obj_idx, obj in enumerate(objects):
        label_element = obj.find('name')
        bbox = obj.find('bndbox')
        if label_element is None:
            warnings.warn(f"Falta el elemento 'name' en el objeto {obj_idx + 1}.")
            continue
        if bbox is None:
            warnings.warn(f"Falta el elemento 'bndbox' en el objeto {obj_idx + 1}.")
            continue
        label = label_element.text
        
        x_min_element = bbox.find('xmin')
        y_min_element = bbox.find('ymin')
        x_max_element = bbox.find('xmax')
        y_max_element = bbox.find('ymax')
        if x_min_element is None or y_min_element is None or x_max_element is None or y_max_element is None:
            warnings.warn(f"Faltan las coordenadas del cuadro delimitador en el objeto {obj_idx + 1}.")
            continue
        x_min = int(x_min_element.text)
        y_min = int(y_min_element.text)
        x_max = int(x_max_element.text)
        y_max = int(y_max_element.text)
        
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(1)
    
    if not boxes:
        warnings.warn(f"No se encontraron cajas delimitadoras en el archivo {xml_file}.")
        return None

    # Rellenar las listas de cajas y etiquetas hasta que tengan la longitud max_objects
    while len(boxes) < max_objects:
        boxes.append([0, 0, 0, 0])  # Rellenar con cajas vacías
        labels.append(0)  # Rellenar con etiquetas vacías

    target = {"boxes": boxes, "labels": labels}
    return target

def load_pretrained_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_model(model, data_loader, num_layers_to_freeze=0, num_epochs=70, weight_decay=0.0005, momentum=0.9):
    # Congelar las primeras num_layers_to_freeze capas
    ct = 0
    for param in model.parameters():
        ct += 1
        if ct <= num_layers_to_freeze:
            param.requires_grad = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0001, weight_decay=weight_decay, momentum=momentum)

    for epoch in range(num_epochs):
        model.train()
        for i, (image_path, images, targets) in enumerate(data_loader):
            if len(images) == 0 or len(targets) == 0:
                warnings.warn("Las imágenes o los objetivos están vacíos.")
                continue
            images = list(img.to(device) for img in images)
            if all(isinstance(t, dict) for t in targets):
                targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
            else:
                warnings.warn("Los objetivos no son diccionarios.")
                continue
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

#FUNCIONES DE PREDICCION Y EVALUACION:
def predict(image, model, nms_threshold=0.3):

    # Comprobar si la imagen es una cadena (ruta de archivo) o un tensor
    if isinstance(image, str):
        # Cargar y preprocesar la imagen
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = F.to_tensor(image)

    if image.dim() == 3:
        image = image.unsqueeze(0)  # añadir una dimensión extra para el lote

    # Mover la imagen al dispositivo correcto
    device = torch.device('cpu')  # Usar GPU si está disponible, de lo contrario usar CPU
    image = image.to(device)

    # Mover el modelo al mismo dispositivo
    model = model.to(device)

    # Poner el modelo en modo de evaluación y obtener las predicciones
    model.eval()
    with torch.no_grad():
        prediction = model(image)

    # Procesar las predicciones
    boxes = prediction[0]['boxes'].cpu().numpy().tolist()
    labels = prediction[0]['labels'].cpu().numpy().tolist()
    scores = prediction[0]['scores'].cpu().numpy().tolist()

    # Aplicar NMS a las cajas, etiquetas y puntuaciones
    boxes, scores = apply_nms(np.array(boxes), np.array(scores), threshold=0.3)
    labels = [labels[i] for i in range(len(boxes))]

    return boxes, labels, scores

def apply_nms(boxes, scores, threshold=0.35):
    selected_boxes = []
    selected_scores = []
    while len(boxes) > 0:
        max_index = np.argmax(scores)
        selected_boxes.append(boxes[max_index])
        selected_scores.append(scores[max_index])
        selected_box = boxes[max_index]

        iou = calculate_iou(selected_box, boxes)
        overlapping_indices = np.where(iou > threshold)[0]

        boxes = np.delete(boxes, overlapping_indices, axis=0)
        scores = np.delete(scores, overlapping_indices)

    # Retornar las detecciones seleccionadas después de NMS
    return np.array(selected_boxes), np.array(selected_scores)

def calculate_iou(box, boxes):
    # Calcular la Intersección sobre Unión (IoU) de una caja con un conjunto de cajas
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    iou = intersection_area / (box_area + boxes_area - intersection_area)
    return iou

def get_true_labels(dataset, model):
    true_labels = []
    for image_path, image, target in dataset:
        _, labels, _ = predict(image, model)
        true_labels.extend(labels)
    return true_labels

def get_pred_labels(dataset, model):
    pred_labels = []
    for image_tuple in dataset:
        image_path, image, _ = image_tuple
        _, labels, _ = predict(image, model)
        pred_labels.extend(labels)
    return pred_labels

def calculate_metrics(true_labels, pred_labels):
    # Calcular la matriz de confusión
    cm = confusion_matrix(true_labels, pred_labels)

    # Calcular la tasa de falsos positivos y falsos negativos
    if cm.shape[1] > 1:
        fp_rate = cm[0, 1] / cm[0].sum()
        fn_rate = cm[1, 0] / cm[1].sum()
    else:
        fp_rate = 0
        fn_rate = 0

    # Aplanar true_labels antes de convertirlo en un conjunto
    true_labels_flattened = [item for sublist in true_labels for item in (sublist if isinstance(sublist, Iterable) else [sublist])]
    true_labels_set = set(true_labels_flattened)

    # Calcular la AUC-ROC
    if len(true_labels_set) > 1:
        true_labels = label_binarize(true_labels_flattened, classes=list(true_labels_set))
        pred_labels = label_binarize(list(pred_labels), classes=list(true_labels_set))
        auc_roc = roc_auc_score(true_labels, pred_labels, multi_class='ovr')
    else:
        auc_roc = None
        
    # Calcular el F1-Score
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    # Calcular la precisión
    precision = precision_score(true_labels, pred_labels, average='weighted')

    # Calcular el recall
    recall = recall_score(true_labels, pred_labels, average='weighted')

    return cm, fp_rate, fn_rate, auc_roc, f1, precision, recall

#FUNCIONES DE APRENDIZAJE ACTIVO:
def get_predictions(image_dir, model):
    test_dataset = load_images_without_annotations(image_dir)
    predictions = []
    for item in test_dataset:
        if len(item) == 2:
            image_path, image_tensor = item
        else:
            image_path = item[0]
            image = Image.open(image_path).convert("RGB")
            image_tensor = transforms.ToTensor()(image)

        boxes, labels, scores = predict(image_tensor, model)
        predictions.append((image_path, boxes, labels, scores))
    return predictions

def analyze_predictions(predictions, true_labels):
    selected_images = []
    for (image_path, boxes, labels, scores), true_label in zip(predictions, true_labels):
        if labels != true_label:
            selected_images.append(image_path)
    return selected_images

def select_samples_for_retraining(selected_images, annotation_dir):
    selected_annotations = [os.path.join(annotation_dir, os.path.basename(img_path).replace(".jpg", ".xml")) for img_path in selected_images]
    return selected_images, selected_annotations

def retrain_model(selected_images, selected_annotations, model, batch_size):
    selected_dataset = load_data(selected_images, selected_annotations)
    data_loader = torch.utils.data.DataLoader(selected_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, data_loader, num_epochs=50)

def evaluate_model(val_dataset, model):
    val_true_labels = get_true_labels(val_dataset, model)
    val_pred_labels = get_pred_labels(val_dataset, model)
    cm, fp_rate, fn_rate, auc_roc, f1, precision, recall = calculate_metrics(val_true_labels, val_pred_labels)
    print(f"Confusion Matrix:\n{cm}")
    print(f"False Positive Rate: {fp_rate}")
    print(f"False Negative Rate: {fn_rate}")
    print(f"AUC-ROC: {auc_roc}")
    print(f"F1-Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

def active_learning(image_dir, annotation_dir, model, num_iterations=30, batch_size=4):
    for iteration in range(num_iterations):
        predictions = get_predictions(image_dir, model)
        selected_images = analyze_predictions(predictions)
        selected_images, selected_annotations = select_samples_for_retraining(selected_images, annotation_dir)
        retrain_model(selected_images, selected_annotations, model, batch_size)
        evaluate_model(val_dataset, model)

#FUNCIONES DE VISUALIZACION Y GUARDADO:
def visualize_predictions(image_path, boxes, labels, scores, output_dir, threshold=0.2):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x_min, y_min, f'{label}: {score:.2f}', color='white',
                     bbox=dict(facecolor='red', edgecolor='red', pad=1))

    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"prediction_{image_name}")
    plt.savefig(output_path)
    plt.close(fig)

def save_results_to_csv(true_labels, pred_labels, path):
    df = pd.DataFrame({"True Labels": true_labels, "Predicted Labels": pred_labels})
    df.to_csv(path, index=False)

def build_confusion_matrix(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de confusión')
    plt.xlabel('Predicho')
    plt.ylabel('Verdadero')
    plt.show()
    return cm

def user_feedback(image, model_prediction, user_label):
    user_data = [(image, user_label)]
    user_data_loader = torch.utils.data.DataLoader(user_data, batch_size=4)

    train_model(model, user_data_loader, num_epochs=100)

if __name__ == "__main__":
    # Cargar los datos de entrenamiento
    train_image_dir = "path/to/train/images"
    train_annotation_dir = "path/to/train/annotations"
    train_dataset = load_data(train_image_dir, train_annotation_dir)

    # Cargar los datos de validación
    val_image_dir = "path/to/valid/images"
    val_annotation_dir = "path/to/valid/annotations"
    val_dataset = load_data(val_image_dir, val_annotation_dir)

    num_classes = 6
    model = load_pretrained_model(num_classes)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    train_model(model, train_data_loader, num_epochs=100)

    # Evaluar en el conjunto de validación
    val_true_labels = get_true_labels(val_dataset, model)
    val_pred_labels = get_pred_labels(val_dataset, model)
    cm, fp_rate, fn_rate, auc_roc, f1, precision, recall = calculate_metrics(val_true_labels, val_pred_labels)

    # Procesar el conjunto de prueba
    test_image_dir = "path/to/test/images"
    test_dataset = load_images_without_annotations(test_image_dir)  # Asegúrate de cargar solo las imágenes

    output_dir = "path/to/predictions"
    os.makedirs(output_dir, exist_ok=True)

    for image_path, image_tensor in test_dataset:
        boxes, labels, scores = predict(image_tensor, model)
        visualize_predictions(image_path, boxes, labels, scores, output_dir)

    test_true_labels = get_true_labels(test_dataset, model)
    test_pred_labels = get_pred_labels(test_dataset, model)
    save_results_to_csv(test_true_labels, test_pred_labels, "path/to/results.csv")