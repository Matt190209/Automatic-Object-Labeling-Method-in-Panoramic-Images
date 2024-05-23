from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m.yaml')  # build a new model from YAML
model = YOLO('/home/pf2024/Escritorio/PF/runs/detect/train5/weights/best.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8m.yaml').load('/home/pf2024/Escritorio/PF/runs/detect/train5/weights/best.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='/home/pf2024/Escritorio/PF/retrain_new_clases/data.yaml', epochs=50, imgsz=640, conf=0.3, batch=16, pretrained=True, lr0=0.0005, lrf=0.0005, save_period=50, save=True, freeze=None)
path = model.export(format="torchscript")  # export the model to pytorch format