from ultralytics import YOLO

# Initialize a YOLO11n model from a YAML configuration file
# This creates a model architecture without loading pre-trained weights
model = YOLO("yolo11n.yaml")

# Alternatively, load a pre-trained YOLO11n model directly
# This loads both the architecture and the weights trained on COCO
# model = YOLO("yolo11n.pt")
# Display model information (architecture, layers, parameters, etc.)
model.info()

# Train the model using the COCO8 dataset (a small subset of COCO) for 100 epochs
results = model.train(data="yolo_train.yaml", epochs=100, imgsz=640)

# Run inference with the trained model on an image
results = model("/Users/user/Desktop/Yolo.jpg")