from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("best-seg.pt")

# Export the model to TensorRT format
model.export(format="engine", dynamic=True,simplify=True, optimize=False, half=True)  # creates 'yolo11n.engine'


# Load the exported TensorRT model
model = YOLO("best-seg.engine", task="segment")

# Run inference
result = model.predict("https://ultralytics.com/images/bus.jpg")