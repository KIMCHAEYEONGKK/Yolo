from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("best4.pt")

# Export the model to TensorRT format
model.export(
    format="engine",
)
# Load the exported TensorRT model 
tensorrt_model = YOLO("best1.engine",task="detect")

# Run inference
results = tensorrt_model("https://ultralytics.com/images/bus.jpg")