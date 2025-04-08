import cv2
import time
import torch
import numpy as np
import onnxruntime
from thop import profile

pt_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
pt_model.eval()
dummy_input = torch.randn(1, 3, 640, 640)
flops, params = profile(pt_model.model, inputs=(dummy_input,), verbose=False)
flops_text = f"FLOPs: {flops / 1e9:.2f} GFLOPs"
params_text = f"Params: {params / 1e6:.2f} M"

onnx_path = "yolov5n.onnx"
session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    img = cv2.resize(frame, (640, 640))
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 640), swapRB=True, crop=False)
    ort_inputs = {input_name: blob}
    outputs = session.run(None, ort_inputs)
    output = np.squeeze(outputs[0])  # (1, N, 85) → (N, 85)

    output_img = frame.copy()
    h, w = frame.shape[:2]

    for det in output:
        conf = float(det[4])
        if conf < 0.3:
            continue
        class_scores = det[5:]
        class_id = np.argmax(class_scores)
        score = float(class_scores[class_id])
        if score * conf < 0.3:
            continue

        cx, cy, bw, bh = det[:4]
        x1 = int((cx - bw / 2) * w / 640)
        y1 = int((cy - bh / 2) * h / 640)
        x2 = int((cx + bw / 2) * w / 640)
        y2 = int((cy + bh / 2) * h / 640)

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        label = CLASSES[class_id]
        cv2.circle(output_img, (center_x, center_y), 6, (0, 0, 255), -1)
        cv2.putText(output_img, f"{label} {score*conf:.2f}", (center_x + 8, center_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # FPS 계산
    end = time.time()
    fps = 1 / (end - start)

    cv2.putText(output_img, f"FPS: {fps:.2f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(output_img, flops_text, (440, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(output_img, params_text, (440, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("YOLOv5n ONNX - Center Only", output_img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
