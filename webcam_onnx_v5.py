import cv2
import numpy as np
import onnxruntime
import onnx
import torch
from thop import profile
import pyrealsense2 as rs
import time


model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 640, 640)
flops, params = profile(model.model, inputs=(dummy_input,), verbose=False)

flops_text = f"FLOPs: {flops / 1e9:.2f} GFLOPs"
params_text = f"Params: {params / 1e6:.2f} M"

session = onnxruntime.InferenceSession("yolov5n.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


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

def preprocess(image):
    img = cv2.resize(image, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dim
    return img

def postprocess(outputs, image, conf_thres=0.3):
    boxes, scores, class_ids = [], [], []
    output = np.squeeze(outputs[0])  # ← 수정

    h, w = image.shape[:2]
    for det in output:
        conf = float(det[4])  # ← float 처리
        if conf < conf_thres:
            continue
        class_scores = det[5:]
        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])
        if score * conf > conf_thres:
            cx, cy, bw, bh = det[:4]
            x1 = int((cx - bw / 2) * w / 640)
            y1 = int((cy - bh / 2) * h / 640)
            x2 = int((cx + bw / 2) * w / 640)
            y2 = int((cy + bh / 2) * h / 640)
            boxes.append([x1, y1, x2, y2])
            scores.append(score * conf)
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, 0.45)
    for i in indices:
        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        box = boxes[i]
        label = f"{CLASSES[class_ids[i]]}: {scores[i]:.2f}"
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(image, label, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    return image



cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라 열기 실패")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    input_tensor = preprocess(frame)
    outputs = session.run(None, {input_name: input_tensor})
    result_frame = postprocess(outputs, frame)
    fps = 1 / (time.time() - start)

    # FPS 및 FLOPs/Params 출력
    cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(result_frame, flops_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(result_frame, params_text, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("YOLOv5n ONNX - Webcam", result_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
