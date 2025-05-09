from ultralytics import YOLO
import torch

# 1. 모델 로드 (.pt 파일)
model = YOLO("yolov8n-seg.pt")  # 훈련한 모델 경로

# 2. ONNX 변환
model.export(format="onnx", opset=12, dynamic=True, simplify=True)

# 결과:
# - best.onnx 파일이 같은 디렉토리에 저장됨
