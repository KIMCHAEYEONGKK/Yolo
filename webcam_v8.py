from ultralytics import YOLO
import cv2
import time
import pyrealsense2 as rs
import torch
from thop import profile

# 1. YOLOv11 모델 로드 (Nano 버전 예시)
model = YOLO("yolov8n.pt")  # 또는 yolo11s.pt, yolo11m.pt 등
pytorch_model = model.model

# FLOPs/Params 미리 계산 (입력 크기 기준)
dummy_input = torch.randn(1, 3, 640, 640)
flops, params = profile(pytorch_model, inputs=(dummy_input,))
flops_text = f"FLOPs: {flops / 1e9:.2f} GFLOPs"
params_text = f"Params: {params / 1e6:.2f} M"

# 2. 웹캠 연결 (0: 기본 카메라, 1 이상은 외장카메라)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 3. 실시간 추론 루프
while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    # 추론
    results = model(frame)

    # 결과 프레임 그리기
    annotated_frame = results[0].plot()

    end = time.time()
    fps = 1 / (end - start)

    # FPS 표시
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, flops_text, (440,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(annotated_frame, params_text, (440, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 프레임 출력
    cv2.imshow("YOLOv11 Webcam", annotated_frame)

    # 'q' 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. 종료 처리
cap.release()
cv2.destroyAllWindows()
