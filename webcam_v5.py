import torch
import cv2
import numpy as np
import time
import pyrealsense2 as rs
from thop import profile

# 모델 로드 (yolov5n.pt는 속도 빠름. 필요시 yolov5s.pt 등으로 변경)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5n.pt', source='github')
pytorch_model = model.model

dummy_input = torch.randn(1, 3, 640, 640)
flops, params = profile(pytorch_model, inputs=(dummy_input,))
flops_text = f"FLOPs: {flops / 1e9:.2f} GFLOPs"
params_text = f"Params: {params / 1e6:.2f} M"

# 2. 웹캠 연결
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

    # YOLOv5 추론
    results = model(frame)

    # 시각화 (bounding box 포함된 이미지)
    annotated_frame = np.squeeze(results.render())

    end = time.time()
    fps = 1 / (end - start)

    # FPS & 모델 정보 표시
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, flops_text, (440, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(annotated_frame, params_text, (440, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 화면 출력
    cv2.imshow("YOLOv5 Webcam", annotated_frame)

    # 'q' 키로 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. 종료 처리
cap.release()
cv2.destroyAllWindows()