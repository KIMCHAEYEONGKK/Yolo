from ultralytics import YOLO
import cv2
import torch
from thop import profile
import time

# 모델 로드
model = YOLO("yolo11n.pt")
pytorch_model = model.model

# FLOPs/Params 미리 계산 (입력 크기 기준)
dummy_input = torch.randn(1, 3, 640, 640)
flops, params = profile(pytorch_model, inputs=(dummy_input,))
flops_text = f"FLOPs: {flops / 1e9:.2f} GFLOPs"
params_text = f"Params: {params / 1e6:.2f} M"

# 웹캠 열기 (0 = 기본 웹캠)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    results = model(frame)
    result = results[0]
    output_img = frame.copy()

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            label = result.names[int(cls)]

            # 중심점
            cv2.circle(output_img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(output_img, f"{label} {conf:.2f}", (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
    end = time.time()
    fps = 1 / (end - start)

    cv2.putText(output_img, f"FPS: {fps:.2f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # FLOPs/Params 텍스트 (좌상단 고정)
    cv2.putText(output_img, flops_text, (440,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(output_img, params_text, (440, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 화면에 출력
    cv2.imshow("YOLOv11 Webcam - Center Points", output_img)

    # ESC 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
