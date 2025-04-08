import torch
import cv2
import time
from thop import profile

# 📦 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.eval()
pytorch_model = model.model

# FLOPs/Params 계산
dummy_input = torch.randn(1, 3, 640, 640)
flops, params = profile(pytorch_model, inputs=(dummy_input,), verbose=False)
flops_text = f"FLOPs: {flops / 1e9:.2f} GFLOPs"
params_text = f"Params: {params / 1e6:.2f} M"

# 📷 웹캠 연결
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    # YOLOv5 추론
    results = model(frame)
    output_img = frame.copy()

    if results.xyxy[0].shape[0] > 0:
        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            label = model.names[int(cls)]

            # ✅ 중심점만 표시 (바운딩 박스 없음)
            cv2.circle(output_img, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(output_img, f"{label} {conf:.2f}", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # FPS 계산
    end = time.time()
    fps = 1 / (end - start)

    # 💬 텍스트 표시
    cv2.putText(output_img, f"FPS: {fps:.2f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(output_img, flops_text, (440, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(output_img, params_text, (440, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 화면 출력
    cv2.imshow("YOLOv5n - Center Only", output_img)

    # ESC 키 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
