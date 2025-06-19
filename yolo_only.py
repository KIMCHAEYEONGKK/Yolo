import os
import cv2
import torch
import numpy as np
import time
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from LiteMono.networks.depth_encoder_only_ghostinCDC import LiteMono
from LiteMono.networks.depth_decoder import DepthDecoder
from LiteMono.layers import disp_to_depth
from ptflops import get_model_complexity_info
import matplotlib.cm as cm
import matplotlib as mpl

def print_model_flops(model, height, width):
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model, (3, height, width), as_strings=True,
            print_per_layer_stat=False, verbose=False)
    print(f"[FLOPs] Encoder MACs: {macs} | Parameters: {params}")

# --- Brightness Enhancement ---
def apply_clahe_rgb(img, clip_limit=3.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)

def adjust_gamma(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def brightness_enhancement(img, clip_limit=3.0, brightness_threshold=5.0):
    h, w = img.shape[:2]
    slice_width = max(w // 10, 1)

    # 중심 및 전체 밝기 기준 계산
    center_slice = img[:, w//2 - slice_width : w//2 + slice_width]
    target_brightness = np.mean(cv2.cvtColor(center_slice, cv2.COLOR_BGR2GRAY))
    global_brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    # CLAHE 대비 보정
    if global_brightness < 60:
        clahe_limit = clip_limit + 1.5
    else:
        clahe_limit = clip_limit
    img = apply_clahe_rgb(img, clip_limit=clahe_limit)

    # 좌우 분할
    left_img = img[:, :w//2]
    right_img = img[:, w//2:]

    left_brightness = np.mean(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY))
    right_brightness = np.mean(cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY))

    # 밝기 보정 함수 정의
    def enhance_region(region_img, region_brightness):
        diff = abs(target_brightness - region_brightness)

        if global_brightness < 50 or diff > brightness_threshold:
            gamma_boost = 1.5 if global_brightness < 40 else 1.2
            estimated_gamma = np.log(target_brightness + 1e-6) / np.log(region_brightness + 1e-6)
            blended_gamma = (estimated_gamma + gamma_boost) / 2
            region_img = np.clip(adjust_gamma(region_img, gamma=blended_gamma), 0, 255).astype(np.uint8)

        elif global_brightness > 180 or region_brightness > 200:
            suppress_gamma = 0.6  # 감마 < 1은 어둡게 만듦
            region_img = np.clip(adjust_gamma(region_img, gamma=suppress_gamma), 0, 255).astype(np.uint8)

        return region_img

    left_img = enhance_region(left_img, left_brightness)
    right_img = enhance_region(right_img, right_brightness)

    return np.hstack((left_img, right_img))

def run_video_inference_yolo_only(video_path):
    yolo_model = YOLO("best4.engine")
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        start_time = time.time()

        enhanced = brightness_enhancement(frame, brightness_threshold=7.0)
        results = yolo_model(enhanced)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw box + center point
            cv2.rectangle(enhanced, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(enhanced, (cx, cy), 4, (255, 0, 0), -1)

        # FPS 표시
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(enhanced, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow("YOLO Only | Real-time Inference", enhanced)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

# --- Entry ---
if __name__ == "__main__":
    video_path = "output.mp4"
    run_video_inference_yolo_only(video_path)
