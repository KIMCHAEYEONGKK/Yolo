import os
import cv2
import torch
import numpy as np
# import serial
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

# --- Camera Parameters ---
f_kitti = 721.5377
B_kitti = 0.5327

# 시리얼 포트 및 보드레이트 설정 (Linux: '/dev/ttyUSB0', Windows: 'COM3')
# arduino = serial.Serial('COM4', 9600, timeout=1)
# time.sleep(2)  # 아두이노 초기화 대기

def print_model_flops(model, height, width):
    # with torch.cuda.device(0):
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

def create_colormap(disp_np, cmap_name='magma'):
    vmin = np.percentile(disp_np, 1)
    vmax = np.percentile(disp_np, 99.5)
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap_name)
    colormapped = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)

    colormapped = np.clip((colormapped.astype(np.float32) / 255) ** 0.9 * 255, 0, 255).astype(np.uint8)
    return colormapped


# --- Model Load --- 
def load_model(weights_folder):
    encoder_path = os.path.join(weights_folder, "encoder.pth")
    decoder_path = os.path.join(weights_folder, "depth.pth")
    encoder_dict = torch.load(encoder_path)
    decoder_dict = torch.load(decoder_path)

    height = encoder_dict['height']
    width = encoder_dict['width']

    encoder = LiteMono(height=height, width=width)
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    encoder.eval().to(device)

    decoder = DepthDecoder(encoder.num_ch_enc, scales=range(3))
    decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in decoder.state_dict()})
    decoder.eval().to(device)
    print_model_flops(encoder, height, width)

    return encoder, decoder, height, width

# --- 추론 ---
def infer_frame(encoder, decoder, frame, feed_width, feed_height, device):
    input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    original_width, original_height = input_image.size
    input_image = input_image.resize((feed_width, feed_height), Image.LANCZOS)
    input_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(input_tensor)
        outputs = decoder(features)
        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate( 
            disp, (original_height, original_width), mode="bilinear", align_corners=False)

    return disp_resized.squeeze().cpu().numpy()

# --- 실시간 추론 및 객체 거리 시각화 ---
def run_video_inference(weights_folder, video_path):
    encoder, decoder, feed_h, feed_w = load_model(weights_folder)
    yolo_model = YOLO("best4.engine")
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        enhanced = brightness_enhancement(frame, brightness_threshold=7.0)

        # FPS 측정 시작
        start_time = time.time()

        disp_np = infer_frame(encoder, decoder, enhanced, feed_w, feed_h, device)
        depth_map = create_colormap(disp_np)
        depth_bgr = cv2.cvtColor(depth_map, cv2.COLOR_RGB2BGR)

        #화면 3등분으로 나누기
        h, w = disp_np.shape
        thirds = np.array_split(disp_np, 3, axis=1)  # axis=1은 width 방향

        region_names = ["Left", "Center", "Right"]
        region_positions = [(int(w * 1 / 6), 50), (int(w * 3 / 6), 50), (int(w * 5 / 6), 50)]
        for region, (x, y), name in zip(thirds, region_positions, region_names):
            mean_disp = np.mean(region)
            if mean_disp > 0:
                distance = (f_kitti * B_kitti) / mean_disp
                label = f"{name}"
            else:
                label = "N/A"
            cv2.putText(enhanced, label, (x - 40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        results = yolo_model(enhanced)[0]
        max_brightness_in_region = {"Left": -1, "Center": -1, "Right": -1}
        region_width = disp_np.shape[1] // 3

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Box 안에서 depth_map RGB의 최대 밝기 색상 찾기
            box_crop = depth_map[max(y1,0):min(y2,depth_map.shape[0]), max(x1,0):min(x2,depth_map.shape[1]), :]
            if box_crop.size > 0:
                brightness_map = box_crop.mean(axis=2)
                max_idx = np.unravel_index(np.argmax(brightness_map, axis=None), brightness_map.shape)
                max_color = box_crop[max_idx[0], max_idx[1], :]
                max_color_value = int(max_color.mean())

                label = f"{max_color_value}"

                # 중심점 대신 max 위치에 시각화
                cx = x1 + max_idx[1]
                cy = y1 + max_idx[0]
                
                # 영역 분할 후 최대 밝기 갱신
                if cx < region_width:
                    region = "Left"
                elif cx < 2 * region_width:
                    region = "Center"
                else:
                    region = "Right" 
                max_brightness_in_region[region] = max(max_brightness_in_region[region], max_color_value)

            else:
                label = "N/A"

            # 시각화
            cv2.circle(enhanced, (cx, cy), 4, (255, 0, 0), -1)
            cv2.putText(enhanced, label, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.circle(depth_bgr, (cx, cy), 4, (255, 0, 0), -1)
            cv2.putText(depth_bgr, label, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # --- 회피 판단 ---
        brightness_threshold = 180

        avoidance_direction = "Center"  # 기본은 직진

        if max_brightness_in_region["Center"] >= brightness_threshold:
            left_brightness = max(max_brightness_in_region["Left"], 0)
            right_brightness = max(max_brightness_in_region["Right"], 0)

            if left_brightness >= brightness_threshold and right_brightness >= brightness_threshold:
                avoidance_direction = "Stop"
            elif left_brightness < right_brightness:
                avoidance_direction = "Left"
            else:
                avoidance_direction = "Right"

        # --- 아두이노에 명령 전송 ---
        # if avoidance_direction == "Left":
        #     print("Sending: L")
        #     arduino.write(b'L\n')
        # elif avoidance_direction == "Right":
        #     print("Sending: R")
        #     arduino.write(b'R\n')
        # elif avoidance_direction =="Stop":
        #     print("Sending: S")
        #     arduino.write(b'S\n')
        # else:
        #     print("Sending: C")
        #     arduino.write(b'C\n')
        #     time.sleep(0.5)


        # FPS 계산 및 표시
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        cv2.putText(enhanced, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(enhanced, f"{avoidance_direction}", (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2)

        combined = np.hstack((enhanced, depth_bgr))
        cv2.imshow("YOLO + LiteMono | Real-time Inference", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    # arduino.close()


# --- Entry ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_folder = "lite-mono_640x192"
    video_path = "output_video.mp4"
    run_video_inference(weights_folder, video_path)