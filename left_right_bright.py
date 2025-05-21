import os
import cv2
import time
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from networks.depth_encoder import LiteMono
from networks.depth_decoder import DepthDecoder
from layers import disp_to_depth
from types import SimpleNamespace
from ptflops import get_model_complexity_info

# --- Camera Parameters ---
f_kitti = 721.5377
B_kitti = 0.5327
f_realsense = 1.33
B_realsense = 0.05

def print_model_flops(model, height, width):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    macs, params = get_model_complexity_info(
        model.to(device), (3, height, width), as_strings=True,
        print_per_layer_stat=False, verbose=False)
    print(f"[FLOPs] Encoder MACs: {macs} | Parameters: {params}")

def load_model(weights_folder):
    encoder_path = os.path.join(weights_folder, "encoder.pth")
    decoder_path = os.path.join(weights_folder, "depth.pth")
    encoder_dict = torch.load(encoder_path, map_location=torch.device('cpu'))
    decoder_dict = torch.load(decoder_path, map_location=torch.device('cpu'))

    height = encoder_dict['height']
    width = encoder_dict['width']

    encoder = LiteMono(height=height, width=width)
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    encoder.eval()

    decoder = DepthDecoder(encoder.num_ch_enc, scales=range(3))
    decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in decoder.state_dict()})
    decoder.eval()

    print_model_flops(encoder, height, width)

    return encoder, decoder, height, width

def apply_clahe_rgb(img, clip_limit=3.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    merged = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

#감마 보정 함수
def adjust_gamma(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def brightness_enhancement(img, clip_limit=3.0, brightness_threshold=5.0):
    h, w = img.shape[:2]
    slice_width = max(w // 10, 1)

    # 중심 20% 영역의 평균 밝기 기준 설정
    center_slice = img[:, w//2 - slice_width : w//2 + slice_width]
    target_brightness = np.mean(cv2.cvtColor(center_slice, cv2.COLOR_BGR2GRAY))

    # CLAHE 먼저 적용
    img = apply_clahe_rgb(img, clip_limit=clip_limit)

    # 좌우 나누기
    left_img = img[:, :w//2]
    right_img = img[:, w//2:]

    left_brightness = np.mean(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY))
    right_brightness = np.mean(cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY))

    # 밝기 차이 확인
    diff_left = abs(target_brightness - left_brightness)
    diff_right = abs(target_brightness - right_brightness)

    # 감마 보정 적용 조건부
    if diff_left > brightness_threshold:
        left_gamma = np.log(target_brightness + 1e-6) / np.log(left_brightness + 1e-6)
        left_img = np.clip(adjust_gamma(left_img, gamma=left_gamma), 0, 255).astype(np.uint8)

    if diff_right > brightness_threshold:
        right_gamma = np.log(target_brightness + 1e-6) / np.log(right_brightness + 1e-6)
        right_img = np.clip(adjust_gamma(right_img, gamma=right_gamma), 0, 255).astype(np.uint8)

    # 병합 후 반환
    return np.hstack((left_img, right_img))


def create_colormap(np_array, cmap_name='magma', vmin=None, vmax=None):
    if vmin is None:
        vmin = np_array.min()
    if vmax is None:
        vmax = np.percentile(np_array, 95)
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap_name)
    colormapped = (mapper.to_rgba(np_array)[:, :, :3] * 255).astype(np.uint8)
    return colormapped

def run_image_inference(image_path, weights_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, feed_height, feed_width = load_model(weights_folder)
    encoder.to(device)
    decoder.to(device)

    yolo_model = YOLO("/Users/user/Desktop/Lite-Mono/bestv11n.onnx")

    paths = [os.path.join(image_path, f) for f in sorted(os.listdir(image_path)) if f.endswith(('.jpg', '.png'))] \
        if os.path.isdir(image_path) else [image_path]

    os.makedirs("output", exist_ok=True)

    for img_path in paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read {img_path}")
            continue

        enhanced_img = brightness_enhancement(img,brightness_threshold=7.0)
        original = enhanced_img.copy()
        yolo_result = yolo_model.predict(source=enhanced_img, device=device, verbose=False)[0]

        pil_image = Image.fromarray(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)).resize((feed_width, feed_height))
        input_tensor = transforms.ToTensor()(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = encoder(input_tensor)
            outputs = decoder(features)
            disp = outputs["disp", 0]
            disp_resized = torch.nn.functional.interpolate(disp, (enhanced_img.shape[0], enhanced_img.shape[1]), mode="bilinear", align_corners=False)

        disp_np = disp_resized.squeeze().cpu().numpy()
        depth_map = create_colormap(disp_np)
        depth_map_bgr = cv2.cvtColor(depth_map, cv2.COLOR_RGB2BGR)

        for box in yolo_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if 0 <= center_y < disp_np.shape[0] and 0 <= center_x < disp_np.shape[1]:
                patch = disp_np[max(center_y - 2, 0):center_y + 3, max(center_x - 2, 0):center_x + 3]
                disparity = np.mean(patch)

                if disparity > 0:
                    distance = (f_kitti * B_kitti) / disparity
                    brightness = int(
                        0.299 * depth_map[center_y, center_x][0] +
                        0.587 * depth_map[center_y, center_x][1] +
                        0.114 * depth_map[center_y, center_x][2]
                    )
                    label = f"{brightness}"
                else:
                    label = "N/A"
            else:
                label = "N/A"


            cv2.circle(original, (center_x, center_y), 4, (255, 0, 0), -1)
            cv2.putText(original, label, (center_x + 10, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.circle(depth_map_bgr, (center_x, center_y), 4, (255, 0, 0), -1)
            cv2.putText(depth_map_bgr, label, (center_x + 10, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        combined = np.hstack((original, depth_map_bgr))
        basename = os.path.splitext(os.path.basename(img_path))[0]
        output_path = f"output/{basename}_depth_yolo.jpg"
        cv2.imwrite(output_path, combined)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    image_path ="/Users/user/Desktop/Lite-Mono/kitti_data/2011_09_26/2011_09_26_drive_0060_sync/image_02/data" 
    weights_folder = "lite-mono_640x192"
    run_image_inference(image_path, weights_folder)
