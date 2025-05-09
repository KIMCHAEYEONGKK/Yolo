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

#조도 기반 전처리
def apply_clahe_rgb(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

#감마 보정 함수
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

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

    yolo_model = YOLO("/Users/user/Desktop/Lite-Mono/yolov8n-seg.onnx")

    paths = [os.path.join(image_path, f) for f in sorted(os.listdir(image_path)) if f.endswith(('.jpg', '.png'))] \
        if os.path.isdir(image_path) else [image_path]

    os.makedirs("output", exist_ok=True)

    for img_path in paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read {img_path}")
            continue

        img = apply_clahe_rgb(img)
        img = adjust_gamma(img, gamma=1.5)

        original = img.copy()
        input_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((feed_width, feed_height), Image.LANCZOS)
        input_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = encoder(input_tensor)
            outputs = decoder(features)
            disp = outputs["disp", 0]
            disp_resized = torch.nn.functional.interpolate(disp, (img.shape[0], img.shape[1]), mode="bilinear", align_corners=False)

        disp_np = disp_resized.squeeze().cpu().numpy()
        depth_map = create_colormap(disp_np)
        depth_map_bgr = cv2.cvtColor(depth_map, cv2.COLOR_RGB2BGR)

        results = yolo_model.predict(source=img, device=device, verbose=False)[0]
        
        if results.masks is None or results.masks.data is None:
            print("마스크가 탐지되지 않았습니다.")
        else:
            for i, mask in enumerate(results.masks.data):
                # 이진 마스크로 변환
                mask_np = (mask > 0.5).cpu().numpy().astype(np.uint8)

                if cv2.countNonZero(mask_np) < 10:
                    continue

                # 마스크 크기 보정
                mask_np = cv2.resize(mask_np, (disp_np.shape[1], disp_np.shape[0]), interpolation=cv2.INTER_NEAREST)

                # disparity 계산
                #마스크 영역에만 해당하는 disparity값들만 추출
                disparity_vals = disp_np[mask_np == 1]
                disparity = np.mean(disparity_vals) if len(disparity_vals) > 0 else 0

                if disparity > 0:
                    distance = (f_kitti * B_kitti) / disparity

                    # 밝기 계산
                    mask_pixels = original[mask_np == 1]
                    brightness_vals = 0.299 * mask_pixels[:, 0] + 0.587 * mask_pixels[:, 1] + 0.114 * mask_pixels[:, 2]
                    brightness = int(np.mean(brightness_vals))
                    label = f"{brightness}"
                else:
                    label = "N/A"

                # 외곽선 시각화
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(original, contours, -1, (255, 0, 0), 2)
                cv2.drawContours(depth_map_bgr, contours, -1, (255, 0, 0), 2)

                #마스크 경계선을 검추라혀 원본이미지와 깊이 이미지에 시각화
                if contours:
                    M = cv2.moments(contours[0])
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(original, label, (cX + 10, cY),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        cv2.putText(depth_map_bgr, label, (cX + 10, cY),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        combined = np.hstack((original, depth_map_bgr))
        basename = os.path.splitext(os.path.basename(img_path))[0]
        output_path = f"output/{basename}_depth_yolo.jpg"
        cv2.imwrite(output_path, combined)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    image_path ="/Users/user/Desktop/Lite-Mono/kitti_data/2011_09_29/2011_09_29_drive_0071_sync/image_02/data"
    weights_folder = "lite-mono_640x192"
    run_image_inference(image_path, weights_folder)
