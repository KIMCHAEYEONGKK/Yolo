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

def compute_shadow_mask(img, v_thresh=130, s_thresh=60, l_thresh=140):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    v, s, l = hsv[..., 2], hsv[..., 1], lab[..., 0]
    mask = (v < v_thresh) & (s > s_thresh) & (l < l_thresh)
    mask = cv2.GaussianBlur(mask.astype(np.uint8) * 255, (11, 11), 2)
    return cv2.normalize(mask, None, 0, 1.0, cv2.NORM_MINMAX).astype(np.float32)

# 2. MSRCR (Retinex)
def MSRCR(img, sigma_list=[15, 80, 250]):
    img = np.float32(img) + 1.0
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        retinex += np.log10(img) - np.log10(blur + 1)
    retinex /= len(sigma_list)
    intensity = np.sum(img, axis=2) / 3.0
    color_restoration = np.log10(125 * (img / (intensity[:, :, None] + 1)))
    msrcr = retinex * color_restoration
    return np.clip(cv2.normalize(msrcr, None, 0, 255, cv2.NORM_MINMAX), 0, 255).astype(np.uint8)

def selective_msrcr(img, mask, sigma_list=[15,80,250]):
    msrcr = MSRCR(img, sigma_list)
    blur = cv2.GaussianBlur(mask, (11,11),3)
    blur = np.clip(blur[...,None],0,1.0) #(H,W,1)
    blended = img.astype(np.float32) * (1.0 - blur) + msrcr.astype(np.float32) * blur
    blended = np.clip(blended, 0,255).astype(np.uint8)
    return blended

def apply_guided_filter(img, radius=8, eps=1e-2):
    return cv2.ximgproc.guidedFilter(guide=img, src=img, radius=radius, eps=eps, dDepth=-1)


def restore_lab_color(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    # A/B 채널 중심 기준 복원 강도 결정
    a_mean, b_mean = np.mean(a), np.mean(b)
    ab_deviation = np.sqrt((a_mean - 128)**2 + (b_mean - 128)**2)

    # 채도가 너무 낮아졌으면 복원 강도 증가
    ab_scale = 1.05 + min(ab_deviation / 30.0, 0.2)  # 1.05~1.25 범위

    # 중심 복원
    a = np.clip(128 + (a - 128) * ab_scale, 0, 255)
    b = np.clip(128 + (b - 128) * ab_scale, 0, 255)
    lab_merged = cv2.merge((l, a, b)).astype(np.uint8)
    return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

# def restore_lab_color(img_bgr, ab_scale_base=1.1):
#     lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
#     l, a, b = cv2.split(lab)

#     # A/B 평균 색 중심 편차 계산
#     a_mean, b_mean = np.mean(a), np.mean(b)
#     ab_deviation = np.sqrt((a_mean - 128)**2 + (b_mean - 128)**2)
#     ab_scale = ab_scale_base + min(ab_deviation / 25.0, 0.25)

#     a = 128 + (a - a_mean) * ab_scale
#     b = 128 + (b - b_mean) * ab_scale

#     a = np.clip(a, 0, 255)
#     b = np.clip(b, 0, 255)

#     lab = cv2.merge((l, a, b)).astype(np.uint8)
#     return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def enhance_shadow_region(img, mask, brightness_gain=1.5, saturation_gain=1.6):
    img_float = img.astype(np.float32) / 255.0
    hsv = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)

    mask = cv2.GaussianBlur(mask, (11, 11), sigmaX=3)
    mask = np.clip(mask, 0, 1.0)

    hsv[..., 2] = np.minimum(hsv[..., 2] * (1 + mask * (brightness_gain - 1)), 230)

    hsv[..., 1] = np.minimum(hsv[..., 1] * (1 + mask * (saturation_gain - 1)), 200)

    # HSV 범위 안정화
    hsv[..., 0] = np.clip(hsv[..., 0], 0, 179)
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def final_saturation_boost(img_bgr, boost_factor=1.25):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= boost_factor
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def adaptive_saturation_boost(img_bgr, mask, factor=1.4):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + mask * (factor - 1)), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def brightness_enhancement(img):
    base = MSRCR(img)

    mask = compute_shadow_mask(base)

    # 3. 밝기 & 채도 복원
    enhanced = enhance_shadow_region(base, mask, brightness_gain=1.2, saturation_gain=1.2)

    # 4. Edge-aware 필터로 질감 보존
    filtered = apply_guided_filter(enhanced)

    # 5. LAB 색공간 기반 색감 복원
    restored = restore_lab_color(filtered)

    # 6. 대비 향상 (CLAHE)
    contrasted = enhance_contrast(restored)

    # 7. 전체 채도 강화
    final = final_saturation_boost(contrasted, boost_factor=1.05)

    final = adaptive_saturation_boost(final, mask, factor=1.2) 

    return final


# def brightness_enhancement(img):
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
#     l_clahe = clahe.apply(l)
#     lab_clahe = cv2.merge((l_clahe, a, b))
#     img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

#     # 2. HSV 채널에서 살짝 채도만 증가
#     hsv = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2HSV).astype(np.float32)
#     hsv[..., 1] *= 1.05  # 채도 증가 (5%)
#     hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
#     img_saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

#     # 3. Guided filter로 부드러운 디테일 보정
#     guided = cv2.ximgproc.guidedFilter(guide=img, src=img_saturated, radius=8, eps=1e-2, dDepth=-1)

#     return guided


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

        enhanced_img = brightness_enhancement(img)
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
    image_path ="/Users/user/Desktop/Lite-Mono/kitti_data/2011_09_26/2011_09_26_drive_0011_sync/image_02/data" 
    weights_folder = "lite-mono_640x192"
    run_image_inference(image_path, weights_folder)
