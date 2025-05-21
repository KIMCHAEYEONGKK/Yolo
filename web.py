import os
import cv2
import torch
import numpy as np
import time
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from networks.depth_encoder import LiteMono
from networks.depth_decoder import DepthDecoder
from layers import disp_to_depth
from ptflops import get_model_complexity_info
import matplotlib.cm as cm
import matplotlib as mpl

# --- Camera Parameters ---
f_kitti = 721.5377
B_kitti = 0.5327

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
    center_slice = img[:, w//2 - slice_width:w//2 + slice_width]
    target_brightness = np.mean(cv2.cvtColor(center_slice, cv2.COLOR_BGR2GRAY))
    img = apply_clahe_rgb(img, clip_limit=clip_limit)
    left, right = img[:, :w//2], img[:, w//2:]
    l_brightness = np.mean(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY))
    r_brightness = np.mean(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))

    if abs(target_brightness - l_brightness) > brightness_threshold:
        gamma = np.log(target_brightness + 1e-6) / np.log(l_brightness + 1e-6)
        left = adjust_gamma(left, gamma=np.clip(gamma, 0.5, 2.5))
    if abs(target_brightness - r_brightness) > brightness_threshold:
        gamma = np.log(target_brightness + 1e-6) / np.log(r_brightness + 1e-6)
        right = adjust_gamma(right, gamma=np.clip(gamma, 0.5, 2.5))

    return np.hstack((left, right)).astype(np.uint8)

# --- Color Mapping ---
def create_colormap(np_array, cmap_name='magma', gamma=0.7):
    np_array = np_array ** gamma
    vmax = np.percentile(np_array, 95)
    normalizer = mpl.colors.Normalize(vmin=np_array.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap_name)
    colormapped = (mapper.to_rgba(np_array)[:, :, :3] * 255).astype(np.uint8)
    return colormapped

# --- Model Load ---
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
def run_video_inference(weights_folder):
    encoder, decoder, feed_h, feed_w = load_model(weights_folder)
    yolo_model = YOLO("bestv11n.onnx")
    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        enhanced = brightness_enhancement(frame, brightness_threshold=7.0)

        # FPS 측정 시작
        start_time = time.time()

        input_tensor = transforms.ToTensor()(Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)).resize((feed_w, feed_h))).unsqueeze(0).to(device)

        with torch.no_grad():
            features = encoder(input_tensor)
            outputs = decoder(features)
            disp = outputs["disp", 0]
            disp_resized = torch.nn.functional.interpolate(disp, (frame.shape[0], frame.shape[1]), mode="bilinear", align_corners=False)

        disp_np = disp_resized.squeeze().cpu().numpy()
        depth_map = create_colormap(disp_np)
        depth_bgr = cv2.cvtColor(depth_map, cv2.COLOR_RGB2BGR)

        results = yolo_model(enhanced)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if 0 <= cx < disp_np.shape[1] and 0 <= cy < disp_np.shape[0]:
                patch = disp_np[max(cy - 2, 0):cy + 3, max(cx - 2, 0):cx + 3]
                disparity = np.mean(patch)
                if disparity > 0:
                    distance = (f_kitti * B_kitti) / disparity
                    brightness = int(np.mean(depth_map[cy, cx]))
                    label = f"{brightness}"
                else:
                    label = "N/A"
                cv2.circle(enhanced, (cx, cy), 4, (255, 0, 0), -1)
                cv2.putText(enhanced, label, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.circle(depth_bgr, (cx, cy), 4, (255, 0, 0), -1)
                cv2.putText(depth_bgr, label, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # FPS 계산 및 표시
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        cv2.putText(enhanced, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        combined = np.hstack((enhanced, depth_bgr))
        cv2.imshow("YOLO + LiteMono | Real-time Inference", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

# --- Entry ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_folder = "lite-mono_640x192"
    run_video_inference(weights_folder)