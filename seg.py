import os
import cv2
import torch
import numpy as np
import time
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms.functional as F
from LiteMono.networks.depth_encoder_only_ghostinCDC import LiteMono
from LiteMono.networks.depth_decoder import DepthDecoder
from LiteMono.layers import disp_to_depth
import matplotlib.cm as cm
import matplotlib as mpl

f_kitti = 721.5377
B_kitti = 0.5327

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
    center_slice = img[:, w//2 - slice_width : w//2 + slice_width]
    target_brightness = np.mean(cv2.cvtColor(center_slice, cv2.COLOR_BGR2GRAY))
    global_brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    img = apply_clahe_rgb(img, clip_limit=clip_limit + 1.5 if global_brightness < 60 else clip_limit)
    left_img, right_img = img[:, :w//2], img[:, w//2:]

    def enhance_region(region_img, region_brightness):
        diff = abs(target_brightness - region_brightness)
        if global_brightness < 50 or diff > brightness_threshold:
            gamma_boost = 1.5 if global_brightness < 40 else 1.2
            estimated_gamma = np.log(target_brightness + 1e-6) / np.log(region_brightness + 1e-6)
            blended_gamma = (estimated_gamma + gamma_boost) / 2
            return adjust_gamma(region_img, gamma=blended_gamma)
        elif global_brightness > 180 or region_brightness > 200:
            return adjust_gamma(region_img, gamma=0.6)
        return region_img

    left_brightness = np.mean(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY))
    right_brightness = np.mean(cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY))
    return np.hstack((enhance_region(left_img, left_brightness), enhance_region(right_img, right_brightness)))

def create_colormap(disp_np):
    vmax = np.percentile(disp_np, 99)
    normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    return (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)

def load_model(weights_folder, device):
    encoder_dict = torch.load(os.path.join(weights_folder, "encoder.pth"))
    decoder_dict = torch.load(os.path.join(weights_folder, "depth.pth"))
    height, width = encoder_dict['height'], encoder_dict['width']
    encoder = LiteMono(height=height, width=width).to(device)
    decoder = DepthDecoder(encoder.num_ch_enc, scales=range(3)).to(device)
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in decoder.state_dict()})
    encoder.eval(), decoder.eval()
    return encoder, decoder, height, width

def preprocess_for_maskrcnn(image, device):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    return F.to_tensor(image_pil).to(device)

def infer_frame(encoder, decoder, frame, feed_width, feed_height, device):
    input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transforms.ToTensor()(input_image.resize((feed_width, feed_height))).unsqueeze(0).to(device)
    with torch.no_grad():
        features = encoder(input_tensor)
        disp = decoder(features)[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(disp, (frame.shape[0], frame.shape[1]), mode="bilinear", align_corners=False)
    return disp_resized.squeeze().cpu().numpy()

def run_video_inference(weights_folder, video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, decoder, feed_h, feed_w = load_model(weights_folder, device)

    maskrcnn = maskrcnn_resnet50_fpn(pretrained=True).to(device).eval()

    cap = cv2.VideoCapture(video_path)


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        start_time = time.time()
        enhanced = brightness_enhancement(frame)
        disp_np = infer_frame(encoder, decoder, enhanced, feed_w, feed_h, device)
        depth_map = create_colormap(disp_np)
        depth_bgr = cv2.cvtColor(depth_map, cv2.COLOR_RGB2BGR)

        max_brightness_in_region = {"Left": -1, "Center": -1, "Right": -1}
        w = frame.shape[1]

        img_tensor = preprocess_for_maskrcnn(enhanced, device)
        with torch.no_grad():
            results = maskrcnn([img_tensor])[0]

        for mask, box, score in zip(results['masks'], results['boxes'], results['scores']):
            if score < 0.5: continue
            mask_np = (mask[0].cpu().numpy() > 0.5).astype(np.uint8)
            x1, y1, x2, y2 = box.int().tolist()
            colored_mask = np.zeros_like(enhanced)
            colored_mask[mask_np > 0] = (0, 255, 0)
            enhanced = cv2.addWeighted(enhanced, 1.0, colored_mask, 0.5, 0)
            depth_bgr = cv2.addWeighted(depth_bgr, 1.0, colored_mask, 0.5, 0)

            region_disp = disp_np[mask_np > 0]
            region_depth_color = depth_map[mask_np > 0]
            label = "N/A"

            if region_disp.size > 0:
                mean_disp = np.mean(region_disp)
                distance = (f_kitti * B_kitti) / mean_disp if mean_disp > 0 else float('inf')
                region_gray = cv2.cvtColor(region_depth_color.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).flatten()
                brightness = int(np.mean(region_gray))
                mean_cx = int(np.mean(np.column_stack(np.where(mask_np > 0))[:, 1]))
                if mean_cx < w // 3:
                    max_brightness_in_region["Left"] = max(max_brightness_in_region["Left"], brightness)
                elif mean_cx < 2 * w // 3:
                    max_brightness_in_region["Center"] = max(max_brightness_in_region["Center"], brightness)
                else:
                    max_brightness_in_region["Right"] = max(max_brightness_in_region["Right"], brightness)
                label = f"{brightness}"

            cv2.putText(enhanced, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(depth_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        direction = "Center"
        if max_brightness_in_region["Center"] >= 200:
            left, right = max_brightness_in_region["Left"], max_brightness_in_region["Right"]
            if left >= 200 and right >= 200:
                direction = "Stop"
            elif left < right:
                direction = "Left"
            else:
                direction = "Right"

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(enhanced, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(enhanced, f"{direction}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2)
        combined = np.hstack((enhanced, depth_bgr))
        cv2.imshow("Mask R-CNN + LiteMono", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_video_inference("lite-mono_640x192", "output_video.mp4")
