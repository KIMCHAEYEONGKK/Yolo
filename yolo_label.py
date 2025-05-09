import os
import shutil
import random

# 경로 설정
kitti_img_dir = "/Users/user/Desktop/Lite-Mono/kitti_data1/image_2"
kitti_lbl_dir = "/Users/user/Desktop/Lite-Mono/kitti_data1/label_02"
output_base = "/Users/user/Desktop/Lite-Mono/kitti_yolo_obstacle"
img_size = (1242, 375)  # KITTI 이미지 크기

# 디렉토리 준비
def prepare_dirs(base_dir):
    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

# KITTI → YOLO (모든 객체 class_id = 0)
def convert_to_obstacle_yolo(label_lines, img_w, img_h):
    yolo_lines = []
    for line in label_lines:
        parts = line.strip().split()
        if parts[0] == "DontCare":
            continue  # DontCare 무시

        x1, y1, x2, y2 = map(float, parts[4:8])
        xc = (x1 + x2) / 2 / img_w
        yc = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return yolo_lines

# 전체 변환 및 분할
def main(split_ratio=0.9):
    prepare_dirs(output_base)
    image_files = sorted([f for f in os.listdir(kitti_img_dir) if f.endswith(".png")])
    random.shuffle(image_files)
    split_idx = int(len(image_files) * split_ratio)

    count = 0
    for i, img_file in enumerate(image_files):
        img_id = os.path.splitext(img_file)[0]
        img_path = os.path.join(kitti_img_dir, img_file)
        lbl_path = os.path.join(kitti_lbl_dir, img_id + ".txt")

        if not os.path.exists(lbl_path):
            continue

        with open(lbl_path, "r") as f:
            lines = f.readlines()

        yolo_labels = convert_to_obstacle_yolo(lines, *img_size)
        if not yolo_labels:
            continue  # 객체 없음 → 건너뜀

        split = "train" if i < split_idx else "val"
        shutil.copy(img_path, os.path.join(output_base, f"images/{split}", img_file))
        with open(os.path.join(output_base, f"labels/{split}", img_id + ".txt"), "w") as f:
            f.write("\n".join(yolo_labels))
        count += 1

    print(f"✅ 변환 완료: {count}개 이미지에서 obstacle로 YOLO 라벨 생성")

    # data.yaml 작성
    with open(os.path.join(output_base, "data.yaml"), "w") as f:
        f.write(f"""train: {os.path.abspath(output_base)}/images/train
val: {os.path.abspath(output_base)}/images/val

nc: 1
names: ['obstacle']
""")

if __name__ == "__main__":
    main()
