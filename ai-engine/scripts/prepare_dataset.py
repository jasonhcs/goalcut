#!/usr/bin/env python3
"""
GoalCut 篮球检测数据集准备脚本

功能：
1. 从项目视频中提取帧
2. 使用 COCO YOLO 模型初步检测
3. 用颜色特征辅助识别篮球和篮筐
4. 生成 YOLO 格式的标注文件

类别定义：
  0: basketball  (篮球)
  1: hoop        (篮筐/篮圈)
  2: net         (篮网)
  3: player      (球员)
"""

import os
import sys
import glob
import json
import subprocess
import random
import shutil
import cv2
import numpy as np

# 添加 ai-engine 到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_ENGINE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(AI_ENGINE_DIR)

# 数据集输出目录
DATASET_DIR = os.path.join(AI_ENGINE_DIR, "datasets", "basketball")
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LBL_DIR = os.path.join(DATASET_DIR, "train", "labels")
VALID_IMG_DIR = os.path.join(DATASET_DIR, "valid", "images")
VALID_LBL_DIR = os.path.join(DATASET_DIR, "valid", "labels")

# 类别映射
CLASSES = {
    "basketball": 0,
    "hoop": 1,
    "net": 2,
    "player": 3,
}


def ensure_dirs():
    for d in [TRAIN_IMG_DIR, TRAIN_LBL_DIR, VALID_IMG_DIR, VALID_LBL_DIR]:
        os.makedirs(d, exist_ok=True)


def extract_frames(video_path, output_dir, fps=1, prefix=""):
    """从视频中按指定 FPS 提取帧"""
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(video_path))[0]
    if prefix:
        basename = f"{prefix}_{basename}"
    
    pattern = os.path.join(output_dir, f"{basename}_frame_%06d.jpg")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        pattern, "-y"
    ]
    subprocess.run(cmd, capture_output=True)
    
    frames = sorted(glob.glob(os.path.join(output_dir, f"{basename}_frame_*.jpg")))
    print(f"  提取 {len(frames)} 帧 from {os.path.basename(video_path)}")
    return frames


def detect_basketball_color(img, bbox=None):
    """使用颜色特征检测篮球（橙色球体）"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 篮球的橙色范围
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # 形态学处理
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    balls = []
    h, w = img.shape[:2]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # 太小忽略
            continue
        
        # 检查圆度
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if circularity > 0.4:  # 较圆的形状
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = bw / max(bh, 1)
            if 0.5 < aspect_ratio < 2.0:  # 接近正方形
                # 转换为 YOLO 格式 (cx, cy, w, h) 归一化
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                if nw > 0.005 and nh > 0.005:  # 不能太小
                    balls.append((cx, cy, nw, nh, area))
    
    return balls


def detect_hoop_color(img):
    """使用颜色特征检测篮筐（红色/橙色环状区域，通常在画面上部）"""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 篮筐的红色/橙色范围
    lower_red1 = np.array([0, 100, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 80])
    upper_red2 = np.array([180, 255, 255])
    lower_orange = np.array([10, 100, 80])
    upper_orange = np.array([25, 255, 255])
    
    mask = cv2.inRange(hsv, lower_red1, upper_red1) | \
           cv2.inRange(hsv, lower_red2, upper_red2) | \
           cv2.inRange(hsv, lower_orange, upper_orange)
    
    # 只看画面上半部分（篮筐通常在上方）
    mask[int(h * 0.6):, :] = 0
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hoops = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = bw / max(bh, 1)
        
        # 篮筐通常是扁平的（宽 > 高）
        if aspect_ratio > 1.2 and bw > 20:
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            if nw > 0.02:
                hoops.append((cx, cy, nw, nh, area))
    
    return hoops


def auto_annotate_frame(frame_path, model=None):
    """自动标注一帧图片"""
    img = cv2.imread(frame_path)
    if img is None:
        return []
    
    h, w = img.shape[:2]
    annotations = []
    
    # 1. 用 YOLO COCO 模型检测人体 (class 0) 和 sports ball (class 32)
    if model is not None:
        try:
            results = model(frame_path, conf=0.2, verbose=False)
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    
                    if cls == 0 and conf > 0.3:  # person → player
                        annotations.append((CLASSES["player"], cx, cy, bw, bh))
                    elif cls == 32 and conf > 0.15:  # sports ball → basketball
                        annotations.append((CLASSES["basketball"], cx, cy, bw, bh))
        except Exception as e:
            pass
    
    # 2. 颜色检测篮球
    balls = detect_basketball_color(img)
    for (cx, cy, bw, bh, area) in balls:
        # 检查是否和 YOLO 检测的球重叠
        is_dup = False
        for ann in annotations:
            if ann[0] == CLASSES["basketball"]:
                dist = ((ann[1] - cx) ** 2 + (ann[2] - cy) ** 2) ** 0.5
                if dist < 0.05:
                    is_dup = True
                    break
        if not is_dup and area > 200:
            annotations.append((CLASSES["basketball"], cx, cy, bw, bh))
    
    # 3. 颜色检测篮筐
    hoops = detect_hoop_color(img)
    for (cx, cy, bw, bh, area) in hoops:
        if area > 500:  # 足够大的篮筐候选
            annotations.append((CLASSES["hoop"], cx, cy, bw, bh))
    
    return annotations


def write_yolo_label(label_path, annotations):
    """写入 YOLO 格式的标注文件"""
    with open(label_path, 'w') as f:
        for ann in annotations:
            cls, cx, cy, w, h = ann
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def create_data_yaml():
    """创建 data.yaml 配置文件"""
    yaml_content = f"""# GoalCut Basketball Detection Dataset
# Auto-generated by prepare_dataset.py

train: {os.path.join(DATASET_DIR, 'train', 'images')}
val: {os.path.join(DATASET_DIR, 'valid', 'images')}

nc: 4
names:
  0: basketball
  1: hoop
  2: net
  3: player
"""
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"data.yaml 已创建: {yaml_path}")
    return yaml_path


def main():
    print("=" * 60)
    print("GoalCut 篮球检测数据集准备")
    print("=" * 60)
    
    ensure_dirs()
    
    # 加载 YOLO COCO 模型
    print("\n加载 YOLO COCO 模型...")
    model = None
    try:
        from ultralytics import YOLO
        model_path = os.path.join(AI_ENGINE_DIR, "yolov8n.pt")
        if os.path.exists(model_path):
            model = YOLO(model_path)
        else:
            model = YOLO("yolov8n.pt")
        print("YOLO 模型加载成功")
    except Exception as e:
        print(f"YOLO 加载失败: {e}（将仅使用颜色检测）")
    
    # 临时帧目录
    tmp_frames = os.path.join(DATASET_DIR, "_tmp_frames")
    os.makedirs(tmp_frames, exist_ok=True)
    
    # 收集所有视频
    videos = []
    
    # 野球场素材（主要来源）
    wild_videos = sorted(glob.glob(os.path.join(PROJECT_DIR, "test/野球场素材/IMG_7225_*.mp4")))
    for v in wild_videos[:8]:  # 取前8个
        videos.append(("wild", v))
    
    # 受控测试视频
    ctrl_videos = sorted(
        glob.glob(os.path.join(PROJECT_DIR, "test/input/*.MP4")) +
        glob.glob(os.path.join(PROJECT_DIR, "test/input/*.mp4"))
    )
    for v in ctrl_videos[:3]:
        videos.append(("ctrl", v))
    
    # 素材视频
    mat_videos = sorted(
        glob.glob(os.path.join(PROJECT_DIR, "test/素材/*.mp4")) +
        glob.glob(os.path.join(PROJECT_DIR, "test/素材/*.MP4"))
    )
    for v in mat_videos[:5]:
        videos.append(("mat", v))
    
    print(f"\n共 {len(videos)} 个视频待处理")
    
    # 提取帧
    all_frames = []
    for tag, video_path in videos:
        print(f"\n处理 [{tag}] {os.path.basename(video_path)}...")
        frames = extract_frames(video_path, tmp_frames, fps=1, prefix=tag)
        all_frames.extend(frames)
    
    print(f"\n总计提取 {len(all_frames)} 帧")
    
    if not all_frames:
        print("错误: 未提取到任何帧")
        return
    
    # 自动标注
    print("\n开始自动标注...")
    annotated_frames = []
    
    for i, frame_path in enumerate(all_frames):
        if i % 50 == 0:
            print(f"  标注进度: {i}/{len(all_frames)}")
        
        annotations = auto_annotate_frame(frame_path, model)
        
        if annotations:  # 只保留有标注的帧
            annotated_frames.append((frame_path, annotations))
    
    print(f"\n有标注的帧: {annotated_frames.__len__()} / {len(all_frames)}")
    
    # 统计类别分布
    class_counts = {name: 0 for name in CLASSES}
    for _, anns in annotated_frames:
        for ann in anns:
            for name, cid in CLASSES.items():
                if ann[0] == cid:
                    class_counts[name] += 1
    
    print("\n类别分布:")
    for name, count in class_counts.items():
        print(f"  {name}: {count}")
    
    # 分割训练集/验证集 (80/20)
    random.seed(42)
    random.shuffle(annotated_frames)
    split_idx = int(len(annotated_frames) * 0.8)
    train_data = annotated_frames[:split_idx]
    valid_data = annotated_frames[split_idx:]
    
    print(f"\n训练集: {len(train_data)} 帧")
    print(f"验证集: {len(valid_data)} 帧")
    
    # 复制帧到数据集目录并写入标注
    for data, img_dir, lbl_dir, tag in [
        (train_data, TRAIN_IMG_DIR, TRAIN_LBL_DIR, "train"),
        (valid_data, VALID_IMG_DIR, VALID_LBL_DIR, "valid"),
    ]:
        for frame_path, annotations in data:
            basename = os.path.basename(frame_path)
            img_dst = os.path.join(img_dir, basename)
            lbl_dst = os.path.join(lbl_dir, os.path.splitext(basename)[0] + ".txt")
            
            shutil.copy2(frame_path, img_dst)
            write_yolo_label(lbl_dst, annotations)
    
    # 清理临时文件
    shutil.rmtree(tmp_frames, ignore_errors=True)
    
    # 创建 data.yaml
    yaml_path = create_data_yaml()
    
    # 统计最终数据集
    train_count = len(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg")))
    valid_count = len(glob.glob(os.path.join(VALID_IMG_DIR, "*.jpg")))
    
    print(f"\n{'=' * 60}")
    print(f"数据集准备完成！")
    print(f"  训练集: {train_count} 张图片")
    print(f"  验证集: {valid_count} 张图片")
    print(f"  data.yaml: {yaml_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
