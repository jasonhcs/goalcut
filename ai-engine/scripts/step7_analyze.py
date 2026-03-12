#!/usr/bin/env python3
"""分析 IMG_7225_1.mp4 上的检测细节：降低阈值看是否有更多检出"""
import glob
import os
from ultralytics import YOLO

FRAMES_DIR = "/tmp/test_frames_7225"
MODEL_PATH = "/root/project/github.com/goalcut/ai-engine/models/basketball_v1/weights/best.pt"

model = YOLO(MODEL_PATH)
frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.jpg")))
print(f"帧数: {len(frames)}, 模型类别: {model.names}")

# 用不同置信度阈值测试
for conf_th in [0.10, 0.15, 0.20, 0.25, 0.30]:
    ball_frames = 0
    rim_frames = 0
    ball_total = 0
    rim_total = 0
    for f in frames:
        results = model(f, conf=conf_th, verbose=False)
        fb, fr = False, False
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_name = model.names[int(box.cls[0])]
                if cls_name in ("ball", "basketball"):
                    ball_total += 1
                    fb = True
                elif cls_name in ("rim", "hoop"):
                    rim_total += 1
                    fr = True
        if fb: ball_frames += 1
        if fr: rim_frames += 1
    n = len(frames)
    print(f"  conf={conf_th:.2f}: ball={ball_frames}/{n}({ball_frames/n:.1%})  rim={rim_frames}/{n}({rim_frames/n:.1%})  ball_det={ball_total}  rim_det={rim_total}")

# 抽样看一些帧的全部检测
print("\n抽样帧检测详情 (conf=0.10):")
sample_indices = [0, 30, 60, 90, 120, 180, 240, 300, 359]
for idx in sample_indices:
    if idx >= len(frames):
        continue
    f = frames[idx]
    results = model(f, conf=0.10, verbose=False)
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            dets.append(f"{cls_name}:{conf:.2f}")
    print(f"  帧{idx:3d}: {', '.join(dets) if dets else '(无检测)'}")
