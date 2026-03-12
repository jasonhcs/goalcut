#!/usr/bin/env python3
"""步骤 7：在项目视频 IMG_7225_1.mp4 上验证检测效果"""
import glob
import os
import subprocess
import sys

# 1. 提取帧
FRAMES_DIR = "/tmp/test_frames_7225"
VIDEO = "/root/project/github.com/goalcut/test/野球场素材/IMG_7225_1.mp4"

if not os.path.exists(VIDEO):
    print(f"错误: 视频文件不存在: {VIDEO}")
    sys.exit(1)

os.makedirs(FRAMES_DIR, exist_ok=True)
existing = glob.glob(os.path.join(FRAMES_DIR, "frame_*.jpg"))
if len(existing) < 10:
    print(f"正在从视频提取帧 (fps=3)...")
    subprocess.run([
        "ffmpeg", "-i", VIDEO, "-vf", "fps=3",
        os.path.join(FRAMES_DIR, "frame_%06d.jpg"),
        "-y", "-loglevel", "warning"
    ], check=True)
    existing = glob.glob(os.path.join(FRAMES_DIR, "frame_*.jpg"))

frames = sorted(existing)
print(f"共 {len(frames)} 帧待检测")

# 2. 加载模型
from ultralytics import YOLO

MODEL_PATH = "/root/project/github.com/goalcut/ai-engine/models/basketball_v1/weights/best.pt"
model = YOLO(MODEL_PATH)
print(f"模型类别: {model.names}")

# 3. 逐帧检测
total_ball = 0
total_rim = 0
total_person = 0
frames_with_ball = 0
frames_with_rim = 0

for i, f in enumerate(frames):
    results = model(f, conf=0.25, verbose=False)
    found_ball = False
    found_rim = False
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            if cls_name in ("ball", "basketball"):
                total_ball += 1
                found_ball = True
            elif cls_name in ("rim", "hoop", "basket"):
                total_rim += 1
                found_rim = True
            elif cls_name in ("person", "player"):
                total_person += 1
    if found_ball:
        frames_with_ball += 1
    if found_rim:
        frames_with_rim += 1
    
    if (i + 1) % 50 == 0 or i == len(frames) - 1:
        print(f"  进度: {i+1}/{len(frames)} 帧")

# 4. 输出结果
print()
print("=" * 60)
print("项目视频检测结果 (IMG_7225_1.mp4)")
print("=" * 60)
n = len(frames)
ball_cov = frames_with_ball / n if n > 0 else 0
rim_cov = frames_with_rim / n if n > 0 else 0

print(f"  总帧数: {n}")
print(f"  球体检测: {total_ball} 次命中, {frames_with_ball}/{n} 帧 = {ball_cov:.1%} 覆盖率")
print(f"  篮筐检测: {total_rim} 次命中, {frames_with_rim}/{n} 帧 = {rim_cov:.1%} 覆盖率")
print(f"  人体检测: {total_person} 次命中")
print()

ok_ball = "✅ 达标" if ball_cov >= 0.20 else "❌ 不达标"
ok_rim = "✅ 达标" if rim_cov >= 0.50 else "❌ 不达标"
print("验收标准:")
print(f"  球体覆盖率 = {ball_cov:.1%}  (要求 >= 20%)  {ok_ball}")
print(f"  篮筐覆盖率 = {rim_cov:.1%}  (要求 >= 50%)  {ok_rim}")
