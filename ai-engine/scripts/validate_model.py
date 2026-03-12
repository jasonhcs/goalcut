#!/usr/bin/env python3
"""
GoalCut 篮球专用模型验证脚本

功能：
1. 检查训练指标（mAP@50）
2. 在项目视频上验证检测效果
3. 对比新模型 vs COCO 模型

Usage:
    python3 validate_model.py
    python3 validate_model.py --video test/野球场素材/IMG_7225_1.mp4
"""
import argparse
import glob
import os
import sys
import subprocess

os.chdir("/root/project/github.com/goalcut")
sys.path.insert(0, "ai-engine")


def find_best_model():
    """查找最优模型"""
    candidates = [
        "ai-engine/models/basketball_v1/weights/best.pt",
        "ai-engine/runs/detect/models/basketball_v1/weights/best.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def validate_metrics(model_path):
    """在验证集上评估模型指标"""
    from ultralytics import YOLO

    model = YOLO(model_path)
    print(f"\n{'=' * 60}")
    print(f"模型: {model_path}")
    print(f"类别: {model.names}")
    print(f"{'=' * 60}")

    data_yaml = "ai-engine/datasets/basketball/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"警告: 数据集配置不存在: {data_yaml}")
        return

    results = model.val(data=data_yaml, imgsz=416)
    print(f"\n验证结果:")
    print(f"  mAP@50:    {results.box.map50:.4f}")
    print(f"  mAP@50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall:    {results.box.mr:.4f}")

    print(f"\n每类 AP@50:")
    for i, ap in enumerate(results.box.ap50):
        name = model.names.get(i, f"class_{i}")
        status = "✅" if ap >= 0.70 else "⚠️" if ap >= 0.50 else "❌"
        print(f"  {status} {name}: {ap:.4f}")

    # 检查是否达标
    ball_ap = results.box.ap50[0] if len(results.box.ap50) > 0 else 0
    rim_idx = None
    for k, v in model.names.items():
        if v == "rim":
            rim_idx = k
            break
    rim_ap = results.box.ap50[rim_idx] if rim_idx is not None and rim_idx < len(results.box.ap50) else 0

    print(f"\n达标检查:")
    print(f"  ball mAP@50 ≥ 80%: {'✅ PASS' if ball_ap >= 0.80 else '❌ FAIL'} ({ball_ap:.1%})")
    print(f"  rim  mAP@50 ≥ 70%: {'✅ PASS' if rim_ap >= 0.70 else '❌ FAIL'} ({rim_ap:.1%})")
    print(f"  整体 mAP@50 ≥ 75%: {'✅ PASS' if results.box.map50 >= 0.75 else '❌ FAIL'} ({results.box.map50:.1%})")


def validate_on_video(model_path, video_path):
    """在项目视频上验证检测效果"""
    from ultralytics import YOLO

    model = YOLO(model_path)
    print(f"\n{'=' * 60}")
    print(f"视频验证: {video_path}")
    print(f"模型: {model_path}")
    print(f"{'=' * 60}")

    # 提取帧
    tmp_dir = "/tmp/validate_frames"
    os.makedirs(tmp_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-i", video_path, "-vf", "fps=3",
        f"{tmp_dir}/frame_%06d.jpg", "-y"
    ], capture_output=True)

    frames = sorted(glob.glob(f"{tmp_dir}/frame_*.jpg"))
    print(f"帧数: {len(frames)}")

    if not frames:
        print("错误: 未提取到帧")
        return

    # 统计
    total_ball = 0
    total_rim = 0
    total_person = 0
    total_made = 0
    total_shoot = 0

    for f in frames:
        results = model(f, conf=0.25, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_name = model.names[int(box.cls[0])]
                if cls_name == "ball":
                    total_ball += 1
                elif cls_name == "rim":
                    total_rim += 1
                elif cls_name == "person":
                    total_person += 1
                elif cls_name == "made":
                    total_made += 1
                elif cls_name == "shoot":
                    total_shoot += 1

    print(f"\n检测统计 ({len(frames)} 帧):")
    print(f"  ball:   {total_ball:4d} 次 = {total_ball / len(frames):.1%} 覆盖率")
    print(f"  rim:    {total_rim:4d} 次 = {total_rim / len(frames):.1%} 覆盖率")
    print(f"  person: {total_person:4d} 次 = {total_person / len(frames):.1%} 覆盖率")
    print(f"  made:   {total_made:4d} 次")
    print(f"  shoot:  {total_shoot:4d} 次")

    print(f"\n达标检查:")
    ball_cov = total_ball / len(frames) if frames else 0
    rim_cov = total_rim / len(frames) if frames else 0
    print(f"  球体覆盖率 ≥ 20%: {'✅ PASS' if ball_cov >= 0.20 else '❌ FAIL'} ({ball_cov:.1%})")
    print(f"  篮筐覆盖率 ≥ 50%: {'✅ PASS' if rim_cov >= 0.50 else '❌ FAIL'} ({rim_cov:.1%})")

    # 清理
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


def compare_models(video_path):
    """对比新模型 vs COCO 模型"""
    from ultralytics import YOLO

    basketball_model = find_best_model()
    coco_model = "ai-engine/yolov8n.pt"

    if not basketball_model:
        print("篮球专用模型不存在，无法对比")
        return

    print(f"\n{'=' * 60}")
    print(f"模型对比: {video_path}")
    print(f"{'=' * 60}")

    # 提取帧
    tmp_dir = "/tmp/compare_frames"
    os.makedirs(tmp_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-i", video_path, "-vf", "fps=3",
        f"{tmp_dir}/frame_%06d.jpg", "-y"
    ], capture_output=True)
    frames = sorted(glob.glob(f"{tmp_dir}/frame_*.jpg"))

    for name, path, ball_classes, hoop_classes in [
        ("COCO (yolov8n.pt)", coco_model, [32], []),
        ("Basketball v1", basketball_model, None, None),
    ]:
        model = YOLO(path)

        if ball_classes is None:
            ball_classes = [k for k, v in model.names.items() if v.lower() in ("ball", "basketball")]
        if hoop_classes is None:
            hoop_classes = [k for k, v in model.names.items() if v.lower() in ("hoop", "rim", "basket")]

        balls = 0
        hoops = 0
        for f in frames:
            results = model(f, conf=0.25, verbose=False)
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls in ball_classes:
                        balls += 1
                    elif cls in hoop_classes:
                        hoops += 1

        print(f"\n{name}:")
        print(f"  Ball 检测: {balls} ({balls / max(len(frames), 1):.1%})")
        print(f"  Hoop 检测: {hoops} ({hoops / max(len(frames), 1):.1%})")

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="test/野球场素材/IMG_7225_1.mp4")
    parser.add_argument("--no-metrics", action="store_true", help="跳过数据集验证")
    parser.add_argument("--compare", action="store_true", help="对比新旧模型")
    args = parser.parse_args()

    model_path = find_best_model()
    if not model_path:
        print("❌ 篮球专用模型尚未生成，请等待训练完成")
        print("检查训练进度: tail -f ai-engine/train.log")
        sys.exit(1)

    print(f"✅ 找到模型: {model_path}")
    print(f"  大小: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")

    if not args.no_metrics:
        validate_metrics(model_path)

    if os.path.exists(args.video):
        validate_on_video(model_path, args.video)

        if args.compare:
            compare_models(args.video)
    else:
        print(f"\n视频不存在: {args.video}")


if __name__ == "__main__":
    main()
