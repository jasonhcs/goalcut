#!/usr/bin/env python3
"""
诊断脚本：YOLO basketball_v1 模型 hoop 检测
1. 打印模型类别列表
2. 对测试视频 14s 和 64s 附近帧运行 hoop 检测，打印 bbox 详情
"""
import os
import sys

# 确保 ai-engine 在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

def main():
    from ultralytics import YOLO
    import cv2
    import tempfile

    # 1. 加载模型并打印类别
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "ai-engine", "models",
        "basketball_v1", "weights", "best.pt"
    )
    if not os.path.exists(model_path):
        print(f"错误: 模型不存在 {model_path}")
        return 1

    model = YOLO(model_path)
    print("=" * 60)
    print("1. YOLO basketball_v1 模型类别列表")
    print("=" * 60)
    print(f"model.names = {model.names}")
    print()

    # 查找 hoop 相关类别
    hoop_classes = []
    for cls_id, name in model.names.items():
        if name.lower() in ('hoop', 'basket', 'rim', 'backboard',
                            'basketball-hoop', 'basketball_hoop',
                            'ring', 'basket-rim'):
            hoop_classes.append(cls_id)
    print(f"hoop 相关类别 ID: {hoop_classes}")
    if hoop_classes:
        for cid in hoop_classes:
            print(f"  - class {cid}: {model.names.get(cid, '?')}")
    print()

    # 2. 从视频提取 14s 和 64s 附近的帧
    video_path = os.path.join(
        os.path.dirname(__file__), "..", "野球场素材", "IMG_7225_1.mp4"
    )
    if not os.path.exists(video_path):
        print(f"错误: 视频不存在 {video_path}")
        return 1

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    target_times = [13.0, 14.0, 15.0, 63.0, 64.0, 65.0]  # 14s 和 64s 附近多帧
    frame_paths = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for t in target_times:
            frame_path = os.path.join(tmpdir, f"frame_{int(t)}s.jpg")
            cmd = [
                "ffmpeg", "-y", "-ss", str(t), "-i", video_path,
                "-vframes", "1", "-q:v", "2", frame_path
            ]
            import subprocess
            subprocess.run(cmd, capture_output=True, timeout=10)
            if os.path.exists(frame_path):
                frame_paths.append((t, frame_path))

        if not frame_paths:
            print("错误: 无法提取视频帧")
            return 1

        print("=" * 60)
        print("2. 对指定时间点帧运行 YOLO hoop 检测")
        print("=" * 60)

        all_hoop_bboxes = []
        for t, fpath in frame_paths:
            print(f"\n--- 时间点 {t}s 附近帧: {fpath} ---")
            results = model(fpath, conf=0.3, imgsz=640, verbose=False)

            for r in results:
                if r.boxes is None:
                    print("  无检测框")
                    continue
                hoop_count = 0
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls in hoop_classes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        w = x2 - x1
                        h = y2 - y1
                        hoop_count += 1
                        bbox_info = {
                            "time": t, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "cx": cx, "cy": cy, "w": w, "h": h, "conf": conf
                        }
                        all_hoop_bboxes.append(bbox_info)
                        print(f"  hoop #{hoop_count}: xyxy=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) "
                              f"cx={cx:.1f} cy={cy:.1f} w={w:.1f} h={h:.1f} conf={conf:.3f}")
                if hoop_count == 0:
                    print("  未检测到 hoop 类别")

        # 3. 汇总 hoop bbox 典型尺寸
        print("\n" + "=" * 60)
        print("3. hoop 检测典型 bbox 尺寸汇总")
        print("=" * 60)
        if all_hoop_bboxes:
            import numpy as np
            arr = np.array([(b["w"], b["h"], b["conf"]) for b in all_hoop_bboxes])
            print(f"  检测数量: {len(all_hoop_bboxes)}")
            print(f"  宽度 w:  min={arr[:,0].min():.1f} max={arr[:,0].max():.1f} "
                  f"mean={arr[:,0].mean():.1f} median={np.median(arr[:,0]):.1f}")
            print(f"  高度 h:  min={arr[:,1].min():.1f} max={arr[:,1].max():.1f} "
                  f"mean={arr[:,1].mean():.1f} median={np.median(arr[:,1]):.1f}")
            print(f"  置信度: min={arr[:,2].min():.3f} max={arr[:,2].max():.3f} "
                  f"mean={arr[:,2].mean():.3f}")
        else:
            print("  无 hoop 检测结果")

    return 0

if __name__ == "__main__":
    sys.exit(main())
