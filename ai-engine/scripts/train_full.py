#!/usr/bin/env python3
"""
GoalCut YOLOv8n 完整训练脚本
运行方式: nohup python3 train_full.py > train.log 2>&1 &

训练参数已针对 CPU 优化:
- epochs=50, patience=15 (early stopping)
- imgsz=416 (与数据集匹配)
- batch=8 (CPU 内存友好)
"""
import os
os.chdir("/root/project/github.com/goalcut/ai-engine")

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="datasets/basketball/data.yaml",
    epochs=50,
    imgsz=416,
    batch=8,
    project="models",
    name="basketball_v1",
    patience=15,
    lr0=0.01,
    lrf=0.01,
    augment=True,
    verbose=True,
    workers=2,
    device="cpu",
    exist_ok=True,
)

print("\n" + "=" * 60)
print("训练完成!")
best = "models/basketball_v1/weights/best.pt"
if os.path.exists(best):
    m = YOLO(best)
    print(f"类别: {m.names}")
    val = m.val(data="datasets/basketball/data.yaml", imgsz=416)
    print(f"mAP@50: {val.box.map50:.4f}")
    print(f"mAP@50-95: {val.box.map:.4f}")
    for i, ap in enumerate(val.box.ap50):
        print(f"  {m.names.get(i, f'class_{i}')}: AP@50 = {ap:.4f}")
else:
    # 检查 runs/detect 下
    alt = "runs/detect/models/basketball_v1/weights/best.pt"
    if os.path.exists(alt):
        print(f"模型保存在: {alt}")
        os.makedirs("models/basketball_v1/weights", exist_ok=True)
        import shutil
        shutil.copy2(alt, best)
        print(f"已复制到: {best}")
print("=" * 60)
