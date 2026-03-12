#!/usr/bin/env python3
"""
GoalCut YOLOv8n 微调训练脚本

Usage:
    # 快速测试（少量数据，5 epochs）
    python3 train_basketball.py --quick

    # 完整训练
    python3 train_basketball.py

    # 自定义参数
    python3 train_basketball.py --epochs 100 --batch 16 --imgsz 640
"""

import argparse
import os
import sys
import shutil
import random
import glob


def create_subset(src_data_yaml, subset_size=500, seed=42):
    """创建小子集用于快速测试"""
    import yaml
    
    with open(src_data_yaml) as f:
        data = yaml.safe_load(f)
    
    base_dir = os.path.dirname(src_data_yaml)
    subset_dir = os.path.join(base_dir, "_subset")
    
    for split in ["train", "valid"]:
        for sub in ["images", "labels"]:
            os.makedirs(os.path.join(subset_dir, split, sub), exist_ok=True)
    
    # 从训练集随机采样
    train_img_dir = data["train"]
    train_imgs = sorted(glob.glob(os.path.join(train_img_dir, "*.jpg")))
    random.seed(seed)
    random.shuffle(train_imgs)
    selected_train = train_imgs[:subset_size]
    
    train_lbl_dir = train_img_dir.replace("/images", "/labels")
    for img_path in selected_train:
        basename = os.path.basename(img_path)
        lbl_name = os.path.splitext(basename)[0] + ".txt"
        lbl_path = os.path.join(train_lbl_dir, lbl_name)
        
        shutil.copy2(img_path, os.path.join(subset_dir, "train", "images", basename))
        if os.path.exists(lbl_path):
            shutil.copy2(lbl_path, os.path.join(subset_dir, "train", "labels", lbl_name))
    
    # 验证集取前100
    val_img_dir = data["val"]
    val_imgs = sorted(glob.glob(os.path.join(val_img_dir, "*.jpg")))[:100]
    val_lbl_dir = val_img_dir.replace("/images", "/labels")
    
    for img_path in val_imgs:
        basename = os.path.basename(img_path)
        lbl_name = os.path.splitext(basename)[0] + ".txt"
        lbl_path = os.path.join(val_lbl_dir, lbl_name)
        
        shutil.copy2(img_path, os.path.join(subset_dir, "valid", "images", basename))
        if os.path.exists(lbl_path):
            shutil.copy2(lbl_path, os.path.join(subset_dir, "valid", "labels", lbl_name))
    
    # 写 subset data.yaml
    subset_yaml = os.path.join(subset_dir, "data.yaml")
    subset_data = {
        "train": os.path.join(subset_dir, "train", "images"),
        "val": os.path.join(subset_dir, "valid", "images"),
        "nc": data["nc"],
        "names": data["names"],
    }
    with open(subset_yaml, 'w') as f:
        yaml.dump(subset_data, f, default_flow_style=False)
    
    print(f"子集创建完成: {len(selected_train)} train + {len(val_imgs)} valid")
    print(f"子集 YAML: {subset_yaml}")
    return subset_yaml


def main():
    parser = argparse.ArgumentParser(description="GoalCut YOLOv8n Basketball Fine-tuning")
    parser.add_argument("--quick", action="store_true", help="快速测试模式（500张训练，5 epochs）")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数（默认50）")
    parser.add_argument("--batch", type=int, default=8, help="批大小（默认8，CPU建议4-8）")
    parser.add_argument("--imgsz", type=int, default=416, help="输入尺寸（默认416）")
    parser.add_argument("--device", type=str, default="cpu", help="设备（cpu/0/1/...）")
    parser.add_argument("--patience", type=int, default=15, help="Early Stopping 耐心值")
    parser.add_argument("--resume", action="store_true", help="从上次中断处恢复训练")
    args = parser.parse_args()
    
    # 路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ai_engine_dir = os.path.dirname(script_dir)
    dataset_dir = os.path.join(ai_engine_dir, "datasets", "basketball")
    data_yaml = os.path.join(dataset_dir, "data.yaml")
    base_model = os.path.join(ai_engine_dir, "yolov8n.pt")
    
    if not os.path.exists(data_yaml):
        print(f"错误: 数据集配置不存在: {data_yaml}")
        print("请先运行 download_dataset2.py 下载数据集")
        sys.exit(1)
    
    if not os.path.exists(base_model):
        base_model = "yolov8n.pt"
        print(f"使用默认模型路径: {base_model}")
    
    # 快速测试模式
    if args.quick:
        print("=" * 60)
        print("快速测试模式")
        print("=" * 60)
        try:
            import yaml
        except ImportError:
            os.system(f"{sys.executable} -m pip install pyyaml -q")
            import yaml
        
        data_yaml = create_subset(data_yaml, subset_size=500)
        args.epochs = 5
        args.patience = 3
    
    # 恢复训练
    if args.resume:
        last_pt = os.path.join(ai_engine_dir, "models", "basketball_v1", "weights", "last.pt")
        if os.path.exists(last_pt):
            base_model = last_pt
            print(f"恢复训练: {last_pt}")
        else:
            print(f"未找到 last.pt，从头开始训练")
    
    print(f"\n{'=' * 60}")
    print(f"GoalCut YOLOv8n 篮球微调训练")
    print(f"{'=' * 60}")
    print(f"  基础模型: {base_model}")
    print(f"  数据集: {data_yaml}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch}")
    print(f"  Image Size: {args.imgsz}")
    print(f"  Device: {args.device}")
    print(f"  Patience: {args.patience}")
    print(f"{'=' * 60}\n")
    
    # 开始训练
    from ultralytics import YOLO
    
    model = YOLO(base_model)
    
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=os.path.join(ai_engine_dir, "models"),
        name="basketball_v1",
        patience=args.patience,
        lr0=0.01,
        lrf=0.01,
        augment=True,
        verbose=True,
        workers=2,
        device=args.device,
        exist_ok=True,  # 允许覆盖同名项目
    )
    
    # 输出训练结果
    print(f"\n{'=' * 60}")
    print(f"训练完成!")
    print(f"{'=' * 60}")
    
    best_pt = os.path.join(ai_engine_dir, "models", "basketball_v1", "weights", "best.pt")
    if os.path.exists(best_pt):
        print(f"最优模型: {best_pt}")
        
        # 验证
        print("\n运行验证...")
        val_model = YOLO(best_pt)
        val_results = val_model.val(data=data_yaml, imgsz=args.imgsz)
        
        print(f"\n验证结果:")
        print(f"  mAP@50: {val_results.box.map50:.4f}")
        print(f"  mAP@50-95: {val_results.box.map:.4f}")
        
        # 打印每个类别的 mAP
        names = val_model.names
        for i, ap in enumerate(val_results.box.ap50):
            print(f"  {names.get(i, f'class_{i}')}: AP@50 = {ap:.4f}")
    else:
        print(f"警告: 未找到 best.pt")


if __name__ == "__main__":
    main()
