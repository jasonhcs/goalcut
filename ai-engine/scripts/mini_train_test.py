#!/usr/bin/env python3
"""极小子集训练测试（50张，1 epoch）- 验证流程可行性"""
import os, shutil, random, glob, yaml

os.chdir("/root/project/github.com/goalcut/ai-engine")

base = "datasets/basketball"
subset = f"{base}/_mini"
for s in ["train", "valid"]:
    for t in ["images", "labels"]:
        os.makedirs(f"{subset}/{s}/{t}", exist_ok=True)

# 50 train + 10 valid
train_imgs = sorted(glob.glob(f"{base}/train/images/*.jpg"))
random.seed(42)
random.shuffle(train_imgs)
for img in train_imgs[:50]:
    bn = os.path.basename(img)
    ln = os.path.splitext(bn)[0] + ".txt"
    shutil.copy2(img, f"{subset}/train/images/{bn}")
    lp = f"{base}/train/labels/{ln}"
    if os.path.exists(lp):
        shutil.copy2(lp, f"{subset}/train/labels/{ln}")

val_imgs = sorted(glob.glob(f"{base}/valid/images/*.jpg"))[:10]
for img in val_imgs:
    bn = os.path.basename(img)
    ln = os.path.splitext(bn)[0] + ".txt"
    shutil.copy2(img, f"{subset}/valid/images/{bn}")
    lp = f"{base}/valid/labels/{ln}"
    if os.path.exists(lp):
        shutil.copy2(lp, f"{subset}/valid/labels/{ln}")

d = {
    "train": os.path.abspath(f"{subset}/train/images"),
    "val": os.path.abspath(f"{subset}/valid/images"),
    "nc": 5,
    "names": {0: "ball", 1: "made", 2: "person", 3: "rim", 4: "shoot"},
}
with open(f"{subset}/data.yaml", "w") as f:
    yaml.dump(d, f)

print(f"Mini subset: 50 train + 10 valid")

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(
    data=f"{subset}/data.yaml",
    epochs=1,
    imgsz=416,
    batch=4,
    project="models",
    name="basketball_test",
    patience=1,
    workers=1,
    device="cpu",
    exist_ok=True,
    verbose=True,
)
print("训练完成!")
best = "models/basketball_test/weights/best.pt"
print(f"best.pt exists: {os.path.exists(best)}")
if os.path.exists(best):
    m = YOLO(best)
    print(f"类别: {m.names}")
