#!/usr/bin/env python3
"""步骤 6：训练后逐类 mAP 验证"""
from ultralytics import YOLO

model = YOLO("models/basketball_v1/weights/best.pt")
print("模型类别:", model.names)
print()

# 运行验证
results = model.val(data="datasets/basketball/data.yaml", imgsz=416, verbose=True)

print()
print("=" * 60)
print("逐类验证结果:")
print("=" * 60)

names = model.names
for i, name in names.items():
    ap50 = results.box.ap50[i] if i < len(results.box.ap50) else 0
    ap = results.box.ap[i] if i < len(results.box.ap) else 0
    print(f"  class {i} ({name:>8s}): mAP@50 = {ap50:.1%}  mAP@50-95 = {ap:.1%}")

print()
print(f"  整体 mAP@50    = {results.box.map50:.1%}")
print(f"  整体 mAP@50-95 = {results.box.map:.1%}")
print()

# 验收判断
ball_ap = results.box.ap50[0]  # ball (class 0)
rim_ap = results.box.ap50[3]   # rim (class 3)
overall = results.box.map50

print("验收标准:")
ok_ball = "✅ 达标" if ball_ap >= 0.80 else "❌ 不达标"
ok_rim = "✅ 达标" if rim_ap >= 0.70 else "❌ 不达标"
ok_all = "✅ 达标" if overall >= 0.75 else "❌ 不达标"
print(f"  ball  mAP@50 = {ball_ap:.1%}  (要求 >= 80%)  {ok_ball}")
print(f"  rim   mAP@50 = {rim_ap:.1%}  (要求 >= 70%)  {ok_rim}")
print(f"  整体  mAP@50 = {overall:.1%}  (要求 >= 75%)  {ok_all}")
