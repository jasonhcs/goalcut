#!/usr/bin/env python3
"""快速评估回归测试结果：比对检测结果与GT"""
import json

GT_PATH = "/root/project/github.com/goalcut/test/ground_truth/goalcut_demo_1.json"
DET_PATH = "/root/project/github.com/goalcut/test/output/regression/goalcut_demo_1_detection.json"
TOLERANCE = 3.0  # 时间容差（秒）

with open(GT_PATH) as f:
    gt_data = json.load(f)
with open(DET_PATH) as f:
    det_events = json.load(f)

gt_goals = [g["timestamp"] for g in gt_data["goals"]]
det_timestamps = [e["timestamp"] for e in det_events]

print(f"GT 进球: {len(gt_goals)} 个, 时间戳: {gt_goals}")
print(f"检测结果: {len(det_events)} 个, 时间戳: {det_timestamps}")
print(f"容差: ±{TOLERANCE}s")
print()

# 匹配
matched_gt = set()
matched_det = set()
for i, gt_t in enumerate(gt_goals):
    for j, det_t in enumerate(det_timestamps):
        if j in matched_det:
            continue
        if abs(gt_t - det_t) <= TOLERANCE:
            matched_gt.add(i)
            matched_det.add(j)
            print(f"  ✅ GT {gt_t:.1f}s ↔ Det {det_t:.2f}s (差={det_t-gt_t:+.2f}s)")
            break

print()
# 漏检
missed = [gt_goals[i] for i in range(len(gt_goals)) if i not in matched_gt]
if missed:
    print(f"  ❌ 漏检: {missed}")

# 误检
false_pos = [det_timestamps[j] for j in range(len(det_timestamps)) if j not in matched_det]
if false_pos:
    print(f"  ⚠️ 误检: {[f'{t:.2f}s' for t in false_pos]}")

# 指标
TP = len(matched_gt)
FP = len(false_pos)
FN = len(missed)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

print()
print("=" * 50)
print(f"Precision = {TP}/{TP+FP} = {precision:.0%}")
print(f"Recall    = {TP}/{TP+FN} = {recall:.0%}")
print(f"F1        = {2*precision*recall/(precision+recall):.0%}" if (precision+recall) > 0 else "F1 = 0%")
print()
if precision == 1.0 and recall == 1.0:
    print("🎉 回归测试通过! P=100%, R=100%")
else:
    print(f"⚠️ 回归测试{'部分' if recall >= 0.75 else ''}未达标 (目标 P=100%, R=100%)")
    if FP > 0:
        print(f"  建议: 调高 confidence_threshold 过滤 {FP} 个误检")
