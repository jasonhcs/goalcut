#!/usr/bin/env python3
"""
GoalCut GT 对比评估脚本

用法：
    python3 eval_gt.py <result.json> <gt.json> [tolerance_sec]

输出 Precision / Recall / F1 以及详细匹配表。
"""
import json
import sys


def evaluate(result_path, gt_path, tolerance=3.0):
    with open(result_path) as f:
        results = json.load(f)
    with open(gt_path) as f:
        gt_data = json.load(f)

    gt_goals = [g["timestamp"] for g in gt_data["goals"]]
    det_times = [e["timestamp"] for e in results]

    # 匹配：贪心最近匹配
    matched_gt = set()
    matched_det = set()

    for gi, gt in enumerate(gt_goals):
        best_di = None
        best_dist = tolerance + 1
        for di, dt in enumerate(det_times):
            if di in matched_det:
                continue
            dist = abs(dt - gt)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_di = di
        if best_di is not None:
            matched_gt.add(gi)
            matched_det.add(best_di)

    tp = len(matched_gt)
    fp = len(results) - tp
    fn = len(gt_goals) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*60}")
    print(f"GT: {gt_path}")
    print(f"Result: {result_path}")
    print(f"Tolerance: ±{tolerance}s")
    print(f"{'='*60}")

    print(f"\n--- GT 匹配详情 ---")
    for gi, gt in enumerate(gt_goals):
        if gi in matched_gt:
            # 找匹配的检测
            for di in matched_det:
                if abs(det_times[di] - gt) <= tolerance:
                    e = results[di]
                    print(f"  ✅ GT {gt:.1f}s → 检测 {e['timestamp']:.2f}s "
                          f"(Δ={e['timestamp']-gt:+.2f}s, conf={e['confidence']:.3f}, "
                          f"type={e.get('goal_type','?')}) {e.get('detail','')[:40]}")
                    break
        else:
            print(f"  ❌ GT {gt:.1f}s → 漏检")

    print(f"\n--- 误检详情 ---")
    for di, e in enumerate(results):
        if di not in matched_det:
            print(f"  ⚠️  {e['timestamp']:.2f}s conf={e['confidence']:.3f} "
                  f"type={e.get('goal_type','?')} causal={e.get('causal_score','?')} "
                  f"{e.get('detail','')[:50]}")

    print(f"\n--- 指标 ---")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"  Precision = {precision:.1%}  ({tp}/{tp+fp})")
    print(f"  Recall    = {recall:.1%}  ({tp}/{tp+fn})")
    print(f"  F1        = {f1:.1%}")

    perfect = (fp == 0 and fn == 0)
    if perfect:
        print(f"\n  🎉 完美匹配！所有 GT 进球检出，无误检。")
    print(f"{'='*60}\n")
    return perfect


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"用法: {sys.argv[0]} <result.json> <gt.json> [tolerance]")
        sys.exit(1)
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    perfect = evaluate(sys.argv[1], sys.argv[2], tol)
    sys.exit(0 if perfect else 1)
