#!/usr/bin/env python3
"""
GoalCut VLM 闭环反馈引擎 v2.0
================================
读取 VLM 验证报告（包括全视频扫描报告），生成结构化的算法调参反馈，供 Agent 消费。

Agent 闭环工作流:
  1. 运行 GoalCut 处理视频 → 得到 detection.json
  2. 运行 verify_vlm.py --scan → 得到 vlm_scan_report.json（VLM 独立发现所有进球）
  3. 运行 vlm_feedback.py → 得到 feedback.json（本脚本）
  4. Agent 读取 feedback.json，据此调整算法参数或代码
  5. 回到步骤 1，形成闭环

用法:
  # 分析全视频扫描报告
  python3 vlm_feedback.py \\
    --report vlm_scan_report.json \\
    --config ../configs/config.yaml \\
    --output feedback.json

  # 分析片段验证报告（兼容 v1.0）
  python3 vlm_feedback.py \\
    --report vlm_report.json \\
    --config ../configs/config.yaml \\
    --output feedback.json
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional


def load_report(path: str) -> dict:
    """加载 VLM 验证报告"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_goalcut_config(path: str) -> dict:
    """加载 GoalCut 配置（YAML → dict）"""
    try:
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        config = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if ":" in line and not line.startswith("#"):
                    parts = line.split(":", 1)
                    key = parts[0].strip()
                    val = parts[1].strip().strip('"')
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                    config[key] = val
        return config


# ============================================================================
# 片段验证模式分析（兼容 v1.0）
# ============================================================================

def analyze_false_positives(results: List[dict]) -> dict:
    """分析疑似误检，提取模式和建议"""
    fp_results = [r for r in results if r.get("classification") == "FP_SUSPECT"]
    if not fp_results:
        return {"count": 0, "patterns": [], "actions": []}

    patterns = []
    actions = []

    low_conf_fps = [r for r in fp_results if r["algorithm_confidence"] < 0.60]
    mid_conf_fps = [r for r in fp_results if 0.60 <= r["algorithm_confidence"] < 0.75]
    high_conf_fps = [r for r in fp_results if r["algorithm_confidence"] >= 0.75]

    if low_conf_fps:
        max_conf = max(r["algorithm_confidence"] for r in low_conf_fps)
        patterns.append({
            "type": "low_confidence_fp",
            "count": len(low_conf_fps),
            "max_confidence": round(max_conf, 3),
            "description": f"{len(low_conf_fps)} 个误检的置信度 < 0.60，"
                          f"说明全局阈值可以安全上调"
        })
        actions.append({
            "action": "raise_confidence_threshold",
            "parameter": "detection.confidence_threshold",
            "current_value": None,
            "suggested_value": round(max_conf + 0.03, 2),
            "impact": "消除低置信度误检，不影响高置信度正确检测",
            "risk": "low",
        })

    if high_conf_fps:
        patterns.append({
            "type": "high_confidence_fp",
            "count": len(high_conf_fps),
            "description": f"{len(high_conf_fps)} 个误检的置信度 >= 0.75，"
                          f"说明存在算法逻辑层面的问题，不能仅靠调阈值解决"
        })
        actions.append({
            "action": "investigate_algorithm_logic",
            "description": "高置信度误检需要深入分析算法逻辑",
            "clips": [
                {
                    "timestamp": r["clip"]["source_event_timestamp"],
                    "algo_conf": r["algorithm_confidence"],
                    "vlm_reasoning": r["vlm_judgment"].get("reasoning", ""),
                    "algo_detail": r["clip"].get("source_event_detail", ""),
                }
                for r in high_conf_fps
            ],
            "risk": "medium",
        })

    channel_stats: Dict[str, int] = {}
    for r in fp_results:
        detail = r["clip"].get("source_event_detail", "")
        for ch_name in ["1a", "1b", "2", "3", "4", "5"]:
            markers = [f"ch{ch_name}", f"channel_{ch_name}", f"ch.{ch_name}"]
            if any(m in detail.lower() for m in markers):
                channel_stats[ch_name] = channel_stats.get(ch_name, 0) + 1

    for ch, count in sorted(channel_stats.items(), key=lambda x: -x[1]):
        patterns.append({
            "type": "channel_fp",
            "channel": ch,
            "count": count,
            "description": f"通道 {ch} 产生了 {count}/{len(fp_results)} 个误检"
        })
        actions.append({
            "action": "tune_channel",
            "channel": ch,
            "description": f"审查通道 {ch} 的检测阈值或降低其跨通道验证权重",
            "risk": "low",
        })

    return {
        "count": len(fp_results),
        "low_conf": len(low_conf_fps),
        "mid_conf": len(mid_conf_fps),
        "high_conf": len(high_conf_fps),
        "patterns": patterns,
        "actions": actions,
    }


def analyze_false_negatives(report: dict) -> dict:
    """分析负样本审查中发现的疑似漏检"""
    neg_checks = report.get("negative_sample_checks", [])
    if not neg_checks:
        return {"count": 0, "missed": [], "actions": []}

    missed = [r for r in neg_checks if r["vlm_judgment"].get("is_goal", False)]
    if not missed:
        return {"count": 0, "missed": [], "actions": []}

    actions = []
    if missed:
        actions.append({
            "action": "lower_threshold_or_enhance_channels",
            "description": f"在 {len(missed)} 个非检测区间发现疑似进球，"
                          f"可能需要降低阈值或增强弱通道灵敏度",
            "time_ranges": [
                r["time_range"] for r in missed
            ],
            "risk": "medium",
        })

    return {
        "count": len(missed),
        "missed": [
            {
                "time_range": r["time_range"],
                "vlm_confidence": r["vlm_judgment"]["confidence"],
                "vlm_reasoning": r["vlm_judgment"].get("reasoning", ""),
            }
            for r in missed
        ],
        "actions": actions,
    }


# ============================================================================
# 全视频扫描模式分析（v2.0 新增）
# ============================================================================

def analyze_scan_comparison(report: dict) -> dict:
    """分析全视频扫描比对结果，生成详细反馈"""
    comparison = report.get("comparison", {})
    if not comparison:
        return {
            "mode": "scan_only",
            "vlm_goals": report.get("vlm_goals", []),
            "message": "仅 VLM 扫描，无算法比对数据",
            "actions": [],
        }

    metrics = comparison.get("metrics", {})
    fps = comparison.get("false_positives", [])
    fns = comparison.get("false_negatives", [])
    tp_pairs = comparison.get("matched_pairs", [])

    actions = []

    # FP 分析：算法误检
    if fps:
        low_conf_fps = [f for f in fps if f.get("algo_confidence", 0) < 0.60]
        high_conf_fps = [f for f in fps if f.get("algo_confidence", 0) >= 0.70]

        if low_conf_fps:
            max_fp_conf = max(f["algo_confidence"] for f in low_conf_fps)
            actions.append({
                "action": "raise_confidence_threshold",
                "parameter": "detection.confidence_threshold",
                "current_value": None,
                "suggested_value": round(max_fp_conf + 0.05, 2),
                "impact": f"消除 {len(low_conf_fps)} 个低置信度误检",
                "risk": "low",
                "fp_timestamps": [f["timestamp"] for f in low_conf_fps],
            })

        if high_conf_fps:
            actions.append({
                "action": "investigate_algorithm_logic",
                "description": f"{len(high_conf_fps)} 个高置信度误检(>=0.70)，"
                              f"需要深入分析算法逻辑",
                "fp_details": [
                    {
                        "timestamp": f["timestamp"],
                        "algo_confidence": f["algo_confidence"],
                        "algo_detail": f.get("algo_detail", ""),
                    }
                    for f in high_conf_fps
                ],
                "risk": "high",
            })

        # 分析误检来源通道
        channel_fps: Dict[str, int] = {}
        for f in fps:
            detail = f.get("algo_detail", "")
            for ch in ["1a", "1b", "2", "3", "4", "5"]:
                if f"ch{ch}" in detail.lower() or f"channel_{ch}" in detail.lower():
                    channel_fps[ch] = channel_fps.get(ch, 0) + 1
        for ch, count in sorted(channel_fps.items(), key=lambda x: -x[1]):
            if count >= 1:
                actions.append({
                    "action": "tune_channel",
                    "channel": ch,
                    "description": f"通道 {ch} 产生了 {count} 个误检",
                    "risk": "low",
                })

    # FN 分析：算法漏检
    if fns:
        high_conf_fns = [f for f in fns if f.get("vlm_confidence", 0) >= 0.80]
        low_conf_fns = [f for f in fns if f.get("vlm_confidence", 0) < 0.80]

        if high_conf_fns:
            actions.append({
                "action": "fix_false_negatives_high_priority",
                "description": f"{len(high_conf_fns)} 个高确信漏检(VLM conf>=0.80)，"
                              f"算法在这些时间段完全未检测到进球",
                "fn_details": [
                    {
                        "timestamp": f["timestamp"],
                        "vlm_confidence": f["vlm_confidence"],
                        "vlm_reasoning": f.get("vlm_reasoning", ""),
                        "scan_window": f.get("scan_window", []),
                    }
                    for f in high_conf_fns
                ],
                "suggested_actions": [
                    "检查这些时间段的原始帧，确认 YOLO 是否检测到篮球和篮筐",
                    "检查追踪器是否在这些区间丢失了球体轨迹",
                    "考虑降低 confidence_threshold 或增强弱通道灵敏度",
                ],
                "risk": "high",
            })

        if low_conf_fns:
            actions.append({
                "action": "investigate_possible_false_negatives",
                "description": f"{len(low_conf_fns)} 个低确信漏检(VLM conf<0.80)，"
                              f"可能是 VLM 误判或边界情况",
                "fn_timestamps": [f["timestamp"] for f in low_conf_fns],
                "risk": "medium",
            })

    # 时间偏差分析
    if tp_pairs:
        time_diffs = [p["time_diff"] for p in tp_pairs]
        avg_diff = sum(time_diffs) / len(time_diffs)
        max_diff = max(time_diffs)
        if avg_diff > 1.5:
            actions.append({
                "action": "improve_timestamp_accuracy",
                "description": f"匹配事件的平均时间偏差 {avg_diff:.1f}s 偏大 "
                              f"(最大 {max_diff:.1f}s)，"
                              f"建议优化进球时间戳的精度",
                "risk": "low",
            })

    # 按 risk 排序
    risk_order = {"low": 0, "medium": 1, "high": 2}
    actions.sort(key=lambda x: -risk_order.get(x.get("risk", "medium"), 1))

    return {
        "mode": "scan_compare",
        "metrics": metrics,
        "fp_count": len(fps),
        "fn_count": len(fns),
        "tp_count": len(tp_pairs),
        "actions": actions,
    }


# ============================================================================
# 通用分析
# ============================================================================

def analyze_agreement_trend(report: dict) -> dict:
    """分析 VLM-算法一致性趋势"""
    mode = report.get("mode", "verify")

    if mode == "scan":
        comparison = report.get("comparison", {})
        metrics = comparison.get("metrics", {})
        tp = metrics.get("tp", 0)
        fp = metrics.get("fp", 0)
        fn = metrics.get("fn", 0)
        total = tp + fp + fn
        agreement_rate = tp / total if total > 0 else 0
    else:
        results = report.get("results", [])
        summary = report.get("summary", {})
        agreement_rate = summary.get("agreement_rate", 0)
        total = summary.get("total_clips", 0)

    if agreement_rate >= 0.9:
        assessment = "excellent"
        message = "算法与 VLM 高度一致，当前检测质量优秀"
    elif agreement_rate >= 0.75:
        assessment = "good"
        message = "算法表现良好，存在少量可优化项"
    elif agreement_rate >= 0.6:
        assessment = "needs_improvement"
        message = "一致性偏低，需要重点排查分歧片段"
    else:
        assessment = "poor"
        message = "一致性严重不足，算法可能存在系统性问题"

    return {
        "agreement_rate": round(agreement_rate, 3),
        "assessment": assessment,
        "message": message,
        "total": total,
    }


def generate_feedback(report: dict, goalcut_config: Optional[dict] = None) -> dict:
    """生成完整的闭环反馈报告（自动识别扫描/验证模式）"""
    mode = report.get("mode", "verify")
    trend = analyze_agreement_trend(report)

    if mode == "scan":
        # 全视频扫描模式分析
        scan_analysis = analyze_scan_comparison(report)
        all_actions = scan_analysis.get("actions", [])
        fp_count = scan_analysis.get("fp_count", 0)
        fn_count = scan_analysis.get("fn_count", 0)

        # 注入当前配置值
        if goalcut_config:
            det_cfg = goalcut_config.get("detection", {})
            for action in all_actions:
                if action.get("parameter") == "detection.confidence_threshold":
                    action["current_value"] = det_cfg.get("confidence_threshold")

        # 整体结论
        if trend["assessment"] == "excellent" and fp_count == 0 and fn_count == 0:
            overall_verdict = "PASS"
            overall_message = "VLM 扫描与算法检测完全一致，算法表现优秀"
        elif trend["assessment"] in ("excellent", "good") and fn_count == 0:
            overall_verdict = "PASS_WITH_SUGGESTIONS"
            overall_message = "算法整体良好，存在少量误检可优化"
        elif fn_count > 0:
            overall_verdict = "NEEDS_TUNING"
            overall_message = f"算法存在 {fn_count} 个漏检，需要优化召回率"
        elif fp_count > 3:
            overall_verdict = "NEEDS_TUNING"
            overall_message = f"算法存在 {fp_count} 个误检，需要优化精确率"
        else:
            overall_verdict = "PASS_WITH_SUGGESTIONS"
            overall_message = "算法表现可接受，有优化空间"

        feedback = {
            "version": "2.0",
            "mode": "scan",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_report": report.get("source_video", ""),
            "vlm_provider": report.get("provider", ""),
            "vlm_model": report.get("model", ""),

            "verdict": {
                "status": overall_verdict,
                "message": overall_message,
                "agreement_rate": trend["agreement_rate"],
                "assessment": trend["assessment"],
            },

            "scan_analysis": scan_analysis,

            "recommended_actions": all_actions,

            "tuning_suggestions": report.get("comparison", {}).get(
                "tuning_suggestions", []),

            "ground_truth_comparison": report.get("ground_truth_comparison"),

            "parameter_changes": [
                {
                    "parameter": a["parameter"],
                    "current": a.get("current_value"),
                    "suggested": a.get("suggested_value"),
                    "reason": a.get("impact", a.get("description", "")),
                }
                for a in all_actions
                if "parameter" in a and "suggested_value" in a
            ],
        }
    else:
        # 原有片段验证模式
        results = report.get("results", [])
        fp_analysis = analyze_false_positives(results)
        fn_analysis = analyze_false_negatives(report)

        all_actions = []
        all_actions.extend(fp_analysis.get("actions", []))
        all_actions.extend(fn_analysis.get("actions", []))

        if goalcut_config:
            det_cfg = goalcut_config.get("detection", {})
            for action in all_actions:
                if action.get("parameter") == "detection.confidence_threshold":
                    action["current_value"] = det_cfg.get("confidence_threshold")

        risk_order = {"low": 0, "medium": 1, "high": 2}
        all_actions.sort(key=lambda x: risk_order.get(x.get("risk", "medium"), 1))

        if trend["assessment"] == "excellent" and fp_analysis["count"] == 0:
            overall_verdict = "PASS"
            overall_message = "检测算法表现优秀，无需调整"
        elif trend["assessment"] in ("excellent", "good") and fp_analysis.get("high_conf", 0) == 0:
            overall_verdict = "PASS_WITH_SUGGESTIONS"
            overall_message = "检测算法整体良好，有小幅优化空间"
        elif trend["assessment"] == "needs_improvement":
            overall_verdict = "NEEDS_TUNING"
            overall_message = "检测算法需要参数调优"
        else:
            overall_verdict = "NEEDS_INVESTIGATION"
            overall_message = "检测算法存在较严重问题，需深入排查"

        feedback = {
            "version": "2.0",
            "mode": "verify",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_report": report.get("source_video", ""),
            "vlm_provider": report.get("provider", ""),
            "vlm_model": report.get("model", ""),

            "verdict": {
                "status": overall_verdict,
                "message": overall_message,
                "agreement_rate": trend["agreement_rate"],
                "assessment": trend["assessment"],
            },

            "false_positive_analysis": fp_analysis,
            "false_negative_analysis": fn_analysis,

            "recommended_actions": all_actions,

            "tuning_suggestions": report.get("tuning_suggestions", []),

            "ground_truth_comparison": report.get("ground_truth_comparison"),

            "parameter_changes": [
                {
                    "parameter": a["parameter"],
                    "current": a.get("current_value"),
                    "suggested": a.get("suggested_value"),
                    "reason": a.get("impact", a.get("description", "")),
                }
                for a in all_actions
                if "parameter" in a and "suggested_value" in a
            ],
        }

    return feedback


def print_feedback(feedback: dict):
    """以人类可读格式打印反馈"""
    mode = feedback.get("mode", "verify")
    v = feedback["verdict"]
    status_emoji = {
        "PASS": "✅", "PASS_WITH_SUGGESTIONS": "✅",
        "NEEDS_TUNING": "⚠️", "NEEDS_INVESTIGATION": "❌"
    }
    emoji = status_emoji.get(v["status"], "❓")

    print(f"\n{'='*60}")
    print(f"  GoalCut VLM 闭环反馈报告 (模式: {mode})")
    print(f"{'='*60}")
    print(f"  {emoji} 结论: {v['status']}")
    print(f"  {v['message']}")
    print(f"  VLM-算法一致率: {v['agreement_rate']:.1%}")

    if mode == "scan":
        scan = feedback.get("scan_analysis", {})
        metrics = scan.get("metrics", {})
        if metrics:
            print(f"\n  扫描比对指标:")
            print(f"    匹配(TP):     {metrics.get('tp', 0)}")
            print(f"    算法误检(FP): {metrics.get('fp', 0)}")
            print(f"    算法漏检(FN): {metrics.get('fn', 0)}")
            print(f"    Precision:    {metrics.get('precision', 0):.1%}")
            print(f"    Recall:       {metrics.get('recall', 0):.1%}")
            print(f"    F1:           {metrics.get('f1', 0):.1%}")
    else:
        fp = feedback.get("false_positive_analysis", {})
        if fp.get("count", 0) > 0:
            print(f"\n  误检分析 ({fp['count']} 个):")
            print(f"    低置信度(<0.60): {fp.get('low_conf', 0)}")
            print(f"    中置信度(0.60~0.75): {fp.get('mid_conf', 0)}")
            print(f"    高置信度(>=0.75): {fp.get('high_conf', 0)}")

        fn = feedback.get("false_negative_analysis", {})
        if fn.get("count", 0) > 0:
            print(f"\n  漏检分析 ({fn['count']} 个):")
            for m in fn.get("missed", []):
                r = m["time_range"]
                print(f"    - {r['start']:.1f}~{r['end']:.1f}s "
                      f"(VLM conf={m['vlm_confidence']:.2f})")

    changes = feedback.get("parameter_changes", [])
    if changes:
        print(f"\n  建议参数修改:")
        for c in changes:
            cur = c.get("current", "?")
            sug = c.get("suggested", "?")
            print(f"    {c['parameter']}: {cur} → {sug}")
            print(f"      理由: {c['reason']}")

    actions = feedback.get("recommended_actions", [])
    if actions:
        print(f"\n  推荐操作 ({len(actions)} 项):")
        for i, a in enumerate(actions, 1):
            risk_label = {"low": "低风险", "medium": "中风险", "high": "高风险"}
            print(f"    {i}. [{risk_label.get(a.get('risk',''), a.get('risk',''))}] "
                  f"{a.get('description', a.get('action', ''))}")

    sug = feedback.get("tuning_suggestions", [])
    if sug:
        print(f"\n  调参建议:")
        for i, s in enumerate(sug, 1):
            print(f"    {i}. {s}")

    gt = feedback.get("ground_truth_comparison")
    if gt:
        print(f"\n  Ground Truth 指标对比:")
        if "vlm_scan_metrics" in gt:
            vlm_m = gt["vlm_scan_metrics"]
            print(f"    VLM 扫描:  P={vlm_m['precision']:.1%} R={vlm_m['recall']:.1%} "
                  f"F1={vlm_m['f1']:.1%}")
        if "algorithm_metrics" in gt:
            algo = gt["algorithm_metrics"]
            print(f"    算法检测:  P={algo['precision']:.1%} R={algo['recall']:.1%} "
                  f"F1={algo['f1']:.1%}")
        if "vlm_filtered_metrics" in gt:
            vlm_f = gt["vlm_filtered_metrics"]
            print(f"    VLM过滤后: P={vlm_f['precision']:.1%} R={vlm_f['recall']:.1%} "
                  f"F1={vlm_f['f1']:.1%}")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="GoalCut VLM 闭环反馈引擎 v2.0 - 支持全视频扫描和片段验证模式"
    )
    parser.add_argument("--report", "-r", required=True,
                        help="VLM 验证报告 JSON 路径 (verify_vlm.py 的输出)")
    parser.add_argument("--config", "-c",
                        help="GoalCut 配置文件路径 (用于获取当前参数值)")
    parser.add_argument("--output", "-o", default="feedback.json",
                        help="反馈报告输出路径 (默认 feedback.json)")

    args = parser.parse_args()

    report = load_report(args.report)
    goalcut_config = None
    if args.config and os.path.exists(args.config):
        goalcut_config = load_goalcut_config(args.config)

    feedback = generate_feedback(report, goalcut_config)

    print_feedback(feedback)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)
    print(f"  反馈报告已保存: {args.output}")

    status = feedback["verdict"]["status"]
    if status in ("NEEDS_TUNING", "NEEDS_INVESTIGATION"):
        sys.exit(1)


if __name__ == "__main__":
    main()
