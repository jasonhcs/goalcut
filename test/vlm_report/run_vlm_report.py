#!/usr/bin/env python3
"""
GoalCut VLM 标注报告生成器
==========================
批量/单次运行 VLM 扫描，将标注结果输出到 test/vlm_report/ 目录。

特性：
  - 报告文件名包含大模型 provider、model 和关键扫描参数
  - 相同 (视频 + 大模型 + 参数) 组合已有报告时自动跳过，不重复跑
  - 支持扫描 test/input/ 下所有视频，或指定单个视频

用法：
  # 扫描单个视频
  python3 run_vlm_report.py --video test/input/goalcut_demo_1.MP4

  # 扫描 test/input/ 下所有视频
  python3 run_vlm_report.py --all

  # 强制重新跑（忽略已有报告）
  python3 run_vlm_report.py --all --force

  # 自定义参数
  python3 run_vlm_report.py --video test/input/xxx.mp4 --window 8 --step 4

环境变量：
  VLM_PROVIDER   - dashscope / openai / gemini / ollama (默认 dashscope)
  VLM_API_KEY    - API Key
  VLM_MODEL      - 模型名称 (默认按 provider 自动选择)
  VLM_BASE_URL   - 自定义 API 端点
"""

import argparse
import hashlib
import json
import os
import sys
import glob
from pathlib import Path

# 项目路径
SCRIPT_DIR = Path(__file__).resolve().parent            # test/vlm_report/
TEST_DIR = SCRIPT_DIR.parent                             # test/
PROJECT_ROOT = TEST_DIR.parent                           # goalcut/
VERIFY_SCRIPT = TEST_DIR / "vlm_verify" / "verify_vlm.py"
CONFIG_FILE = TEST_DIR / "vlm_verify" / "verify_vlm_config.yaml"
INPUT_DIR = TEST_DIR / "input"
OUTPUT_DIR = TEST_DIR / "output"
GT_DIR = TEST_DIR / "ground_truth"
REPORT_DIR = SCRIPT_DIR                                  # test/vlm_report/

# 默认扫描参数（与 verify_vlm_config.yaml 一致）
DEFAULT_WINDOW = 6.0
DEFAULT_STEP = 3.0
DEFAULT_THRESHOLD = 0.5
DEFAULT_MERGE_WINDOW = 4.0
DEFAULT_NUM_FRAMES = 8


def load_config():
    """从 verify_vlm_config.yaml 读取默认配置"""
    if not CONFIG_FILE.exists():
        return {}
    try:
        import yaml
        with open(CONFIG_FILE, "r") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # 没有 pyyaml，手动解析关键字段
        config = {}
        with open(CONFIG_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or ":" not in line:
                    continue
                key, _, val = line.partition(":")
                val = val.strip().strip('"').strip("'")
                if val:
                    try:
                        config[key.strip()] = float(val) if "." in val else val
                    except ValueError:
                        config[key.strip()] = val
        return config


def get_scan_params(args):
    """获取最终扫描参数（命令行 > 配置文件 > 默认值）"""
    config = load_config()
    scan_cfg = config.get("scan", {}) if isinstance(config, dict) else {}

    window = args.window or (scan_cfg.get("window_seconds") if isinstance(scan_cfg, dict) else None) or DEFAULT_WINDOW
    step = args.step or (scan_cfg.get("step_seconds") if isinstance(scan_cfg, dict) else None) or DEFAULT_STEP
    threshold = args.threshold or (scan_cfg.get("goal_confidence_threshold") if isinstance(scan_cfg, dict) else None) or DEFAULT_THRESHOLD
    merge_window = args.merge_window or (scan_cfg.get("merge_window") if isinstance(scan_cfg, dict) else None) or DEFAULT_MERGE_WINDOW
    num_frames = args.num_frames or DEFAULT_NUM_FRAMES

    return {
        "window_seconds": float(window),
        "step_seconds": float(step),
        "goal_confidence_threshold": float(threshold),
        "merge_window": float(merge_window),
        "num_frames": int(num_frames),
    }


def get_vlm_params():
    """获取 VLM 模型参数"""
    provider = os.environ.get("VLM_PROVIDER", "dashscope")
    model = os.environ.get("VLM_MODEL", "")
    if not model:
        defaults = {
            "openai": "gpt-4o",
            "gemini": "gemini-1.5-pro",
            "ollama": "llava-13b",
            "dashscope": "qwen-vl-max",
        }
        model = defaults.get(provider, "gpt-4o")
    return provider, model


def make_report_filename(video_name: str, provider: str, model: str,
                         scan_params: dict) -> str:
    """
    生成带有模型和参数信息的报告文件名。
    格式: {video_name}__{provider}_{model}__w{window}s{step}.json
    """
    safe_model = model.replace("/", "-").replace(":", "-")
    window = scan_params["window_seconds"]
    step = scan_params["step_seconds"]
    return f"{video_name}__{provider}_{safe_model}__w{window:.0f}s{step:.0f}.json"


def find_existing_report(video_name: str, provider: str, model: str,
                         scan_params: dict) -> str:
    """检查是否已有相同参数的报告，返回路径或空字符串"""
    filename = make_report_filename(video_name, provider, model, scan_params)
    report_path = REPORT_DIR / filename
    if report_path.exists():
        # 进一步验证文件内容中的参数是否完全匹配
        try:
            with open(report_path, "r") as f:
                report = json.load(f)
            rp = report.get("vlm_config", {})
            rs = report.get("scan_config", {})
            cfg = rp if rp else rs
            if (report.get("provider") == provider and
                report.get("model") == model and
                cfg.get("window_seconds") == scan_params["window_seconds"] and
                cfg.get("step_seconds") == scan_params["step_seconds"]):
                return str(report_path)
        except (json.JSONDecodeError, KeyError):
            pass
    return ""


def find_detection(video_name: str) -> str:
    """查找视频对应的 detection.json"""
    for pattern in [
        OUTPUT_DIR / f"{video_name}_detection.json",
        OUTPUT_DIR / f"{video_name}_goalcut_detection.json",
    ]:
        if pattern.exists():
            return str(pattern)
    return ""


def find_ground_truth(video_name: str) -> str:
    """查找视频对应的 Ground Truth"""
    gt_path = GT_DIR / f"{video_name}.json"
    return str(gt_path) if gt_path.exists() else ""


def run_vlm_scan(video_path: str, provider: str, model: str,
                 scan_params: dict, force: bool = False) -> bool:
    """
    对单个视频运行 VLM 扫描并输出报告。
    如果已有相同参数的报告且不 force，则跳过。
    返回 True 表示成功或跳过，False 表示失败。
    """
    video_path = str(Path(video_path).resolve())
    video_name = Path(video_path).stem

    # 检查去重
    if not force:
        existing = find_existing_report(video_name, provider, model, scan_params)
        if existing:
            print(f"  [跳过] {video_name}: 已有相同模型+参数的报告")
            print(f"         {existing}")
            return True

    # 准备输出路径
    report_filename = make_report_filename(video_name, provider, model, scan_params)
    report_path = REPORT_DIR / report_filename

    # 查找 detection 和 ground truth
    detection = find_detection(video_name)
    ground_truth = find_ground_truth(video_name)

    print(f"\n{'='*60}")
    print(f"  视频: {video_name}")
    print(f"  模型: {provider}/{model}")
    print(f"  参数: window={scan_params['window_seconds']}s step={scan_params['step_seconds']}s")
    print(f"  报告: {report_path}")
    if detection:
        print(f"  检测: {detection}")
    if ground_truth:
        print(f"  GT:   {ground_truth}")
    print(f"{'='*60}")

    # 构建命令行
    # 直接导入 verify_vlm 的模块来避免子进程问题
    sys.path.insert(0, str(TEST_DIR / "vlm_verify"))
    import verify_vlm as vv

    vlm_client = vv.VLMClient(
        provider=provider,
        model=model,
        api_key=os.environ.get("VLM_API_KEY", ""),
        base_url=os.environ.get("VLM_BASE_URL", ""),
    )

    frame_sampler = vv.FrameSampler(
        num_frames=scan_params["num_frames"],
        max_dimension=768,
    )

    scanner = vv.VLMScanner(
        vlm_client=vlm_client,
        frame_sampler=frame_sampler,
        window_seconds=scan_params["window_seconds"],
        step_seconds=scan_params["step_seconds"],
        goal_confidence_threshold=scan_params["goal_confidence_threshold"],
        merge_window=scan_params["merge_window"],
    )

    # VLM 全视频扫描
    vlm_goals = scanner.scan_video(video_path)

    # 算法比对
    comparison = None
    detection_events = []
    if detection:
        with open(detection, "r") as f:
            detection_events = json.load(f)
        comparison = scanner.cross_compare(vlm_goals, detection_events, tolerance=3.0)
        vv._print_scan_report(comparison)

    # Ground Truth 比对
    gt_comparison = None
    if ground_truth:
        with open(ground_truth, "r") as f:
            gt = json.load(f)
        gt_timestamps = [g["timestamp"] for g in gt.get("goals", [])]
        vlm_ts = [g.timestamp for g in vlm_goals]

        def match_ts(dets, gts, tol=3.0):
            tp_count, matched = 0, set()
            for d in sorted(dets):
                for gi, g in enumerate(gts):
                    if gi not in matched and abs(d - g) <= tol:
                        tp_count += 1
                        matched.add(gi)
                        break
            fp_count = len(dets) - tp_count
            fn_count = len(gts) - tp_count
            p = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
            r = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            return {"tp": tp_count, "fp": fp_count, "fn": fn_count,
                    "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}

        gt_comparison = {
            "ground_truth_goals": len(gt_timestamps),
            "vlm_scan_metrics": match_ts(vlm_ts, gt_timestamps),
        }
        if detection_events:
            algo_ts = [e.get("timestamp", 0) for e in detection_events]
            gt_comparison["algorithm_metrics"] = match_ts(algo_ts, gt_timestamps)

        print(f"\n  Ground Truth 对比:")
        vlm_m = gt_comparison["vlm_scan_metrics"]
        print(f"    VLM:  P={vlm_m['precision']:.1%} R={vlm_m['recall']:.1%} F1={vlm_m['f1']:.1%}")
        if "algorithm_metrics" in gt_comparison:
            algo_m = gt_comparison["algorithm_metrics"]
            print(f"    算法: P={algo_m['precision']:.1%} R={algo_m['recall']:.1%} F1={algo_m['f1']:.1%}")

    # 组装报告
    import time as _time
    scan_report = {
        "version": "2.0",
        "mode": "scan",
        "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "provider": provider,
        "model": model,
        "source_video": str(Path(video_path).relative_to(PROJECT_ROOT)),
        "vlm_config": {
            "provider": provider,
            "model": model,
            "window_seconds": scan_params["window_seconds"],
            "step_seconds": scan_params["step_seconds"],
            "goal_confidence_threshold": scan_params["goal_confidence_threshold"],
            "merge_window": scan_params["merge_window"],
            "num_frames": scan_params["num_frames"],
        },
        "scan_config": {
            "window_seconds": scan_params["window_seconds"],
            "step_seconds": scan_params["step_seconds"],
            "goal_confidence_threshold": scan_params["goal_confidence_threshold"],
            "merge_window": scan_params["merge_window"],
            "num_frames": scan_params["num_frames"],
        },
        "vlm_goals": [
            {
                "index": i,
                "timestamp": g.timestamp,
                "confidence": g.confidence,
                "reasoning": g.reasoning,
                "window": [g.window_start, g.window_end],
                "goal_frame_index": g.goal_frame_index,
            }
            for i, g in enumerate(vlm_goals)
        ],
    }
    if comparison:
        scan_report["comparison"] = comparison
    if gt_comparison:
        scan_report["ground_truth_comparison"] = gt_comparison

    # 写入报告
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(scan_report, f, ensure_ascii=False, indent=2)

    print(f"\n  报告已保存: {report_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="GoalCut VLM 标注报告生成器 — 输出到 test/vlm_report/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--video", "-v", metavar="PATH",
                        help="指定单个视频路径")
    parser.add_argument("--all", "-a", action="store_true",
                        help="扫描 test/input/ 下所有视频")
    parser.add_argument("--force", "-f", action="store_true",
                        help="强制重新跑，忽略已有报告")

    # 扫描参数（可选，覆盖配置文件）
    parser.add_argument("--window", type=float, default=None,
                        help="扫描窗口大小（秒）")
    parser.add_argument("--step", type=float, default=None,
                        help="扫描步长（秒）")
    parser.add_argument("--threshold", type=float, default=None,
                        help="VLM 进球判定最低置信度")
    parser.add_argument("--merge-window", type=float, default=None,
                        help="去重合并窗口（秒）")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="每窗口采样帧数")

    args = parser.parse_args()

    if not args.video and not args.all:
        parser.print_help()
        print("\n错误: 请指定 --video 或 --all")
        sys.exit(1)

    # 获取参数
    provider, model = get_vlm_params()
    scan_params = get_scan_params(args)

    print(f"VLM 标注报告生成器")
    print(f"  Provider: {provider}")
    print(f"  Model:    {model}")
    print(f"  Window:   {scan_params['window_seconds']}s")
    print(f"  Step:     {scan_params['step_seconds']}s")
    print(f"  输出目录: {REPORT_DIR}")

    # 收集视频列表
    videos = []
    if args.video:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = PROJECT_ROOT / video_path
        if not video_path.exists():
            print(f"错误: 视频不存在: {video_path}")
            sys.exit(1)
        videos.append(str(video_path))
    elif args.all:
        for ext in ["*.mp4", "*.MP4", "*.avi", "*.mov"]:
            videos.extend(glob.glob(str(INPUT_DIR / ext)))
        videos.sort()

    if not videos:
        print("未找到任何视频文件")
        sys.exit(1)

    print(f"\n共 {len(videos)} 个视频待处理")

    # 逐个处理
    success = 0
    skipped = 0
    failed = 0

    for video in videos:
        try:
            video_name = Path(video).stem
            if not args.force:
                existing = find_existing_report(video_name, provider, model, scan_params)
                if existing:
                    print(f"\n  [跳过] {video_name}: 已有相同模型+参数的报告")
                    print(f"         {existing}")
                    skipped += 1
                    continue

            ok = run_vlm_scan(video, provider, model, scan_params, force=args.force)
            if ok:
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n  [错误] {Path(video).stem}: {e}")
            failed += 1

    # 汇总
    print(f"\n{'='*60}")
    print(f"  完成: 成功={success} 跳过={skipped} 失败={failed}")
    print(f"  报告目录: {REPORT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
