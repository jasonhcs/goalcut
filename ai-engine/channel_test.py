#!/usr/bin/env python3
"""
GoalCut 通道独立测试工具

支持对每个检测通道单独运行、单独输出结果，便于逐通道调参和优化。

用法:
    # 测试全部通道（独立输出每个通道结果）
    python channel_test.py --frames-dir /tmp/frames --video-duration 300

    # 只测试指定通道（逗号分隔）
    python channel_test.py --frames-dir /tmp/frames --video-duration 300 --channels 1a,5

    # 测试音频通道并调参
    python channel_test.py --frames-dir /tmp/frames --video-duration 300 \\
        --channels 5 --audio-file /tmp/audio.wav \\
        --audio-burst-ratio 4.0 --audio-max-duration 200

    # 对比 ground truth
    python channel_test.py --frames-dir /tmp/frames --video-duration 300 \\
        --ground-truth gt.json

通道编号:
    1a  - YOLO球体检测 + IOU追踪 + 几何判定
    1b  - 启发式检测（球在篮筐区域消失）
    2   - 人体运动模式检测
    3   - 局部运动突变检测（帧差分）
    4   - 篮网形变检测（光流法）
    5   - 音频事件检测（入网声 + 哨声）
"""

import argparse
import glob
import json
import os
import sys
import time
import numpy as np
from typing import List, Dict, Optional, Tuple

# 确保能 import 同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def log(msg):
    print(f"[channel_test] {msg}", flush=True)


# ============================================================
# 公共基础设施：篮筐检测、帧加载
# ============================================================

def load_frames_and_model(frames_dir: str, yolo_confidence: float):
    """加载帧文件和 YOLO 模型，返回公共上下文"""
    import cv2

    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not frame_files:
        log("错误: 未找到帧文件")
        return None

    sample_img = cv2.imread(frame_files[0])
    if sample_img is None:
        log("错误: 无法读取帧图片")
        return None

    img_h, img_w = sample_img.shape[:2]
    log(f"帧数: {len(frame_files)}, 尺寸: {img_w}x{img_h}")

    # 加载 YOLO（优先使用篮球专用模型）
    model = None
    _using_basketball_model = False
    try:
        from ultralytics import YOLO
        _ai_engine_dir = os.path.dirname(os.path.abspath(__file__))
        _basketball_model_candidates = [
            os.path.join(_ai_engine_dir, "models", "basketball_v1", "weights", "best.pt"),
            os.path.join(_ai_engine_dir, "runs", "detect", "models", "basketball_v1", "weights", "best.pt"),
        ]
        _BASKETBALL_MODEL = None
        for _candidate in _basketball_model_candidates:
            if os.path.exists(_candidate):
                _BASKETBALL_MODEL = _candidate
                break
        if _BASKETBALL_MODEL:
            model = YOLO(_BASKETBALL_MODEL)
            _using_basketball_model = True
            log(f"篮球专用 YOLO 模型加载成功")
            log(f"  模型类别: {model.names}")
        else:
            model = YOLO("yolov8n.pt")
            log("YOLOv8n COCO 模型加载成功")
    except Exception as e:
        log(f"YOLO 加载失败: {e} (通道 1a/1b/2 不可用)")

    return {
        "frame_files": frame_files,
        "img_w": img_w,
        "img_h": img_h,
        "model": model,
        "_using_basketball_model": _using_basketball_model,
    }


def detect_hoop_and_balls(ctx: dict, sample_fps: float, yolo_confidence: float):
    """运行 YOLO 球体检测 + 篮筐定位（通道 1a/1b 的前置步骤）"""
    from detect import HoopDetector

    frame_files = ctx["frame_files"]
    model = ctx["model"]
    img_w, img_h = ctx["img_w"], ctx["img_h"]

    if model is None:
        log("YOLO 不可用，跳过球体检测")
        return None, None, None

    # 根据模型类别自动判断球体 class ID
    _using_basketball_model = ctx.get("_using_basketball_model", False)
    if _using_basketball_model:
        BALL_CLASS = None
        for cls_id, name in model.names.items():
            if name.lower() in ("basketball", "ball", "sports ball"):
                BALL_CLASS = cls_id
                break
        if BALL_CLASS is None:
            BALL_CLASS = 0
        log(f"球体类别: class {BALL_CLASS} ({model.names.get(BALL_CLASS, '?')})")
    else:
        BALL_CLASS = 32  # COCO sports ball
    ball_yolo_conf = min(yolo_confidence, 0.25)
    ball_detections = []

    log("YOLO 球体检测中...")
    for i, frame_path in enumerate(frame_files):
        if i % 50 == 0:
            log(f"  进度: {i+1}/{len(frame_files)}")
        try:
            results = model(frame_path, conf=ball_yolo_conf, imgsz=640, verbose=False)
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls == BALL_CLASS:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        w = x2 - x1
                        h = y2 - y1
                        conf = float(box.conf[0])
                        ball_detections.append((i, cx, cy, w, h, conf))
        except Exception:
            continue

    log(f"球体检测完成: {len(ball_detections)} 次出现")

    # 篮筐检测
    hoop_detector = HoopDetector(img_w, img_h)
    hoop_detector.detect_from_frames(
        frame_files, model=model,
        ball_detections=ball_detections if ball_detections else None
    )
    hoop_cx, hoop_cy, hoop_w, hoop_h = hoop_detector.get_hoop_region()
    hoop_y_min, hoop_y_max = hoop_detector.get_hoop_y_range()
    log(f"篮筐: method={hoop_detector.detection_method}, "
        f"cx={hoop_cx:.0f}, cy={hoop_cy:.0f}, w={hoop_w:.0f}")

    hoop_info = {
        "cx": hoop_cx, "cy": hoop_cy, "w": hoop_w, "h": hoop_h,
        "y_min": hoop_y_min, "y_max": hoop_y_max,
        "method": hoop_detector.detection_method,
    }

    return ball_detections, hoop_info, hoop_detector


# ============================================================
# 各通道独立运行函数
# ============================================================

def run_channel_1a(ctx: dict, ball_detections, hoop_info,
                   sample_fps: float) -> List[Dict]:
    """通道 1a: IOU 追踪 + 几何判定"""
    if not ball_detections:
        log("通道 1a: 无球体检测数据，跳过")
        return []

    from tracker import IOUTracker, Detection as TrackerDetection
    from detect import judge_goal_by_trajectory, judge_goal_by_zone_disappear

    img_w, img_h = ctx["img_w"], ctx["img_h"]
    frame_files = ctx["frame_files"]

    max_lost = max(3, int(sample_fps * 1.5))
    dist_threshold = max(50, min(img_w, img_h) * 0.15)
    tracker = IOUTracker(
        iou_threshold=0.15, max_lost=max_lost,
        distance_threshold=dist_threshold,
    )

    frame_dets = {}
    frame_balls = {}
    for (fidx, cx, cy, w, h, conf) in ball_detections:
        if fidx not in frame_dets:
            frame_dets[fidx] = []
            frame_balls[fidx] = []
        frame_dets[fidx].append(TrackerDetection(fidx, cx, cy, w, h, conf))
        frame_balls[fidx].append((cx, cy, w, h, conf))

    total_frames = len(frame_files)
    for fidx in range(total_frames):
        tracker.update(frame_dets.get(fidx, []))

    all_tracks = tracker.get_all_tracks()
    log(f"通道 1a: {len(all_tracks)} 条有效轨迹")

    events = []
    hoop_cx = hoop_info["cx"]
    hoop_cy = hoop_info["cy"]
    hoop_w = hoop_info["w"]
    hoop_h = hoop_info["h"]

    for track in all_tracks:
        points = [(p.frame_idx, p.cx, p.cy, p.conf) for p in track.points]
        event = judge_goal_by_trajectory(
            points, hoop_cx, hoop_cy, hoop_w, hoop_h, sample_fps)
        if event:
            events.append(event)
            continue
        event = judge_goal_by_zone_disappear(
            points, hoop_cy, (hoop_info["y_min"], hoop_info["y_max"]),
            total_frames, sample_fps, frame_balls)
        if event:
            events.append(event)

    return events


def run_channel_1b(ctx: dict, ball_detections, hoop_info,
                   sample_fps: float, video_duration: float,
                   conf_threshold: float) -> List[Dict]:
    """通道 1b: 启发式检测"""
    if not ball_detections:
        log("通道 1b: 无球体检测数据，跳过")
        return []

    from detect import fallback_heuristic_v2

    frame_balls = {}
    for (fidx, cx, cy, w, h, conf) in ball_detections:
        if fidx not in frame_balls:
            frame_balls[fidx] = []
        frame_balls[fidx].append((cx, cy, w, h, conf))

    events = fallback_heuristic_v2(
        frame_balls, ctx["img_w"], ctx["img_h"],
        len(ctx["frame_files"]), sample_fps, video_duration,
        hoop_info["y_min"], hoop_info["y_max"], conf_threshold
    )
    return events


def run_channel_2(ctx: dict, sample_fps: float,
                  yolo_confidence: float) -> List[Dict]:
    """通道 2: 人体运动模式检测"""
    if ctx["model"] is None:
        log("通道 2: YOLO 不可用，跳过")
        return []

    from detect import detect_goals_by_person_motion
    events = detect_goals_by_person_motion(
        ctx["frame_files"], ctx["model"],
        ctx["img_w"], ctx["img_h"],
        sample_fps, yolo_confidence
    )
    return events


def run_channel_3(ctx: dict, sample_fps: float,
                  video_duration: float) -> List[Dict]:
    """通道 3: 局部运动突变检测"""
    from detect import detect_goals_by_motion
    events = detect_goals_by_motion(
        ctx["frame_files"], ctx["img_w"], ctx["img_h"],
        sample_fps, video_duration
    )
    return events


def run_channel_4(ctx: dict, hoop_info, sample_fps: float,
                  cooldown_s: float = 3.0,
                  burst_ratio: float = 2.0,
                  min_purity: float = 1.5,
                  use_legacy: bool = False) -> List[Dict]:
    """通道 4: 篮网有向光流穿越检测 (T-1.9.13 算法C)

    默认使用有向光流检测，传 use_legacy=True 回退到旧版平均幅度。
    """
    if hoop_info is None:
        log("通道 4: 无篮筐信息，跳过")
        return []

    if use_legacy:
        from net_deform import detect_net_deformation
        events = detect_net_deformation(
            ctx["frame_files"],
            hoop_info["cx"], hoop_info["cy"], hoop_info["w"],
            sample_fps=sample_fps,
            cooldown_s=cooldown_s,
            min_burst_ratio=burst_ratio,
        )
    else:
        from net_deform import detect_downward_flow_through_net
        events = detect_downward_flow_through_net(
            ctx["frame_files"],
            hoop_info["cx"], hoop_info["cy"], hoop_info["w"],
            sample_fps=sample_fps,
            net_height_ratio=0.8,
            min_burst_ratio=burst_ratio,
            min_purity=min_purity,
            cooldown_s=cooldown_s,
        )
    return events


def run_channel_5(audio_file: str, video_duration: float,
                  burst_ratio: float = 3.0,
                  max_duration_ms: float = 300.0) -> List[Dict]:
    """通道 5: 音频事件检测"""
    if not audio_file or not os.path.exists(audio_file):
        log("通道 5: 音频文件不存在，跳过")
        return []

    from audio_detect import detect_swish_events, detect_whistle_events

    swish = detect_swish_events(
        audio_file, video_duration,
        min_burst_ratio=burst_ratio,
        max_duration_ms=max_duration_ms,
    )
    whistle = detect_whistle_events(audio_file, video_duration)

    # 合并去重
    events = swish + whistle
    if not events:
        return []
    events.sort(key=lambda e: e["timestamp"])
    deduped = [events[0]]
    for ev in events[1:]:
        if ev["timestamp"] - deduped[-1]["timestamp"] > 2.0:
            deduped.append(ev)
        elif ev["confidence"] > deduped[-1]["confidence"]:
            deduped[-1] = ev
    return deduped


# ============================================================
# 评测：对比 Ground Truth
# ============================================================

def evaluate_channel(events: List[Dict], ground_truth: List[float],
                     tolerance: float = 3.0) -> Dict:
    """
    评测通道检测结果，对比 ground truth。

    Args:
        events:       通道输出的事件列表
        ground_truth: 真实进球时间戳列表（秒）
        tolerance:    匹配容差（秒），默认 ±3s

    Returns:
        {"tp", "fp", "fn", "precision", "recall", "f1", "matched", "missed", "false_alarms"}
    """
    detected_times = sorted([e["timestamp"] for e in events])
    gt_times = sorted(ground_truth)

    gt_matched = set()
    det_matched = set()

    # 贪心匹配：对每个检测事件找最近的未匹配 GT
    for di, dt in enumerate(detected_times):
        best_gi = -1
        best_dist = tolerance + 1
        for gi, gt in enumerate(gt_times):
            if gi in gt_matched:
                continue
            dist = abs(dt - gt)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_gi = gi
        if best_gi >= 0:
            gt_matched.add(best_gi)
            det_matched.add(di)

    tp = len(gt_matched)
    fp = len(detected_times) - len(det_matched)
    fn = len(gt_times) - len(gt_matched)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 详细信息
    matched = []
    for gi in sorted(gt_matched):
        # 找匹配的检测时间
        for di in sorted(det_matched):
            if abs(detected_times[di] - gt_times[gi]) <= tolerance:
                matched.append({
                    "gt": gt_times[gi],
                    "detected": detected_times[di],
                    "offset": round(detected_times[di] - gt_times[gi], 2),
                })
                break

    missed = [gt_times[gi] for gi in range(len(gt_times)) if gi not in gt_matched]
    false_alarms = [detected_times[di] for di in range(len(detected_times))
                    if di not in det_matched]

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "matched": matched,
        "missed": missed,
        "false_alarms": [round(t, 2) for t in false_alarms],
    }


# ============================================================
# 结果输出
# ============================================================

def print_channel_result(channel_name: str, events: List[Dict],
                         elapsed_s: float, gt: Optional[List[float]] = None):
    """格式化输出单通道结果"""
    print(f"\n{'='*60}")
    print(f"  通道 {channel_name}: {len(events)} 个候选事件  "
          f"(耗时 {elapsed_s:.1f}s)")
    print(f"{'='*60}")

    if not events:
        print("  (无事件)")
    else:
        events_sorted = sorted(events, key=lambda e: e["timestamp"])
        for i, e in enumerate(events_sorted):
            print(f"  [{i+1:2d}] {e['timestamp']:7.2f}s  "
                  f"conf={e['confidence']:.3f}  {e['detail']}")

    if gt is not None:
        metrics = evaluate_channel(events, gt)
        print(f"\n  --- 评测结果 (容差 ±3s) ---")
        print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}")
        print(f"  Precision={metrics['precision']:.1%}  "
              f"Recall={metrics['recall']:.1%}  F1={metrics['f1']:.1%}")
        if metrics["matched"]:
            match_strs = [f'{m["gt"]}s→{m["detected"]}s({m["offset"]:+.1f}s)' for m in metrics['matched']]
            print(f"  匹配: {', '.join(match_strs)}")
        if metrics["missed"]:
            miss_strs = [f'{t}s' for t in metrics['missed']]
            print(f"  漏检: {', '.join(miss_strs)}")
        if metrics["false_alarms"]:
            fa_strs = [f'{t}s' for t in metrics['false_alarms']]
            print(f"  误报: {', '.join(fa_strs)}")

    print()


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="GoalCut 通道独立测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
通道编号:
  1a  YOLO球体 + IOU追踪 + 几何判定
  1b  启发式检测（球消失）
  2   人体运动模式检测
  3   局部运动突变（帧差分）
  4   篮网形变（光流法）
  5   音频事件检测
  all 全部通道

示例:
  python channel_test.py --frames-dir /tmp/frames --video-duration 300 --channels 1a,5
  python channel_test.py --frames-dir /tmp/frames --video-duration 300 --channels 5 \\
      --audio-file /tmp/audio.wav --audio-burst-ratio 4.0
        """)

    parser.add_argument("--frames-dir", required=True, help="帧图片目录")
    parser.add_argument("--video-duration", type=float, required=True,
                        help="视频时长（秒）")
    parser.add_argument("--channels", default="all",
                        help="要测试的通道（逗号分隔），如: 1a,1b,2,3,4,5 或 all")
    parser.add_argument("--sample-fps", type=float, default=3.0,
                        help="采样帧率 (default: 3.0)")
    parser.add_argument("--confidence-threshold", type=float, default=0.55,
                        help="置信度阈值 (default: 0.55)")
    parser.add_argument("--yolo-confidence", type=float, default=0.25,
                        help="YOLO 检测置信度 (default: 0.25)")

    # 音频通道参数
    parser.add_argument("--audio-file", default="",
                        help="WAV 音频文件路径（通道 5 需要）")
    parser.add_argument("--audio-burst-ratio", type=float, default=3.0,
                        help="音频突发阈值倍数 (default: 3.0)")
    parser.add_argument("--audio-max-duration", type=float, default=300.0,
                        help="入网声最大持续时间 ms (default: 300)")

    # 篮网形变参数
    parser.add_argument("--net-burst-ratio", type=float, default=2.5,
                        help="篮网形变突发阈值倍数 (default: 2.5)")
    parser.add_argument("--net-cooldown", type=float, default=3.0,
                        help="篮网形变冷静期秒数 (default: 3.0)")

    # Ground truth
    parser.add_argument("--ground-truth", default="",
                        help="Ground truth JSON 文件路径")

    # 输出
    parser.add_argument("--output", default="",
                        help="结果输出 JSON 文件路径（可选）")

    args = parser.parse_args()

    # 解析通道列表
    all_channels = ["1a", "1b", "2", "3", "4", "5"]
    if args.channels.lower() == "all":
        channels = all_channels
    else:
        channels = [c.strip() for c in args.channels.split(",")]
        invalid = [c for c in channels if c not in all_channels]
        if invalid:
            log(f"未知通道: {invalid}，可用: {all_channels}")
            sys.exit(1)

    log(f"测试通道: {channels}")
    log(f"帧目录: {args.frames_dir}")
    log(f"视频时长: {args.video_duration}s")

    # 加载 Ground Truth
    gt = None
    if args.ground_truth:
        try:
            with open(args.ground_truth, 'r') as f:
                gt_data = json.load(f)
            # 支持两种格式：
            #   纯列表: [41.0, 52.0, ...]
            #   对象列表: [{"timestamp": 41.0, "note": "..."}, ...]
            #   字典包装: {"goals": [...]}
            raw = gt_data
            if isinstance(gt_data, dict) and "goals" in gt_data:
                raw = gt_data["goals"]
            if isinstance(raw, list) and len(raw) > 0:
                if isinstance(raw[0], dict):
                    gt = [float(g["timestamp"]) for g in raw]
                else:
                    gt = [float(t) for t in raw]
            else:
                gt = []
            log(f"加载 Ground Truth: {len(gt)} 个进球")
        except Exception as e:
            log(f"加载 Ground Truth 失败: {e}")

    # 判断需要哪些前置数据
    need_yolo = any(c in channels for c in ["1a", "1b", "2"])
    need_hoop = any(c in channels for c in ["1a", "1b", "4"])

    # 加载帧和模型
    ctx = load_frames_and_model(args.frames_dir, args.yolo_confidence)
    if ctx is None:
        sys.exit(1)

    # 球体检测和篮筐定位（共享数据，避免重复检测）
    ball_detections = None
    hoop_info = None
    if need_yolo or need_hoop:
        ball_detections, hoop_info, _ = detect_hoop_and_balls(
            ctx, args.sample_fps, args.yolo_confidence)

    # 逐通道运行
    results = {}

    for ch in channels:
        log(f"--- 运行通道 {ch} ---")
        t0 = time.time()

        if ch == "1a":
            events = run_channel_1a(
                ctx, ball_detections, hoop_info, args.sample_fps)
        elif ch == "1b":
            events = run_channel_1b(
                ctx, ball_detections, hoop_info,
                args.sample_fps, args.video_duration,
                args.confidence_threshold)
        elif ch == "2":
            events = run_channel_2(
                ctx, args.sample_fps, args.yolo_confidence)
        elif ch == "3":
            events = run_channel_3(
                ctx, args.sample_fps, args.video_duration)
        elif ch == "4":
            events = run_channel_4(
                ctx, hoop_info, args.sample_fps,
                cooldown_s=args.net_cooldown,
                burst_ratio=args.net_burst_ratio)
        elif ch == "5":
            events = run_channel_5(
                args.audio_file, args.video_duration,
                burst_ratio=args.audio_burst_ratio,
                max_duration_ms=args.audio_max_duration)
        else:
            events = []

        elapsed = time.time() - t0
        results[ch] = {
            "events": events,
            "count": len(events),
            "elapsed_s": round(elapsed, 1),
        }

        print_channel_result(f"通道{ch}", events, elapsed, gt)

    # 汇总
    print(f"\n{'='*60}")
    print(f"  汇总")
    print(f"{'='*60}")
    total_events = 0
    for ch in channels:
        r = results[ch]
        total_events += r["count"]
        print(f"  通道 {ch:3s}: {r['count']:3d} 个事件  ({r['elapsed_s']:.1f}s)")
    print(f"  {'─'*40}")
    print(f"  合计:     {total_events:3d} 个候选事件")

    if gt is not None:
        # 合并所有通道的事件做整体评测
        all_events = []
        for ch in channels:
            all_events.extend(results[ch]["events"])
        # 简单去重（3s 窗口取最高置信度）
        if all_events:
            all_events.sort(key=lambda e: e["timestamp"])
            merged = [all_events[0]]
            for ev in all_events[1:]:
                if ev["timestamp"] - merged[-1]["timestamp"] <= 3.0:
                    if ev["confidence"] > merged[-1]["confidence"]:
                        merged[-1] = ev
                else:
                    merged.append(ev)
            metrics = evaluate_channel(merged, gt)
            print(f"\n  --- 合并后整体评测 ({len(merged)} 个事件，容差 ±3s) ---")
            print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}")
            print(f"  Precision={metrics['precision']:.1%}  "
                  f"Recall={metrics['recall']:.1%}  F1={metrics['f1']:.1%}")
    print()

    # 输出到文件
    if args.output:
        output_data = {
            "channels_tested": channels,
            "params": {
                "sample_fps": args.sample_fps,
                "confidence_threshold": args.confidence_threshold,
                "yolo_confidence": args.yolo_confidence,
                "audio_burst_ratio": args.audio_burst_ratio,
                "audio_max_duration": args.audio_max_duration,
                "net_burst_ratio": args.net_burst_ratio,
                "net_cooldown": args.net_cooldown,
            },
            "results": {},
        }
        for ch in channels:
            r = results[ch]
            ch_data = {
                "count": r["count"],
                "elapsed_s": r["elapsed_s"],
                "events": r["events"],
            }
            if gt is not None:
                ch_data["metrics"] = evaluate_channel(r["events"], gt)
            output_data["results"][ch] = ch_data

        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        log(f"结果已保存: {args.output}")


if __name__ == "__main__":
    main()
