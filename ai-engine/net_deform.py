#!/usr/bin/env python3
"""
GoalCut 篹网形变检测模块 (T-1.9.7)

原理：
  进球时球穿过篹网，导致篹网发生形变（晃动）。
  使用 Farneback 稠密光流法计算篹网 ROI 区域的运动幅度，
  检测运动幅度的突发峰值作为进球辅助信号。

野球场场景中，视觉信号较弱时，篹网形变是独立的补充信号。
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def get_net_roi(hoop_cx: float, hoop_cy: float, hoop_w: float,
                img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    根据篹筐位置推算篹网 ROI。

    篹网在篹筐正下方，高度约为篹筐宽度的 0.8 倍。
    返回 (x1, y1, x2, y2)
    """
    net_w = hoop_w * 1.2
    net_h = hoop_w * 0.8

    x1 = max(0, int(hoop_cx - net_w / 2))
    y1 = max(0, int(hoop_cy))
    x2 = min(img_w, int(hoop_cx + net_w / 2))
    y2 = min(img_h, int(hoop_cy + net_h))

    # ROI 太小则无意义
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    return (x1, y1, x2, y2)


def compute_optical_flow_magnitude(prev_gray: np.ndarray,
                                   curr_gray: np.ndarray,
                                   roi: Tuple[int, int, int, int]) -> float:
    """
    计算 ROI 区域的 Farneback 光流平均幅度。
    """
    x1, y1, x2, y2 = roi
    prev_roi = prev_gray[y1:y2, x1:x2]
    curr_roi = curr_gray[y1:y2, x1:x2]

    if prev_roi.size == 0 or curr_roi.size == 0:
        return 0.0

    try:
        flow = cv2.calcOpticalFlowFarneback(
            prev_roi, curr_roi,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return float(np.mean(mag))
    except Exception:
        return 0.0


def detect_net_deformation(
    frame_files: List[str],
    hoop_cx: float,
    hoop_cy: float,
    hoop_w: float,
    sample_fps: float,
    min_burst_ratio: float = 2.5,
    cooldown_s: float = 3.0,
) -> List[dict]:
    """
    检测篹网形变事件，返回候选进球事件列表。

    Args:
        frame_files:      帧文件列表（按顺序）
        hoop_cx/cy/w:     篹筐中心及宽度（像素）
        sample_fps:       采样帧率
        min_burst_ratio:  触发阈值 = 均值 + min_burst_ratio * 标准差
        cooldown_s:       两次事件最小间隔（秒）

    Returns:
        [{"frame_index", "timestamp", "confidence", "detail"}, ...]
    """
    if not frame_files:
        return []

    # 读取第一帧获取图像尺寸
    first = cv2.imread(frame_files[0], cv2.IMREAD_GRAYSCALE)
    if first is None:
        return []
    img_h, img_w = first.shape[:2]

    net_roi = get_net_roi(hoop_cx, hoop_cy, hoop_w, img_w, img_h)
    if net_roi is None:
        print(f"[net_deform] 篹网 ROI 无效，跳过形变检测", flush=True)
        return []

    x1, y1, x2, y2 = net_roi
    print(f"[net_deform] 篹网 ROI: ({x1},{y1})~({x2},{y2}), "
          f"hoop=({hoop_cx:.0f},{hoop_cy:.0f},w={hoop_w:.0f})", flush=True)

    # 计算每对相邻帧的光流幅度
    flow_scores = []  # [(frame_idx, score)]
    prev_gray = None

    for i, path in enumerate(frame_files):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            prev_gray = None
            continue

        if prev_gray is not None:
            score = compute_optical_flow_magnitude(prev_gray, gray, net_roi)
            flow_scores.append((i, score))

        prev_gray = gray

    if len(flow_scores) < 5:
        print("[net_deform] 有效帧数不足，跳过", flush=True)
        return []

    vals = np.array([s for _, s in flow_scores])
    mean_v = float(np.mean(vals))
    std_v = float(np.std(vals))

    # 标准差过小说明画面静止，无法区分信号
    if std_v < 0.3:
        print(f"[net_deform] 光流变化过小(std={std_v:.3f})，跳过", flush=True)
        return []

    burst_threshold = mean_v + min_burst_ratio * std_v
    print(f"[net_deform] 光流统计: mean={mean_v:.3f}, std={std_v:.3f}, "
          f"threshold={burst_threshold:.3f}", flush=True)

    # 检测突发峰值
    candidate_events = []
    last_event_frame = -int(cooldown_s * sample_fps) - 1

    for fidx, score in flow_scores:
        if fidx - last_event_frame < int(cooldown_s * sample_fps):
            continue
        if score <= burst_threshold:
            continue

        # 验证：峰值后是否有衰减（光流减弱 = 运动完成）
        next_scores = [s for f, s in flow_scores if fidx < f <= fidx + 3]
        if next_scores:
            avg_next = float(np.mean(next_scores))
            # 要求后续光流明显减弱（衰减到 70% 以下）
            if avg_next >= score * 0.7:
                continue  # 持续运动，不是瞬间进球晃动

        timestamp = fidx / sample_fps
        burst_ratio = (score - mean_v) / (std_v + 1e-6)
        # 置信度：突发比例越高越可信，最高 0.70（辅助信号，不超过主判定）
        confidence = min(0.70, 0.35 + 0.05 * burst_ratio)

        candidate_events.append({
            "frame_index": fidx,
            "timestamp": round(timestamp, 2),
            "confidence": round(confidence, 3),
            "detail": (f"篹网形变: flow={score:.2f}, "
                       f"threshold={burst_threshold:.2f}, "
                       f"ratio={burst_ratio:.1f}x"),
        })
        last_event_frame = fidx

    print(f"[net_deform] 检测到 {len(candidate_events)} 个篹网形变事件", flush=True)
    return candidate_events


if __name__ == "__main__":
    # 简单自测
    import sys
    if len(sys.argv) < 2:
        print("用法: python net_deform.py <frames_dir> [hoop_cx] [hoop_cy] [hoop_w]")
        sys.exit(1)

    import glob
    frames_dir = sys.argv[1]
    frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.jpg"))

    hoop_cx = float(sys.argv[2]) if len(sys.argv) > 2 else 640
    hoop_cy = float(sys.argv[3]) if len(sys.argv) > 3 else 300
    hoop_w  = float(sys.argv[4]) if len(sys.argv) > 4 else 120

    events = detect_net_deformation(frame_files, hoop_cx, hoop_cy, hoop_w,
                                    sample_fps=3.0)
    import json
    print(json.dumps(events, ensure_ascii=False, indent=2))
