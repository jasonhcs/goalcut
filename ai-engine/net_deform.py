#!/usr/bin/env python3
"""
GoalCut 篹网形变检测模块 (T-1.9.7 + T-1.9.13)

原理：
  进球时球穿过篹网，导致篹网发生形变（晃动）。
  v1 (T-1.9.7): Farneback 稠密光流法计算篹网 ROI 区域的运动幅度（平均幅度）。
  v2 (T-1.9.13): 有向光流穿越检测 — 计算垂直向下光流分量与横向分量之比（方向纯净度），
                  区分进球（垂直向下）与篮板争抢（横向混乱），降低通道4误报。

野球场场景中，视觉信号较弱时，篹网形变是独立的补充信号。
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def get_net_roi(hoop_cx: float, hoop_cy: float, hoop_w: float,
                img_w: int, img_h: int,
                height_ratio: float = 0.8) -> Tuple[int, int, int, int]:
    """
    根据篹筐位置推算篹网 ROI。

    篹网在篹筐正下方，高度约为篹筐宽度的 height_ratio 倍。
    返回 (x1, y1, x2, y2)

    Args:
        height_ratio: 篹网高度与篹筐宽度之比（v1 默认 0.8，T-1.9.14 建议 1.2→0.8 不变，
                      但外层可通过参数控制）
    """
    net_w = hoop_w * 1.2
    net_h = hoop_w * height_ratio

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


def compute_directional_flow(prev_gray: np.ndarray,
                             curr_gray: np.ndarray,
                             roi: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
    """
    计算 ROI 区域的有向光流特征（T-1.9.13 算法C）。

    返回 (downward_score, horizontal_noise, directional_purity):
      - downward_score:  垂直向下光流分量均值（越大=越强的向下运动）
      - horizontal_noise: 横向光流分量绝对值均值（越大=越杂乱）
      - directional_purity: 方向纯净度 = downward / (horizontal + eps)
        进球时 >> 1（纯向下穿越）
        篮板争抢时 ≈ 0.5~1.0（横向混乱）
    """
    x1, y1, x2, y2 = roi
    prev_roi = prev_gray[y1:y2, x1:x2]
    curr_roi = curr_gray[y1:y2, x1:x2]

    if prev_roi.size == 0 or curr_roi.size == 0:
        return 0.0, 0.0, 0.0

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
        vx = flow[..., 0]
        vy = flow[..., 1]

        # 只统计向下运动（vy > 0 在图像坐标系中表示向下）
        downward_score = float(np.mean(np.clip(vy, 0, None)))
        horizontal_noise = float(np.mean(np.abs(vx)))
        directional_purity = downward_score / (horizontal_noise + 1e-6)

        return downward_score, horizontal_noise, directional_purity
    except Exception:
        return 0.0, 0.0, 0.0


def detect_downward_flow_through_net(
    frame_files: List[str],
    hoop_cx: float,
    hoop_cy: float,
    hoop_w: float,
    sample_fps: float,
    net_height_ratio: float = 0.8,
    min_burst_ratio: float = 2.0,
    min_purity: float = 1.5,
    cooldown_s: float = 3.0,
) -> List[dict]:
    """
    有向光流穿越检测（T-1.9.13 算法C）。

    与 detect_net_deformation 的区别：
    - 旧版：计算篹网 ROI 的平均光流幅度 → 无法区分进球与篮板争抢
    - 新版：计算垂直向下光流分量与横向分量之比（方向纯净度）
      - 进球：球从上到下穿网，垂直向下运动占主导 → directional_purity >> 1
      - 篮板争抢：多人激烈争抢，横向运动杂乱 → directional_purity ≈ 0.5~1.0

    Args:
        frame_files:      帧文件列表（按顺序）
        hoop_cx/cy/w:     篹筐中心及宽度（像素）
        sample_fps:       采样帧率
        net_height_ratio: 篹网 ROI 高度 = hoop_w * net_height_ratio
        min_burst_ratio:  下行光流突发阈值 = 均值 + ratio * 标准差
        min_purity:       方向纯净度最低要求（向下分量需是横向的 min_purity 倍）
        cooldown_s:       两次事件最小间隔（秒）

    Returns:
        [{"frame_index", "timestamp", "confidence", "detail"}, ...]
    """
    if not frame_files:
        return []

    first = cv2.imread(frame_files[0], cv2.IMREAD_GRAYSCALE)
    if first is None:
        return []
    img_h, img_w = first.shape[:2]

    net_roi = get_net_roi(hoop_cx, hoop_cy, hoop_w, img_w, img_h,
                          height_ratio=net_height_ratio)
    if net_roi is None:
        print(f"[net_deform] 篹网 ROI 无效，跳过有向光流检测", flush=True)
        return []

    x1, y1, x2, y2 = net_roi
    print(f"[net_deform] 有向光流 ROI: ({x1},{y1})~({x2},{y2}), "
          f"hoop=({hoop_cx:.0f},{hoop_cy:.0f},w={hoop_w:.0f})", flush=True)

    # 计算每对相邻帧的有向光流特征
    flow_data = []  # [(frame_idx, downward_score, horizontal_noise, purity)]
    prev_gray = None

    for i, path in enumerate(frame_files):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            prev_gray = None
            continue

        if prev_gray is not None:
            ds, hn, dp = compute_directional_flow(prev_gray, gray, net_roi)
            flow_data.append((i, ds, hn, dp))

        prev_gray = gray

    if len(flow_data) < 5:
        print("[net_deform] 有效帧数不足，跳过有向光流检测", flush=True)
        return []

    downward_vals = np.array([d[1] for d in flow_data])
    mean_down = float(np.mean(downward_vals))
    std_down = float(np.std(downward_vals))

    if std_down < 0.2:
        print(f"[net_deform] 下行光流变化过小(std={std_down:.3f})，跳过", flush=True)
        return []

    burst_threshold = mean_down + min_burst_ratio * std_down
    print(f"[net_deform] 有向光流统计: mean_down={mean_down:.3f}, "
          f"std_down={std_down:.3f}, burst_threshold={burst_threshold:.3f}, "
          f"min_purity={min_purity}", flush=True)

    # 检测满足条件的事件：下行光流突发 + 方向纯净度高
    candidate_events = []
    last_event_frame = -int(cooldown_s * sample_fps) - 1

    for fidx, ds, hn, dp in flow_data:
        if fidx - last_event_frame < int(cooldown_s * sample_fps):
            continue
        if ds <= burst_threshold:
            continue
        if dp < min_purity:
            # 方向纯净度不足 — 可能是篮板争抢而非进球
            continue

        # 验证：峰值后是否有衰减（球穿过后运动减弱）
        next_data = [(d[1], d[3]) for d in flow_data if fidx < d[0] <= fidx + 3]
        if next_data:
            avg_next_down = float(np.mean([d[0] for d in next_data]))
            # 要求后续下行光流减弱到 80% 以下（比旧版 70% 宽松些，因为有方向过滤）
            if avg_next_down >= ds * 0.8:
                continue

        timestamp = fidx / sample_fps
        burst_ratio = (ds - mean_down) / (std_down + 1e-6)

        # 置信度：结合突发强度和方向纯净度
        # 基础分 0.35 + 突发贡献 + 纯净度贡献
        purity_bonus = min(0.15, (dp - min_purity) * 0.05)
        confidence = min(0.70, 0.35 + 0.04 * burst_ratio + purity_bonus)

        candidate_events.append({
            "frame_index": fidx,
            "timestamp": round(timestamp, 2),
            "confidence": round(confidence, 3),
            "detail": (f"篹网有向穿越: down={ds:.2f}, horiz={hn:.2f}, "
                       f"purity={dp:.2f}, burst={burst_ratio:.1f}x"),
        })
        last_event_frame = fidx

    print(f"[net_deform] 有向光流检测到 {len(candidate_events)} 个篹网穿越事件",
          flush=True)
    return candidate_events


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
        print("用法: python net_deform.py <frames_dir> [hoop_cx] [hoop_cy] [hoop_w] [--directional]")
        print("  --directional  使用有向光流检测（T-1.9.13 算法C，默认）")
        print("  --legacy       使用旧版平均幅度检测（T-1.9.7）")
        sys.exit(1)

    import glob
    frames_dir = sys.argv[1]
    frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.jpg"))

    # 解析位置参数（跳过 flag 参数）
    pos_args = [a for a in sys.argv[2:] if not a.startswith("--")]
    hoop_cx = float(pos_args[0]) if len(pos_args) > 0 else 640
    hoop_cy = float(pos_args[1]) if len(pos_args) > 1 else 300
    hoop_w  = float(pos_args[2]) if len(pos_args) > 2 else 120

    use_legacy = "--legacy" in sys.argv

    import json
    if use_legacy:
        print("=== 旧版平均幅度检测 (T-1.9.7) ===")
        events = detect_net_deformation(frame_files, hoop_cx, hoop_cy, hoop_w,
                                        sample_fps=3.0)
    else:
        print("=== 有向光流穿越检测 (T-1.9.13 算法C) ===")
        events = detect_downward_flow_through_net(
            frame_files, hoop_cx, hoop_cy, hoop_w, sample_fps=3.0)

    print(json.dumps(events, ensure_ascii=False, indent=2))
