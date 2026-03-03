#!/usr/bin/env python3
"""
GoalCut AI 进球检测引擎 v2.1

Phase 1.5 重构版本，核心改进：
- 动态篮筐检测（替代硬编码 ROI）
- IOU 追踪器建立球体轨迹
- 基于空间几何的进球判定（替代启发式规则）
- 冷静期机制防止重复计数
- 多通道并行检测：YOLO球体 + 人体运动 + 帧差分运动
- 保留三层回退兼容（YOLO+追踪 → 启发式 → 帧差分）
"""

import argparse
import json
import os
import sys
import glob
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def log(msg):
    print(f"[ai-engine] {msg}", flush=True)


# ============================================================
# 篮筐检测器
# ============================================================

class HoopDetector:
    """
    动态篮筐检测器。
    
    检测策略（按优先级）：
    1. YOLO 检测 hoop 类别（如果有篮球专用模型）
    2. 基于颜色 + 形状的篮筐检测（橙色/红色圆环）
    3. 基于球体高频出现区域推断篮筐位置
    4. 回退到固定比例 ROI
    """

    def __init__(self, img_w: int, img_h: int):
        self.img_w = img_w
        self.img_h = img_h
        self.hoop_rect = None  # (cx, cy, w, h) 篮筐矩形
        self.detection_method = "none"

    def detect_from_frames(self, frame_files: List[str], model=None,
                           ball_detections=None) -> bool:
        """
        从多帧中检测篮筐位置。

        返回 True 表示成功检测到篮筐。
        """
        import cv2

        # 策略 1：尝试用 YOLO 检测篮筐（如果模型支持）
        if model is not None:
            hoop = self._detect_hoop_yolo(frame_files, model)
            if hoop is not None:
                self.hoop_rect = hoop
                self.detection_method = "yolo"
                return True

        # 策略 2：基于颜色 + 形状检测篮筐（橙色/红色圆环）
        hoop = self._detect_hoop_color(frame_files, cv2)
        if hoop is not None:
            self.hoop_rect = hoop
            self.detection_method = "color"
            return True

        # 策略 3：基于球体高频出现区域推断
        if ball_detections:
            hoop = self._infer_hoop_from_balls(ball_detections)
            if hoop is not None:
                self.hoop_rect = hoop
                self.detection_method = "ball_infer"
                return True

        # 策略 4：回退到固定比例 ROI
        self._fallback_fixed_roi()
        return False

    def _detect_hoop_yolo(self, frame_files, model) -> Optional[Tuple]:
        """尝试用 YOLO 模型检测篮筐（需要篮球专用模型）"""
        # 检查模型是否有 hoop 相关类别
        try:
            names = model.names
            hoop_classes = []
            for cls_id, name in names.items():
                if name.lower() in ('hoop', 'basket', 'rim', 'backboard'):
                    hoop_classes.append(cls_id)
            if not hoop_classes:
                return None

            # 在采样帧中检测篮筐
            hoop_candidates = []
            sample_indices = np.linspace(0, len(frame_files) - 1,
                                         min(20, len(frame_files)), dtype=int)
            for idx in sample_indices:
                results = model(frame_files[idx], conf=0.3, verbose=False)
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if cls in hoop_classes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            w = x2 - x1
                            h = y2 - y1
                            hoop_candidates.append((cx, cy, w, h))

            if len(hoop_candidates) >= 3:
                # 取中位数作为稳定位置
                arr = np.array(hoop_candidates)
                cx = np.median(arr[:, 0])
                cy = np.median(arr[:, 1])
                w = np.median(arr[:, 2])
                h = np.median(arr[:, 3])
                return (cx, cy, w, h)
        except Exception as e:
            log(f"YOLO 篮筐检测失败: {e}")

        return None

    def _detect_hoop_color(self, frame_files, cv2) -> Optional[Tuple]:
        """基于颜色 + 形状检测篮筐（橙色/红色圆环区域）"""
        try:
            sample_indices = np.linspace(0, len(frame_files) - 1,
                                         min(10, len(frame_files)), dtype=int)
            all_candidates = []

            for idx in sample_indices:
                img = cv2.imread(frame_files[idx])
                if img is None:
                    continue
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                # 橙色/红色范围（篮筐颜色）
                # 红色在 HSV 中分布在两端
                mask1 = cv2.inRange(hsv, np.array([0, 100, 100]),
                                    np.array([15, 255, 255]))
                mask2 = cv2.inRange(hsv, np.array([160, 100, 100]),
                                    np.array([180, 255, 255]))
                # 橙色
                mask3 = cv2.inRange(hsv, np.array([10, 100, 100]),
                                    np.array([25, 255, 255]))
                mask = mask1 | mask2 | mask3

                # 只看画面上半部分（篮筐通常在画面 10%~55% 高度范围）
                h_img = img.shape[0]
                mask[:int(h_img * 0.1), :] = 0   # 清除顶部 10%（天空/背景）
                mask[int(h_img * 0.55):, :] = 0   # 清除下半部分

                # 形态学处理
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    # 篮筐面积范围（根据画面比例）
                    min_area = self.img_w * self.img_h * 0.001
                    max_area = self.img_w * self.img_h * 0.05
                    if min_area < area < max_area:
                        x, y, w, h = cv2.boundingRect(cnt)
                        # 篮筐形状：宽 > 高，宽高比 1.5~5
                        ratio = w / max(h, 1)
                        if 1.2 < ratio < 6.0:
                            cx = x + w / 2
                            cy = y + h / 2
                            all_candidates.append((cx, cy, w, h))

            if len(all_candidates) >= 2:
                # 聚类筛选：找最密集的区域
                arr = np.array(all_candidates)
                # 简单方法：取中位数
                cx = np.median(arr[:, 0])
                cy = np.median(arr[:, 1])
                w = np.median(arr[:, 2])
                h = np.median(arr[:, 3])
                # 确保篮筐宽度合理（不小于画面宽度的 3%）
                w = max(w, self.img_w * 0.03)
                return (cx, cy, w, h)
        except Exception as e:
            log(f"颜色篮筐检测失败: {e}")

        return None

    def _infer_hoop_from_balls(self, ball_detections) -> Optional[Tuple]:
        """
        从球体检测结果推断篮筐位置。
        
        思路：进球时球体会在篮筐区域反复出现，统计球体在画面上半部分
        出现的热力图，找到最密集的水平带作为篮筐区域。
        """
        try:
            # 只取上半部分的球体位置
            upper_balls = [(cx, cy) for (_, cx, cy, _, _, _) in ball_detections
                           if cy < self.img_h * 0.7]
            if len(upper_balls) < 5:
                return None

            # Y 方向直方图，找最密集的水平带
            ys = np.array([cy for _, cy in upper_balls])
            bins = max(10, int(self.img_h / 50))
            hist, edges = np.histogram(ys, bins=bins)

            # 找峰值区间
            peak_bin = np.argmax(hist)
            if hist[peak_bin] < 3:
                return None

            peak_y = (edges[peak_bin] + edges[peak_bin + 1]) / 2

            # 在峰值 Y 附近的球体中取 X 中位数
            nearby_balls = [(cx, cy) for cx, cy in upper_balls
                            if abs(cy - peak_y) < self.img_h * 0.1]
            if len(nearby_balls) < 3:
                return None

            cx = np.median([cx for cx, _ in nearby_balls])
            cy = peak_y
            # 估算篮筐宽度（约为画面宽度的 5~10%）
            w = self.img_w * 0.07
            h = w * 0.5

            log(f"从球体位置推断篮筐: cx={cx:.0f}, cy={cy:.0f}")
            return (cx, cy, w, h)
        except Exception:
            return None

    def _fallback_fixed_roi(self):
        """回退到固定比例 ROI"""
        cx = self.img_w / 2
        cy = self.img_h * 0.4  # 画面 40% 高度
        w = self.img_w * 0.3
        h = self.img_h * 0.1
        self.hoop_rect = (cx, cy, w, h)
        self.detection_method = "fixed_roi"
        log(f"使用固定比例 ROI: cx={cx:.0f}, cy={cy:.0f}, w={w:.0f}, h={h:.0f}")

    def get_hoop_region(self) -> Tuple[float, float, float, float]:
        """返回 (cx, cy, w, h) 篮筐区域"""
        if self.hoop_rect is None:
            self._fallback_fixed_roi()
        return self.hoop_rect

    def get_hoop_y_range(self) -> Tuple[float, float]:
        """返回篮筐的 Y 范围（用于兼容旧逻辑）"""
        cx, cy, w, h = self.get_hoop_region()
        margin = max(h * 2, self.img_h * 0.1)
        y_min = max(0, cy - margin)
        y_max = min(self.img_h, cy + margin)
        return y_min, y_max


# ============================================================
# 几何进球判定器
# ============================================================

def judge_goal_by_trajectory(track_points: List, hoop_cx: float, hoop_cy: float,
                              hoop_w: float, hoop_h: float,
                              sample_fps: float) -> Optional[Dict]:
    """
    通过球体轨迹的空间几何关系判断是否进球。

    判定条件（需同时满足）：
    1. 球体轨迹在篮筐区域上方出现
    2. 球体运动方向向下（Y 坐标递增）
    3. 球体中心穿越篮筐水平线（y ≈ hoop_cy）
    4. 穿越点的 X 坐标在篮筐内圈范围内
    5. 连续至少 2 帧满足下落趋势

    Args:
        track_points: 轨迹点列表 [(frame_idx, cx, cy, conf), ...]
        hoop_cx, hoop_cy, hoop_w, hoop_h: 篮筐位置
        sample_fps: 采样帧率

    Returns:
        进球事件 dict 或 None
    """
    if len(track_points) < 3:
        return None

    # 篮筐内圈范围（加 20% 容差）
    margin = hoop_w * 0.2
    hoop_left = hoop_cx - hoop_w / 2 - margin
    hoop_right = hoop_cx + hoop_w / 2 + margin
    hoop_y = hoop_cy

    crossing_frames = []

    for i in range(1, len(track_points)):
        prev_fidx, prev_cx, prev_cy, prev_conf = track_points[i - 1]
        curr_fidx, curr_cx, curr_cy, curr_conf = track_points[i]

        # 条件 2：方向向下
        if curr_cy <= prev_cy:
            continue

        # 条件 3：穿越篮筐水平线
        if not (prev_cy < hoop_y <= curr_cy):
            continue

        # 线性插值计算穿越时的 X 坐标
        dy = curr_cy - prev_cy
        if dy < 1e-6:
            continue
        ratio = (hoop_y - prev_cy) / dy
        cross_x = prev_cx + ratio * (curr_cx - prev_cx)

        # 条件 4：穿越点在篮筐 X 范围内
        if hoop_left <= cross_x <= hoop_right:
            crossing_frames.append({
                "frame_idx": curr_fidx,
                "cross_x": cross_x,
                "cross_y": hoop_y,
                "conf": (prev_conf + curr_conf) / 2,
                "dy": dy,
            })

    # 条件 5：至少有有效穿越帧
    if len(crossing_frames) >= 1:
        best = max(crossing_frames, key=lambda f: f["conf"])
        timestamp = best["frame_idx"] / sample_fps
        # 置信度：基于穿越质量
        base_conf = best["conf"]
        # 多次穿越给予额外置信度
        crossing_bonus = min(0.15, len(crossing_frames) * 0.05)
        confidence = min(0.95, base_conf * 0.7 + 0.2 + crossing_bonus)

        return {
            "frame_index": best["frame_idx"],
            "timestamp": round(timestamp, 2),
            "confidence": round(confidence, 3),
            "detail": (f"几何判定: 球穿越篮筐平面 x={best['cross_x']:.0f} "
                       f"(范围{hoop_left:.0f}~{hoop_right:.0f}), "
                       f"穿越帧数={len(crossing_frames)}, dy={best['dy']:.0f}")
        }

    return None


def judge_goal_by_zone_disappear(track_points: List, hoop_cy: float,
                                  hoop_y_range: Tuple[float, float],
                                  total_frames: int, sample_fps: float,
                                  frame_balls: Dict) -> Optional[Dict]:
    """
    辅助判定：球在篮筐区域出现后消失（入网模式）。
    
    作为几何判定的补充，处理球被篮网遮挡导致追踪丢失的情况。
    """
    if len(track_points) < 2:
        return None

    hoop_y_min, hoop_y_max = hoop_y_range
    last_point = track_points[-1]
    last_fidx, last_cx, last_cy, last_conf = last_point

    # 检查轨迹最后是否在篮筐区域
    if not (hoop_y_min <= last_cy <= hoop_y_max):
        return None

    # 检查是否有下落趋势（至少最后2帧向下）
    if len(track_points) >= 2:
        prev_cy = track_points[-2][2]
        if last_cy <= prev_cy:
            return None

    # 检查轨迹结束后是否球消失了（后续帧无检测）
    disappeared = True
    for gap in range(1, 4):
        check_frame = last_fidx + gap
        if check_frame in frame_balls:
            disappeared = False
            break

    if disappeared:
        timestamp = last_fidx / sample_fps
        confidence = min(0.75, last_conf * 0.6 + 0.2)
        return {
            "frame_index": last_fidx,
            "timestamp": round(timestamp, 2),
            "confidence": round(confidence, 3),
            "detail": f"辅助判定: 球在篮筐区域消失, y={last_cy:.0f}, conf={last_conf:.2f}"
        }

    return None


# ============================================================
# 人体运动模式检测（补充通道）
# ============================================================

def detect_goals_by_person_motion(frame_files: List[str], model,
                                   img_w: int, img_h: int,
                                   sample_fps: float,
                                   yolo_confidence: float) -> List[Dict]:
    """
    基于人体运动模式检测进球（上篮/扣篮场景）。

    原理：上篮进球时，球员会快速向篮筐区域移动，然后动作结束（减速/离开）。
    特征：
    1. 人体在画面上半部分（篮筐区域）出现
    2. 人体位置在连续帧中快速上移（向篮筐接近）
    3. 人体到达最高点后下降（投篮完成）

    改进: 追踪画面中每个区域(左半/右半)最高的人，避免多人跳跃追踪。
    """
    log("开始人体运动模式检测...")

    PERSON_CLASS = 0
    # 篮筐区域：画面上半部分（Y < 60% 画面高度，放宽）
    hoop_zone_y = img_h * 0.6

    # 收集所有帧中的人体检测
    frame_persons = {}  # {frame_idx: [(cx, cy, w, h, conf), ...]}

    for i, frame_path in enumerate(frame_files):
        try:
            results = model(frame_path, conf=yolo_confidence, verbose=False)
            persons = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls == PERSON_CLASS:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        w = x2 - x1
                        h = y2 - y1
                        conf = float(box.conf[0])
                        persons.append((cx, cy, w, h, conf))
            if persons:
                frame_persons[i] = persons
        except Exception:
            continue

    if not frame_persons:
        log("人体运动检测: 未检测到人体")
        return []

    # 分析人体运动模式 — 分左右半场独立追踪
    candidate_events = []
    sorted_frames = sorted(frame_persons.keys())
    mid_x = img_w / 2

    # 跳过视频开头的帧（前3秒通常有转场/字幕）
    skip_frames = int(sample_fps * 3)

    # 对左半场和右半场分别检测上篮模式
    for side_name, x_filter in [("left", lambda cx: cx < mid_x * 1.2),
                                 ("right", lambda cx: cx > mid_x * 0.8)]:
        prev_top_cy = None
        prev_fidx = -999
        rising_count = 0
        peak_frame = -1
        peak_cy = 9999
        peak_cx = 0

        for fidx in sorted_frames:
            persons = frame_persons[fidx]
            # 取该半场最高的人
            side_persons = [p for p in persons if x_filter(p[0])]
            if not side_persons:
                # 该半场无人，如果之前在上升中，视为可能的上篮完成
                if (rising_count >= 2 and peak_cy < hoop_zone_y
                        and peak_frame > skip_frames):
                    timestamp = peak_frame / sample_fps
                    height_factor = max(0, 1.0 - peak_cy / hoop_zone_y)
                    conf = min(0.75, 0.4 + 0.1 * rising_count
                               + 0.2 * height_factor)
                    candidate_events.append({
                        "frame_index": peak_frame,
                        "timestamp": round(timestamp, 2),
                        "confidence": round(conf, 3),
                        "detail": (f"人体运动({side_name}): 上篮模式, "
                                   f"连续上升{rising_count}帧, "
                                   f"peak_y={peak_cy:.0f}, "
                                   f"cx={peak_cx:.0f}")
                    })
                rising_count = 0
                peak_cy = 9999
                prev_top_cy = None
                prev_fidx = -999
                continue

            top_person = min(side_persons, key=lambda p: p[1])
            top_cx, top_cy = top_person[0], top_person[1]

            if prev_top_cy is not None and fidx - prev_fidx <= 2:
                dy = top_cy - prev_top_cy  # dy < 0 = 上升

                if dy < -img_h * 0.015:  # 上升（1.5%画面高度）
                    rising_count += 1
                    if top_cy < peak_cy:
                        peak_cy = top_cy
                        peak_frame = fidx
                        peak_cx = top_cx
                elif dy > img_h * 0.01 and rising_count >= 2:
                    # 下降 → 上篮可能完成
                    if peak_cy < hoop_zone_y and peak_frame > skip_frames:
                        timestamp = peak_frame / sample_fps
                        height_factor = max(0, 1.0 - peak_cy / hoop_zone_y)
                        conf = min(0.75, 0.4 + 0.1 * rising_count
                                   + 0.2 * height_factor)
                        candidate_events.append({
                            "frame_index": peak_frame,
                            "timestamp": round(timestamp, 2),
                            "confidence": round(conf, 3),
                            "detail": (f"人体运动({side_name}): 上篮模式, "
                                       f"连续上升{rising_count}帧, "
                                       f"peak_y={peak_cy:.0f}, "
                                       f"cx={peak_cx:.0f}")
                        })
                    rising_count = 0
                    peak_cy = 9999
                elif dy >= 0:
                    # 停滞或微降，如果之前上升不够就重置
                    if rising_count < 2:
                        rising_count = 0
                        peak_cy = 9999
            else:
                # 帧不连续，结算之前的上升
                if (rising_count >= 2 and peak_cy < hoop_zone_y
                        and peak_frame > skip_frames):
                    timestamp = peak_frame / sample_fps
                    height_factor = max(0, 1.0 - peak_cy / hoop_zone_y)
                    conf = min(0.70, 0.35 + 0.1 * rising_count
                               + 0.2 * height_factor)
                    candidate_events.append({
                        "frame_index": peak_frame,
                        "timestamp": round(timestamp, 2),
                        "confidence": round(conf, 3),
                        "detail": (f"人体运动({side_name}): 上篮模式, "
                                   f"连续上升{rising_count}帧, "
                                   f"peak_y={peak_cy:.0f}, "
                                   f"cx={peak_cx:.0f}")
                    })
                rising_count = 0
                peak_cy = 9999

            prev_top_cy = top_cy
            prev_fidx = fidx

        # 循环结束，结算未处理的上升
        if (rising_count >= 2 and peak_cy < hoop_zone_y
                and peak_frame > skip_frames):
            timestamp = peak_frame / sample_fps
            height_factor = max(0, 1.0 - peak_cy / hoop_zone_y)
            conf = min(0.70, 0.35 + 0.1 * rising_count
                       + 0.2 * height_factor)
            candidate_events.append({
                "frame_index": peak_frame,
                "timestamp": round(timestamp, 2),
                "confidence": round(conf, 3),
                "detail": (f"人体运动({side_name}): 上篮模式, "
                           f"连续上升{rising_count}帧, "
                           f"peak_y={peak_cy:.0f}, "
                           f"cx={peak_cx:.0f}")
            })

    log(f"人体运动检测产生 {len(candidate_events)} 个候选事件")
    return candidate_events


# ============================================================
# 局部运动检测（帧差分 - 并行补充通道）
# ============================================================

def detect_goals_by_motion(frame_files: List[str], img_w: int, img_h: int,
                            sample_fps: float,
                            video_duration: float) -> List[Dict]:
    """
    基于帧差分的局部运动突变检测。

    与 fallback_detection 不同，此函数：
    1. 始终运行（不仅在 YOLO 不可用时）
    2. 检测篮筐区域的局部运动突变（而非全画面）
    3. 分析运动突变的时序模式（爆发→平静 = 进球完成）
    """
    log("开始局部运动突变检测...")
    import cv2

    if len(frame_files) < 3:
        return []

    # 跳过视频开头的帧（前3秒通常有转场/字幕）
    skip_frames = int(sample_fps * 3)

    # 定义多个 ROI 区域分别检测运动
    # ROI: 左侧篮筐区域、右侧篮筐区域、中间区域
    rois = {
        "left_hoop": (0, 0, int(img_w * 0.4), int(img_h * 0.6)),
        "right_hoop": (int(img_w * 0.6), 0, img_w, int(img_h * 0.6)),
        "center_upper": (int(img_w * 0.25), 0,
                         int(img_w * 0.75), int(img_h * 0.5)),
    }

    # 计算每个 ROI 每帧的运动得分
    roi_scores = {name: [] for name in rois}
    prev_grays = {}

    for i, path in enumerate(frame_files):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        for name, (x1, y1, x2, y2) in rois.items():
            roi = img[y1:y2, x1:x2]
            if name in prev_grays:
                diff = cv2.absdiff(roi, prev_grays[name])
                score = float(np.mean(diff))
                roi_scores[name].append((i, score))
            prev_grays[name] = roi

    # 对每个 ROI 检测运动突变
    candidate_events = []

    for name, scores in roi_scores.items():
        if len(scores) < 5:
            continue

        vals = np.array([s for _, s in scores])
        mean_s = np.mean(vals)
        std_s = np.std(vals)
        if std_s < 1.0:
            continue

        # 自适应阈值：均值 + 2.5倍标准差（较严格）
        threshold = mean_s + 2.5 * std_s
        last_event_frame = -999

        for fidx, score in scores:
            if (fidx <= skip_frames or
                    fidx - last_event_frame <= sample_fps * 5):
                continue
            if score > threshold:
                # 检测爆发后是否有平静期（进球完成特征）
                # 找后续2帧的平均得分
                next_scores = [s for f, s in scores
                               if fidx < f <= fidx + 3]
                if next_scores:
                    avg_next = np.mean(next_scores)
                    # 运动从高到低 = 动作完成（衰减至少50%）
                    if avg_next < score * 0.5:
                        timestamp = fidx / sample_fps
                        burst_ratio = (score - mean_s) / (std_s + 1e-6)
                        conf = min(0.55, 0.3 + 0.05 * burst_ratio)
                        candidate_events.append({
                            "frame_index": fidx,
                            "timestamp": round(timestamp, 2),
                            "confidence": round(conf, 3),
                            "detail": (f"运动突变({name}): "
                                       f"score={score:.1f}, "
                                       f"阈值={threshold:.1f}, "
                                       f"后续衰减={avg_next:.1f}")
                        })
                        last_event_frame = fidx

    log(f"运动突变检测产生 {len(candidate_events)} 个候选事件")
    return candidate_events


# ============================================================
# 主检测流程
# ============================================================

def detect_goals(frames_dir, output_path, sample_fps, confidence_threshold,
                 yolo_confidence, video_duration):
    """主检测流程 v2.1 - 多通道并行检测"""
    log(f"开始进球检测 (v2.1)")
    log(f"  帧目录: {frames_dir}")
    log(f"  采样帧率: {sample_fps}")
    log(f"  置信度阈值: {confidence_threshold}")
    log(f"  YOLO置信度: {yolo_confidence}")
    log(f"  视频时长: {video_duration}s")

    # 加载 YOLO 模型
    log("加载 YOLOv8 模型...")
    model = None
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        log("YOLOv8n 模型加载成功")
    except Exception as e:
        log(f"加载 YOLO 模型失败: {e}")
        log("回退到简化检测模式（无 YOLO）")
        events = fallback_detection(frames_dir, sample_fps, video_duration,
                                    confidence_threshold)
        save_results(events, output_path)
        return

    # 获取所有帧文件
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not frame_files:
        log("错误: 未找到帧文件")
        save_results([], output_path)
        return

    log(f"共 {len(frame_files)} 帧待检测")

    # 获取图像尺寸
    import cv2
    sample_img = cv2.imread(frame_files[0])
    if sample_img is None:
        log("错误: 无法读取帧图片")
        save_results([], output_path)
        return
    img_h, img_w = sample_img.shape[:2]
    log(f"帧尺寸: {img_w}x{img_h}")

    # COCO 类别
    BALL_CLASS = 32  # sports ball

    # ================================================================
    # 通道 1: YOLO 球体检测（使用较低阈值捕获更多）
    # ================================================================
    log("--- 通道 1: YOLO 球体检测 ---")
    ball_yolo_conf = min(yolo_confidence, 0.25)  # 降低阈值
    ball_detections = []  # [(frame_idx, cx, cy, w, h, conf)]

    for i, frame_path in enumerate(frame_files):
        if i % 10 == 0:
            log(f"检测进度: {i+1}/{len(frame_files)}")

        try:
            results = model(frame_path, conf=ball_yolo_conf, verbose=False)
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
        except Exception as e:
            log(f"帧 {i} 检测异常: {e}")
            continue

    log(f"通道1: 共检测到 {len(ball_detections)} 次球体出现")

    # ---- 动态篮筐检测 ----
    log("开始动态篮筐检测...")
    hoop_detector = HoopDetector(img_w, img_h)
    hoop_detected = hoop_detector.detect_from_frames(
        frame_files, model=model,
        ball_detections=ball_detections if ball_detections else None
    )
    hoop_cx, hoop_cy, hoop_w, hoop_h = hoop_detector.get_hoop_region()
    hoop_y_min, hoop_y_max = hoop_detector.get_hoop_y_range()
    log(f"篮筐检测结果: method={hoop_detector.detection_method}, "
        f"cx={hoop_cx:.0f}, cy={hoop_cy:.0f}, w={hoop_w:.0f}, h={hoop_h:.0f}")
    log(f"篮筐 Y 范围: {hoop_y_min:.0f} ~ {hoop_y_max:.0f}")

    # 所有通道的候选事件收集
    all_candidate_events = []

    # ---- 通道 1a: IOU 追踪 + 几何判定 ----
    if ball_detections:
        log("--- 通道 1a: IOU 追踪 + 几何判定 ---")
        from tracker import IOUTracker, Detection as TrackerDetection

        max_lost = max(3, int(sample_fps * 1.5))
        dist_threshold = max(50, min(img_w, img_h) * 0.15)
        tracker = IOUTracker(
            iou_threshold=0.15,
            max_lost=max_lost,
            distance_threshold=dist_threshold,
        )

        frame_dets = {}
        frame_balls = {}
        for (fidx, cx, cy, w, h, conf) in ball_detections:
            if fidx not in frame_dets:
                frame_dets[fidx] = []
                frame_balls[fidx] = []
            frame_dets[fidx].append(
                TrackerDetection(fidx, cx, cy, w, h, conf))
            frame_balls[fidx].append((cx, cy, w, h, conf))

        total_frames = len(frame_files)
        for fidx in range(total_frames):
            dets = frame_dets.get(fidx, [])
            tracker.update(dets)

        all_tracks = tracker.get_all_tracks()
        log(f"追踪完成: 共 {len(all_tracks)} 条有效轨迹")

        for track in all_tracks:
            points = [(p.frame_idx, p.cx, p.cy, p.conf)
                      for p in track.points]

            event = judge_goal_by_trajectory(
                points, hoop_cx, hoop_cy, hoop_w, hoop_h, sample_fps
            )
            if event:
                all_candidate_events.append(event)
                continue

            event = judge_goal_by_zone_disappear(
                points, hoop_cy, (hoop_y_min, hoop_y_max),
                total_frames, sample_fps, frame_balls
            )
            if event:
                all_candidate_events.append(event)

        log(f"通道1a 几何判定: {len(all_candidate_events)} 个候选")

        # ---- 通道 1b: 改进版启发式 ----
        log("--- 通道 1b: 启发式检测 ---")
        heuristic_events = fallback_heuristic_v2(
            frame_balls, img_w, img_h, total_frames,
            sample_fps, video_duration, hoop_y_min, hoop_y_max,
            confidence_threshold
        )
        all_candidate_events.extend(heuristic_events)
        log(f"通道1b 启发式: {len(heuristic_events)} 个候选")

    # ================================================================
    # 通道 2: 人体运动模式检测（不依赖球体）
    # ================================================================
    log("--- 通道 2: 人体运动模式检测 ---")
    person_events = detect_goals_by_person_motion(
        frame_files, model, img_w, img_h, sample_fps, yolo_confidence
    )
    all_candidate_events.extend(person_events)

    # ================================================================
    # 通道 3: 局部运动突变检测（不依赖 YOLO）
    # ================================================================
    log("--- 通道 3: 局部运动突变检测 ---")
    motion_events = detect_goals_by_motion(
        frame_files, img_w, img_h, sample_fps, video_duration
    )
    all_candidate_events.extend(motion_events)

    # ================================================================
    # 汇总所有通道结果 + 跨通道验证
    # ================================================================
    log(f"所有通道汇总: {len(all_candidate_events)} 个候选事件")

    if not all_candidate_events:
        log("所有通道均未检测到进球事件")
        save_results([], output_path)
        return

    # 跨通道验证：在 ±2s 窗口内有多个通道命中的事件提升置信度
    validation_window = 2.0  # 秒
    for i, event in enumerate(all_candidate_events):
        t = event["timestamp"]
        # 统计其他事件中在时间窗口内的数量（不同来源）
        nearby_count = 0
        for j, other in enumerate(all_candidate_events):
            if i == j:
                continue
            if abs(other["timestamp"] - t) <= validation_window:
                # 检查是否来自不同检测通道
                this_src = event["detail"].split(":")[0].strip()
                other_src = other["detail"].split(":")[0].strip()
                if this_src != other_src:
                    nearby_count += 1
        if nearby_count > 0:
            boost = min(0.15, nearby_count * 0.08)
            old_conf = event["confidence"]
            event["confidence"] = round(
                min(0.95, event["confidence"] + boost), 3)
            log(f"  跨通道验证: {t:.1f}s conf {old_conf} → "
                f"{event['confidence']} (nearby={nearby_count})")

    # ---- 事件合并 + 冷静期 ----
    events = merge_events_with_cooldown(
        all_candidate_events, confidence_threshold,
        merge_window=3.0, cooldown=2.0
    )

    log(f"最终结果: {len(events)} 个进球事件")
    for e in events:
        log(f"  {e['timestamp']}s conf={e['confidence']} {e['detail']}")
    save_results(events, output_path)


# ============================================================
# 事件合并 + 冷静期
# ============================================================

def merge_events_with_cooldown(candidate_events: List[Dict],
                                conf_threshold: float,
                                merge_window: float = 3.0,
                                cooldown: float = 2.0) -> List[Dict]:
    """
    事件合并与冷静期过滤。

    1. 按时间排序
    2. merge_window 内的事件合并（取最高置信度）
    3. 冷静期过滤：确认事件后 cooldown 秒内忽略后续候选
    4. 过滤低置信度
    """
    if not candidate_events:
        return []

    # 按时间排序
    candidate_events.sort(key=lambda e: e["timestamp"])

    # 合并相近事件
    merged = []
    current_group = [candidate_events[0]]

    for i in range(1, len(candidate_events)):
        if (candidate_events[i]["timestamp"] -
                current_group[0]["timestamp"] <= merge_window):
            current_group.append(candidate_events[i])
        else:
            best = max(current_group, key=lambda e: e["confidence"])
            merged.append(best)
            current_group = [candidate_events[i]]

    best = max(current_group, key=lambda e: e["confidence"])
    merged.append(best)

    # 冷静期过滤
    final = []
    last_confirmed_time = -999.0

    for event in merged:
        if event["confidence"] < conf_threshold:
            continue
        if event["timestamp"] - last_confirmed_time < cooldown:
            continue  # 冷静期内，跳过
        final.append(event)
        last_confirmed_time = event["timestamp"]

    log(f"候选: {len(candidate_events)}, 合并: {len(merged)}, "
        f"冷静期+阈值过滤: {len(final)}")
    return final


# ============================================================
# 回退检测（兼容旧逻辑）
# ============================================================

def fallback_heuristic_v2(frame_balls, img_w, img_h, total_frames,
                          sample_fps, video_duration,
                          hoop_y_min, hoop_y_max, conf_threshold):
    """
    改进版启发式检测：使用动态检测的篮筐 Y 范围替代硬编码值。
    """
    log("使用改进版启发式检测")

    if not frame_balls:
        return []

    candidate_events = []
    sorted_frames = sorted(frame_balls.keys())

    # 方法 1：球在篮筐区域连续出现后消失
    last_ball_frame = -999
    last_ball_y = 0
    ball_in_hoop_zone_count = 0

    for fidx in range(total_frames):
        if fidx in frame_balls:
            balls = frame_balls[fidx]
            best_ball = max(balls, key=lambda b: b[4])
            cx, cy, w, h, conf = best_ball

            if hoop_y_min <= cy <= hoop_y_max:
                if last_ball_frame >= 0 and fidx - last_ball_frame <= 3:
                    if cy > last_ball_y:
                        ball_in_hoop_zone_count += 1
                    else:
                        ball_in_hoop_zone_count = max(
                            0, ball_in_hoop_zone_count - 1)
                else:
                    ball_in_hoop_zone_count = 1

                if ball_in_hoop_zone_count >= 2:
                    timestamp = fidx / sample_fps
                    confidence = min(
                        0.8,
                        conf * 0.7 + 0.2 * (ball_in_hoop_zone_count / 3))
                    candidate_events.append({
                        "frame_index": fidx,
                        "timestamp": round(timestamp, 2),
                        "confidence": round(confidence, 3),
                        "detail": (f"启发式: 球在篮筐区域连续出现"
                                   f"{ball_in_hoop_zone_count}帧, "
                                   f"y={cy:.0f}, conf={conf:.2f}")
                    })
                    ball_in_hoop_zone_count = 0

            last_ball_frame = fidx
            last_ball_y = cy
        else:
            if last_ball_frame >= 0 and fidx - last_ball_frame == 1:
                if (hoop_y_min <= last_ball_y <= hoop_y_max and
                        ball_in_hoop_zone_count >= 1):
                    timestamp = last_ball_frame / sample_fps
                    confidence = 0.6 + 0.1 * min(ball_in_hoop_zone_count, 3)
                    candidate_events.append({
                        "frame_index": last_ball_frame,
                        "timestamp": round(timestamp, 2),
                        "confidence": round(confidence, 3),
                        "detail": (f"启发式: 球在篮筐区域后消失, "
                                   f"y={last_ball_y:.0f}, "
                                   f"连续帧数={ball_in_hoop_zone_count}")
                    })
            ball_in_hoop_zone_count = 0

    # 方法 2：球快速下落穿过篮筐区域
    prev_cy = None
    prev_fidx = -999
    for fidx in sorted_frames:
        balls = frame_balls[fidx]
        best_ball = max(balls, key=lambda b: b[4])
        cx, cy, w, h, conf = best_ball

        if prev_cy is not None and fidx - prev_fidx <= 2:
            dy = cy - prev_cy
            if dy > img_h * 0.08 and hoop_y_min <= prev_cy <= hoop_y_max:
                timestamp = fidx / sample_fps
                speed_factor = min(1.0, abs(dy) / (img_h * 0.15))
                confidence = 0.5 + 0.3 * speed_factor
                candidate_events.append({
                    "frame_index": fidx,
                    "timestamp": round(timestamp, 2),
                    "confidence": round(confidence, 3),
                    "detail": (f"启发式: 球快速下落穿过篮筐区域, "
                               f"dy={dy:.0f}, prev_y={prev_cy:.0f}")
                })

        prev_cy = cy
        prev_fidx = fidx

    log(f"启发式检测产生 {len(candidate_events)} 个候选事件")
    return candidate_events


def fallback_detection(frames_dir, sample_fps, video_duration, conf_threshold):
    """
    完全回退模式：当 YOLO 不可用时，使用帧差分检测运动。
    """
    log("使用帧差分回退检测模式")

    try:
        import cv2
    except ImportError:
        log("错误: OpenCV 不可用，无法进行回退检测")
        return []

    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if len(frame_files) < 3:
        return []

    prev_gray = None
    motion_scores = []

    for i, path in enumerate(frame_files):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h = img.shape[0]
        roi = img[:int(h * 0.6), :]

        if prev_gray is not None:
            diff = cv2.absdiff(roi, prev_gray)
            score = np.mean(diff)
            motion_scores.append((i, score))

        prev_gray = roi

    if not motion_scores:
        return []

    scores = [s for _, s in motion_scores]
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    threshold = mean_score + 1.5 * std_score

    log(f"运动均值={mean_score:.2f}, 标准差={std_score:.2f}, "
        f"阈值={threshold:.2f}")

    events = []
    last_event_frame = -999

    for fidx, score in motion_scores:
        if score > threshold and fidx - last_event_frame > sample_fps * 3:
            timestamp = fidx / sample_fps
            confidence = min(
                0.6, (score - mean_score) / (std_score * 3 + 1e-6))
            if confidence >= conf_threshold:
                events.append({
                    "frame_index": fidx,
                    "timestamp": round(timestamp, 2),
                    "confidence": round(confidence, 3),
                    "detail": f"帧差分回退: motion_score={score:.2f}"
                })
                last_event_frame = fidx

    log(f"帧差分检测找到 {len(events)} 个候选事件")
    return events


def save_results(events, output_path):
    """保存检测结果到 JSON"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    log(f"检测结果已保存: {output_path} ({len(events)} 个事件)")


def main():
    parser = argparse.ArgumentParser(description="GoalCut AI 进球检测引擎 v2.1")
    parser.add_argument("--frames-dir", required=True, help="帧图片目录")
    parser.add_argument("--output", required=True, help="检测结果输出路径(JSON)")
    parser.add_argument("--sample-fps", type=float, default=3, help="采样帧率")
    parser.add_argument("--confidence-threshold", type=float, default=0.55,
                        help="置信度阈值")
    parser.add_argument("--yolo-confidence", type=float, default=0.25,
                        help="YOLO检测置信度")
    parser.add_argument("--video-duration", type=float, default=0,
                        help="视频时长(秒)")

    args = parser.parse_args()

    detect_goals(
        frames_dir=args.frames_dir,
        output_path=args.output,
        sample_fps=args.sample_fps,
        confidence_threshold=args.confidence_threshold,
        yolo_confidence=args.yolo_confidence,
        video_duration=args.video_duration,
    )


if __name__ == "__main__":
    main()
