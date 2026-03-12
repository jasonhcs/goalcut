"""
GoalCut 庆祝动作检测（通道 2 改造版）

替代原通道 2（上篮动作检测）：
- 原逻辑：检测"人体上升→下降"（投篮过程）→ 天然 50%+ 误报
- 新逻辑：检测"候选进球时间点后 1~4s 内的庆祝反应"→ 作为确认信号

检测的庆祝行为：
1. 急停（运动速度骤降）：进球后球员停下来看球进网
2. 聚集（两人靠近）：进球后击掌/拥抱
3. 举手（暂未实现，需要姿态估计模型）

此通道不独立产生事件，而是对已有候选事件做确认/加权。
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


def detect_player_stops(
    person_detections_by_frame: Dict[int, List[Tuple[float, float, float, float]]],
    candidate_timestamp: float,
    sample_fps: float,
    window_after: float = 4.0,
    min_speed_ratio: float = 0.3,
) -> float:
    """检测候选进球后 1~4s 内是否有球员急停。

    Args:
        person_detections_by_frame: {帧索引: [(cx, cy, w, h), ...]}
        candidate_timestamp: 候选进球时间戳
        sample_fps: 采样帧率
        window_after: 检测窗口（进球后几秒内）
        min_speed_ratio: 急停判定：速度降至之前的 30% 以下

    Returns:
        庆祝置信度 [0, 1]
    """
    start_frame = int(candidate_timestamp * sample_fps)
    before_start = max(0, start_frame - int(2.0 * sample_fps))
    after_end = start_frame + int(window_after * sample_fps)

    # 计算进球前 2s 内的平均运动量
    before_movements = []
    for f in range(before_start, start_frame):
        if f in person_detections_by_frame and (f + 1) in person_detections_by_frame:
            persons_f = person_detections_by_frame[f]
            persons_f1 = person_detections_by_frame[f + 1]
            for p in persons_f:
                for p1 in persons_f1:
                    dist = np.sqrt((p[0] - p1[0])**2 + (p[1] - p1[1])**2)
                    if dist < 200:  # 同一人的匹配距离阈值
                        before_movements.append(dist)

    if not before_movements:
        return 0.0

    avg_speed_before = np.mean(before_movements)
    if avg_speed_before < 5.0:  # 进球前就很静止，无法判断急停
        return 0.0

    # 计算进球后 1~4s 内的运动量
    after_start = start_frame + int(1.0 * sample_fps)
    after_movements = []
    for f in range(after_start, min(after_end, max(person_detections_by_frame.keys(), default=0))):
        if f in person_detections_by_frame and (f + 1) in person_detections_by_frame:
            persons_f = person_detections_by_frame[f]
            persons_f1 = person_detections_by_frame[f + 1]
            for p in persons_f:
                for p1 in persons_f1:
                    dist = np.sqrt((p[0] - p1[0])**2 + (p[1] - p1[1])**2)
                    if dist < 200:
                        after_movements.append(dist)

    if not after_movements:
        return 0.0

    avg_speed_after = np.mean(after_movements)
    speed_ratio = avg_speed_after / (avg_speed_before + 1e-6)

    # 速度骤降 = 急停，可能是庆祝
    if speed_ratio < min_speed_ratio:
        return min(0.50, 0.20 + 0.30 * (1.0 - speed_ratio / min_speed_ratio))

    return 0.0


def detect_player_gathering(
    person_detections_by_frame: Dict[int, List[Tuple[float, float, float, float]]],
    candidate_timestamp: float,
    sample_fps: float,
    window_after: float = 4.0,
    gathering_dist: float = 80.0,
) -> float:
    """检测候选进球后 1~4s 内是否有球员聚集（击掌/拥抱）。

    Args:
        person_detections_by_frame: {帧索引: [(cx, cy, w, h), ...]}
        candidate_timestamp: 候选进球时间戳
        sample_fps: 采样帧率
        window_after: 检测窗口
        gathering_dist: 两人聚集的距离阈值（像素）

    Returns:
        庆祝置信度 [0, 1]
    """
    start_frame = int(candidate_timestamp * sample_fps)
    before_start = max(0, start_frame - int(2.0 * sample_fps))
    after_start = start_frame + int(1.0 * sample_fps)
    after_end = start_frame + int(window_after * sample_fps)

    def count_close_pairs(frame_range_start, frame_range_end):
        close_count = 0
        total_pairs = 0
        for f in range(frame_range_start, frame_range_end):
            if f not in person_detections_by_frame:
                continue
            persons = person_detections_by_frame[f]
            for i in range(len(persons)):
                for j in range(i + 1, len(persons)):
                    dist = np.sqrt(
                        (persons[i][0] - persons[j][0])**2 +
                        (persons[i][1] - persons[j][1])**2
                    )
                    total_pairs += 1
                    if dist < gathering_dist:
                        close_count += 1
        return close_count, total_pairs

    before_close, before_total = count_close_pairs(before_start, start_frame)
    after_close, after_total = count_close_pairs(after_start, after_end)

    if before_total == 0 or after_total == 0:
        return 0.0

    before_ratio = before_close / before_total
    after_ratio = after_close / after_total

    # 进球后聚集比例显著增加
    if after_ratio > before_ratio + 0.15:
        increase = after_ratio - before_ratio
        return min(0.50, 0.15 + increase * 1.5)

    return 0.0


def evaluate_celebration(
    person_detections_by_frame: Dict[int, List[Tuple[float, float, float, float]]],
    candidate_events: List[Dict],
    sample_fps: float,
) -> List[Dict]:
    """对候选事件列表评估庆祝动作，返回带庆祝置信度的事件。

    此函数不产生新事件，而是对已有候选事件增加庆祝确认分数。

    Args:
        person_detections_by_frame: YOLO 检测到的人体位置（按帧索引）
        candidate_events: 已有的候选进球事件列表
        sample_fps: 采样帧率

    Returns:
        庆祝事件列表（仅包含检测到庆祝行为的事件）
    """
    if not person_detections_by_frame or not candidate_events:
        return []

    celebration_events = []

    for event in candidate_events:
        ts = event["timestamp"]

        stop_score = detect_player_stops(
            person_detections_by_frame, ts, sample_fps
        )
        gather_score = detect_player_gathering(
            person_detections_by_frame, ts, sample_fps
        )

        combined = max(stop_score, gather_score)
        if combined > 0.15:
            celebration_events.append({
                "frame_index": event.get("frame_index", int(ts * sample_fps)),
                "timestamp": round(ts, 2),
                "confidence": round(combined, 3),
                "detail": (
                    f"庆祝动作: 急停={stop_score:.2f}, "
                    f"聚集={gather_score:.2f}"
                ),
            })

    print(f"[celebration] 检测到 {len(celebration_events)} 个庆祝动作事件", flush=True)
    return celebration_events
