"""
GoalCut 运动学分析模块

基于"第一性原理-连续帧进球检测算法"文档实现：
- 逐帧运动学变量计算（位置/速度/诊断加速度）
- 抛物线拟合残差碰撞检测（替代不可用的二阶差分方案，见 G-2/R-1）
- 速度方向变化检测（弹回/侧向逃逸）
- 消失-重现模式分析
- 反证法通道（通道 8）
- σ_pos 动态估计

坐标系约定（全文统一）：
  原点：图像左上角
  x 轴：向右为正
  y 轴：向下为正
  vy > 0 表示球向下运动，vy < 0 表示球向上运动
  重力方向：+y
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from math import sqrt, atan2


HOOP_DIAMETER_M = 0.4572
BALL_DIAMETER_M = 0.2426


@dataclass
class KinematicsPoint:
    frame_idx: int
    time: float
    x: float
    y: float
    conf: float = 1.0
    vx: Optional[float] = None
    vy: Optional[float] = None
    speed: Optional[float] = None
    ax: Optional[float] = None
    ay: Optional[float] = None


@dataclass
class CollisionEvent:
    frame_idx: int
    time: float
    residual_px: float
    residual_sigma: float
    in_hoop_region: bool
    in_backboard_region: bool
    vy_reversed: bool
    vx_reversed: bool
    lateral_escape: bool


@dataclass
class TrackSegment:
    """连续追踪片段"""
    points: List[KinematicsPoint]
    start_frame: int
    end_frame: int

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def n_points(self) -> int:
        return len(self.points)


@dataclass
class DisappearPattern:
    pattern: str  # "goal", "bounce_out", "unknown"
    confidence: float
    detail: str
    gap_start_frame: int = -1
    gap_end_frame: int = -1
    time_gap: float = 0.0


@dataclass
class GoalJudgment:
    is_goal: bool
    goal_type: str = ""
    confidence: float = 0.0
    reason: str = ""
    crossing_time: float = 0.0


def estimate_gravity_pixel(hoop_w_pixels: float, fps: float) -> float:
    """从篮筐像素宽度动态估算像素空间重力加速度 (px/frame²)"""
    if hoop_w_pixels <= 0 or fps <= 0:
        return 0.0
    pixel_per_meter = hoop_w_pixels / HOOP_DIAMETER_M
    return 9.81 * pixel_per_meter / (fps ** 2)


def estimate_sigma_pos(points: List[KinematicsPoint],
                       default: float = 3.0,
                       min_points: int = 8) -> float:
    """从近似匀速直线运动阶段动态估计位置噪声 σ_pos。

    方法：取连续 N 帧位置对 x(t) 和 y(t) 分别做线性拟合，
    残差标准差即为 σ_pos 估计值。
    """
    if len(points) < min_points:
        return default

    t = np.array([p.time for p in points])
    x = np.array([p.x for p in points])
    y = np.array([p.y for p in points])

    try:
        cx = np.polyfit(t, x, 1)
        cy = np.polyfit(t, y, 2)  # y 含重力，用二次拟合
        rx = x - np.polyval(cx, t)
        ry = y - np.polyval(cy, t)
        sigma = max(np.std(rx), np.std(ry))
        return max(sigma, 1.0)
    except (np.linalg.LinAlgError, ValueError):
        return default


def compute_kinematics(track_points: List[dict],
                       fps: float) -> List[KinematicsPoint]:
    """对球的追踪序列计算逐帧运动学变量。

    Args:
        track_points: [{"frame_idx": int, "cx": float, "cy": float, "conf": float}, ...]
        fps: 视频帧率
    """
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
    result = []

    for i, pt in enumerate(track_points):
        kp = KinematicsPoint(
            frame_idx=pt["frame_idx"],
            time=pt["frame_idx"] * dt,
            x=pt["cx"],
            y=pt["cy"],
            conf=pt.get("conf", 1.0),
        )

        if i >= 1:
            prev = track_points[i - 1]
            frames_gap = pt["frame_idx"] - prev["frame_idx"]
            if frames_gap > 0:
                actual_dt = frames_gap * dt
                kp.vx = (pt["cx"] - prev["cx"]) / actual_dt
                kp.vy = (pt["cy"] - prev["cy"]) / actual_dt
                kp.speed = sqrt(kp.vx ** 2 + kp.vy ** 2)

        if i >= 2:
            prev = track_points[i - 1]
            pprev = track_points[i - 2]
            gap1 = pt["frame_idx"] - prev["frame_idx"]
            gap2 = prev["frame_idx"] - pprev["frame_idx"]
            if gap1 > 0 and gap2 > 0:
                dt1 = gap1 * dt
                dt2 = gap2 * dt
                prev_vx = (prev["cx"] - pprev["cx"]) / dt2
                prev_vy = (prev["cy"] - pprev["cy"]) / dt2
                kp.ax = (kp.vx - prev_vx) / ((dt1 + dt2) / 2) if kp.vx is not None else None
                kp.ay = (kp.vy - prev_vy) / ((dt1 + dt2) / 2) if kp.vy is not None else None

        result.append(kp)

    return result


def split_into_segments(points: List[KinematicsPoint],
                        max_gap_frames: int = 2) -> List[TrackSegment]:
    """将追踪序列按帧间隔拆分为连续片段"""
    if not points:
        return []

    segments = []
    current = [points[0]]

    for i in range(1, len(points)):
        gap = points[i].frame_idx - points[i - 1].frame_idx
        if gap > max_gap_frames:
            if len(current) >= 2:
                segments.append(TrackSegment(
                    points=current,
                    start_frame=current[0].frame_idx,
                    end_frame=current[-1].frame_idx,
                ))
            current = [points[i]]
        else:
            current.append(points[i])

    if len(current) >= 2:
        segments.append(TrackSegment(
            points=current,
            start_frame=current[0].frame_idx,
            end_frame=current[-1].frame_idx,
        ))

    return segments


def fit_parabola(segment: TrackSegment) -> Tuple[np.ndarray, np.ndarray]:
    """对连续片段拟合抛物线 y(t) = at² + bt + c，x(t) = dt + e。

    Returns:
        (x_coeffs, y_coeffs): 多项式系数
    """
    t = np.array([p.time for p in segment.points])
    x = np.array([p.x for p in segment.points])
    y = np.array([p.y for p in segment.points])

    t_norm = t - t[0]

    x_coeffs = np.polyfit(t_norm, x, 1)
    y_coeffs = np.polyfit(t_norm, y, 2)

    return x_coeffs, y_coeffs


def compute_residuals(segment: TrackSegment,
                      x_coeffs: np.ndarray,
                      y_coeffs: np.ndarray) -> List[float]:
    """计算每个点相对于拟合抛物线的残差（欧几里得距离，单位 px）"""
    t = np.array([p.time for p in segment.points])
    t_norm = t - t[0]

    x_fit = np.polyval(x_coeffs, t_norm)
    y_fit = np.polyval(y_coeffs, t_norm)

    x_actual = np.array([p.x for p in segment.points])
    y_actual = np.array([p.y for p in segment.points])

    residuals = np.sqrt((x_actual - x_fit) ** 2 + (y_actual - y_fit) ** 2)
    return residuals.tolist()


def detect_collision_events(kinematics: List[KinematicsPoint],
                            hoop_rect: Dict,
                            sigma_pos: float,
                            baseline_window: int = 5,
                            residual_sigma_single: float = 3.0,
                            residual_sigma_double: float = 2.0,
                            residual_floor_px: float = 3.0,
                            ) -> Tuple[List[CollisionEvent], Dict]:
    """检测球轨迹中的碰撞/弹回事件。

    主信号：
    1. 抛物线拟合残差
    2. 速度方向变化

    Returns:
        (events, stats): 碰撞事件列表, 统计信息
    """
    hoop_cx = hoop_rect.get("cx", 0)
    hoop_cy = hoop_rect.get("cy", 0)
    hoop_w = hoop_rect.get("w", 100)
    hoop_h = hoop_rect.get("h", 50)

    hoop_left = hoop_cx - hoop_w * 0.75
    hoop_right = hoop_cx + hoop_w * 0.75
    hoop_top = hoop_cy - hoop_h * 1.5
    hoop_bottom = hoop_cy + hoop_h * 1.5

    backboard_top = hoop_cy - hoop_h * 3.0
    backboard_bottom = hoop_cy - hoop_h * 0.5
    backboard_left = hoop_cx - hoop_w * 0.8
    backboard_right = hoop_cx + hoop_w * 0.8

    events = []
    stats = {
        "total_segments": 0,
        "analyzed_segments": 0,
        "max_residual_px": 0.0,
        "mean_residual_px": 0.0,
        "residuals_all": [],
    }

    segments = split_into_segments(kinematics, max_gap_frames=2)
    stats["total_segments"] = len(segments)

    effective_threshold_single = max(residual_sigma_single * sigma_pos,
                                     residual_floor_px)
    effective_threshold_double = max(residual_sigma_double * sigma_pos,
                                     residual_floor_px)

    for seg in segments:
        if seg.n_points < baseline_window:
            continue

        stats["analyzed_segments"] += 1

        try:
            x_coeffs, y_coeffs = fit_parabola(seg)
            residuals = compute_residuals(seg, x_coeffs, y_coeffs)
        except (np.linalg.LinAlgError, ValueError):
            continue

        stats["residuals_all"].extend(residuals)
        if residuals:
            stats["max_residual_px"] = max(stats["max_residual_px"],
                                            max(residuals))

        for i in range(1, len(seg.points) - 1):
            residual_px = residuals[i]
            residual_sigma = residual_px / max(sigma_pos, 1e-6)

            pt = seg.points[i]
            prev_pt = seg.points[i - 1]
            next_pt = seg.points[i + 1]

            in_hoop = (hoop_left <= pt.x <= hoop_right and
                       hoop_top <= pt.y <= hoop_bottom)
            in_backboard = (backboard_left <= pt.x <= backboard_right and
                            backboard_top <= pt.y <= backboard_bottom)

            vy_reversed = (prev_pt.vy is not None and next_pt.vy is not None and
                           prev_pt.vy > 0 and next_pt.vy < 0)
            vx_reversed = (prev_pt.vx is not None and next_pt.vx is not None and
                           abs(prev_pt.vx) > 1e-3 and abs(next_pt.vx) > 1e-3 and
                           prev_pt.vx * next_pt.vx < 0)
            lateral_escape = (next_pt.vx is not None and next_pt.vy is not None and
                              abs(next_pt.vx) > abs(next_pt.vy) * 1.2)

            is_multi_frame = (i > 0 and i < len(residuals) - 1 and
                              residuals[i - 1] > effective_threshold_double)

            if (residual_px >= effective_threshold_single or
                    (residual_px >= effective_threshold_double and is_multi_frame)):
                events.append(CollisionEvent(
                    frame_idx=pt.frame_idx,
                    time=pt.time,
                    residual_px=residual_px,
                    residual_sigma=residual_sigma,
                    in_hoop_region=in_hoop,
                    in_backboard_region=in_backboard,
                    vy_reversed=vy_reversed,
                    vx_reversed=vx_reversed,
                    lateral_escape=lateral_escape,
                ))

    if stats["residuals_all"]:
        stats["mean_residual_px"] = float(np.mean(stats["residuals_all"]))

    return events, stats


def find_tracking_gaps(kinematics: List[KinematicsPoint],
                       min_gap_frames: int = 2) -> List[Dict]:
    """找到追踪序列中的消失间隔"""
    gaps = []
    for i in range(1, len(kinematics)):
        frame_gap = kinematics[i].frame_idx - kinematics[i - 1].frame_idx
        if frame_gap >= min_gap_frames:
            gaps.append({
                "start_idx": i - 1,
                "end_idx": i,
                "start_frame": kinematics[i - 1].frame_idx,
                "end_frame": kinematics[i].frame_idx,
                "gap_frames": frame_gap,
                "time_gap": kinematics[i].time - kinematics[i - 1].time,
            })
    return gaps


def analyze_disappear_reappear(kinematics: List[KinematicsPoint],
                                hoop_rect: Dict,
                                fps: float,
                                reappear_time_range: Tuple[float, float] = (0.10, 1.0),
                                position_tolerance: float = 0.5,
                                ) -> List[DisappearPattern]:
    """分析球在篮筐区域的消失-重现模式。

    v6 收紧：最小间隔从 0.06s 提高到 0.10s（过滤检测闪烁），
    位置容差从 0.6 降到 0.5，要求消失时球向下运动。
    """
    hoop_cx = hoop_rect.get("cx", 0)
    hoop_cy = hoop_rect.get("cy", 0)
    hoop_w = hoop_rect.get("w", 100)
    hoop_h = hoop_rect.get("h", 50)
    hoop_bottom = hoop_cy + hoop_h

    results = []
    gaps = find_tracking_gaps(kinematics, min_gap_frames=3)

    for gap in gaps:
        last_visible = kinematics[gap["start_idx"]]
        first_reappear = kinematics[gap["end_idx"]]
        time_gap = gap["time_gap"]

        near_hoop = (abs(last_visible.x - hoop_cx) < hoop_w * 0.8 and
                     abs(last_visible.y - hoop_cy) < hoop_h * 2.0)
        if not near_hoop:
            continue

        reappear_below = first_reappear.y > hoop_bottom
        reappear_near_center = abs(first_reappear.x - hoop_cx) < hoop_w * position_tolerance
        reappear_downward = (first_reappear.vy is not None and first_reappear.vy > 0)
        disappear_downward = (last_visible.vy is not None and last_visible.vy > 0)

        if (reappear_time_range[0] <= time_gap <= reappear_time_range[1] and
                reappear_below and reappear_near_center and reappear_downward
                and disappear_downward):
            results.append(DisappearPattern(
                pattern="goal",
                confidence=0.75,
                detail=f"消失在篮筐区域, {time_gap:.3f}s后从正下方垂直落出",
                gap_start_frame=gap["start_frame"],
                gap_end_frame=gap["end_frame"],
                time_gap=time_gap,
            ))
        elif (time_gap < 0.15 and not reappear_downward and
              last_visible.speed is not None and first_reappear.speed is not None and
              first_reappear.speed > last_visible.speed * 0.4):
            results.append(DisappearPattern(
                pattern="bounce_out",
                confidence=0.80,
                detail=f"快速弹出 ({time_gap:.3f}s), 速度方向反转",
                gap_start_frame=gap["start_frame"],
                gap_end_frame=gap["end_frame"],
                time_gap=time_gap,
            ))
        else:
            results.append(DisappearPattern(
                pattern="unknown",
                confidence=0.0,
                detail=f"消失 {time_gap:.3f}s, 模式不明",
                gap_start_frame=gap["start_frame"],
                gap_end_frame=gap["end_frame"],
                time_gap=time_gap,
            ))

    return results


def channel_8_no_bounce_evidence(kinematics: List[KinematicsPoint],
                                  hoop_rect: Dict,
                                  total_frames: int,
                                  fps: float,
                                  window_s: float = 1.0,
                                  bounce_speed_threshold: float = 5.0,
                                  ) -> Optional[Dict]:
    """反证法通道：检查球到达篮筐后是否存在弹回证据"""
    hoop_cx = hoop_rect.get("cx", 0)
    hoop_cy = hoop_rect.get("cy", 0)
    hoop_w = hoop_rect.get("w", 100)
    hoop_h = hoop_rect.get("h", 50)
    hoop_top = hoop_cy - hoop_h

    approach_point = None
    approach_idx = -1
    for i, p in enumerate(kinematics):
        dist = sqrt((p.x - hoop_cx) ** 2 + (p.y - hoop_cy) ** 2)
        if dist < hoop_w * 0.8 and (p.vy is None or p.vy > 0):
            approach_point = p
            approach_idx = i
            break

    if approach_point is None:
        return None

    post_frames = [p for p in kinematics
                   if approach_point.time < p.time <= approach_point.time + window_s]

    bounce_evidence = []
    for p in post_frames:
        if p.vy is not None and p.vy < -bounce_speed_threshold:
            bounce_evidence.append(("velocity_reversal", p.time))
        if p.y < hoop_top:
            bounce_evidence.append(("upper_reappear", p.time))
        if abs(p.x - hoop_cx) > hoop_w * 0.5:
            bounce_evidence.append(("lateral_escape", p.time))

    if bounce_evidence:
        return None

    expected_frames = int(window_s * fps)
    actual_frames = len(post_frames)
    viewport_coverage = min(1.0, actual_frames / max(expected_frames, 1))

    if viewport_coverage < 0.5:
        return None

    base_confidence = 0.45 if viewport_coverage >= 0.8 else 0.30
    return {
        "timestamp": approach_point.time,
        "confidence": base_confidence,
        "detail": (f"反证法: 覆盖率 {viewport_coverage:.0%}, "
                   f"{actual_frames} 帧已检查, 未观察到弹回"),
        "channel": "8",
    }


def analyze_track_continuity(kinematics: List[KinematicsPoint],
                              hoop_rect: Dict,
                              fps: float,
                              min_continuous: int = 5,
                              ) -> Dict:
    """分析追踪连续性统计数据。

    Returns:
        {
            "total_points": int,
            "total_segments": int,
            "segments": list of segment info,
            "max_continuous": int,
            "near_hoop_segments": int,
            "near_hoop_max_continuous": int,
            "continuity_rate": float,  # 关键区连续追踪率
        }
    """
    hoop_cx = hoop_rect.get("cx", 0)
    hoop_cy = hoop_rect.get("cy", 0)
    hoop_w = hoop_rect.get("w", 100)
    hoop_h = hoop_rect.get("h", 50)

    segments = split_into_segments(kinematics, max_gap_frames=2)
    seg_info = []
    near_hoop_segs = []

    for seg in segments:
        any_near_hoop = any(
            abs(p.x - hoop_cx) < hoop_w * 1.5 and
            abs(p.y - hoop_cy) < hoop_h * 3.0
            for p in seg.points
        )
        info = {
            "start_frame": seg.start_frame,
            "end_frame": seg.end_frame,
            "n_points": seg.n_points,
            "near_hoop": any_near_hoop,
        }
        seg_info.append(info)
        if any_near_hoop:
            near_hoop_segs.append(seg)

    max_cont = max((s.n_points for s in segments), default=0)
    near_hoop_max = max((s.n_points for s in near_hoop_segs), default=0)

    n_qualified = sum(1 for s in near_hoop_segs if s.n_points >= min_continuous)
    continuity_rate = (n_qualified / max(len(near_hoop_segs), 1)
                       if near_hoop_segs else 0.0)

    return {
        "total_points": len(kinematics),
        "total_segments": len(segments),
        "segments": seg_info,
        "max_continuous": max_cont,
        "near_hoop_segments": len(near_hoop_segs),
        "near_hoop_max_continuous": near_hoop_max,
        "continuity_rate": continuity_rate,
        "qualified_segments_ge5": n_qualified,
    }


def judge_goal_by_continuity(kinematics: List[KinematicsPoint],
                              collisions: List[CollisionEvent],
                              hoop_rect: Dict,
                              fps: float,
                              sigma_pos: float = 3.0,
                              ) -> GoalJudgment:
    """基于连续帧运动学分析判断是否进球"""
    hoop_cx = hoop_rect.get("cx", 0)
    hoop_cy = hoop_rect.get("cy", 0)
    hoop_w = hoop_rect.get("w", 100)
    hoop_h = hoop_rect.get("h", 50)
    hoop_left = hoop_cx - hoop_w * 0.5
    hoop_right = hoop_cx + hoop_w * 0.5

    approach_frames = [p for p in kinematics
                       if abs(p.y - hoop_cy) < hoop_h * 1.5 and
                       abs(p.x - hoop_cx) < hoop_w * 1.0]
    if not approach_frames:
        return GoalJudgment(is_goal=False, reason="球未接近篮筐区域")

    downward_frames = sum(1 for f in approach_frames
                          if f.vy is not None and f.vy > 0)
    if downward_frames == 0:
        return GoalJudgment(is_goal=False, reason="球在篮筐区域未呈现向下运动")

    crossing = None
    for i in range(1, len(kinematics)):
        prev, curr = kinematics[i - 1], kinematics[i]
        if (prev.y < hoop_cy <= curr.y and
                hoop_left <= curr.x <= hoop_right):
            crossing = curr
            break

    if crossing is None:
        return GoalJudgment(is_goal=False, reason="球未穿越篮筐平面")

    board_hits = [c for c in collisions if c.in_backboard_region]
    rim_hits = [c for c in collisions
                if c.in_hoop_region and not c.in_backboard_region]

    strong_reject = any(
        c.in_hoop_region and (c.vy_reversed or c.lateral_escape or c.vx_reversed)
        for c in collisions
    )
    if strong_reject:
        return GoalJudgment(
            is_goal=False,
            reason="检测到显著弹回/侧向逃逸证据",
        )

    if board_hits:
        post_crossing = [p for p in kinematics
                         if crossing.time < p.time <= crossing.time + 0.3]
        still_downward = any(p.vy is not None and p.vy > 0 for p in post_crossing)
        if still_downward:
            return GoalJudgment(
                is_goal=True, goal_type="bankshot", confidence=0.82,
                reason="碰板后恢复向下并完成穿越",
                crossing_time=crossing.time,
            )

    if not rim_hits and not board_hits:
        return GoalJudgment(
            is_goal=True, goal_type="swish_or_clean", confidence=0.95,
            reason="未观察到显著打铁证据, 球平滑穿越篮筐平面",
            crossing_time=crossing.time,
        )

    post_crossing = [p for p in kinematics
                     if crossing.time < p.time <= crossing.time + 0.3]
    still_downward = any(p.vy is not None and p.vy > 0 for p in post_crossing)
    if still_downward:
        return GoalJudgment(
            is_goal=True, goal_type="rim_in", confidence=0.80,
            reason="存在接触/微扰, 但穿越后仍保持向下运动",
            crossing_time=crossing.time,
        )

    no_bounce = channel_8_no_bounce_evidence(
        kinematics=kinematics, hoop_rect=hoop_rect,
        total_frames=kinematics[-1].frame_idx if kinematics else 0,
        fps=fps,
    )
    if no_bounce is not None:
        return GoalJudgment(
            is_goal=True, goal_type="inferred", confidence=no_bounce["confidence"],
            reason=no_bounce["detail"], crossing_time=crossing.time,
        )

    return GoalJudgment(is_goal=False, reason="证据不足")


def compute_adaptive_params(img_w: int, img_h: int,
                             native_fps: int, bitrate_kbps: int,
                             hoop_ratio: float = 0.07) -> dict:
    """根据视频规格自动计算最优检测参数"""
    from math import ceil

    hoop_w_px = img_w * hoop_ratio
    ball_d_px = hoop_w_px * (BALL_DIAMETER_M / HOOP_DIAMETER_M)
    bpp = (bitrate_kbps * 1000) / max(img_w * img_h * native_fps, 1)

    if native_fps <= 15:
        coarse_fps = native_fps
    elif img_w >= 3840:
        coarse_fps = 3
    elif img_w >= 1920:
        coarse_fps = 3
    elif img_w >= 1280:
        coarse_fps = 4
    else:
        coarse_fps = 6

    fine_fps = min(native_fps, 30) if native_fps > 24 else native_fps

    g_pixel = 9.81 * (hoop_w_px / HOOP_DIAMETER_M) / (fine_fps ** 2)

    noise_factor = 0.08 if bpp > 0.05 else (0.12 if bpp > 0.02 else 0.15)
    measurement_noise = max(1.5, ball_d_px * noise_factor)

    if bpp < 0.02 or ball_d_px < 15:
        residual_sigma_single = 4.0
        residual_sigma_double = 3.0
    elif bpp < 0.05 or ball_d_px < 25:
        residual_sigma_single = 3.5
        residual_sigma_double = 2.5
    else:
        residual_sigma_single = 3.0
        residual_sigma_double = 2.0

    return {
        "coarse_fps": coarse_fps,
        "fine_fps": fine_fps,
        "gravity_pixel": round(g_pixel, 3),
        "tracker": "kalman" if ball_d_px >= 20 and fine_fps >= 24 else "iou",
        "process_noise": round(g_pixel * 0.5, 2),
        "measurement_noise": round(measurement_noise, 1),
        "distance_threshold": round(max(30, ball_d_px * 2.5), 0),
        "max_lost": max(3, int(0.8 * coarse_fps)),
        "residual_sigma_single": residual_sigma_single,
        "residual_sigma_double": residual_sigma_double,
        "residual_floor_px": round(ball_d_px * 0.12, 1),
        "sigma_pos_default": round(measurement_noise * 0.5, 1),
        "ball_d_px": round(ball_d_px, 1),
        "hoop_w_px": round(hoop_w_px, 1),
        "bpp": round(bpp, 4),
        "kinematics_feasible": fine_fps >= 24,
    }


def cross_track_disappear_reappear(all_kinematics: List[List[KinematicsPoint]],
                                    hoop_rect: Dict,
                                    fps: float,
                                    max_time_gap: float = 1.5,
                                    ) -> List[Dict]:
    """跨轨迹消失-重现分析。

    检测两种进球模式：
    模式1（标准）: 球从上方飞向篮筐消失 -> 从篮筐下方重现
    模式2（落下）: 球在篮筐正下方以向下速度出现（穿筐后的落下段）
    """
    hoop_cx = hoop_rect.get("cx", 0)
    hoop_cy = hoop_rect.get("cy", 0)
    hoop_w = hoop_rect.get("w", 100)
    hoop_h = hoop_rect.get("h", 50)

    track_info = []
    for ti, kin in enumerate(all_kinematics):
        if len(kin) < 2:
            continue
        end_pt = kin[-1]
        start_pt = kin[0]

        end_vys = [p.vy for p in kin[-min(4, len(kin)):] if p.vy is not None]
        avg_end_vy = sum(end_vys) / len(end_vys) if end_vys else 0

        approaching_hoop = (
            abs(end_pt.x - hoop_cx) < hoop_w * 0.8 and
            end_pt.y < hoop_cy and
            avg_end_vy > 50 and
            len(kin) >= 4
        )

        x_trend_toward = False
        if len(kin) >= 3:
            dx = end_pt.x - kin[-3].x
            x_gap = end_pt.x - hoop_cx
            x_trend_toward = abs(x_gap) < hoop_w * 0.8 or (x_gap > 0 and dx < 0) or (x_gap < 0 and dx > 0)

        near_hoop_end = approaching_hoop and x_trend_toward

        below_hoop_start = (
            start_pt.y > hoop_cy + hoop_h * 0.3 and
            abs(start_pt.x - hoop_cx) < hoop_w * 1.0
        )

        track_info.append({
            "track_idx": ti,
            "end_time": end_pt.time,
            "end_x": end_pt.x,
            "end_y": end_pt.y,
            "end_vy": avg_end_vy,
            "end_near_hoop": near_hoop_end,
            "start_time": start_pt.time,
            "start_x": start_pt.x,
            "start_y": start_pt.y,
            "start_vy": start_pt.vy,
            "start_below_hoop": below_hoop_start,
            "n_points": len(kin),
        })

    results = []

    # 模式1: 消失-重现匹配
    for ep_a in track_info:
        if not ep_a["end_near_hoop"]:
            continue

        for ep_b in track_info:
            if ep_b["track_idx"] == ep_a["track_idx"]:
                continue
            if not ep_b["start_below_hoop"]:
                continue

            time_gap = ep_b["start_time"] - ep_a["end_time"]
            if time_gap < 0.03 or time_gap > max_time_gap:
                continue

            reappear_near_center = abs(ep_b["start_x"] - hoop_cx) < hoop_w * 0.7
            reappear_downward = ep_b["start_vy"] is None or ep_b["start_vy"] > -20

            if reappear_near_center and reappear_downward:
                confidence = 0.75
                if time_gap <= 0.5:
                    confidence = 0.82
                if time_gap <= 0.25:
                    confidence = 0.88

                event_time = (ep_a["end_time"] + ep_b["start_time"]) / 2
                results.append({
                    "type": "cross_track_goal",
                    "timestamp": event_time,
                    "confidence": confidence,
                    "disappear_track": ep_a["track_idx"],
                    "reappear_track": ep_b["track_idx"],
                    "disappear_time": ep_a["end_time"],
                    "disappear_pos": (ep_a["end_x"], ep_a["end_y"]),
                    "reappear_time": ep_b["start_time"],
                    "reappear_pos": (ep_b["start_x"], ep_b["start_y"]),
                    "time_gap": round(time_gap, 4),
                    "detail": (f"跨轨迹进球: 消失t={ep_a['end_time']:.2f}s "
                               f"y={ep_a['end_y']:.0f} -> "
                               f"重现t={ep_b['start_time']:.2f}s "
                               f"y={ep_b['start_y']:.0f} "
                               f"gap={time_gap:.3f}s"),
                })

    # 模式2: 篮筐正下方落下检测
    for ep in track_info:
        start_y = ep["start_y"] if ep["start_below_hoop"] else None
        if start_y is None:
            continue

        ti = ep["track_idx"]
        kin = all_kinematics[ti]
        if len(kin) < 3:
            continue

        in_drop_zone = (
            abs(kin[0].x - hoop_cx) < hoop_w * 0.4 and
            hoop_cy + hoop_h * 0.5 < kin[0].y < hoop_cy + hoop_h * 3.0
        )
        if not in_drop_zone:
            continue

        drop_vys = [p.vy for p in kin[:min(5, len(kin))] if p.vy is not None]
        avg_drop_vy = sum(drop_vys) / len(drop_vys) if drop_vys else 0
        mostly_down = (sum(1 for v in drop_vys if v > 50) >= len(drop_vys) * 0.7
                       if drop_vys else False)

        if mostly_down and len(drop_vys) >= 3 and avg_drop_vy > 80:
            already_matched = any(
                r["reappear_track"] == ti for r in results
            )
            if not already_matched:
                results.append({
                    "type": "drop_below_hoop",
                    "timestamp": kin[0].time,
                    "confidence": 0.55,
                    "disappear_track": -1,
                    "reappear_track": ti,
                    "disappear_time": kin[0].time - 0.2,
                    "disappear_pos": (hoop_cx, hoop_cy),
                    "reappear_time": kin[0].time,
                    "reappear_pos": (kin[0].x, kin[0].y),
                    "time_gap": 0,
                    "detail": (f"篮筐下方落下: t={kin[0].time:.2f}s "
                               f"({kin[0].x:.0f},{kin[0].y:.0f}) "
                               f"avg_vy={sum(drop_vys)/len(drop_vys):.0f}"),
                })

    results.sort(key=lambda r: r["timestamp"])

    deduped = []
    for r in results:
        if not deduped or abs(r["timestamp"] - deduped[-1]["timestamp"]) > 2.0:
            deduped.append(r)
        elif r["confidence"] > deduped[-1]["confidence"]:
            deduped[-1] = r

    return deduped
