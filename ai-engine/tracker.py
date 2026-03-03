"""
GoalCut 轻量级 IOU 追踪器

基于 IOU（Intersection over Union）的帧间目标关联，为篮球建立跨帧轨迹。
不依赖外部追踪库（如 ByteTrack），仅使用 numpy。

核心逻辑：
1. 对每帧的检测框，计算与上一帧已有轨迹最后位置的 IOU
2. IOU 超过阈值则关联到已有轨迹，否则创建新轨迹
3. 轨迹连续丢失超过 max_lost 帧则终止

输出：带 track_id 的轨迹序列，供几何进球判定使用。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Detection:
    """单帧检测结果"""
    frame_idx: int
    cx: float      # 中心 x
    cy: float      # 中心 y
    w: float       # 宽度
    h: float       # 高度
    conf: float    # 置信度

    @property
    def x1(self) -> float:
        return self.cx - self.w / 2

    @property
    def y1(self) -> float:
        return self.cy - self.h / 2

    @property
    def x2(self) -> float:
        return self.cx + self.w / 2

    @property
    def y2(self) -> float:
        return self.cy + self.h / 2


@dataclass
class TrackPoint:
    """轨迹点"""
    frame_idx: int
    cx: float
    cy: float
    conf: float


@dataclass
class Track:
    """一条完整轨迹"""
    track_id: int
    points: List[TrackPoint] = field(default_factory=list)
    lost_count: int = 0          # 连续丢失帧数
    last_detection: Optional[Detection] = None  # 最后一次匹配到的检测

    @property
    def last_point(self) -> Optional[TrackPoint]:
        return self.points[-1] if self.points else None

    def predict_position(self) -> Tuple[float, float]:
        """基于最近两个点预测下一帧位置（简单线性外推）"""
        if len(self.points) < 2:
            p = self.points[-1]
            return p.cx, p.cy
        p1, p2 = self.points[-2], self.points[-1]
        dx = p2.cx - p1.cx
        dy = p2.cy - p1.cy
        return p2.cx + dx, p2.cy + dy

    def add_point(self, det: Detection):
        self.points.append(TrackPoint(
            frame_idx=det.frame_idx,
            cx=det.cx,
            cy=det.cy,
            conf=det.conf,
        ))
        self.last_detection = det
        self.lost_count = 0


def compute_iou(det: Detection, track: Track) -> float:
    """计算检测框与轨迹最后位置的 IOU"""
    if track.last_detection is None:
        return 0.0

    last = track.last_detection

    # 如果轨迹已丢失，使用预测位置构造虚拟框
    if track.lost_count > 0:
        pred_cx, pred_cy = track.predict_position()
        # 使用最后检测的宽高
        ax1 = pred_cx - last.w / 2
        ay1 = pred_cy - last.h / 2
        ax2 = pred_cx + last.w / 2
        ay2 = pred_cy + last.h / 2
    else:
        ax1, ay1, ax2, ay2 = last.x1, last.y1, last.x2, last.y2

    bx1, by1, bx2, by2 = det.x1, det.y1, det.x2, det.y2

    # 交集
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def compute_distance(det: Detection, track: Track) -> float:
    """计算检测框中心与轨迹预测位置的欧式距离（归一化）"""
    pred_cx, pred_cy = track.predict_position()
    dx = det.cx - pred_cx
    dy = det.cy - pred_cy
    return np.sqrt(dx * dx + dy * dy)


class IOUTracker:
    """
    基于 IOU 的轻量级追踪器。

    参数：
        iou_threshold: IOU 匹配阈值，低于此值不关联
        max_lost: 连续丢失帧数上限，超过则终止轨迹
        distance_threshold: 当 IOU 为 0 时的距离匹配阈值（像素）
    """

    def __init__(self, iou_threshold: float = 0.2, max_lost: int = 5,
                 distance_threshold: float = 100.0):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.distance_threshold = distance_threshold
        self.next_id = 1
        self.active_tracks: List[Track] = []
        self.finished_tracks: List[Track] = []

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        用当前帧的检测结果更新追踪器状态。

        Args:
            detections: 当前帧的所有检测结果

        Returns:
            当前活跃的所有轨迹
        """
        if not detections:
            # 没有检测结果，所有轨迹丢失计数 +1
            for track in self.active_tracks:
                track.lost_count += 1
            self._prune_lost_tracks()
            return self.active_tracks

        if not self.active_tracks:
            # 没有已有轨迹，所有检测创建新轨迹
            for det in detections:
                self._create_track(det)
            return self.active_tracks

        # 计算 IOU 矩阵 + 距离矩阵
        n_tracks = len(self.active_tracks)
        n_dets = len(detections)
        iou_matrix = np.zeros((n_tracks, n_dets))
        dist_matrix = np.full((n_tracks, n_dets), float('inf'))

        for i, track in enumerate(self.active_tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = compute_iou(det, track)
                dist_matrix[i, j] = compute_distance(det, track)

        # 贪心匹配：优先 IOU 最高的配对
        matched_tracks = set()
        matched_dets = set()

        # 第一阶段：IOU 匹配
        while True:
            # 找未匹配中 IOU 最大的
            best_iou = self.iou_threshold
            best_i, best_j = -1, -1
            for i in range(n_tracks):
                if i in matched_tracks:
                    continue
                for j in range(n_dets):
                    if j in matched_dets:
                        continue
                    if iou_matrix[i, j] > best_iou:
                        best_iou = iou_matrix[i, j]
                        best_i, best_j = i, j

            if best_i < 0:
                break

            self.active_tracks[best_i].add_point(detections[best_j])
            matched_tracks.add(best_i)
            matched_dets.add(best_j)

        # 第二阶段：距离匹配（针对 IOU 为 0 但距离近的情况，如遮挡后重现）
        for i in range(n_tracks):
            if i in matched_tracks:
                continue
            best_dist = self.distance_threshold
            best_j = -1
            for j in range(n_dets):
                if j in matched_dets:
                    continue
                if dist_matrix[i, j] < best_dist:
                    best_dist = dist_matrix[i, j]
                    best_j = j

            if best_j >= 0:
                self.active_tracks[i].add_point(detections[best_j])
                matched_tracks.add(i)
                matched_dets.add(best_j)

        # 未匹配的轨迹：丢失计数 +1
        for i in range(n_tracks):
            if i not in matched_tracks:
                self.active_tracks[i].lost_count += 1

        # 未匹配的检测：创建新轨迹
        for j in range(n_dets):
            if j not in matched_dets:
                self._create_track(detections[j])

        self._prune_lost_tracks()
        return self.active_tracks

    def _create_track(self, det: Detection) -> Track:
        track = Track(track_id=self.next_id)
        self.next_id += 1
        track.add_point(det)
        self.active_tracks.append(track)
        return track

    def _prune_lost_tracks(self):
        """移除丢失过久的轨迹"""
        still_active = []
        for track in self.active_tracks:
            if track.lost_count > self.max_lost:
                if len(track.points) >= 2:
                    self.finished_tracks.append(track)
            else:
                still_active.append(track)
        self.active_tracks = still_active

    def get_all_tracks(self) -> List[Track]:
        """获取所有轨迹（活跃 + 已结束），至少 2 个点"""
        all_tracks = []
        for t in self.finished_tracks:
            if len(t.points) >= 2:
                all_tracks.append(t)
        for t in self.active_tracks:
            if len(t.points) >= 2:
                all_tracks.append(t)
        return all_tracks
