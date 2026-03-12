#!/usr/bin/env python3
"""
GoalCut AI 进球检测引擎 v2.7

Phase 1.5 P4 增强版本，新增（T-1.9.13 + T-1.9.14 + O-23）：
- 因果推理链验证器（算法E，O-23）：在跨通道互证后对候选事件做时序因果检查，
  区分投篮/上篮/罚球三种得分方式，孤立噪声事件惩罚 -0.15，强因果链 boost +0.12
- 通道4改为有向光流穿越检测（算法C），区分进球（垂直向下）与篮板争抢（横向混乱）
- 精扫阶段全帧采样（frame_step=1），解决3fps采样率与球穿越时间窗口的物理冲突
- 通道3（运动突变）置信度上限从0.50降至0.35，仅作候选触发信号而非独立进球信号
- 篮网ROI高度系数从1.5优化为1.2，减少无关区域噪声

v2.4 已有：
- 自适应融合策略（通道1a失效时启用VLM混合模式）
- VLM逐事件确认 + VLM独立扫描补充漏检

v2.2 已有：
- 篮网形变检测（光流法，net_deform.py，通道 4）
- 入网声/哨声检测（audio_detect.py，通道 5）
- 两阶段采样策略（粗扫定位候选区间，精扫几何判定，--two-stage）

v2.1 已有：
- 动态篮筐检测（替代硬编码 ROI）
- IOU 追踪器建立球体轨迹
- 基于空间几何的进球判定（替代启发式规则）
- 冷静期机制防止重复计数
- 多通道并行检测：YOLO球体 + 人体运动 + 帧差分运动
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
# 算法配置（Algorithm Profile）
# ============================================================

class AlgorithmProfile:
    """算法配置：定义通道开关、置信度上限、互证参数、VLM 模式。

    加载来源（优先级从高到低）：
    1. --algorithm-config <yaml 文件路径>  独立配置文件
    2. --algorithm-profile <名称>          config.yaml 中的预定义方案
    3. 命令行 --channel 参数               向后兼容的通道过滤
    4. 代码默认值                          全通道启用
    """

    # 所有已知通道 ID
    ALL_CHANNELS = {"1a", "1b", "2", "3", "4", "5", "6", "7"}

    # 代码内置默认值
    DEFAULT_CAPS = {
        "1a": 0.95,
        "1b": 0.60,
        "2": 0.55,
        "3": 0.35,
        "4": 0.70,
        "5": 0.70,
        "6": 0.90,
        "7": 0.50,
    }

    # 通道 ID → detail 前缀的映射（用于匹配 WEAK_CHANNEL_CAPS）
    CHANNEL_DETAIL_MAP = {
        "1a": ["几何判定", "辅助判定"],
        "1b": ["启发式"],
        "2": ["人体运动"],
        "3": ["运动突变"],
        "4": ["篹网形变", "篹网有向穿越", "篮网形变", "篮网有向穿越"],
        "5": ["入网声", "哨声", "叫好声", "击掌声"],
        "6": ["ROI分类器"],
        "7": ["庆祝动作"],
    }

    def __init__(self):
        self.name = "default"
        self.description = "v2.5 默认全通道配置"
        self.channels = {}  # channel_id -> {"enabled": bool, "confidence_cap": float}
        self.cross_validation = {
            "window": 2.0,
            "boost_per_channel": 0.08,
            "max_boost": 0.15,
        }
        self.vlm = {
            "enabled": False,
            "mode": "fallback",
            "confirm_range": [0.35, 0.70],
        }
        self.detection_overrides = {}

        for ch_id in self.ALL_CHANNELS:
            self.channels[ch_id] = {
                "enabled": True,
                "confidence_cap": self.DEFAULT_CAPS.get(ch_id, 0.95),
            }

    def is_enabled(self, channel_id: str) -> bool:
        ch = self.channels.get(channel_id)
        return ch is not None and ch.get("enabled", True)

    def get_cap(self, channel_id: str) -> float:
        ch = self.channels.get(channel_id)
        if ch:
            return ch.get("confidence_cap", 0.95)
        return 0.95

    def get_cap_by_detail(self, detail_prefix: str) -> float:
        """根据事件 detail 前缀查找置信度上限"""
        for ch_id, prefixes in self.CHANNEL_DETAIL_MAP.items():
            if detail_prefix in prefixes:
                return self.get_cap(ch_id)
        return 0.95

    def build_weak_channel_caps(self) -> Dict[str, float]:
        """生成 WEAK_CHANNEL_CAPS 字典（detail 前缀 → 置信度上限）"""
        caps = {}
        for ch_id, prefixes in self.CHANNEL_DETAIL_MAP.items():
            cap = self.get_cap(ch_id)
            if cap < 0.95:
                for prefix in prefixes:
                    caps[prefix] = cap
        # 击掌声单独降权：野球场击掌声密度高、误报多，
        # 单独设置更低的 cap 以减少其作为独立事件的误报
        caps["击掌声"] = min(caps.get("击掌声", 0.95), 0.50)
        # 入网声降权：野球场环境下球碰篮板/篮筐也可能触发高频能量突发
        caps["入网声"] = min(caps.get("入网声", 0.95), 0.60)
        return caps

    def get_enabled_set(self) -> set:
        """返回启用的通道 ID 集合（用于 ch_enabled 检查）"""
        return {ch_id for ch_id, cfg in self.channels.items()
                if cfg.get("enabled", True)}

    def summary(self) -> str:
        lines = [f"算法配置: {self.name} — {self.description}"]
        for ch_id in sorted(self.channels.keys(),
                            key=lambda x: (x.replace("a","0").replace("b","1"))):
            cfg = self.channels[ch_id]
            status = "启用" if cfg.get("enabled") else "禁用"
            cap = cfg.get("confidence_cap", 0.95)
            lines.append(f"  通道 {ch_id}: {status} (上限={cap})")
        vlm_status = "启用" if self.vlm.get("enabled") else "禁用"
        vlm_mode = self.vlm.get("mode", "fallback")
        lines.append(f"  VLM: {vlm_status} (模式={vlm_mode})")
        cv = self.cross_validation
        lines.append(f"  互证: 窗口={cv['window']}s, "
                      f"boost={cv['boost_per_channel']}/通道, "
                      f"上限={cv['max_boost']}")
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict, name: str = "custom") -> "AlgorithmProfile":
        """从字典（YAML 解析结果）创建配置"""
        profile = cls()
        profile.name = name
        profile.description = data.get("description", name)

        if "channels" in data:
            for ch_id, ch_cfg in data["channels"].items():
                ch_id = str(ch_id)
                if ch_id not in profile.channels:
                    profile.channels[ch_id] = {}
                if isinstance(ch_cfg, dict):
                    profile.channels[ch_id]["enabled"] = ch_cfg.get("enabled", True)
                    if "confidence_cap" in ch_cfg:
                        profile.channels[ch_id]["confidence_cap"] = ch_cfg["confidence_cap"]
                elif isinstance(ch_cfg, bool):
                    profile.channels[ch_id]["enabled"] = ch_cfg

        if "cross_validation" in data:
            profile.cross_validation.update(data["cross_validation"])

        if "vlm" in data:
            profile.vlm.update(data["vlm"])

        if "detection_overrides" in data:
            profile.detection_overrides = data["detection_overrides"]

        return profile

    @classmethod
    def load(cls, profile_name: str = None,
             config_path: str = None) -> "AlgorithmProfile":
        """加载算法配置。

        Args:
            profile_name: 预定义方案名称（从 config.yaml 的 algorithm_profiles 中读取）
            config_path: 独立算法配置 YAML 文件路径
        """
        if config_path and os.path.exists(config_path):
            try:
                import yaml
            except ImportError:
                log("警告: 加载 YAML 需要 pyyaml (pip install pyyaml)")
                return cls()
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            name = os.path.basename(config_path).replace(".yaml", "")
            if "channels" in data:
                return cls.from_dict(data, name)
            if "algorithm_profiles" in data and profile_name:
                pdata = data["algorithm_profiles"].get(profile_name)
                if pdata:
                    return cls.from_dict(pdata, profile_name)
            return cls()

        if profile_name:
            candidates = [
                os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml"),
                os.path.join(os.path.dirname(__file__), "configs", "config.yaml"),
                "configs/config.yaml",
            ]
            for cfg_path in candidates:
                if os.path.exists(cfg_path):
                    try:
                        import yaml
                        with open(cfg_path, 'r', encoding='utf-8') as f:
                            data = yaml.safe_load(f)
                        profiles = data.get("algorithm_profiles", {})
                        if profile_name in profiles:
                            profile = cls.from_dict(profiles[profile_name],
                                                     profile_name)
                            # 用全局 vlm 段填充方案中未设置的 VLM 凭证
                            global_vlm = data.get("vlm", {})
                            for key in ("provider", "model", "api_key"):
                                if not profile.vlm.get(key):
                                    profile.vlm[key] = global_vlm.get(key, "")
                            return profile
                        else:
                            available = list(profiles.keys())
                            log(f"警告: 未找到算法配置 '{profile_name}'，"
                                f"可选: {available}")
                    except ImportError:
                        log("警告: 加载 YAML 需要 pyyaml")
                    except Exception as e:
                        log(f"警告: 加载配置失败: {e}")
                    break

        return cls()

    @classmethod
    def from_channel_filter(cls, channel_str: str) -> "AlgorithmProfile":
        """从 --channel 参数向后兼容创建配置（如 '1a,3,5'）"""
        profile = cls()
        profile.name = "custom_filter"
        profile.description = f"命令行通道过滤: {channel_str}"
        enabled = set(c.strip() for c in channel_str.split(","))
        for ch_id in profile.channels:
            profile.channels[ch_id]["enabled"] = ch_id in enabled
        return profile


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
                if name.lower() in ('hoop', 'basket', 'rim', 'backboard',
                                     'basketball-hoop', 'basketball_hoop',
                                     'ring', 'basket-rim'):
                    hoop_classes.append(cls_id)
            if not hoop_classes:
                return None

            # 在采样帧中检测篮筐
            hoop_candidates = []
            sample_indices = np.linspace(0, len(frame_files) - 1,
                                         min(20, len(frame_files)), dtype=int)
            for idx in sample_indices:
                results = model(frame_files[idx], conf=0.3, imgsz=640, verbose=False)
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

    改进v2: 提高上升帧数阈值(3→)，缩小篮筐区域，降低置信度上限，
    减少左右半场重叠，增加位移幅度要求。
    """
    log("开始人体运动模式检测...")

    # 根据模型类别自动判断人体 class ID
    # 通过检查 model.names 判断是否为篮球专用模型
    _is_basketball = any(
        n.lower() in ("ball", "basketball") for n in model.names.values()
    )
    if _is_basketball:
        PERSON_CLASS = None
        for cls_id, name in model.names.items():
            if name.lower() in ("person", "player"):
                PERSON_CLASS = cls_id
                break
        if PERSON_CLASS is None:
            log("警告: 篮球专用模型中未找到 person/player 类别")
            PERSON_CLASS = -1  # 不存在，跳过人体检测
        else:
            log(f"人体类别: class {PERSON_CLASS} ({model.names.get(PERSON_CLASS, '?')})")
    else:
        PERSON_CLASS = 0  # COCO person
    # 篮筐区域：画面上部（Y < 45% 画面高度），比之前更严格
    hoop_zone_y = img_h * 0.45

    # 收集所有帧中的人体检测
    frame_persons = {}  # {frame_idx: [(cx, cy, w, h, conf), ...]}

    for i, frame_path in enumerate(frame_files):
        try:
            results = model(frame_path, conf=yolo_confidence, imgsz=640, verbose=False)
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

    # 对左半场和右半场分别检测上篮模式（减少重叠区域）
    for side_name, x_filter in [("left", lambda cx: cx < mid_x * 1.05),
                                 ("right", lambda cx: cx > mid_x * 0.95)]:
        prev_top_cy = None
        prev_fidx = -999
        rising_count = 0
        total_rise = 0
        peak_frame = -1
        peak_cy = 9999
        peak_cx = 0

        for fidx in sorted_frames:
            persons = frame_persons[fidx]
            # 取该半场最高的人
            side_persons = [p for p in persons if x_filter(p[0])]
            if not side_persons:
                # 该半场无人，如果之前在上升中，视为可能的上篮完成
                if (rising_count >= 3 and peak_cy < hoop_zone_y
                        and peak_frame > skip_frames
                        and total_rise > img_h * 0.06):
                    timestamp = peak_frame / sample_fps
                    height_factor = max(0, 1.0 - peak_cy / hoop_zone_y)
                    conf = min(0.55, 0.30 + 0.05 * rising_count
                               + 0.15 * height_factor)
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
                total_rise = 0
                prev_top_cy = None
                prev_fidx = -999
                continue

            top_person = min(side_persons, key=lambda p: p[1])
            top_cx, top_cy = top_person[0], top_person[1]

            if prev_top_cy is not None and fidx - prev_fidx <= 2:
                dy = top_cy - prev_top_cy  # dy < 0 = 上升

                if dy < -img_h * 0.02:  # 上升（2%画面高度，更严格）
                    rising_count += 1
                    total_rise += abs(dy)
                    if top_cy < peak_cy:
                        peak_cy = top_cy
                        peak_frame = fidx
                        peak_cx = top_cx
                elif dy > img_h * 0.01 and rising_count >= 3:
                    # 下降 → 上篮可能完成
                    if (peak_cy < hoop_zone_y and peak_frame > skip_frames
                            and total_rise > img_h * 0.06):
                        timestamp = peak_frame / sample_fps
                        height_factor = max(0, 1.0 - peak_cy / hoop_zone_y)
                        conf = min(0.55, 0.30 + 0.05 * rising_count
                                   + 0.15 * height_factor)
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
                    total_rise = 0
                elif dy >= 0:
                    # 停滞或微降，如果之前上升不够就重置
                    if rising_count < 3:
                        rising_count = 0
                        peak_cy = 9999
                        total_rise = 0
            else:
                # 帧不连续，结算之前的上升
                if (rising_count >= 3 and peak_cy < hoop_zone_y
                        and peak_frame > skip_frames
                        and total_rise > img_h * 0.06):
                    timestamp = peak_frame / sample_fps
                    height_factor = max(0, 1.0 - peak_cy / hoop_zone_y)
                    conf = min(0.50, 0.28 + 0.05 * rising_count
                               + 0.12 * height_factor)
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
                total_rise = 0

            prev_top_cy = top_cy
            prev_fidx = fidx

        # 循环结束，结算未处理的上升
        if (rising_count >= 3 and peak_cy < hoop_zone_y
                and peak_frame > skip_frames
                and total_rise > img_h * 0.06):
            timestamp = peak_frame / sample_fps
            height_factor = max(0, 1.0 - peak_cy / hoop_zone_y)
            conf = min(0.50, 0.28 + 0.05 * rising_count
                       + 0.12 * height_factor)
            candidate_events.append({
                "frame_index": peak_frame,
                "timestamp": round(timestamp, 2),
                "confidence": round(conf, 3),
                "detail": (f"人体运动({side_name}): 上篮模式, "
                           f"连续上升{rising_count}帧, "
                           f"peak_y={peak_cy:.0f}, "
                           f"cx={peak_cx:.0f}")
            })

    # 去重：左右半场重叠区域可能检测到同一事件
    if len(candidate_events) > 1:
        candidate_events.sort(key=lambda e: e["timestamp"])
        deduped = [candidate_events[0]]
        for ev in candidate_events[1:]:
            if ev["timestamp"] - deduped[-1]["timestamp"] > 3.0:
                deduped.append(ev)
            elif ev["confidence"] > deduped[-1]["confidence"]:
                deduped[-1] = ev
        candidate_events = deduped

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
                        # T-1.9.14: 通道3上限从0.50降至0.35，仅作触发信号
                        conf = min(0.35, 0.2 + 0.04 * burst_ratio)
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

def two_stage_filter_frames(frame_files: List[str], sample_fps: float,
                             rough_events: List[Dict],
                             window_s: float = 4.0) -> List[str]:
    """
    两阶段采样 - 阶段2：根据粗扫候选区间筛选精扫帧。

    在每个候选事件前后 window_s 秒范围内保留全部帧；
    其余区域跳过（已在阶段1粗扫过）。

    Args:
        frame_files:   全部帧（按顺序）
        sample_fps:    帧率
        rough_events:  阶段1候选事件列表
        window_s:      候选事件前后保留范围（秒）

    Returns:
        精扫帧文件列表（比全量少，聚焦候选区间）
    """
    if not rough_events:
        return frame_files

    # 建立候选区间集合（帧索引范围）
    candidate_ranges = []
    for ev in rough_events:
        t = ev["timestamp"]
        start_f = max(0, int((t - window_s) * sample_fps))
        end_f = int((t + window_s) * sample_fps) + 1
        candidate_ranges.append((start_f, end_f))

    # 合并重叠区间
    candidate_ranges.sort()
    merged = [candidate_ranges[0]]
    for s, e in candidate_ranges[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # 筛选帧
    fine_frames = []
    for i, path in enumerate(frame_files):
        for s, e in merged:
            if s <= i <= e:
                fine_frames.append(path)
                break

    log(f"两阶段采样: 全量 {len(frame_files)} 帧 → 精扫 {len(fine_frames)} 帧 "
        f"(节省 {100*(1-len(fine_frames)/max(1,len(frame_files))):.0f}%)")
    return fine_frames


def detect_goals(frames_dir, output_path, sample_fps, confidence_threshold,
                 yolo_confidence, video_duration,
                 audio_file=None, enable_net_deform=False, two_stage=False,
                 enabled_channels=None, source_video=None, vlm_confirm=False,
                 algorithm_profile=None):
    """主检测流程 v2.5 - 多通道并行检测（有向光流通道4 + 全帧精扫 + 通道3降权）
    
    v2.5 变更：
    - 通道4改为有向光流穿越检测（detect_downward_flow_through_net）
    - 精扫阶段全帧不跳过（frame_step=1），确保捕捉穿越帧
    - 通道3（运动突变）置信度上限降至0.35
    - 篮网ROI高度系数优化
    
    v2.6 新增：
    - algorithm_profile: AlgorithmProfile 对象，定义通道开关/置信度上限/互证参数/VLM模式
    - 支持预定义方案（pickup_no_net, pickup_with_net, minimal, vlm_primary）
    - 向后兼容：未传 profile 时行为与 v2.5 完全一致
    
    Args:
        enabled_channels: 指定要运行的通道集合，如 {"1a","2","5"}。None=全部。
                         （向后兼容，优先级低于 algorithm_profile）
        source_video: 源视频路径（VLM确认时需要，用于采样帧）
        vlm_confirm: 是否在通道1a失效时启用VLM逐事件确认
        algorithm_profile: AlgorithmProfile 对象。传入后覆盖 enabled_channels/vlm_confirm
                          等参数，实现完整的算法组合自定义。
    """
    # 算法配置：profile 优先，其次 enabled_channels，最后全部启用
    profile = algorithm_profile or AlgorithmProfile()
    if algorithm_profile:
        log(f"\n{profile.summary()}")
        # profile 中的通道开关覆盖 enabled_channels
        enabled_channels = profile.get_enabled_set()
        # profile 中的 VLM 设置覆盖命令行
        if profile.vlm.get("enabled"):
            vlm_confirm = True
        # 将 VLM 凭证注入模块级变量（方案级覆盖全局）
        global _vlm_config
        _vlm_config = {
            "provider": profile.vlm.get("provider", ""),
            "model": profile.vlm.get("model", ""),
            "api_key": profile.vlm.get("api_key", ""),
        }
        # profile 中的检测参数覆盖
        overrides = profile.detection_overrides
        if "confidence_threshold" in overrides:
            confidence_threshold = overrides["confidence_threshold"]
        if "yolo_confidence" in overrides:
            yolo_confidence = overrides["yolo_confidence"]
        if "sample_fps" in overrides:
            sample_fps = overrides["sample_fps"]
        # profile 中通道 4 的开关覆盖 enable_net_deform
        if profile.is_enabled("4"):
            enable_net_deform = True
        else:
            enable_net_deform = False

    log(f"开始进球检测 (v2.6)")
    log(f"  帧目录: {frames_dir}")
    log(f"  采样帧率: {sample_fps}")
    log(f"  置信度阈值: {confidence_threshold}")
    log(f"  YOLO置信度: {yolo_confidence}")
    log(f"  视频时长: {video_duration}s")
    log(f"  音频文件: {audio_file or '未提供'}")
    log(f"  篮网形变检测: {'启用' if enable_net_deform else '禁用'}")
    log(f"  两阶段采样: {'启用' if two_stage else '禁用'}")
    if enabled_channels:
        log(f"  通道过滤: 仅运行 {sorted(enabled_channels)}")

    def ch_enabled(ch_name):
        """检查通道是否启用"""
        return enabled_channels is None or ch_name in enabled_channels

    # 加载 YOLO 模型（优先使用篮球专用模型）
    log("加载 YOLO 模型...")
    model = None
    _using_basketball_model = False
    try:
        from ultralytics import YOLO

        # 篮球专用模型路径（检查多个可能位置）
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

        _COCO_MODEL = os.path.join(_ai_engine_dir, "yolov8n.pt")

        if _BASKETBALL_MODEL:
            model = YOLO(_BASKETBALL_MODEL)
            _using_basketball_model = True
            log(f"篮球专用 YOLO 模型加载成功: {_BASKETBALL_MODEL}")
            log(f"  模型类别: {model.names}")
        else:
            model = YOLO(_COCO_MODEL if os.path.exists(_COCO_MODEL) else "yolov8n.pt")
            log(f"篮球专用模型不存在，回退到 COCO 预训练: yolov8n.pt")
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

    # 两阶段采样：阶段1 使用稀疏帧（每隔1帧，约1.5fps）快速粗扫
    if two_stage and len(frame_files) > 20:
        log("--- 两阶段采样: 阶段1 粗扫（每隔1帧）---")
        rough_frames = frame_files[::2]  # 每隔1帧取1帧，~1.5fps
        log(f"  粗扫帧数: {len(rough_frames)} / {len(frame_files)}")
    else:
        rough_frames = frame_files

    # 获取图像尺寸
    import cv2
    sample_img = cv2.imread(frame_files[0])
    if sample_img is None:
        log("错误: 无法读取帧图片")
        save_results([], output_path)
        return
    img_h, img_w = sample_img.shape[:2]
    log(f"帧尺寸: {img_w}x{img_h}")

    # 根据模型类别自动判断球体 class ID
    if _using_basketball_model:
        BALL_CLASS = None
        for cls_id, name in model.names.items():
            if name.lower() in ("basketball", "ball", "sports ball"):
                BALL_CLASS = cls_id
                break
        if BALL_CLASS is None:
            log("警告: 篮球专用模型中未找到 basketball/ball 类别，回退到 class 0")
            BALL_CLASS = 0
        log(f"球体类别: class {BALL_CLASS} ({model.names.get(BALL_CLASS, '?')})")
    else:
        BALL_CLASS = 32  # COCO sports ball

    # ================================================================
    # 通道 1: YOLO 球体检测（阶段1使用粗扫帧，阶段2精扫候选区间）
    # ================================================================
    need_ball = ch_enabled("1a") or ch_enabled("1b")
    ball_yolo_conf = min(yolo_confidence, 0.25)  # 降低阈值
    ball_detections = []  # [(frame_idx, cx, cy, w, h, conf)]
    frame_path_to_idx = {path: i for i, path in enumerate(frame_files)}

    if need_ball:
        log("--- 通道 1: YOLO 球体检测 ---")

        # 两阶段时，先用粗扫帧；否则用全量帧
        detect_frames = rough_frames if two_stage else frame_files

        for frame_path in detect_frames:
            # 获取原始帧索引（保证时间戳计算正确）
            i = frame_path_to_idx.get(frame_path, 0)
            if i % 10 == 0:
                log(f"检测进度: {i+1}/{len(frame_files)}")

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
            except Exception as e:
                log(f"帧 {i} 检测异常: {e}")
                continue

        log(f"通道1: 共检测到 {len(ball_detections)} 次球体出现")
    else:
        log("--- 通道 1: YOLO 球体检测 (已跳过) ---")

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

    # ---- 球体检测覆盖率评估 ----
    ball_coverage = len(ball_detections) / max(1, len(frame_files))
    ch1a_healthy = ball_coverage >= 0.10  # 通道1a是否有效工作
    log(f"球体检测覆盖率: {ball_coverage:.1%} ({len(ball_detections)}/{len(frame_files)}) "
        f"→ 通道1a {'正常' if ch1a_healthy else '失效（弱通道模式）'}")

    # 所有通道的候选事件收集
    all_candidate_events = []

    # ---- 通道 1a: 追踪 + 几何判定 ----
    if ball_detections and ch_enabled("1a"):
        # v2.6: 根据配置选择追踪器（Kalman 或 IOU）
        use_kalman = profile.detection_overrides.get("tracker", "iou") == "kalman"
        tracker_name = "Kalman" if use_kalman else "IOU"
        log(f"--- 通道 1a: {tracker_name} 追踪 + 几何判定 ---")

        from tracker import IOUTracker, KalmanTracker, Detection as TrackerDetection

        max_lost = max(3, int(sample_fps * 1.5))
        dist_threshold = max(50, min(img_w, img_h) * 0.15)
        if use_kalman:
            tracker = KalmanTracker(
                iou_threshold=0.15,
                max_lost=max_lost,
                distance_threshold=dist_threshold,
                sample_fps=sample_fps,
                gravity_scale=profile.detection_overrides.get("gravity_scale", 2.0),
            )
        else:
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
        if ch_enabled("1b"):
            log("--- 通道 1b: 启发式检测 ---")
            heuristic_events = fallback_heuristic_v2(
                frame_balls, img_w, img_h, total_frames,
                sample_fps, video_duration, hoop_y_min, hoop_y_max,
                confidence_threshold, hoop_cx=hoop_cx, hoop_cy=hoop_cy
            )
            all_candidate_events.extend(heuristic_events)
            log(f"通道1b 启发式: {len(heuristic_events)} 个候选")

    # ================================================================
    # 两阶段采样阶段2: 若粗扫已找到候选，对候选区间进行精扫补充 ball 检测
    # ================================================================
    if two_stage and all_candidate_events and len(rough_frames) < len(frame_files):
        log("--- 两阶段采样: 阶段2 精扫候选区间 ---")
        fine_for_ball = two_stage_filter_frames(
            frame_files, sample_fps, all_candidate_events, window_s=4.0
        )
        # 只扫描粗扫没覆盖的帧
        rough_set = set(rough_frames)
        extra_frames = [f for f in fine_for_ball if f not in rough_set]
        if extra_frames:
            log(f"  精扫额外帧: {len(extra_frames)} 帧")
            for frame_path in extra_frames:
                i = frame_path_to_idx.get(frame_path, 0)
                try:
                    results = model(frame_path, conf=ball_yolo_conf, imgsz=640, verbose=False)
                    for r in results:
                        if r.boxes is None:
                            continue
                        for box in r.boxes:
                            cls = int(box.cls[0])
                            if cls == BALL_CLASS:
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                cx_b = (x1 + x2) / 2
                                cy_b = (y1 + y2) / 2
                                w_b = x2 - x1
                                h_b = y2 - y1
                                conf_b = float(box.conf[0])
                                ball_detections.append((i, cx_b, cy_b, w_b, h_b, conf_b))
                except Exception:
                    pass
            # 用新的 ball_detections 重新追踪
            if extra_frames:
                log("  重新运行追踪 + 几何判定...")
                from tracker import IOUTracker, Detection as TrackerDetection
                frame_dets2 = {}
                frame_balls2 = {}
                for (fidx, cx, cy, w, h, conf) in ball_detections:
                    if fidx not in frame_dets2:
                        frame_dets2[fidx] = []
                        frame_balls2[fidx] = []
                    frame_dets2[fidx].append(TrackerDetection(fidx, cx, cy, w, h, conf))
                    frame_balls2[fidx].append((cx, cy, w, h, conf))
                tracker2 = IOUTracker(
                    iou_threshold=0.15,
                    max_lost=max(3, int(sample_fps * 1.5)),
                    distance_threshold=max(50, min(img_w, img_h) * 0.15),
                )
                for fidx in range(len(frame_files)):
                    tracker2.update(frame_dets2.get(fidx, []))
                for track in tracker2.get_all_tracks():
                    pts = [(p.frame_idx, p.cx, p.cy, p.conf) for p in track.points]
                    event = judge_goal_by_trajectory(
                        pts, hoop_cx, hoop_cy, hoop_w, hoop_h, sample_fps)
                    if event:
                        all_candidate_events.append(event)

                # 精扫后用完整 frame_balls 重新运行启发式（O-24）
                # 粗扫跳帧可能导致方法2（球快速下落）遗漏——
                # 例如帧200有球、帧201有球但被粗扫跳过，方法2的连续帧条件不满足
                if ch_enabled("1b") and frame_balls2:
                    log("  精扫后重新运行启发式检测...")
                    heuristic_events2 = fallback_heuristic_v2(
                        frame_balls2, img_w, img_h, len(frame_files),
                        sample_fps, video_duration, hoop_y_min, hoop_y_max,
                        confidence_threshold, hoop_cx=hoop_cx, hoop_cy=hoop_cy
                    )
                    # 与粗扫启发式去重：±1s 内不重复添加
                    existing_times = {e["timestamp"] for e in all_candidate_events
                                      if "启发式" in e.get("detail", "")}
                    new_heuristic = []
                    for he in heuristic_events2:
                        is_dup = any(abs(he["timestamp"] - et) < 1.0
                                     for et in existing_times)
                        if not is_dup:
                            new_heuristic.append(he)
                    if new_heuristic:
                        all_candidate_events.extend(new_heuristic)
                        log(f"  精扫启发式新增: {len(new_heuristic)} 个候选")

    # ================================================================
    # 通道 2: 人体运动模式检测（不依赖球体）
    # ================================================================
    if ch_enabled("2"):
        log("--- 通道 2: 人体运动模式检测 ---")
        # 通道2 始终使用全量帧：人体运动检测依赖连续帧的位移变化，
        # 降帧会破坏运动连续性，导致上升帧计数不足从而漏检
        person_events = detect_goals_by_person_motion(
            frame_files,
            model, img_w, img_h, sample_fps, yolo_confidence
        )
        all_candidate_events.extend(person_events)
    else:
        log("--- 通道 2: 人体运动模式检测 (已跳过) ---")

    # ================================================================
    # 通道 3: 局部运动突变检测（不依赖 YOLO）
    # ================================================================
    if ch_enabled("3"):
        log("--- 通道 3: 局部运动突变检测 ---")

        # 两阶段采样：先用粗扫帧（每隔1帧）找候选，再精扫候选区间
        fine_frame_files = frame_files
        if two_stage and all_candidate_events:
            log("--- 两阶段采样: 基于已有候选区间精扫 ---")
            fine_frame_files = two_stage_filter_frames(
                frame_files, sample_fps, all_candidate_events, window_s=4.0
            )

        motion_events = detect_goals_by_motion(
            fine_frame_files, img_w, img_h, sample_fps, video_duration
        )
        all_candidate_events.extend(motion_events)
    else:
        log("--- 通道 3: 局部运动突变检测 (已跳过) ---")
        fine_frame_files = frame_files

    # ================================================================
    # 通道 4: 篮网有向光流穿越检测（T-1.9.13 算法C，替代旧版平均幅度）
    # ================================================================
    if enable_net_deform and ch_enabled("4"):
        log("--- 通道 4: 篮网有向光流穿越检测 (算法C) ---")
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from net_deform import detect_downward_flow_through_net

            # T-1.9.14: 篮网ROI高度系数从默认0.8优化（保持紧凑，减少噪声）
            net_events = detect_downward_flow_through_net(
                fine_frame_files, hoop_cx, hoop_cy, hoop_w,
                sample_fps=sample_fps,
                net_height_ratio=0.8,   # 紧凑ROI，减少无关区域噪声
                min_burst_ratio=2.0,    # 下行光流突发阈值
                min_purity=1.5,         # 方向纯净度：下行需是横向的1.5倍
                cooldown_s=3.0,
            )
            all_candidate_events.extend(net_events)
            log(f"通道4 有向光流穿越: {len(net_events)} 个候选")
        except Exception as e:
            log(f"通道4 有向光流穿越检测失败（跳过）: {e}")
            # 降级到旧版平均幅度检测
            try:
                from net_deform import detect_net_deformation
                log("通道4 降级: 回退到旧版平均幅度检测")
                net_events = detect_net_deformation(
                    fine_frame_files, hoop_cx, hoop_cy, hoop_w,
                    sample_fps=sample_fps,
                    cooldown_s=3.0,
                )
                all_candidate_events.extend(net_events)
                log(f"通道4 旧版形变: {len(net_events)} 个候选")
            except Exception as e2:
                log(f"通道4 旧版也失败（跳过）: {e2}")
    else:
        log("--- 通道 4: 篮网有向光流穿越检测 (已禁用，传 --enable-net-deform 启用) ---")

    # ================================================================
    # 通道 5: 音频事件检测（入网声 + 哨声）
    # ================================================================
    if audio_file and os.path.exists(audio_file) and ch_enabled("5"):
        log(f"--- 通道 5: 音频事件检测 ({audio_file}) ---")
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from audio_detect import detect_audio_events
            audio_events = detect_audio_events(audio_file, video_duration)
            all_candidate_events.extend(audio_events)
            log(f"通道5 音频检测: {len(audio_events)} 个候选")
        except Exception as e:
            log(f"通道5 音频检测失败（跳过）: {e}")
    else:
        log("--- 通道 5: 音频检测 (未提供音频文件，传 --audio-file 启用) ---")

    # ================================================================
    # 通道 6: ROI 多帧分类器（需要预训练 ONNX 模型）
    # ================================================================
    if ch_enabled("6"):
        log("--- 通道 6: ROI 多帧分类器 ---")
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from roi_classifier import detect_goals_by_roi_classifier
            roi_events = detect_goals_by_roi_classifier(
                fine_frame_files if two_stage else frame_files,
                hoop_cx, hoop_cy, hoop_w,
                sample_fps,
                confidence_cap=profile.get_cap("6"),
            )
            all_candidate_events.extend(roi_events)
            log(f"通道6 ROI分类器: {len(roi_events)} 个候选")
        except Exception as e:
            log(f"通道6 ROI分类器失败（跳过）: {e}")
    else:
        log("--- 通道 6: ROI 多帧分类器 (已禁用) ---")

    # ================================================================
    # 通道 7: 庆祝动作检测（对已有候选做确认，不独立产生事件）
    # ================================================================
    if ch_enabled("7") and all_candidate_events and ball_detections:
        log("--- 通道 7: 庆祝动作检测 ---")
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from celebration_detect import evaluate_celebration

            # 整理 YOLO 人体检测结果（按帧索引分组）
            person_by_frame = {}
            for det in ball_detections:
                # ball_detections 只有球，需要从 YOLO 结果中获取人体
                pass

            # 从 YOLO 模型重新获取人体检测（复用已加载的模型）
            if model is not None:
                _frames_for_person = frame_files[:min(len(frame_files), 300)]
                for fidx, fpath in enumerate(_frames_for_person):
                    try:
                        results = model(fpath, verbose=False, conf=0.3, imgsz=640)
                        for r in results:
                            for box in r.boxes:
                                cls_id = int(box.cls[0])
                                if cls_id == 0:  # COCO person class
                                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                                    cx_p = (x1 + x2) / 2
                                    cy_p = (y1 + y2) / 2
                                    w_p = x2 - x1
                                    h_p = y2 - y1
                                    if fidx not in person_by_frame:
                                        person_by_frame[fidx] = []
                                    person_by_frame[fidx].append((cx_p, cy_p, w_p, h_p))
                    except Exception:
                        pass

            if person_by_frame:
                celeb_events = evaluate_celebration(
                    person_by_frame, all_candidate_events, sample_fps
                )
                all_candidate_events.extend(celeb_events)
                log(f"通道7 庆祝动作: {len(celeb_events)} 个确认事件")
            else:
                log("通道7 庆祝动作: 无人体检测数据，跳过")
        except Exception as e:
            log(f"通道7 庆祝动作检测失败（跳过）: {e}")
    else:
        log("--- 通道 7: 庆祝动作检测 (已禁用或无候选事件) ---")

    # ================================================================
    # 汇总所有通道结果 + 跨通道验证
    # ================================================================
    log(f"所有通道汇总: {len(all_candidate_events)} 个候选事件")

    vlm_mode = profile.vlm.get("mode", "fallback")
    if not all_candidate_events and vlm_mode != "only":
        log("所有通道均未检测到进球事件")
        save_results([], output_path)
        return

    # 跨通道验证：仅当 ±2s 窗口内有 >=1 个【不同类型通道】命中时才提升置信度
    # 单通道事件保持原始置信度，避免弱信号通道（如通道3）被误提升
    def _channel_type(detail: str) -> str:
        """提取通道类型（去掉括号中的 left/right 等方向信息）"""
        src = detail.split(":")[0].strip()
        # "人体运动(left)" → "人体运动", "运动突变(left_hoop)" → "运动突变"
        paren_idx = src.find("(")
        if paren_idx > 0:
            src = src[:paren_idx]
        return src

    # 弱通道置信度上限：从 AlgorithmProfile 动态生成
    # v2.6: 通过 profile.build_weak_channel_caps() 从配置中读取
    # 向后兼容：未传 profile 时使用默认值（与 v2.5 一致）
    WEAK_CHANNEL_CAPS = profile.build_weak_channel_caps()
    log(f"  通道置信度上限: {WEAK_CHANNEL_CAPS}")

    cv_cfg = profile.cross_validation
    validation_window = cv_cfg.get("window", 2.0)
    boost_per_ch = cv_cfg.get("boost_per_channel", 0.08)
    max_boost = cv_cfg.get("max_boost", 0.15)

    for i, event in enumerate(all_candidate_events):
        t = event["timestamp"]
        this_type = _channel_type(event["detail"])
        # 收集时间窗口内不同通道来源的集合
        nearby_types = set()
        for j, other in enumerate(all_candidate_events):
            if i == j:
                continue
            if abs(other["timestamp"] - t) <= validation_window:
                other_type = _channel_type(other["detail"])
                if other_type != this_type:
                    nearby_types.add(other_type)
        # 仅当有不同类型通道互证时才 boost
        if len(nearby_types) >= 1:
            boost = min(max_boost, len(nearby_types) * boost_per_ch)
            old_conf = event["confidence"]
            cap = WEAK_CHANNEL_CAPS.get(this_type, 0.95)
            event["confidence"] = round(
                min(cap, event["confidence"] + boost), 3)
            log(f"  跨通道验证: {t:.1f}s conf {old_conf} → "
                f"{event['confidence']} (互证通道={nearby_types})")
        else:
            log(f"  单通道无互证: {t:.1f}s conf={event['confidence']:.3f} "
                f"来源={this_type}, 不做boost")

    # ================================================================
    # v2.9: 强制通道 cap — 确保弱通道事件的 conf 不超过上限
    # 跨通道互证只对有互证的事件做 cap，单通道事件可能绕过。
    # 此处对所有事件按 WEAK_CHANNEL_CAPS 做一次强制 cap。
    # 注意：因果链验证可以在 cap 后继续提升 conf（如强因果链 +0.12）
    # ================================================================
    for event in all_candidate_events:
        this_type = _channel_type(event["detail"])
        cap = WEAK_CHANNEL_CAPS.get(this_type, 0.95)
        if event["confidence"] > cap:
            old_conf = event["confidence"]
            event["confidence"] = round(cap, 3)
            log(f"  强制cap: {event['timestamp']:.1f}s conf {old_conf} → "
                f"{event['confidence']} (通道={this_type}, cap={cap})")

    # ================================================================
    # v2.7: 因果推理链验证（O-23 算法E）
    # 在跨通道互证之后、事件合并之前，检查每个候选事件的时序因果完整性。
    # 真进球：前有投篮动作/球体飞行，后有入网声/欢呼/庆祝 → boost
    # 孤立噪声：前后无因果信号 → 惩罚
    # ================================================================
    try:
        from causal_chain import apply_causal_validation
        log("--- 因果推理链验证 (O-23) ---")
        apply_causal_validation(
            all_candidate_events, video_duration, profile
        )
    except ImportError:
        log("因果推理链模块未找到（causal_chain.py），跳过")
    except Exception as e:
        log(f"因果推理链验证失败（跳过）: {e}")

    # ---- v2.6: 自适应融合策略（支持 VLM always / only 模式）----
    vlm_mode = profile.vlm.get("mode", "fallback")
    vlm_enabled = vlm_confirm and source_video
    use_vlm_only = vlm_enabled and vlm_mode == "only"
    use_vlm_always = vlm_enabled and vlm_mode == "always"
    use_vlm_fallback = vlm_enabled and vlm_mode == "fallback" and not ch1a_healthy

    if use_vlm_only:
        # 纯 VLM 模式：忽略所有通道结果，直接用 VLM 滑动窗口扫描全视频
        scan_window = profile.vlm.get("scan_window", 6.0)
        scan_step = profile.vlm.get("scan_step", 3.0)
        scan_frames = profile.vlm.get("scan_frames", 8)
        log(f"--- 纯 VLM 模式: 滑动窗口扫描全视频 "
            f"(窗口={scan_window}s, 步长={scan_step}s, 帧数={scan_frames}) ---")
        events = vlm_scan_video(
            source_video, video_duration,
            window_seconds=scan_window,
            step_seconds=scan_step,
            num_frames=scan_frames,
        )
        log(f"VLM 扫描完成: {len(events)} 个进球事件")

    elif use_vlm_always:
        # VLM 常态过滤模式：对指定置信度范围内的候选事件做 VLM 确认
        vlm_range = profile.vlm.get("confirm_range", [0.35, 0.70])
        log(f"--- VLM 常态过滤模式: 确认置信度 [{vlm_range[0]}, {vlm_range[1]}) 的候选 ---")
        merged = merge_events_with_cooldown(
            all_candidate_events, min(confidence_threshold, vlm_range[0]),
            merge_window=3.0, cooldown=2.0
        )
        log(f"合并后 {len(merged)} 个候选事件（含低置信度）")

        # 高置信度直接保留，中间段走 VLM 确认
        final_events = []
        vlm_candidates = []
        for e in merged:
            if e["confidence"] >= vlm_range[1]:
                final_events.append(e)
            elif e["confidence"] >= vlm_range[0]:
                vlm_candidates.append(e)

        if vlm_candidates:
            log(f"VLM 确认 {len(vlm_candidates)} 个候选事件...")
            confirmed = vlm_confirm_events(vlm_candidates, source_video)
            final_events.extend(confirmed)

        final_events.sort(key=lambda x: x["timestamp"])
        events = [e for e in final_events
                  if e["confidence"] >= confidence_threshold]

    elif use_vlm_fallback:
        # 通道1a失效 + VLM回退模式
        log("--- 自适应融合: 通道1a失效，启用VLM混合策略 ---")
        merged = merge_events_with_cooldown(
            all_candidate_events, confidence_threshold,
            merge_window=3.0, cooldown=2.0
        )
        log(f"合并后 {len(merged)} 个候选事件")

        log("步骤1: VLM逐事件确认...")
        confirmed = vlm_confirm_events(merged, source_video)

        log("步骤2: VLM独立扫描补充漏检...")
        scan_goals = vlm_scan_video(source_video, video_duration)

        events = vlm_merge_confirmed_and_scanned(
            confirmed, scan_goals, match_tolerance=5.0
        )
    elif not ch1a_healthy:
        log("--- 自适应融合: 通道1a失效，VLM未启用，标准合并（降级模式） ---")
        events = merge_events_with_cooldown(
            all_candidate_events, confidence_threshold,
            merge_window=3.0, cooldown=2.0
        )
    else:
        events = merge_events_with_cooldown(
            all_candidate_events, confidence_threshold,
            merge_window=3.0, cooldown=2.0
        )

    log(f"最终结果: {len(events)} 个进球事件")
    for e in events:
        goal_type = e.get('goal_type', '')
        type_tag = f" [{goal_type}]" if goal_type else ""
        log(f"  {e['timestamp']}s conf={e['confidence']}{type_tag} {e['detail']}")
    save_results(events, output_path)


# ============================================================
# VLM 配置读取（v2.6）
# ============================================================

# 模块级变量：存储从配置文件传入的 VLM 凭证
_vlm_config = {}


def _resolve_vlm_config() -> Tuple[str, str, str]:
    """读取 VLM 凭证，配置文件优先，环境变量回退。

    配置来源（按优先级）：
    1. AlgorithmProfile 中的 vlm 段（方案级覆盖）
    2. configs/config.yaml 顶层 vlm 段（全局共享）
    3. 环境变量 VLM_PROVIDER / VLM_API_KEY / VLM_MODEL（回退）

    Returns:
        (provider, api_key, model)
    """
    provider = (_vlm_config.get("provider") or
                os.environ.get("VLM_PROVIDER") or "dashscope").lower()
    api_key = (_vlm_config.get("api_key") or
               os.environ.get("VLM_API_KEY") or "")
    model = (_vlm_config.get("model") or
             os.environ.get("VLM_MODEL") or "")

    if not model:
        defaults = {
            "openai": "gpt-4o",
            "gemini": "gemini-1.5-pro",
            "ollama": "llava:13b",
            "dashscope": "qwen-vl-max",
        }
        model = defaults.get(provider, "gpt-4o")

    return provider, api_key, model


# ============================================================
# VLM 逐事件确认（v2.4）
# ============================================================

def vlm_confirm_events(candidate_events: List[Dict],
                       source_video: str,
                       window_before: float = 3.0,
                       window_after: float = 3.0,
                       num_frames: int = 8,
                       vlm_conf_threshold: float = 0.5) -> List[Dict]:
    """
    使用VLM对每个候选事件进行独立确认。
    
    对每个事件取 [timestamp - window_before, timestamp + window_after] 区间，
    采样 num_frames 帧送给VLM判断是否为进球。
    只保留VLM确认为进球的事件。
    
    Returns:
        过滤后的事件列表
    """
    import subprocess
    import tempfile
    import base64
    import time

    provider, api_key, model = _resolve_vlm_config()

    if not api_key:
        log("  [warn] VLM API Key 未设置（配置文件和环境变量均为空），跳过VLM确认")
        return candidate_events

    log(f"  VLM确认: provider={provider}, model={model}, "
        f"events={len(candidate_events)}")

    # VLM确认prompt — 针对单个候选事件的精确判定
    confirm_prompt = """你是一位专业的篮球视频审查员。以下是从一段篮球比赛视频中按时间顺序截取的连续帧画面（约6秒）。

请仔细观察这组画面，判断其中是否发生了"篮球进球"事件。

**进球的定义**：篮球从篮筐上方穿过篮网落下。包括：
- 投篮命中（jumper / three-pointer）
- 上篮命中（layup）
- 扣篮命中（dunk）

**不算进球的情况**：
- 球碰到篮筐/篮板但弹出
- 球员运球经过篮筐附近
- 球停在篮筐边缘但未穿过
- 球员做出投篮动作但球未进筐
- 快攻跑动、防守动作等非得分场景

请以严格的 JSON 格式回答（不要添加 markdown 代码块标记）:
{
  "is_goal": true或false,
  "confidence": 0.0到1.0之间的浮点数（你对判断的确信程度，不确定时给低值）,
  "reasoning": "用1-2句话简述你观察到的关键画面证据"
}"""

    confirmed = []
    
    for idx, event in enumerate(candidate_events):
        ts = event["timestamp"]
        start = max(0, ts - window_before)
        end = ts + window_after
        
        log(f"  VLM确认 [{idx+1}/{len(candidate_events)}] "
            f"t={ts:.1f}s ({event['detail'][:40]})")
        
        # 采样帧
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_paths = _sample_frames_from_video(
                source_video, start, end, tmpdir, num_frames
            )
            
            if len(frame_paths) < 3:
                log(f"    帧采样不足({len(frame_paths)}帧)，保留事件")
                confirmed.append(event)
                continue
            
            # 调用VLM
            try:
                judgment = _call_vlm(
                    provider, model, api_key, frame_paths, confirm_prompt
                )
            except Exception as e:
                log(f"    VLM调用失败: {e}，保留事件")
                confirmed.append(event)
                continue
            
            is_goal = judgment.get("is_goal", False)
            vlm_conf = judgment.get("confidence", 0.0)
            reasoning = judgment.get("reasoning", "")
            
            if is_goal and vlm_conf >= vlm_conf_threshold:
                confirmed.append(event)
                log(f"    ✓ VLM确认进球 (conf={vlm_conf:.2f}): {reasoning[:60]}")
            else:
                log(f"    ✗ VLM否决 (is_goal={is_goal}, conf={vlm_conf:.2f}): "
                    f"{reasoning[:60]}")
        
        # Rate limiting
        if provider == "dashscope":
            time.sleep(1.0)
        elif provider == "gemini":
            time.sleep(5.0)
        else:
            time.sleep(0.5)
    
    log(f"  VLM确认结果: {len(confirmed)}/{len(candidate_events)} 个事件通过")
    return confirmed


def vlm_scan_video(source_video: str, video_duration: float,
                   window_seconds: float = 6.0, step_seconds: float = 3.0,
                   num_frames: int = 8,
                   goal_confidence_threshold: float = 0.5,
                   merge_window: float = 4.0) -> List[Dict]:
    """
    VLM独立扫描全视频，找出所有进球时间点。
    
    用滑动窗口逐段扫描，每个窗口采样帧让VLM判断是否有进球。
    """
    import tempfile
    import time
    
    provider, api_key, model = _resolve_vlm_config()

    if not api_key:
        log("  [warn] VLM API Key 未设置（配置文件和环境变量均为空），跳过VLM扫描")
        return []
    
    scan_prompt = (
        "你是一位专业的篮球视频审查员。以下是从一段篮球比赛视频中按时间顺序截取的"
        "连续帧画面（时间跨度约 {ws} 秒）。\n\n"
        "请仔细观察这组画面，判断其中是否发生了「篮球进球」事件。\n\n"
        "**进球的定义**：篮球从篮筐上方穿过篮网落下。包括：\n"
        "- 投篮命中（jumper / three-pointer）\n"
        "- 上篮命中（layup）\n"
        "- 扣篮命中（dunk）\n\n"
        "**不算进球的情况**：\n"
        "- 球碰到篮筐/篮板但弹出\n"
        "- 球员运球经过篮筐附近\n"
        "- 球停在篮筐边缘但未穿过\n"
        "- 球员做出投篮动作但球未进筐\n"
        "- 快攻跑动、防守动作等非得分场景\n\n"
        "注意：一个窗口内可能包含 0 个或 1 个进球，不会有多个。\n"
        "请以严格的 JSON 格式回答（不要添加 markdown 代码块标记）:\n"
        '{{\n'
        '  "is_goal": true或false,\n'
        '  "confidence": 0.0到1.0之间的浮点数,\n'
        '  "reasoning": "用1-2句话简述你观察到的关键画面证据",\n'
        '  "goal_frame_index": null或0到N的整数\n'
        '}}'
    ).format(ws=window_seconds)
    
    # 获取视频时长
    if video_duration <= 0:
        import subprocess
        try:
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
                   "-show_format", source_video]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, timeout=10)
            info = json.loads(proc.stdout.decode("utf-8"))
            video_duration = float(info.get("format", {}).get("duration", 0))
        except Exception:
            video_duration = 300.0
    
    log(f"  VLM扫描: 窗口={window_seconds}s 步长={step_seconds}s "
        f"视频={video_duration:.0f}s")
    
    # 滑动窗口扫描
    raw_goals = []
    t = 0.0
    total_windows = int((video_duration - window_seconds) / step_seconds) + 1
    win_idx = 0
    
    while t + window_seconds <= video_duration + 0.1:
        win_idx += 1
        win_start = t
        win_end = min(t + window_seconds, video_duration)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            frames = _sample_frames_from_video(
                source_video, win_start, win_end, tmpdir, num_frames
            )
            
            if len(frames) < 3:
                t += step_seconds
                continue
            
            try:
                result = _call_vlm(provider, model, api_key, frames, scan_prompt)
            except Exception as e:
                log(f"    窗口 {win_start:.0f}-{win_end:.0f}s VLM失败: {e}")
                t += step_seconds
                continue
            
            is_goal = result.get("is_goal", False)
            conf = result.get("confidence", 0.0)
            
            if is_goal and conf >= goal_confidence_threshold:
                # 计算进球时间戳
                gfi = result.get("goal_frame_index")
                if gfi is not None and isinstance(gfi, (int, float)):
                    goal_ts = win_start + (win_end - win_start) * gfi / max(1, num_frames - 1)
                else:
                    goal_ts = (win_start + win_end) / 2
                
                raw_goals.append({
                    "timestamp": round(goal_ts, 2),
                    "confidence": conf,
                    "detail": f"VLM扫描: {result.get('reasoning', '')[:80]}",
                    "window": [win_start, win_end]
                })
                log(f"    窗口 {win_start:.0f}-{win_end:.0f}s → 进球! "
                    f"t={goal_ts:.1f}s conf={conf:.2f}")
        
        # Rate limiting
        if provider == "dashscope":
            time.sleep(1.0)
        elif provider == "gemini":
            time.sleep(5.0)
        else:
            time.sleep(0.5)
        
        t += step_seconds
    
    log(f"  VLM扫描完成: {len(raw_goals)} 个原始进球")
    
    # 合并相邻窗口的重复检测
    if not raw_goals:
        return []
    
    raw_goals.sort(key=lambda g: g["timestamp"])
    merged = [raw_goals[0]]
    for g in raw_goals[1:]:
        if g["timestamp"] - merged[-1]["timestamp"] < merge_window:
            if g["confidence"] > merged[-1]["confidence"]:
                merged[-1] = g
        else:
            merged.append(g)
    
    log(f"  VLM扫描合并后: {len(merged)} 个进球")
    for g in merged:
        log(f"    {g['timestamp']:.1f}s conf={g['confidence']:.2f}")
    
    return merged


def vlm_merge_confirmed_and_scanned(confirmed_events: List[Dict],
                                     scanned_goals: List[Dict],
                                     match_tolerance: float = 5.0) -> List[Dict]:
    """
    合并VLM确认的算法事件与VLM扫描发现的进球。
    
    策略：VLM扫描发现的进球中，如果没有被确认事件覆盖的，
    作为补充漏检加入最终结果。
    """
    # 找出确认事件已覆盖的VLM扫描进球
    covered_scan = set()
    for ce in confirmed_events:
        for si, sg in enumerate(scanned_goals):
            if abs(ce["timestamp"] - sg["timestamp"]) <= match_tolerance:
                covered_scan.add(si)
    
    # 未覆盖的扫描进球作为补充
    supplement = []
    for si, sg in enumerate(scanned_goals):
        if si not in covered_scan:
            supplement.append(sg)
    
    # 合并
    result = list(confirmed_events) + supplement
    result.sort(key=lambda e: e["timestamp"])
    
    log(f"  混合合并: {len(confirmed_events)} 确认 + {len(supplement)} 补充 "
        f"= {len(result)} 最终事件")
    
    return result


def _sample_frames_from_video(video_path: str, start: float, end: float,
                               output_dir: str, num_frames: int = 8,
                               max_dimension: int = 768) -> List[str]:
    """从视频指定时间区间采样帧"""
    import subprocess
    
    duration = end - start
    if duration <= 0:
        return []
    
    margin = min(0.05, duration * 0.03)
    timestamps = []
    for i in range(num_frames):
        t = start + margin + (duration - 2 * margin) * i / max(1, num_frames - 1)
        timestamps.append(min(t, end - 0.01))
    
    frame_paths = []
    for i, ts in enumerate(timestamps):
        frame_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
        scale_filter = (f"scale=min({max_dimension}\\,iw):"
                        f"min({max_dimension}\\,ih):"
                        f"force_original_aspect_ratio=decrease")
        cmd = [
            "ffmpeg",
            "-ss", f"{ts:.3f}",
            "-i", video_path,
            "-vframes", "1",
            "-vf", scale_filter,
            "-q:v", "2",
            "-y", "-update", "1",
            frame_path
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           check=True, timeout=10)
            if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                frame_paths.append(frame_path)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    
    return frame_paths


def _call_vlm(provider: str, model: str, api_key: str,
              frame_paths: List[str], prompt: str) -> Dict:
    """调用VLM判断进球，返回解析后的dict"""
    import urllib.request
    import urllib.error
    import base64
    
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    if provider == "dashscope":
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        content = [{"text": prompt}]
        for fp in frame_paths:
            b64 = encode_image(fp)
            content.append({"image": f"data:image/jpeg;base64,{b64}"})
        
        payload = {
            "model": model,
            "input": {
                "messages": [{"role": "user", "content": content}]
            },
            "parameters": {"temperature": 0.1, "max_tokens": 300}
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
        text = resp_data["output"]["choices"][0]["message"]["content"][0]["text"].strip()
    
    elif provider == "openai":
        base_url = os.environ.get("VLM_BASE_URL", "https://api.openai.com/v1")
        url = f"{base_url}/chat/completions"
        content = [{"type": "text", "text": prompt}]
        for fp in frame_paths:
            b64 = encode_image(fp)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
            })
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 300, "temperature": 0.1
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
        text = resp_data["choices"][0]["message"]["content"].strip()
    
    elif provider == "gemini":
        base_url = os.environ.get("VLM_BASE_URL",
                                   "https://generativelanguage.googleapis.com/v1beta")
        url = f"{base_url}/models/{model}:generateContent?key={api_key}"
        parts = [{"text": prompt}]
        for fp in frame_paths:
            b64 = encode_image(fp)
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 300}
        }
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
        text = resp_data["candidates"][0]["content"]["parts"][0]["text"].strip()
    
    else:
        raise ValueError(f"不支持的VLM provider: {provider}")
    
    # 解析JSON响应
    cleaned = text
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1]
    if "```" in cleaned:
        cleaned = cleaned.split("```", 1)[0]
    cleaned = cleaned.strip()
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 尝试容错解析
        is_goal = any(kw in text.lower() for kw in
                      ["is_goal\": true", "\"is_goal\":true", "is_goal\": true"])
        return {"is_goal": is_goal, "confidence": 0.5,
                "reasoning": f"[JSON解析异常] {text[:200]}"}


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
                          hoop_y_min, hoop_y_max, conf_threshold,
                          hoop_cx=None, hoop_cy=None):
    """
    改进版启发式检测：使用动态检测的篮筐 Y 范围替代硬编码值。

    v2.9 改进（Round 5）：
    - 方法1 收紧：球在篮筐区域消失的触发条件从 ≥1 帧提高到 ≥2 帧，
      过滤仅单帧出现的假阳性
    - 方法3 增加 dx_min 约束：球与篮筐 X 距离必须 > img_w * 0.05，
      排除篮筐正上方极近处的篮筐/篮网误检

    v2.8 改进：
    - 方法1&2 增加 X 坐标约束：球的 X 坐标必须接近篮筐 X，
      避免远离篮筐的球运动误触发（野球场多人场景）
    - 新增方法3：球在篮筐上方出现后消失（投篮入筐模式），
      解决上篮/近距离投篮时 YOLO 无法追踪球穿筐的问题
    """
    log("使用改进版启发式检测")

    if not frame_balls:
        return []

    # X 坐标约束：球的 X 必须在篮筐 X ± margin 范围内
    # 默认 margin = 画面宽度的 20%（约 384px for 1920）
    x_margin = img_w * 0.20
    # 方法3 的参数
    y_above_min = (hoop_cy - img_h * 0.25) if hoop_cy else 0
    y_above_max = (hoop_cy - img_h * 0.04) if hoop_cy else 0  # v2.9: 0.02→0.04，排除太靠近篮筐Y的误检

    candidate_events = []
    sorted_frames = sorted(frame_balls.keys())

    # 方法 1：球在篮筐区域连续出现后消失
    last_ball_frame = -999
    last_ball_y = 0
    last_ball_x = 0
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
                    # X 约束：球必须接近篮筐
                    dx = abs(cx - hoop_cx) if hoop_cx else 0
                    if hoop_cx is None or dx < x_margin:
                        timestamp = fidx / sample_fps
                        confidence = min(
                            0.60,
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
            last_ball_x = cx
        else:
            if last_ball_frame >= 0 and fidx - last_ball_frame == 1:
                if (hoop_y_min <= last_ball_y <= hoop_y_max and
                        ball_in_hoop_zone_count >= 2):
                    # X 约束
                    dx = abs(last_ball_x - hoop_cx) if hoop_cx else 0
                    if hoop_cx is None or dx < x_margin:
                        timestamp = last_ball_frame / sample_fps
                        confidence = min(0.60, 0.5 + 0.1 * min(ball_in_hoop_zone_count, 3))
                        candidate_events.append({
                            "frame_index": last_ball_frame,
                            "timestamp": round(timestamp, 2),
                            "confidence": round(confidence, 3),
                            "detail": (f"启发式: 球在篮筐区域后消失, "
                                       f"y={last_ball_y:.0f}, "
                                       f"连续帧数={ball_in_hoop_zone_count}")
                        })
            ball_in_hoop_zone_count = 0

    # 方法 2：球快速下落穿过篮筐区域（增加 X 约束）
    prev_cy = None
    prev_cx = None
    prev_fidx = -999
    for fidx in sorted_frames:
        balls = frame_balls[fidx]
        best_ball = max(balls, key=lambda b: b[4])
        cx, cy, w, h, conf = best_ball

        if prev_cy is not None and fidx - prev_fidx <= 2:
            dy = cy - prev_cy
            if dy > img_h * 0.08 and hoop_y_min <= prev_cy <= hoop_y_max:
                # X 约束：prev_cx 必须接近篮筐
                dx = abs(prev_cx - hoop_cx) if hoop_cx else 0
                if hoop_cx is None or dx < x_margin:
                    timestamp = fidx / sample_fps
                    speed_factor = min(1.0, abs(dy) / (img_h * 0.15))
                    confidence = min(0.60, 0.4 + 0.2 * speed_factor)
                    candidate_events.append({
                        "frame_index": fidx,
                        "timestamp": round(timestamp, 2),
                        "confidence": round(confidence, 3),
                        "detail": (f"启发式: 球快速下落穿过篮筐区域, "
                                   f"dy={dy:.0f}, prev_y={prev_cy:.0f}")
                    })

        prev_cy = cy
        prev_cx = cx
        prev_fidx = fidx

    # 方法 3：球在篮筐上方出现后消失（投篮入筐模式）
    # 适用于上篮/近距离投篮：球在篮筐正上方出现（conf >= 0.25），
    # 之后1~2帧在篮筐附近无球检测 → 球穿入篮筐消失
    # v2.9: 增加 dx_min 约束，排除篮筐正上方极近处的篮筐/篮网误检
    dx_min = img_w * 0.05  # 球与篮筐 X 距离最小值（约 96px for 1920w）
    if hoop_cx is not None and hoop_cy is not None:
        for fidx in sorted_frames:
            balls = frame_balls[fidx]
            for cx, cy, w, h, conf in balls:
                # 条件1：Y 在篮筐上方合理范围（不会太远）
                if not (y_above_min <= cy <= y_above_max):
                    continue
                # 条件2：X 接近篮筐（但不能太近，太近更可能是篮筐/篮网误检）
                dx = abs(cx - hoop_cx)
                if dx > x_margin or dx < dx_min:
                    continue
                # 条件3：后续1~2帧在篮筐附近无球
                disappeared = True
                for gap in range(1, 3):
                    check_fidx = fidx + gap
                    if check_fidx in frame_balls:
                        for bcx, bcy, bw, bh, bconf in frame_balls[check_fidx]:
                            bdx = abs(bcx - hoop_cx)
                            bdy = abs(bcy - hoop_cy)
                            if bdx < x_margin and bdy < img_h * 0.25:
                                disappeared = False
                                break
                    if not disappeared:
                        break

                if disappeared:
                    timestamp = fidx / sample_fps
                    # conf 基于球的检测置信度和距篮筐的接近程度
                    proximity_factor = max(0, 1.0 - dx / x_margin)
                    confidence = min(0.60, 0.35 + 0.15 * proximity_factor + 0.10 * min(conf, 0.8))
                    candidate_events.append({
                        "frame_index": fidx,
                        "timestamp": round(timestamp, 2),
                        "confidence": round(confidence, 3),
                        "detail": (f"启发式: 球在篮筐上方消失, "
                                   f"x={cx:.0f}, y={cy:.0f}, "
                                   f"dx={dx:.0f}")
                    })

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
    parser = argparse.ArgumentParser(description="GoalCut AI 进球检测引擎 v2.7")
    parser.add_argument("--frames-dir", required=True, help="帧图片目录")
    parser.add_argument("--output", required=True, help="检测结果输出路径(JSON)")
    parser.add_argument("--sample-fps", type=float, default=3, help="采样帧率")
    parser.add_argument("--confidence-threshold", type=float, default=0.55,
                        help="置信度阈值")
    parser.add_argument("--yolo-confidence", type=float, default=0.25,
                        help="YOLO检测置信度")
    parser.add_argument("--video-duration", type=float, default=0,
                        help="视频时长(秒)")
    # v2.2 新增参数
    parser.add_argument("--audio-file", default="",
                        help="提取的音频 WAV 文件路径（启用音频检测通道）")
    parser.add_argument("--enable-net-deform", action="store_true",
                        help="启用篮网形变检测（光流法，需要 OpenCV）")
    parser.add_argument("--two-stage", action="store_true",
                        help="启用两阶段采样（粗扫候选区间后精扫，提升时间精度）")
    # v2.3 通道选择（向后兼容，优先级低于 --algorithm-profile）
    parser.add_argument("--channel", default="",
                        help="只运行指定通道（逗号分隔），如: 1a,2,5。空=全部。"
                             "优先级低于 --algorithm-profile")
    # v2.4 VLM确认
    parser.add_argument("--source-video", default="",
                        help="源视频路径（VLM确认时从中采样帧）")
    parser.add_argument("--vlm-confirm", action="store_true",
                        help="启用VLM逐事件确认（需设置VLM_API_KEY）")
    # v2.6 算法配置
    parser.add_argument("--algorithm-profile", default="",
                        help="算法配置方案名称，从 configs/config.yaml 的 "
                             "algorithm_profiles 中读取。"
                             "可选: default, pickup_with_net, pickup_no_net, "
                             "minimal, vlm_primary")
    parser.add_argument("--algorithm-config", default="",
                        help="独立算法配置 YAML 文件路径（优先级高于 --algorithm-profile）")
    parser.add_argument("--list-profiles", action="store_true",
                        help="列出所有可用的算法配置方案并退出")

    args = parser.parse_args()

    # 列出可用方案
    if args.list_profiles:
        _list_available_profiles()
        return

    # 加载算法配置（优先级: --algorithm-config > --algorithm-profile > --channel > 默认）
    profile = None
    if args.algorithm_config:
        profile = AlgorithmProfile.load(config_path=args.algorithm_config)
        log(f"从文件加载算法配置: {args.algorithm_config}")
    elif args.algorithm_profile:
        profile = AlgorithmProfile.load(profile_name=args.algorithm_profile)
        log(f"使用预定义算法配置: {args.algorithm_profile}")
    elif args.channel:
        profile = AlgorithmProfile.from_channel_filter(args.channel)
        log(f"从 --channel 参数创建配置: {args.channel}")

    # 向后兼容：未使用 profile 时走原逻辑
    enabled_channels = None
    if profile is None and args.channel:
        enabled_channels = set(c.strip() for c in args.channel.split(","))
        log(f"仅运行通道: {enabled_channels}")

    detect_goals(
        frames_dir=args.frames_dir,
        output_path=args.output,
        sample_fps=args.sample_fps,
        confidence_threshold=args.confidence_threshold,
        yolo_confidence=args.yolo_confidence,
        video_duration=args.video_duration,
        audio_file=args.audio_file if args.audio_file else None,
        enable_net_deform=args.enable_net_deform,
        two_stage=args.two_stage,
        enabled_channels=enabled_channels,
        source_video=args.source_video if args.source_video else None,
        vlm_confirm=args.vlm_confirm,
        algorithm_profile=profile,
    )


def _list_available_profiles():
    """列出所有可用的算法配置方案"""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml"),
        "configs/config.yaml",
    ]
    for cfg_path in candidates:
        if os.path.exists(cfg_path):
            try:
                import yaml
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                profiles = data.get("algorithm_profiles", {})
                print("\n可用的算法配置方案:")
                print("=" * 60)
                for name, pdata in profiles.items():
                    desc = pdata.get("description", "")
                    channels = pdata.get("channels", {})
                    enabled = [str(ch) for ch, cfg in channels.items()
                               if isinstance(cfg, dict) and cfg.get("enabled", True)]
                    disabled = [str(ch) for ch, cfg in channels.items()
                                if isinstance(cfg, dict) and not cfg.get("enabled", True)]
                    vlm = pdata.get("vlm", {})
                    vlm_status = "启用" if vlm.get("enabled") else "禁用"

                    print(f"\n  {name}")
                    print(f"    描述: {desc}")
                    print(f"    启用通道: {', '.join(str(c) for c in sorted(enabled))}")
                    if disabled:
                        print(f"    禁用通道: {', '.join(str(c) for c in sorted(disabled))}")
                    print(f"    VLM: {vlm_status}")
                print("\n" + "=" * 60)
                print(f"使用方式: python detect.py --algorithm-profile <名称> ...")
                print()
            except ImportError:
                print("错误: 需要 pyyaml (pip install pyyaml)")
            except Exception as e:
                print(f"错误: {e}")
            return
    print("未找到 configs/config.yaml")


if __name__ == "__main__":
    main()
