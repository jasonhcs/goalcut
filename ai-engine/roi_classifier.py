"""
GoalCut ROI 多帧分类器（通道 6）

以篮筐为中心裁剪 2× 扩展 ROI，将连续 N 帧在通道维度堆叠，
输入轻量级分类网络（MobileNetV3-Small），直接输出进球概率。

半场野球场的最佳通道：
- 固定机位 + 单篮筐 → ROI 位置极稳定
- 不依赖篮网（学习球运动、光照变化等多维时序特征）
- 端到端学习，无需手工设计规则

使用方式：
1. 训练：python roi_classifier.py train --data-dir <数据目录>
2. 推理：在 detect.py 中作为通道 6 调用 classify_goal_roi()

模型文件：ai-engine/models/roi_classifier.onnx
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional


def crop_hoop_roi(frame: np.ndarray,
                  hoop_cx: int, hoop_cy: int, hoop_w: int,
                  expand_ratio: float = 2.0,
                  output_size: int = 128) -> np.ndarray:
    """以篮筐中心裁剪扩展 ROI 区域。

    Args:
        frame: 原始帧 (H, W, 3)
        hoop_cx, hoop_cy: 篮筐中心坐标
        hoop_w: 篮筐宽度
        expand_ratio: ROI 扩展倍率（2.0 = 篮筐宽度的 2 倍）
        output_size: 输出图像尺寸

    Returns:
        裁剪并缩放后的 ROI 图像 (output_size, output_size, 3)
    """
    h, w = frame.shape[:2]
    roi_half = int(hoop_w * expand_ratio / 2)

    x1 = max(0, hoop_cx - roi_half)
    y1 = max(0, hoop_cy - roi_half)
    x2 = min(w, hoop_cx + roi_half)
    y2 = min(h, hoop_cy + roi_half)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((output_size, output_size, 3), dtype=np.uint8)

    return cv2.resize(roi, (output_size, output_size))


def stack_roi_frames(frame_files: List[str],
                     start_idx: int,
                     n_frames: int,
                     hoop_cx: int, hoop_cy: int, hoop_w: int,
                     output_size: int = 128) -> Optional[np.ndarray]:
    """从连续帧中裁剪并堆叠 ROI。

    Returns:
        堆叠后的张量 (output_size, output_size, n_frames * 3)，
        如果帧不足则返回 None
    """
    if start_idx + n_frames > len(frame_files):
        return None

    rois = []
    for i in range(n_frames):
        frame = cv2.imread(frame_files[start_idx + i])
        if frame is None:
            return None
        roi = crop_hoop_roi(frame, hoop_cx, hoop_cy, hoop_w,
                            output_size=output_size)
        rois.append(roi)

    return np.concatenate(rois, axis=-1)  # (128, 128, n*3)


class ROIClassifier:
    """ROI 多帧分类器推理接口。

    加载 ONNX 模型，对篮筐 ROI 多帧堆叠做二分类推理。
    """

    def __init__(self, model_path: str, n_frames: int = 12):
        self.model_path = model_path
        self.n_frames = n_frames
        self.session = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[roi_classifier] 模型文件不存在: {self.model_path}", flush=True)
            return

        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )
            print(f"[roi_classifier] 模型已加载: {self.model_path}", flush=True)
        except ImportError:
            print("[roi_classifier] 需要 onnxruntime (pip install onnxruntime)", flush=True)
        except Exception as e:
            print(f"[roi_classifier] 模型加载失败: {e}", flush=True)

    @property
    def is_available(self) -> bool:
        return self.session is not None

    def predict(self, stacked_roi: np.ndarray) -> float:
        """对堆叠 ROI 推理，返回进球概率 [0, 1]。

        Args:
            stacked_roi: (128, 128, n_frames * 3) uint8 图像

        Returns:
            进球概率
        """
        if not self.is_available:
            return 0.0

        # 预处理：归一化到 [0, 1]，转换为 NCHW
        x = stacked_roi.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # (C, H, W)
        x = np.expand_dims(x, 0)         # (1, C, H, W)

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: x})

        prob = float(outputs[0][0])
        return max(0.0, min(1.0, prob))


def detect_goals_by_roi_classifier(
    frame_files: List[str],
    hoop_cx: int, hoop_cy: int, hoop_w: int,
    sample_fps: float,
    model_path: str = None,
    n_frames: int = 12,
    stride: int = 6,
    confidence_cap: float = 0.90,
    cooldown_s: float = 3.0,
) -> List[dict]:
    """通道 6：使用 ROI 多帧分类器检测进球。

    Args:
        frame_files: 帧文件路径列表
        hoop_cx, hoop_cy, hoop_w: 篮筐位置和宽度
        sample_fps: 采样帧率
        model_path: ONNX 模型路径（None 则使用默认路径）
        n_frames: 每次输入的帧数
        stride: 滑动窗口步长（帧）
        confidence_cap: 置信度上限
        cooldown_s: 两次检测最小间隔

    Returns:
        候选事件列表
    """
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(__file__), "models", "roi_classifier.onnx"
        )

    classifier = ROIClassifier(model_path, n_frames)
    if not classifier.is_available:
        print("[roi_classifier] 模型不可用，跳过通道 6", flush=True)
        return []

    print(f"[roi_classifier] 开始 ROI 分类检测: "
          f"{len(frame_files)} 帧, 窗口={n_frames}, 步长={stride}", flush=True)

    events = []
    cooldown_frames = int(cooldown_s * sample_fps)
    last_event_idx = -cooldown_frames - 1

    for start in range(0, len(frame_files) - n_frames + 1, stride):
        if start - last_event_idx < cooldown_frames:
            continue

        stacked = stack_roi_frames(
            frame_files, start, n_frames,
            hoop_cx, hoop_cy, hoop_w
        )
        if stacked is None:
            continue

        prob = classifier.predict(stacked)
        if prob >= 0.5:
            center_idx = start + n_frames // 2
            timestamp = center_idx / sample_fps
            confidence = min(confidence_cap, prob)

            events.append({
                "frame_index": center_idx,
                "timestamp": round(timestamp, 2),
                "confidence": round(confidence, 3),
                "detail": f"ROI分类器: prob={prob:.3f}",
            })
            last_event_idx = center_idx

    print(f"[roi_classifier] 检测到 {len(events)} 个候选事件", flush=True)
    return events
