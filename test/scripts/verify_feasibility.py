#!/usr/bin/env python3
"""
GoalCut 算法可行性验证脚本

按照"第一性原理-连续帧进球检测算法"文档第十四章的实验计划执行：
  实验1: 验证粗扫候选召回率 (A-4)
  实验2: 验证关键区追踪连续性 (A-2)
  实验3: 残差分布实测 (A-3)
  实验4: 验证消失-重现与反证法 (A-1 部分)
  实验5: 融合与证据不足策略验证 (A-5)

使用方式:
  cd goalcut
  python3 test/scripts/verify_feasibility.py \
    --video test/野球场素材/IMG_7225_1.mp4 \
    --gt test/ground_truth/IMG_7225_1_gt.json \
    --output test/output/feasibility_report.json
"""

import argparse
import json
import os
import sys
import time
import tempfile
import subprocess
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
AI_ENGINE_DIR = os.path.join(PROJECT_ROOT, "ai-engine")
sys.path.insert(0, AI_ENGINE_DIR)


def log(msg):
    print(f"[feasibility] {msg}", flush=True)


def probe_video(video_path: str) -> Dict:
    """获取视频元信息"""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    vs = [s for s in data["streams"] if s["codec_type"] == "video"][0]
    fmt = data["format"]
    fps_parts = vs.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
    return {
        "width": int(vs["width"]),
        "height": int(vs["height"]),
        "fps": fps,
        "duration": float(fmt["duration"]),
        "bitrate_kbps": int(fmt.get("bit_rate", 0)) // 1000,
        "codec": vs["codec_name"],
    }


def extract_frames(video_path: str, output_dir: str, fps: float,
                    start_s: float = 0, duration_s: float = -1) -> List[str]:
    """提取视频帧"""
    os.makedirs(output_dir, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", video_path]
    if start_s > 0:
        cmd = ["ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-i", video_path]
    if duration_s > 0:
        cmd.extend(["-t", f"{duration_s:.3f}"])
    cmd.extend([
        "-vf", f"fps={fps}",
        "-q:v", "2",
        os.path.join(output_dir, "frame_%06d.jpg"),
    ])
    subprocess.run(cmd, capture_output=True, check=True)
    return sorted(glob.glob(os.path.join(output_dir, "frame_*.jpg")))


def load_ground_truth(gt_path: str) -> List[Dict]:
    """加载 GT 标注"""
    with open(gt_path, "r") as f:
        data = json.load(f)
    return data.get("goals", [])


def match_events_to_gt(events: List[Dict], gt_goals: List[Dict],
                       tolerance_s: float = 3.0) -> Dict:
    """将检测事件与 GT 匹配，计算召回/精确率"""
    matched_gt = set()
    matched_events = set()
    true_positives = 0

    for gi, goal in enumerate(gt_goals):
        gt_t = goal["timestamp"]
        best_dist = float("inf")
        best_ei = -1
        for ei, ev in enumerate(events):
            if ei in matched_events:
                continue
            dist = abs(ev["timestamp"] - gt_t)
            if dist < best_dist and dist <= tolerance_s:
                best_dist = dist
                best_ei = ei
        if best_ei >= 0:
            true_positives += 1
            matched_gt.add(gi)
            matched_events.add(best_ei)

    precision = true_positives / max(len(events), 1)
    recall = true_positives / max(len(gt_goals), 1)
    f1 = (2 * precision * recall / max(precision + recall, 1e-9)
          if precision + recall > 0 else 0)

    missed = [gt_goals[i] for i in range(len(gt_goals)) if i not in matched_gt]
    false_alarms = [events[i] for i in range(len(events)) if i not in matched_events]

    return {
        "true_positives": true_positives,
        "false_positives": len(false_alarms),
        "false_negatives": len(missed),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "missed_goals": missed,
        "false_alarms": [{"timestamp": e["timestamp"],
                          "detail": e.get("detail", "")} for e in false_alarms],
    }


class FeasibilityVerifier:
    """可行性验证器"""

    def __init__(self, video_path: str, gt_path: str, output_dir: str):
        self.video_path = video_path
        self.gt_path = gt_path
        self.output_dir = output_dir
        self.video_info = None
        self.gt_goals = []
        self.model = None
        self.ball_class = 32
        self.using_basketball_model = False
        self.report = {
            "meta": {},
            "exp1_coarse_recall": {},
            "exp2_tracking_continuity": {},
            "exp3_residual_distribution": {},
            "exp4_disappear_reappear": {},
            "exp5_overall_judgment": {},
        }

    def setup(self):
        """初始化"""
        log("=" * 60)
        log("GoalCut 算法可行性验证")
        log("=" * 60)

        self.video_info = probe_video(self.video_path)
        log(f"视频: {os.path.basename(self.video_path)}")
        log(f"  分辨率: {self.video_info['width']}x{self.video_info['height']}")
        log(f"  帧率: {self.video_info['fps']:.2f} fps")
        log(f"  时长: {self.video_info['duration']:.2f}s")
        log(f"  码率: {self.video_info['bitrate_kbps']} kbps")

        if self.gt_path and os.path.exists(self.gt_path):
            self.gt_goals = load_ground_truth(self.gt_path)
            log(f"GT: {len(self.gt_goals)} 个进球")
            for g in self.gt_goals:
                log(f"  {g['timestamp']:.1f}s - {g.get('note', '')}")

        from kinematics import compute_adaptive_params
        self.adaptive_params = compute_adaptive_params(
            self.video_info["width"], self.video_info["height"],
            int(self.video_info["fps"]), self.video_info["bitrate_kbps"],
        )
        log(f"自适应参数:")
        for k, v in self.adaptive_params.items():
            log(f"  {k}: {v}")

        self._load_model()

        self.report["meta"] = {
            "experiment_name": "首轮算法可行性验证",
            "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "video": os.path.basename(self.video_path),
            "video_info": self.video_info,
            "adaptive_params": self.adaptive_params,
            "gt_goals": len(self.gt_goals),
            "code_version": "feasibility_v1",
        }

        os.makedirs(self.output_dir, exist_ok=True)

    def _load_model(self):
        """加载 YOLO 模型"""
        log("加载 YOLO 模型...")
        try:
            from ultralytics import YOLO
            candidates = [
                os.path.join(AI_ENGINE_DIR, "models", "basketball_v1",
                             "weights", "best.pt"),
            ]
            basketball_model = None
            for c in candidates:
                if os.path.exists(c):
                    basketball_model = c
                    break

            if basketball_model:
                self.model = YOLO(basketball_model)
                self.using_basketball_model = True
                for cls_id, name in self.model.names.items():
                    if name.lower() in ("basketball", "ball", "sports ball"):
                        self.ball_class = cls_id
                        break
                log(f"篮球专用模型: {basketball_model}")
                log(f"  球体类别: class {self.ball_class}")
            else:
                coco_path = os.path.join(AI_ENGINE_DIR, "yolov8n.pt")
                self.model = YOLO(coco_path if os.path.exists(coco_path) else "yolov8n.pt")
                self.ball_class = 32
                log("使用 COCO 预训练模型")
        except Exception as e:
            log(f"YOLO 加载失败: {e}")
            raise

    def _detect_balls_in_frames(self, frame_files: List[str],
                                 yolo_conf: float = 0.25,
                                 use_roi: bool = False) -> List[Tuple]:
        """在帧中检测球体。返回 [(frame_idx, cx, cy, w, h, conf), ...]

        当 use_roi=True 时，启用方案 A+B:
        - 方案 A: 全图用 imgsz=960, conf=0.15
        - 方案 B: 篮筐 ROI 裁剪放大后二次检测
        """
        import cv2
        detections = []
        hoop_rect = getattr(self, '_hoop_rect', None)

        base_imgsz = 960 if use_roi else 640
        base_conf = min(yolo_conf, 0.15) if use_roi else yolo_conf

        ball_d_expect = self.adaptive_params.get("ball_d_px", 60)
        min_ball_d = ball_d_expect * 0.2
        max_ball_d = ball_d_expect * 3.0

        for fi, fpath in enumerate(frame_files):
            if fi % 50 == 0:
                log(f"  YOLO 检测进度: {fi}/{len(frame_files)}")

            frame_dets = set()

            try:
                results = self.model(fpath, conf=base_conf,
                                     imgsz=base_imgsz, verbose=False)
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if cls == self.ball_class:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            w = x2 - x1
                            h = y2 - y1
                            d = max(w, h)
                            if d < min_ball_d or d > max_ball_d:
                                continue
                            conf = float(box.conf[0])
                            detections.append((fi, cx, cy, w, h, conf))
                            frame_dets.add((round(cx), round(cy)))
            except Exception:
                continue

            if use_roi and hoop_rect:
                try:
                    img = cv2.imread(fpath)
                    if img is None:
                        continue
                    ih, iw = img.shape[:2]
                    hcx = hoop_rect["cx"]
                    hcy = hoop_rect["cy"]
                    hw = hoop_rect["w"]

                    roi_half = int(hw * 2.5)
                    rx1 = max(0, int(hcx - roi_half))
                    ry1 = max(0, int(hcy - int(roi_half * 1.5)))
                    rx2 = min(iw, int(hcx + roi_half))
                    ry2 = min(ih, int(hcy + int(roi_half * 1.5)))
                    roi = img[ry1:ry2, rx1:rx2]
                    if roi.size == 0:
                        continue

                    roi_h, roi_w = roi.shape[:2]
                    roi_resized = cv2.resize(roi, (640, 640))
                    scale_x = roi_w / 640.0
                    scale_y = roi_h / 640.0

                    tmp_path = '/tmp/_roi_detect.jpg'
                    cv2.imwrite(tmp_path, roi_resized)

                    results = self.model(tmp_path, conf=0.10,
                                         imgsz=640, verbose=False)
                    for r in results:
                        if r.boxes is None:
                            continue
                        for box in r.boxes:
                            if int(box.cls[0]) != self.ball_class:
                                continue
                            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                            cx_roi = (bx1 + bx2) / 2 * scale_x + rx1
                            cy_roi = (by1 + by2) / 2 * scale_y + ry1
                            w_roi = (bx2 - bx1) * scale_x
                            h_roi = (by2 - by1) * scale_y
                            conf_roi = float(box.conf[0])

                            key = (round(cx_roi), round(cy_roi))
                            already = any(abs(key[0] - d[0]) < 15 and
                                         abs(key[1] - d[1]) < 15
                                         for d in frame_dets)
                            if not already:
                                detections.append(
                                    (fi, cx_roi, cy_roi, w_roi, h_roi,
                                     conf_roi * 0.9))
                                frame_dets.add(key)
                except Exception:
                    continue

        return detections

    def _detect_hoop(self, frame_files: List[str],
                     ball_detections: List[Tuple]) -> Dict:
        """检测篮筐位置，并修正 bbox 尺寸。

        YOLO rim 类别仅检测金属篮圈（~33px），但物理篮筐直径 45.72cm
        在画面中应占更大区域。用自适应参数的预期值作为有效搜索区域下限。
        """
        from detect import HoopDetector
        img_w = self.video_info["width"]
        img_h = self.video_info["height"]
        hoop_det = HoopDetector(img_w, img_h)
        hoop_det.detect_from_frames(frame_files, model=self.model,
                                     ball_detections=ball_detections)
        cx, cy, w, h = hoop_det.get_hoop_region()
        log(f"篮筐原始检测: method={hoop_det.detection_method}, "
            f"cx={cx:.0f}, cy={cy:.0f}, w={w:.0f}, h={h:.0f}")

        min_hoop_w = self.adaptive_params.get("hoop_w_px", 100)
        if w < min_hoop_w * 0.5:
            old_w, old_h = w, h
            w = max(w, min_hoop_w)
            h = max(h, w * 0.6)
            log(f"篮筐尺寸修正: {old_w:.0f}x{old_h:.0f} -> {w:.0f}x{h:.0f} "
                f"(预期最小宽度 {min_hoop_w:.0f}px)")

        return {"cx": cx, "cy": cy, "w": w, "h": h,
                "method": hoop_det.detection_method}

    def _build_tracks(self, detections: List[Tuple], fps: float) -> List:
        """建立追踪序列"""
        from tracker import IOUTracker, Detection as TrackerDet
        tracker = IOUTracker(
            iou_threshold=0.15,
            max_lost=max(5, int(fps * 0.25)),
            distance_threshold=max(60, self.adaptive_params.get("ball_d_px", 60) * 2.0),
        )
        frame_dets = {}
        for (fi, cx, cy, w, h, conf) in detections:
            frame_dets.setdefault(fi, []).append(
                TrackerDet(frame_idx=fi, cx=cx, cy=cy, w=w, h=h, conf=conf)
            )
        max_frame = max((d[0] for d in detections), default=0)
        for fi in range(max_frame + 1):
            tracker.update(frame_dets.get(fi, []))

        return tracker.get_all_tracks()

    # ================================================================
    # 实验 1: 粗扫候选召回率 (A-4)
    # ================================================================
    def exp1_coarse_recall(self):
        """验证粗扫候选是否能捕获所有真实进球"""
        log("")
        log("=" * 60)
        log("实验 1: 粗扫候选召回率 (A-4)")
        log("=" * 60)

        coarse_fps = self.adaptive_params["coarse_fps"]
        log(f"粗扫帧率: {coarse_fps} fps")

        coarse_dir = os.path.join(self.output_dir, "frames_coarse")
        log("提取粗扫帧...")
        coarse_frames = extract_frames(self.video_path, coarse_dir, coarse_fps)
        log(f"粗扫帧数: {len(coarse_frames)}")

        log("运行 YOLO 球体检测...")
        coarse_dets = self._detect_balls_in_frames(coarse_frames, yolo_conf=0.20)
        log(f"粗扫球体检测: {len(coarse_dets)} 次")

        hoop_rect = self._detect_hoop(coarse_frames, coarse_dets)

        from detect import fallback_heuristic_v2
        img_w = self.video_info["width"]
        img_h = self.video_info["height"]
        duration = self.video_info["duration"]

        hoop_cy = hoop_rect["cy"]
        hoop_h = hoop_rect["h"]
        hoop_y_min = hoop_cy - max(hoop_h * 1.5, img_h * 0.10)
        hoop_y_max = hoop_cy + max(hoop_h * 1.5, img_h * 0.10)

        frame_balls = {}
        for (fi, cx, cy, w, h, conf) in coarse_dets:
            frame_balls.setdefault(fi, []).append((cx, cy, w, h, conf))

        heuristic_events = fallback_heuristic_v2(
            frame_balls, img_w, img_h,
            len(coarse_frames), coarse_fps, duration,
            hoop_y_min, hoop_y_max,
            conf_threshold=0.30,
            hoop_cx=hoop_rect["cx"], hoop_cy=hoop_rect["cy"],
        )

        tracks = self._build_tracks(coarse_dets, coarse_fps)
        from detect import judge_goal_by_trajectory
        traj_events = []
        for track in tracks:
            pts = [(p.frame_idx, p.cx, p.cy, p.conf)
                   for p in track.points]
            result = judge_goal_by_trajectory(
                pts, hoop_rect["cx"], hoop_rect["cy"],
                hoop_rect["w"], hoop_rect["h"], coarse_fps,
            )
            if result:
                traj_events.append(result)

        # 通道 3: 运动突变检测
        from detect import detect_goals_by_motion
        try:
            motion_events = detect_goals_by_motion(
                coarse_frames, img_w, img_h, coarse_fps, duration,
            )
            for ev in motion_events:
                ev["detail"] = ev.get("detail", "运动突变")
            log(f"运动突变候选: {len(motion_events)} 个")
        except Exception as e:
            log(f"运动突变检测失败: {e}")
            motion_events = []

        # YOLO 'made' 类别检测（basketball_v1 模型有 made 类别）
        made_events = []
        if self.using_basketball_model:
            made_class = None
            for cls_id, name in self.model.names.items():
                if name.lower() == "made":
                    made_class = cls_id
                    break
            if made_class is not None:
                for fi, fpath in enumerate(coarse_frames):
                    try:
                        results = self.model(fpath, conf=0.25, imgsz=640, verbose=False)
                        for r in results:
                            if r.boxes is None:
                                continue
                            for box in r.boxes:
                                if int(box.cls[0]) == made_class:
                                    t = fi / max(coarse_fps, 1)
                                    conf = float(box.conf[0])
                                    made_events.append({
                                        "timestamp": t,
                                        "confidence": min(conf, 0.70),
                                        "detail": f"YOLO made class (conf={conf:.2f})",
                                    })
                    except Exception:
                        continue
                log(f"YOLO 'made' 候选: {len(made_events)} 个")

        all_coarse_events = heuristic_events + traj_events + motion_events + made_events
        for ev in all_coarse_events:
            if "timestamp" not in ev:
                ev["timestamp"] = ev.get("time", 0)

        # 去重：合并 ±2s 内的相似事件
        all_coarse_events.sort(key=lambda e: e.get("timestamp", 0))
        deduped = []
        for ev in all_coarse_events:
            if not deduped or abs(ev["timestamp"] - deduped[-1]["timestamp"]) > 2.0:
                deduped.append(ev)
            elif ev.get("confidence", 0) > deduped[-1].get("confidence", 0):
                deduped[-1] = ev
        all_coarse_events = deduped

        log(f"粗扫候选事件: {len(all_coarse_events)} 个")
        for ev in all_coarse_events:
            log(f"  t={ev['timestamp']:.1f}s conf={ev.get('confidence', 0):.2f} "
                f"{ev.get('detail', '')[:50]}")

        candidate_windows = []
        window_s = 4.0
        for ev in all_coarse_events:
            t = ev["timestamp"]
            candidate_windows.append((max(0, t - window_s), t + window_s))

        merged_windows = self._merge_windows(candidate_windows)

        gt_recall = {"total_goals": len(self.gt_goals), "recalled": 0,
                     "missed": [], "windows": []}
        for g in self.gt_goals:
            gt_t = g["timestamp"]
            found = any(w[0] <= gt_t <= w[1] for w in merged_windows)
            if found:
                gt_recall["recalled"] += 1
            else:
                gt_recall["missed"].append(g)

        recall_rate = gt_recall["recalled"] / max(len(self.gt_goals), 1)
        gt_recall["recall_rate"] = round(recall_rate, 4)
        gt_recall["candidate_count"] = len(all_coarse_events)
        gt_recall["merged_windows"] = len(merged_windows)
        gt_recall["pass"] = recall_rate >= 0.95

        match_result = match_events_to_gt(all_coarse_events, self.gt_goals)
        gt_recall["event_match"] = match_result

        log(f"\n--- 实验 1 结果 ---")
        log(f"候选窗口召回率: {recall_rate:.1%} "
            f"({gt_recall['recalled']}/{len(self.gt_goals)})")
        log(f"候选事件数: {len(all_coarse_events)}")
        log(f"合并窗口数: {len(merged_windows)}")
        log(f"事件级 Precision: {match_result['precision']:.2f}")
        log(f"事件级 Recall: {match_result['recall']:.2f}")
        log(f"事件级 F1: {match_result['f1']:.2f}")
        if gt_recall["missed"]:
            log(f"漏检进球: {gt_recall['missed']}")
        log(f"通过线 (≥95%): {'✅ 通过' if gt_recall['pass'] else '❌ 未通过'}")

        self.report["exp1_coarse_recall"] = gt_recall
        self._hoop_rect = hoop_rect
        self._merged_windows = merged_windows
        self._coarse_events = all_coarse_events
        return gt_recall

    def _merge_windows(self, windows: List[Tuple]) -> List[Tuple]:
        if not windows:
            return []
        sorted_w = sorted(windows)
        merged = [sorted_w[0]]
        for s, e in sorted_w[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged

    # ================================================================
    # 实验 2: 关键区追踪连续性 (A-2)
    # ================================================================
    def exp2_tracking_continuity(self):
        """验证碰撞前是否能获得 ≥5 帧连续追踪"""
        log("")
        log("=" * 60)
        log("实验 2: 关键区追踪连续性 (A-2)")
        log("=" * 60)

        if not hasattr(self, "_merged_windows"):
            log("需要先运行实验 1")
            return

        native_fps = self.video_info["fps"]
        fine_fps = min(native_fps, 30)
        hoop_rect = self._hoop_rect

        all_fine_dets = []
        all_fine_frames = []
        window_frame_offsets = []

        for wi, (ws, we) in enumerate(self._merged_windows):
            fine_dir = os.path.join(self.output_dir, f"frames_fine_w{wi}")
            log(f"窗口 {wi}: [{ws:.1f}s, {we:.1f}s], 提取 {fine_fps:.0f}fps 帧...")
            frames = extract_frames(self.video_path, fine_dir, fine_fps,
                                     start_s=ws, duration_s=we - ws)
            log(f"  提取 {len(frames)} 帧")

            frame_offset = int(ws * fine_fps)
            window_frame_offsets.append(frame_offset)

            log(f"  运行 YOLO 检测 (imgsz=960 + ROI裁剪)...")
            dets = self._detect_balls_in_frames(frames, yolo_conf=0.15,
                                                 use_roi=True)
            adjusted_dets = [(fi + frame_offset, cx, cy, w, h, conf)
                             for (fi, cx, cy, w, h, conf) in dets]
            all_fine_dets.extend(adjusted_dets)
            all_fine_frames.extend(frames)
            log(f"  检测到 {len(dets)} 个球体")

        log(f"全部精扫: {len(all_fine_dets)} 个球体检测, {len(all_fine_frames)} 帧")

        # 近篮筐球检测覆盖率统计(第五轮核心指标)
        near_hoop_det_count = 0
        near_hoop_frames = set()
        if hasattr(self, '_hoop_rect'):
            hr = self._hoop_rect
            for (fi, cx, cy, w, h, conf) in all_fine_dets:
                if (abs(cx - hr["cx"]) < hr["w"] * 1.5 and
                        abs(cy - hr["cy"]) < hr["w"] * 2.0):
                    near_hoop_det_count += 1
                    near_hoop_frames.add(fi)
            near_rate = len(near_hoop_frames) / max(len(all_fine_frames), 1)
            log(f"近篮筐球检测: {near_hoop_det_count}次, "
                f"{len(near_hoop_frames)}帧有球 "
                f"({near_rate:.0%} of {len(all_fine_frames)}帧)")

        tracks = self._build_tracks(all_fine_dets, fine_fps)
        log(f"追踪轨迹数: {len(tracks)}")

        from kinematics import (compute_kinematics, analyze_track_continuity,
                                 estimate_sigma_pos)

        continuity_stats = []
        all_kinematics = []

        for ti, track in enumerate(tracks):
            pts = [{"frame_idx": p.frame_idx, "cx": p.cx, "cy": p.cy,
                     "conf": p.conf} for p in track.points]
            kin = compute_kinematics(pts, fine_fps)
            all_kinematics.append(kin)

            stats = analyze_track_continuity(kin, hoop_rect, fine_fps)
            stats["track_id"] = track.track_id
            stats["track_points"] = len(track.points)
            continuity_stats.append(stats)

            if stats["near_hoop_segments"] > 0:
                log(f"  Track {track.track_id}: "
                    f"{stats['total_points']}pts, "
                    f"{stats['total_segments']}segs, "
                    f"篮筐附近{stats['near_hoop_segments']}seg "
                    f"(最长{stats['near_hoop_max_continuous']}pt), "
                    f"≥5帧段{stats['qualified_segments_ge5']}个, "
                    f"连续率{stats['continuity_rate']:.0%}")

        total_near_hoop = sum(s["near_hoop_segments"] for s in continuity_stats)
        total_qualified = sum(s["qualified_segments_ge5"] for s in continuity_stats)
        overall_rate = total_qualified / max(total_near_hoop, 1)

        sigma_estimates = []
        for kin in all_kinematics:
            if len(kin) >= 8:
                sig = estimate_sigma_pos(kin)
                sigma_estimates.append(sig)

        result = {
            "fine_fps": fine_fps,
            "total_tracks": len(tracks),
            "total_near_hoop_segments": total_near_hoop,
            "total_qualified_ge5": total_qualified,
            "overall_continuity_rate": round(overall_rate, 4),
            "pass": overall_rate >= 0.60,
            "target": "≥60%",
            "sigma_pos_estimates": {
                "count": len(sigma_estimates),
                "mean": round(float(sum(sigma_estimates) / max(len(sigma_estimates), 1)), 2),
                "min": round(min(sigma_estimates, default=0), 2),
                "max": round(max(sigma_estimates, default=0), 2),
            },
            "tracks": [{
                "track_id": s["track_id"],
                "total_points": s["total_points"],
                "total_segments": s["total_segments"],
                "near_hoop_segments": s["near_hoop_segments"],
                "near_hoop_max_continuous": s["near_hoop_max_continuous"],
                "qualified_ge5": s["qualified_segments_ge5"],
                "continuity_rate": s["continuity_rate"],
            } for s in continuity_stats],
        }

        log(f"\n--- 实验 2 结果 ---")
        log(f"关键区连续追踪率: {overall_rate:.1%} "
            f"(≥5帧段 {total_qualified}/{total_near_hoop})")
        log(f"σ_pos 估计: mean={result['sigma_pos_estimates']['mean']}px")
        log(f"通过线 (≥60%): {'✅ 通过' if result['pass'] else '❌ 未通过'}")

        self.report["exp2_tracking_continuity"] = result
        self._all_kinematics = all_kinematics
        self._all_tracks = tracks
        self._fine_fps = fine_fps
        return result

    # ================================================================
    # 实验 3: 残差分布实测 (A-3)
    # ================================================================
    def exp3_residual_distribution(self):
        """验证抛物线拟合残差是否能区分打铁与非打铁"""
        log("")
        log("=" * 60)
        log("实验 3: 残差分布实测 (A-3)")
        log("=" * 60)

        if not hasattr(self, "_all_kinematics"):
            log("需要先运行实验 2")
            return

        from kinematics import (detect_collision_events, estimate_sigma_pos,
                                 split_into_segments, fit_parabola,
                                 compute_residuals)

        hoop_rect = self._hoop_rect
        fine_fps = self._fine_fps
        ap = self.adaptive_params

        all_residuals = []
        all_collision_events = []
        segment_residual_stats = []

        for ti, kin in enumerate(self._all_kinematics):
            if len(kin) < 5:
                continue

            sigma_pos = estimate_sigma_pos(kin)
            collisions, stats = detect_collision_events(
                kin, hoop_rect, sigma_pos,
                residual_sigma_single=ap["residual_sigma_single"],
                residual_sigma_double=ap["residual_sigma_double"],
                residual_floor_px=ap["residual_floor_px"],
            )

            all_collision_events.extend(collisions)
            all_residuals.extend(stats["residuals_all"])

            segments = split_into_segments(kin, max_gap_frames=2)
            for seg in segments:
                if seg.n_points < 5:
                    continue
                try:
                    xc, yc = fit_parabola(seg)
                    residuals = compute_residuals(seg, xc, yc)
                    near_hoop = any(
                        abs(p.x - hoop_rect["cx"]) < hoop_rect["w"] * 1.5 and
                        abs(p.y - hoop_rect["cy"]) < hoop_rect["h"] * 3.0
                        for p in seg.points
                    )
                    segment_residual_stats.append({
                        "track_idx": ti,
                        "n_points": seg.n_points,
                        "near_hoop": near_hoop,
                        "mean_residual": round(float(sum(residuals) / len(residuals)), 2),
                        "max_residual": round(float(max(residuals)), 2),
                        "std_residual": round(float(
                            (sum((r - sum(residuals)/len(residuals))**2
                                 for r in residuals) / len(residuals)) ** 0.5
                        ), 2),
                    })
                except Exception:
                    continue

        import numpy as np
        residual_array = np.array(all_residuals) if all_residuals else np.array([0.0])

        result = {
            "total_residuals": len(all_residuals),
            "total_collision_events": len(all_collision_events),
            "residual_stats": {
                "mean": round(float(np.mean(residual_array)), 2),
                "std": round(float(np.std(residual_array)), 2),
                "p50": round(float(np.percentile(residual_array, 50)), 2),
                "p90": round(float(np.percentile(residual_array, 90)), 2),
                "p99": round(float(np.percentile(residual_array, 99)), 2),
                "max": round(float(np.max(residual_array)), 2),
            },
            "collision_events": [{
                "frame_idx": c.frame_idx,
                "time": round(c.time, 3),
                "residual_px": round(c.residual_px, 2),
                "residual_sigma": round(c.residual_sigma, 2),
                "in_hoop_region": c.in_hoop_region,
                "in_backboard_region": c.in_backboard_region,
                "vy_reversed": c.vy_reversed,
                "vx_reversed": c.vx_reversed,
                "lateral_escape": c.lateral_escape,
            } for c in all_collision_events],
            "segment_stats": segment_residual_stats,
            "near_hoop_segments": [s for s in segment_residual_stats if s["near_hoop"]],
        }

        hoop_segs = result["near_hoop_segments"]
        if hoop_segs:
            hoop_residuals = [s["max_residual"] for s in hoop_segs]
            result["near_hoop_residual_stats"] = {
                "count": len(hoop_segs),
                "mean_max_residual": round(float(np.mean(hoop_residuals)), 2),
                "std_max_residual": round(float(np.std(hoop_residuals)), 2),
            }

        log(f"\n--- 实验 3 结果 ---")
        log(f"总残差数据点: {len(all_residuals)}")
        log(f"残差统计: mean={result['residual_stats']['mean']:.2f}px, "
            f"p50={result['residual_stats']['p50']:.2f}px, "
            f"p90={result['residual_stats']['p90']:.2f}px, "
            f"max={result['residual_stats']['max']:.2f}px")
        log(f"碰撞事件: {len(all_collision_events)} 个")
        for c in all_collision_events:
            log(f"  frame={c.frame_idx} residual={c.residual_px:.1f}px "
                f"({c.residual_sigma:.1f}σ) hoop={c.in_hoop_region} "
                f"vy_rev={c.vy_reversed} vx_rev={c.vx_reversed}")
        log(f"篮筐附近片段: {len(hoop_segs)} 个")

        self.report["exp3_residual_distribution"] = result
        self._collision_events = all_collision_events
        return result

    # ================================================================
    # 实验 4: 消失-重现与反证法 (A-1 部分)
    # ================================================================
    def exp4_disappear_reappear(self):
        """验证消失-重现模式和反证法"""
        log("")
        log("=" * 60)
        log("实验 4: 消失-重现与反证法 (A-1 部分)")
        log("=" * 60)

        if not hasattr(self, "_all_kinematics"):
            log("需要先运行实验 2")
            return

        from kinematics import (analyze_disappear_reappear,
                                 channel_8_no_bounce_evidence,
                                 find_tracking_gaps,
                                 cross_track_disappear_reappear)

        hoop_rect = self._hoop_rect
        fine_fps = self._fine_fps

        all_patterns = []
        all_gaps = []
        all_no_bounce = []

        for ti, kin in enumerate(self._all_kinematics):
            if len(kin) < 3:
                continue

            gaps = find_tracking_gaps(kin, min_gap_frames=2)
            for gap in gaps:
                gap["track_idx"] = ti
                all_gaps.append(gap)

            patterns = analyze_disappear_reappear(kin, hoop_rect, fine_fps)
            for p in patterns:
                p_dict = {
                    "track_idx": ti,
                    "pattern": p.pattern,
                    "confidence": p.confidence,
                    "detail": p.detail,
                    "gap_start_frame": p.gap_start_frame,
                    "gap_end_frame": p.gap_end_frame,
                    "time_gap": round(p.time_gap, 4),
                }
                all_patterns.append(p_dict)

            no_bounce = channel_8_no_bounce_evidence(
                kin, hoop_rect,
                total_frames=kin[-1].frame_idx if kin else 0,
                fps=fine_fps,
            )
            if no_bounce:
                no_bounce["track_idx"] = ti
                all_no_bounce.append(no_bounce)

        # 跨轨迹消失-重现
        cross_track_events = cross_track_disappear_reappear(
            self._all_kinematics, hoop_rect, fine_fps,
        )

        goal_patterns = [p for p in all_patterns if p["pattern"] == "goal"]
        bounce_patterns = [p for p in all_patterns if p["pattern"] == "bounce_out"]
        unknown_patterns = [p for p in all_patterns if p["pattern"] == "unknown"]

        result = {
            "total_gaps": len(all_gaps),
            "total_patterns": len(all_patterns),
            "goal_patterns": len(goal_patterns),
            "bounce_patterns": len(bounce_patterns),
            "unknown_patterns": len(unknown_patterns),
            "no_bounce_evidence": len(all_no_bounce),
            "patterns": all_patterns,
            "gaps": [{
                "track_idx": g["track_idx"],
                "start_frame": g["start_frame"],
                "end_frame": g["end_frame"],
                "gap_frames": g["gap_frames"],
                "time_gap": round(g["time_gap"], 4),
            } for g in all_gaps],
            "no_bounce_results": all_no_bounce,
            "cross_track_events": cross_track_events,
        }

        log(f"\n--- 实验 4 结果 ---")
        log(f"追踪间隔总数: {len(all_gaps)}")
        log(f"消失-重现模式:")
        log(f"  进球模式: {len(goal_patterns)}")
        log(f"  弹出模式: {len(bounce_patterns)}")
        log(f"  未知模式: {len(unknown_patterns)}")
        for p in all_patterns:
            log(f"  [{p['pattern']}] track={p['track_idx']} "
                f"gap={p['time_gap']:.3f}s {p['detail'][:60]}")
        log(f"反证法触发: {len(all_no_bounce)} 次")
        for nb in all_no_bounce:
            log(f"  track={nb['track_idx']} t={nb['timestamp']:.2f}s "
                f"conf={nb['confidence']:.2f} {nb['detail'][:60]}")
        log(f"跨轨迹进球候选: {len(cross_track_events)} 个")
        for ct in cross_track_events:
            log(f"  t={ct['timestamp']:.2f}s conf={ct['confidence']:.2f} "
                f"{ct['detail'][:70]}")

        self.report["exp4_disappear_reappear"] = result
        return result

    # ================================================================
    # 实验 5: 综合判定验证 (A-5)
    # ================================================================
    def exp5_overall_judgment(self):
        """综合运行全链路判定，对比 GT"""
        log("")
        log("=" * 60)
        log("实验 5: 综合判定验证 (A-5)")
        log("=" * 60)

        if not hasattr(self, "_all_kinematics"):
            log("需要先运行实验 2")
            return

        from kinematics import (judge_goal_by_continuity, detect_collision_events,
                                 estimate_sigma_pos, analyze_disappear_reappear,
                                 channel_8_no_bounce_evidence,
                                 cross_track_disappear_reappear, GoalJudgment)

        hoop_rect = self._hoop_rect
        fine_fps = self._fine_fps
        ap = self.adaptive_params

        all_judgments = []

        for ti, kin in enumerate(self._all_kinematics):
            if len(kin) < 3:
                continue

            sigma_pos = estimate_sigma_pos(kin)
            collisions, _ = detect_collision_events(
                kin, hoop_rect, sigma_pos,
                residual_sigma_single=ap["residual_sigma_single"],
                residual_sigma_double=ap["residual_sigma_double"],
                residual_floor_px=ap["residual_floor_px"],
            )

            judgment = judge_goal_by_continuity(
                kin, collisions, hoop_rect, fine_fps, sigma_pos,
            )
            if judgment.is_goal and sigma_pos > 30.0:
                judgment = GoalJudgment(
                    is_goal=False,
                    reason=f"{judgment.reason}; sigma={sigma_pos:.0f}px过高,穿越判定降级",
                )

            patterns = analyze_disappear_reappear(kin, hoop_rect, fine_fps)
            goal_pattern = any(p.pattern == "goal" for p in patterns)

            no_bounce = channel_8_no_bounce_evidence(
                kin, hoop_rect,
                total_frames=kin[-1].frame_idx if kin else 0,
                fps=fine_fps,
            )

            best_conf = judgment.confidence
            combined_reason = judgment.reason

            sigma_quality_ok = sigma_pos < 25.0
            if goal_pattern and not judgment.is_goal and sigma_quality_ok:
                best_conf = max(best_conf, 0.75)
                combined_reason += "; 消失-重现支持进球"
            elif goal_pattern and not judgment.is_goal:
                combined_reason += f"; 消失-重现(sigma={sigma_pos:.0f}px过高,降权)"

            if no_bounce and not judgment.is_goal and sigma_quality_ok:
                best_conf = max(best_conf, no_bounce["confidence"])
                combined_reason += f"; {no_bounce['detail']}"

            final_goal = judgment.is_goal or best_conf >= 0.60

            if kin:
                first_time = kin[0].time
                last_time = kin[-1].time
                mid_time = (first_time + last_time) / 2
                crossing_time = judgment.crossing_time if judgment.crossing_time > 0 else mid_time
            else:
                crossing_time = 0

            result_entry = {
                "track_idx": ti,
                "is_goal": final_goal,
                "confidence": round(best_conf, 3),
                "goal_type": judgment.goal_type,
                "reason": combined_reason,
                "timestamp": round(crossing_time, 2),
                "continuity_judgment": judgment.is_goal,
                "disappear_pattern_goal": goal_pattern,
                "no_bounce_support": no_bounce is not None,
                "n_collisions": len(collisions),
            }
            all_judgments.append(result_entry)

        # 跨轨迹消失-重现分析
        cross_track_events = cross_track_disappear_reappear(
            self._all_kinematics, hoop_rect, fine_fps,
        )
        log(f"跨轨迹进球候选: {len(cross_track_events)} 个")
        for ct in cross_track_events:
            log(f"  {ct['detail']}")
            all_judgments.append({
                "track_idx": -1,
                "is_goal": True,
                "confidence": ct["confidence"],
                "goal_type": "cross_track",
                "reason": ct["detail"],
                "timestamp": round(ct["timestamp"], 2),
                "continuity_judgment": False,
                "disappear_pattern_goal": True,
                "no_bounce_support": False,
                "n_collisions": 0,
            })

        goal_judgments = [j for j in all_judgments if j["is_goal"]]
        non_goal_judgments = [j for j in all_judgments if not j["is_goal"]]

        from detect import merge_events_with_cooldown
        merged_goals = merge_events_with_cooldown(
            goal_judgments, conf_threshold=0.30,
            merge_window=3.0, cooldown=1.5,
        )

        gt_match = match_events_to_gt(merged_goals, self.gt_goals, tolerance_s=5.0)

        result = {
            "total_tracks_analyzed": len(all_judgments),
            "goal_tracks": len(goal_judgments),
            "non_goal_tracks": len(non_goal_judgments),
            "merged_goal_events": len(merged_goals),
            "gt_match": gt_match,
            "judgments": all_judgments,
            "merged_events": [{
                "timestamp": e.get("timestamp", 0),
                "confidence": e.get("confidence", 0),
                "detail": e.get("reason", "")[:100],
            } for e in merged_goals],
        }

        log(f"\n--- 实验 5 结果 ---")
        log(f"分析轨迹: {len(all_judgments)} 条")
        log(f"判定进球: {len(goal_judgments)} 条")
        log(f"合并后事件: {len(merged_goals)} 个")
        log(f"GT 匹配:")
        log(f"  Precision: {gt_match['precision']:.2f}")
        log(f"  Recall: {gt_match['recall']:.2f}")
        log(f"  F1: {gt_match['f1']:.2f}")
        log(f"  TP: {gt_match['true_positives']}, "
            f"FP: {gt_match['false_positives']}, "
            f"FN: {gt_match['false_negatives']}")
        if gt_match["missed_goals"]:
            log(f"  漏检: {gt_match['missed_goals']}")

        for j in all_judgments:
            if j["is_goal"]:
                log(f"  ✅ Track {j['track_idx']}: t={j['timestamp']:.1f}s "
                    f"conf={j['confidence']:.2f} type={j['goal_type']} "
                    f"{j['reason'][:60]}")

        self.report["exp5_overall_judgment"] = result
        return result

    # ================================================================
    # 生成报告
    # ================================================================
    def generate_report(self):
        """生成完整可行性验证报告"""
        log("")
        log("=" * 60)
        log("生成可行性验证报告")
        log("=" * 60)

        summary = {
            "conclusion": "",
            "pass_items": [],
            "fail_items": [],
            "bottleneck": "",
        }

        exp1 = self.report.get("exp1_coarse_recall", {})
        if exp1.get("pass"):
            summary["pass_items"].append(
                f"粗扫候选召回率: {exp1.get('recall_rate', 0):.0%} ≥ 95%"
            )
        else:
            summary["fail_items"].append(
                f"粗扫候选召回率: {exp1.get('recall_rate', 0):.0%} < 95%"
            )

        exp2 = self.report.get("exp2_tracking_continuity", {})
        if exp2.get("pass"):
            summary["pass_items"].append(
                f"关键区连续追踪率: {exp2.get('overall_continuity_rate', 0):.0%} ≥ 60%"
            )
        else:
            summary["fail_items"].append(
                f"关键区连续追踪率: {exp2.get('overall_continuity_rate', 0):.0%} < 60%"
            )

        exp5 = self.report.get("exp5_overall_judgment", {})
        gt_match = exp5.get("gt_match", {})
        f1 = gt_match.get("f1", 0)
        summary["event_f1"] = f1

        if summary["fail_items"]:
            summary["conclusion"] = "部分验证未通过，需优化"
            summary["bottleneck"] = summary["fail_items"][0]
        else:
            summary["conclusion"] = "首轮验证基本通过，可继续优化"

        self.report["summary"] = summary

        report_path = os.path.join(self.output_dir, "feasibility_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, ensure_ascii=False, indent=2, default=str)
        log(f"报告已保存: {report_path}")

        log("")
        log("=" * 60)
        log("验证总结")
        log("=" * 60)
        log(f"结论: {summary['conclusion']}")
        if summary["pass_items"]:
            log("通过项:")
            for item in summary["pass_items"]:
                log(f"  ✅ {item}")
        if summary["fail_items"]:
            log("未通过项:")
            for item in summary["fail_items"]:
                log(f"  ❌ {item}")
        log(f"事件级 F1: {f1:.2f}")

        return report_path


def main():
    parser = argparse.ArgumentParser(description="GoalCut 算法可行性验证")
    parser.add_argument("--video", required=True, help="测试视频路径")
    parser.add_argument("--gt", required=True, help="GT 标注 JSON 路径")
    parser.add_argument("--output", default="test/output/feasibility",
                        help="输出目录")
    parser.add_argument("--skip-exp", nargs="*", default=[],
                        help="跳过的实验编号 (1-5)")
    args = parser.parse_args()

    verifier = FeasibilityVerifier(args.video, args.gt, args.output)
    verifier.setup()

    skip = set(args.skip_exp)

    t0 = time.time()

    if "1" not in skip:
        verifier.exp1_coarse_recall()

    if "2" not in skip:
        verifier.exp2_tracking_continuity()

    if "3" not in skip:
        verifier.exp3_residual_distribution()

    if "4" not in skip:
        verifier.exp4_disappear_reappear()

    if "5" not in skip:
        verifier.exp5_overall_judgment()

    report_path = verifier.generate_report()

    elapsed = time.time() - t0
    log(f"\n总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
