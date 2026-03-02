#!/usr/bin/env python3
"""
GoalCut AI 进球检测引擎
使用 YOLOv8 检测篮球和篮筐，通过球与篮筐的位置关系判断进球。

简化版 MVP 逻辑：
1. 用预训练 YOLOv8 检测 "sports ball" (class 32) 
2. 检测篮筐区域（通过颜色/形状或用 sports ball 在高位出现作为辅助）
3. 当球出现在篮筐附近区域且满足一定条件时，判定为进球
"""

import argparse
import json
import os
import sys
import glob
import numpy as np
from pathlib import Path


def log(msg):
    print(f"[ai-engine] {msg}", flush=True)


def detect_goals(frames_dir, output_path, sample_fps, confidence_threshold, 
                 yolo_confidence, video_duration):
    """主检测流程"""
    log(f"开始进球检测")
    log(f"  帧目录: {frames_dir}")
    log(f"  采样帧率: {sample_fps}")
    log(f"  置信度阈值: {confidence_threshold}")
    log(f"  YOLO置信度: {yolo_confidence}")
    log(f"  视频时长: {video_duration}s")

    # 加载 YOLO 模型
    log("加载 YOLOv8 模型...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")  # 使用预训练的 nano 模型
        log("YOLOv8n 模型加载成功")
    except Exception as e:
        log(f"加载 YOLO 模型失败: {e}")
        # 回退到简化模式
        log("回退到简化检测模式（无 YOLO）")
        events = fallback_detection(frames_dir, sample_fps, video_duration, confidence_threshold)
        save_results(events, output_path)
        return

    # 获取所有帧文件
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not frame_files:
        log("错误: 未找到帧文件")
        save_results([], output_path)
        return
    
    log(f"共 {len(frame_files)} 帧待检测")

    # COCO 类别中与篮球相关的
    # 32: sports ball
    BALL_CLASS = 32

    # 第一遍：检测所有帧中的 sports ball
    ball_detections = []  # [(frame_idx, cx, cy, w, h, conf)]
    
    for i, frame_path in enumerate(frame_files):
        if i % 10 == 0:
            log(f"检测进度: {i+1}/{len(frame_files)}")
        
        try:
            results = model(frame_path, conf=yolo_confidence, verbose=False)
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

    log(f"共检测到 {len(ball_detections)} 次球体出现")

    if not ball_detections:
        log("未检测到任何球体，尝试回退检测模式")
        events = fallback_detection(frames_dir, sample_fps, video_duration, confidence_threshold)
        save_results(events, output_path)
        return

    # 获取图像尺寸
    import cv2
    sample_img = cv2.imread(frame_files[0])
    if sample_img is None:
        log("错误: 无法读取帧图片")
        save_results([], output_path)
        return
    img_h, img_w = sample_img.shape[:2]
    log(f"帧尺寸: {img_w}x{img_h}")

    # 分析球的运动轨迹，寻找进球事件
    events = analyze_ball_trajectory(
        ball_detections, img_w, img_h, 
        len(frame_files), sample_fps, video_duration,
        confidence_threshold
    )

    log(f"轨迹分析完成，检测到 {len(events)} 个进球事件")
    save_results(events, output_path)


def analyze_ball_trajectory(detections, img_w, img_h, total_frames, 
                           sample_fps, video_duration, conf_threshold):
    """
    分析球的轨迹判断进球。
    
    进球特征（简化版）：
    1. 球出现在画面上半部分（篮筐通常在上方）
    2. 球有向下运动的趋势（投篮后下落）
    3. 球在某区域短暂停留或消失（入网）
    4. 球的 y 坐标突然变化（穿过篮筐）
    """
    events = []
    
    if not detections:
        return events

    # 按帧分组
    frame_balls = {}
    for (fidx, cx, cy, w, h, conf) in detections:
        if fidx not in frame_balls:
            frame_balls[fidx] = []
        frame_balls[fidx].append((cx, cy, w, h, conf))

    # 定义篮筐可能的区域：画面上部 1/3 到 2/3 高度
    hoop_y_min = img_h * 0.15
    hoop_y_max = img_h * 0.65

    log(f"篮筐搜索区域: y={hoop_y_min:.0f}~{hoop_y_max:.0f} (图像高度={img_h})")

    # 寻找球消失-出现模式（进球后球会短暂消失在网中）
    sorted_frames = sorted(frame_balls.keys())
    
    # 检测方法1: 球在篮筐区域出现后短暂消失
    last_ball_frame = -999
    last_ball_y = 0
    ball_in_hoop_zone_count = 0
    candidate_events = []

    for fidx in range(total_frames):
        if fidx in frame_balls:
            balls = frame_balls[fidx]
            # 取置信度最高的球
            best_ball = max(balls, key=lambda b: b[4])
            cx, cy, w, h, conf = best_ball
            
            # 球在篮筐区域
            if hoop_y_min <= cy <= hoop_y_max:
                # 检查球是否从上方进入（向下运动）
                if last_ball_frame >= 0 and fidx - last_ball_frame <= 3:
                    if cy > last_ball_y:  # 球在下落
                        ball_in_hoop_zone_count += 1
                    else:
                        ball_in_hoop_zone_count = max(0, ball_in_hoop_zone_count - 1)
                else:
                    ball_in_hoop_zone_count = 1

                # 如果球在篮筐区域连续出现，可能是进球
                if ball_in_hoop_zone_count >= 2:
                    timestamp = fidx / sample_fps
                    confidence = min(0.8, conf * 0.7 + 0.2 * (ball_in_hoop_zone_count / 3))
                    candidate_events.append({
                        "frame_index": fidx,
                        "timestamp": timestamp,
                        "confidence": round(confidence, 3),
                        "detail": f"球在篮筐区域连续出现{ball_in_hoop_zone_count}帧, y={cy:.0f}, conf={conf:.2f}"
                    })
                    ball_in_hoop_zone_count = 0  # 重置

            last_ball_frame = fidx
            last_ball_y = cy
        else:
            # 球消失帧
            if last_ball_frame >= 0 and fidx - last_ball_frame == 1:
                # 球刚刚消失，检查之前是否在篮筐区域
                if hoop_y_min <= last_ball_y <= hoop_y_max and ball_in_hoop_zone_count >= 1:
                    timestamp = last_ball_frame / sample_fps
                    confidence = 0.6 + 0.1 * min(ball_in_hoop_zone_count, 3)
                    candidate_events.append({
                        "frame_index": last_ball_frame,
                        "timestamp": timestamp,
                        "confidence": round(confidence, 3),
                        "detail": f"球在篮筐区域后消失, y={last_ball_y:.0f}, 连续帧数={ball_in_hoop_zone_count}"
                    })
            ball_in_hoop_zone_count = 0

    # 检测方法2: 球的 y 坐标在篮筐区域出现急剧变化（穿过篮筐）
    prev_cy = None
    prev_fidx = -999
    for fidx in sorted_frames:
        balls = frame_balls[fidx]
        best_ball = max(balls, key=lambda b: b[4])
        cx, cy, w, h, conf = best_ball
        
        if prev_cy is not None and fidx - prev_fidx <= 2:
            dy = cy - prev_cy
            # 球快速下落穿过篮筐区域
            if dy > img_h * 0.08 and hoop_y_min <= prev_cy <= hoop_y_max:
                timestamp = fidx / sample_fps
                speed_factor = min(1.0, abs(dy) / (img_h * 0.15))
                confidence = 0.5 + 0.3 * speed_factor
                candidate_events.append({
                    "frame_index": fidx,
                    "timestamp": timestamp,
                    "confidence": round(confidence, 3),
                    "detail": f"球快速下落穿过篮筐区域, dy={dy:.0f}, prev_y={prev_cy:.0f}"
                })
        
        prev_cy = cy
        prev_fidx = fidx

    # 去重和合并：在时间窗口内合并相近事件
    if not candidate_events:
        log("未发现候选进球事件")
        # 如果完全没有候选事件，尝试用宽松标准
        return fallback_heuristic(frame_balls, img_w, img_h, total_frames, 
                                   sample_fps, video_duration, conf_threshold)

    # 按时间排序
    candidate_events.sort(key=lambda e: e["timestamp"])
    
    # 合并窗口内的事件（取置信度最高的）
    merge_window = 3.0  # 3秒内的事件合并
    merged = []
    current_group = [candidate_events[0]]
    
    for i in range(1, len(candidate_events)):
        if candidate_events[i]["timestamp"] - current_group[0]["timestamp"] <= merge_window:
            current_group.append(candidate_events[i])
        else:
            # 取组内置信度最高的
            best = max(current_group, key=lambda e: e["confidence"])
            merged.append(best)
            current_group = [candidate_events[i]]
    
    # 最后一组
    best = max(current_group, key=lambda e: e["confidence"])
    merged.append(best)

    # 过滤低置信度
    events = [e for e in merged if e["confidence"] >= conf_threshold]
    
    log(f"候选事件: {len(candidate_events)}, 合并后: {len(merged)}, 过滤后: {len(events)}")
    return events


def fallback_heuristic(frame_balls, img_w, img_h, total_frames,
                       sample_fps, video_duration, conf_threshold):
    """
    宽松的启发式检测：当标准检测未找到事件时使用。
    寻找球在画面上半部分活跃出现的时间段。
    """
    log("使用宽松启发式检测模式")
    
    if not frame_balls:
        return []

    # 统计每个时间窗口内球在上半部分出现的频率
    window_size = int(sample_fps * 2)  # 2秒窗口
    if window_size < 1:
        window_size = 1
    
    activity_scores = {}
    for fidx, balls in frame_balls.items():
        best = max(balls, key=lambda b: b[4])
        cx, cy, w, h, conf = best
        # 球在上半部分
        if cy < img_h * 0.6:
            window_idx = fidx // window_size
            if window_idx not in activity_scores:
                activity_scores[window_idx] = 0
            activity_scores[window_idx] += conf

    if not activity_scores:
        return []

    # 找活跃度最高的窗口
    sorted_windows = sorted(activity_scores.items(), key=lambda x: x[1], reverse=True)
    
    events = []
    used_times = set()
    
    for window_idx, score in sorted_windows[:5]:  # 最多取5个
        timestamp = (window_idx * window_size + window_size / 2) / sample_fps
        # 检查是否与已有事件太近
        too_close = False
        for t in used_times:
            if abs(timestamp - t) < 4:
                too_close = True
                break
        if too_close:
            continue
            
        confidence = min(0.7, score / (window_size * 0.8))
        if confidence >= conf_threshold:
            events.append({
                "frame_index": window_idx * window_size,
                "timestamp": round(timestamp, 2),
                "confidence": round(confidence, 3),
                "detail": f"启发式检测: 窗口活跃度={score:.2f}"
            })
            used_times.add(timestamp)

    log(f"启发式检测找到 {len(events)} 个候选事件")
    return events


def fallback_detection(frames_dir, sample_fps, video_duration, conf_threshold):
    """
    完全回退模式：当 YOLO 不可用时，使用简单的帧差分检测运动。
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

    # 帧差分检测运动剧烈的时刻
    prev_gray = None
    motion_scores = []

    for i, path in enumerate(frame_files):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # 只关注上半部分（篮筐区域）
        h = img.shape[0]
        roi = img[:int(h * 0.6), :]
        
        if prev_gray is not None:
            diff = cv2.absdiff(roi, prev_gray)
            score = np.mean(diff)
            motion_scores.append((i, score))
        
        prev_gray = roi

    if not motion_scores:
        return []

    # 找运动峰值
    scores = [s for _, s in motion_scores]
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    threshold = mean_score + 1.5 * std_score

    log(f"运动均值={mean_score:.2f}, 标准差={std_score:.2f}, 阈值={threshold:.2f}")

    events = []
    last_event_frame = -999
    
    for fidx, score in motion_scores:
        if score > threshold and fidx - last_event_frame > sample_fps * 3:
            timestamp = fidx / sample_fps
            confidence = min(0.6, (score - mean_score) / (std_score * 3 + 1e-6))
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
    parser = argparse.ArgumentParser(description="GoalCut AI 进球检测引擎")
    parser.add_argument("--frames-dir", required=True, help="帧图片目录")
    parser.add_argument("--output", required=True, help="检测结果输出路径(JSON)")
    parser.add_argument("--sample-fps", type=float, default=3, help="采样帧率")
    parser.add_argument("--confidence-threshold", type=float, default=0.3, help="置信度阈值")
    parser.add_argument("--yolo-confidence", type=float, default=0.3, help="YOLO检测置信度")
    parser.add_argument("--video-duration", type=float, default=0, help="视频时长(秒)")
    
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
