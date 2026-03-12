#!/usr/bin/env python3
"""
GoalCut 通道准确率评估脚本

端到端自动化流程：
  1. 用 ffmpeg 从视频提取帧（和音频）
  2. 调用 channel_test.py 对各通道独立评估
  3. 汇总输出 Precision / Recall / F1 报告
  4. 支持多视频批量评估

用法:
    # 单视频评估（全通道）
    python test/scripts/eval.py \\
        --video test/素材/formal_game_01.mp4 \\
        --gt test/ground_truth/formal_game_01.json

    # 只评估指定通道
    python test/scripts/eval.py \\
        --video test/素材/formal_game_01.mp4 \\
        --gt test/ground_truth/formal_game_01.json \\
        --channels 1a,1b,3

    # 启用音频通道（需提前提取 WAV）
    python test/scripts/eval.py \\
        --video test/素材/formal_game_01.mp4 \\
        --gt test/ground_truth/formal_game_01.json \\
        --channels 1a,5 --with-audio

    # 批量评估 ground_truth/ 目录下所有视频
    python test/scripts/eval.py \\
        --batch test/ground_truth/ \\
        --video-dir test/素材/

    # 保存详细结果到 JSON
    python test/scripts/eval.py \\
        --video test/素材/formal_game_01.mp4 \\
        --gt test/ground_truth/formal_game_01.json \\
        --output /tmp/eval_result.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# 工具函数
# ============================================================

def log(msg: str):
    print(f"[eval] {msg}", flush=True)


def run_cmd(cmd: List[str], desc: str = "") -> Tuple[bool, str]:
    """运行外部命令，返回 (成功, 输出)"""
    if desc:
        log(desc)
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr or e.stdout


def get_video_duration(video_path: str) -> float:
    """用 ffprobe 获取视频时长"""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", video_path,
    ]
    ok, out = run_cmd(cmd)
    if not ok:
        return 0.0
    try:
        data = json.loads(out)
        return float(data.get("format", {}).get("duration", 0))
    except Exception:
        return 0.0


def find_ai_engine_dir() -> str:
    """定位 ai-engine 目录（相对脚本位置自动推断）"""
    script_dir = Path(__file__).parent.resolve()
    # test/scripts/ → ../../../ → project root
    candidates = [
        script_dir / ".." / ".." / "ai-engine",
        script_dir / ".." / ".." / ".." / "goalcut" / "ai-engine",
    ]
    for c in candidates:
        c = c.resolve()
        if (c / "channel_test.py").exists():
            return str(c)
    # 向上搜索
    p = script_dir
    for _ in range(6):
        candidate = p / "ai-engine"
        if (candidate / "channel_test.py").exists():
            return str(candidate)
        p = p.parent
    return ""


def find_video_file(video_dir: str, gt_filename_stem: str) -> Optional[str]:
    """
    在 video_dir 中按 stem 匹配视频文件。
    支持不同扩展名：.mp4 .MP4 .mov .avi .mkv
    """
    exts = [".mp4", ".MP4", ".mov", ".MOV", ".avi", ".mkv", ".MKV"]
    for ext in exts:
        path = os.path.join(video_dir, gt_filename_stem + ext)
        if os.path.exists(path):
            return path
    # 模糊匹配（去掉已知后缀重试）
    for fname in os.listdir(video_dir):
        stem = Path(fname).stem
        if stem == gt_filename_stem and Path(fname).suffix.lower() in [e.lower() for e in exts]:
            return os.path.join(video_dir, fname)
    return None


# ============================================================
# ffmpeg 预处理
# ============================================================

def extract_frames(video_path: str, output_dir: str, sample_fps: float) -> bool:
    """提取视频帧，命名格式为 frame_NNNNNN.jpg"""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"fps={sample_fps}",
        "-q:v", "3",
        os.path.join(output_dir, "frame_%06d.jpg"),
    ]
    ok, err = run_cmd(cmd, f"提取帧 (fps={sample_fps}) → {output_dir}")
    if not ok:
        log(f"  ffmpeg 错误: {err[:200]}")
    return ok


def extract_audio(video_path: str, output_wav: str) -> bool:
    """提取音频为 16kHz 单声道 WAV"""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        output_wav,
    ]
    ok, err = run_cmd(cmd, f"提取音频 → {output_wav}")
    if not ok:
        log(f"  ffmpeg 音频错误: {err[:200]}")
    return ok


# ============================================================
# 调用 channel_test.py
# ============================================================

def run_channel_test(
    ai_engine_dir: str,
    frames_dir: str,
    video_duration: float,
    channels: str,
    gt_path: str,
    output_json: str,
    audio_file: str = "",
    sample_fps: float = 3.0,
    confidence_threshold: float = 0.55,
    yolo_confidence: float = 0.25,
    extra_args: List[str] = None,
) -> bool:
    """调用 channel_test.py 执行评估，结果写入 output_json"""
    # 所有路径转为绝对路径，避免 cwd 切换后找不到文件
    frames_dir = os.path.abspath(frames_dir)
    output_json = os.path.abspath(output_json)
    gt_path_abs = os.path.abspath(gt_path) if gt_path else ""
    audio_file_abs = os.path.abspath(audio_file) if audio_file else ""

    # 优先使用 miniconda python3.10（支持 ultralytics），回退到 sys.executable
    _python = "/opt/miniconda3/bin/python3.10"
    if not os.path.exists(_python):
        _python = sys.executable
    cmd = [
        _python,
        os.path.join(ai_engine_dir, "channel_test.py"),
        "--frames-dir", frames_dir,
        "--video-duration", str(video_duration),
        "--channels", channels,
        "--sample-fps", str(sample_fps),
        "--confidence-threshold", str(confidence_threshold),
        "--yolo-confidence", str(yolo_confidence),
        "--output", output_json,
    ]
    if gt_path_abs and os.path.exists(gt_path_abs):
        cmd += ["--ground-truth", gt_path_abs]
    if audio_file_abs and os.path.exists(audio_file_abs):
        cmd += ["--audio-file", audio_file_abs]
    if extra_args:
        cmd += extra_args

    log(f"运行 channel_test.py: channels={channels}")
    try:
        result = subprocess.run(cmd, cwd=ai_engine_dir, text=True)
        return result.returncode == 0
    except Exception as e:
        log(f"  调用失败: {e}")
        return False


# ============================================================
# 报告渲染
# ============================================================

CHANNEL_NAMES = {
    "1a": "YOLO球体+IOU追踪+几何判定",
    "1b": "启发式检测（球消失）",
    "2":  "人体运动模式检测",
    "3":  "局部运动突变（帧差分）",
    "4":  "篮网形变（光流法）",
    "5":  "音频事件检测",
}


def render_report(result_json_path: str, video_name: str):
    """读取 channel_test 输出 JSON，渲染可读报告"""
    if not os.path.exists(result_json_path):
        log(f"结果文件不存在: {result_json_path}")
        return None

    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    channels = data.get("channels_tested", [])
    results = data.get("results", {})

    print(f"\n{'='*65}")
    print(f"  评估报告: {video_name}")
    print(f"{'='*65}")
    print(f"  {'通道':<6} {'名称':<22} {'事件数':>5} {'TP':>4} {'FP':>4} "
          f"{'FN':>4} {'Prec':>7} {'Rec':>7} {'F1':>7} {'耗时':>6}")
    print(f"  {'-'*62}")

    summary_rows = []
    for ch in channels:
        r = results.get(ch, {})
        count = r.get("count", 0)
        elapsed = r.get("elapsed_s", 0)
        m = r.get("metrics")
        ch_name = CHANNEL_NAMES.get(ch, ch)

        if m:
            row = {
                "channel": ch,
                "name": ch_name,
                "count": count,
                "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "elapsed_s": elapsed,
                "missed": m.get("missed", []),
                "false_alarms": m.get("false_alarms", []),
                "matched": m.get("matched", []),
            }
            print(f"  {ch:<6} {ch_name:<22} {count:>5} {m['tp']:>4} {m['fp']:>4} "
                  f"{m['fn']:>4} {m['precision']:>7.1%} {m['recall']:>7.1%} "
                  f"{m['f1']:>7.1%} {elapsed:>5.1f}s")
        else:
            row = {
                "channel": ch, "name": ch_name, "count": count,
                "elapsed_s": elapsed,
                "tp": None, "fp": None, "fn": None,
                "precision": None, "recall": None, "f1": None,
            }
            print(f"  {ch:<6} {ch_name:<22} {count:>5} {'—':>4} {'—':>4} "
                  f"{'—':>4} {'—':>7} {'—':>7} {'—':>7} {elapsed:>5.1f}s")
        summary_rows.append(row)

    print(f"  {'='*62}")

    # 详细漏检/误报
    has_metrics = any(r.get("tp") is not None for r in summary_rows)
    if has_metrics:
        print()
        for r in summary_rows:
            if r.get("tp") is None:
                continue
            ch = r["channel"]
            missed = r.get("missed", [])
            false_alarms = r.get("false_alarms", [])
            matched = r.get("matched", [])
            if missed or false_alarms:
                print(f"  [通道{ch}] ", end="")
                parts = []
                if matched:
                    parts.append(f"命中 {len(matched)} 个")
                if missed:
                    ts_str = ", ".join(f"{t:.1f}s" for t in missed)
                    parts.append(f"漏检 {len(missed)} 个 ({ts_str})")
                if false_alarms:
                    ts_str = ", ".join(f"{t:.1f}s" for t in false_alarms)
                    parts.append(f"误报 {len(false_alarms)} 个 ({ts_str})")
                print("；".join(parts))

    print()
    return summary_rows


# ============================================================
# 单视频评估
# ============================================================

def eval_single(
    video_path: str,
    gt_path: str,
    ai_engine_dir: str,
    channels: str,
    with_audio: bool,
    sample_fps: float,
    confidence_threshold: float,
    yolo_confidence: float,
    output_json: str = "",
    keep_frames: bool = False,
    work_dir: str = "",
) -> Optional[List[Dict]]:

    video_name = Path(video_path).name
    log(f"开始评估: {video_name}")
    log(f"Ground Truth: {gt_path}")

    # 获取视频时长
    duration = get_video_duration(video_path)
    if duration <= 0:
        log("无法获取视频时长，退出")
        return None
    log(f"视频时长: {duration:.1f}s")

    # 工作目录
    if work_dir:
        tmp_dir = work_dir
        os.makedirs(tmp_dir, exist_ok=True)
        cleanup = False
    else:
        tmp_dir = tempfile.mkdtemp(prefix="goalcut_eval_")
        cleanup = not keep_frames

    frames_dir = os.path.join(tmp_dir, "frames")
    audio_wav = os.path.join(tmp_dir, "audio.wav")
    result_json = output_json or os.path.join(tmp_dir, "channel_result.json")

    try:
        # 1. 提取帧
        t0 = time.time()
        if not extract_frames(video_path, frames_dir, sample_fps):
            log("帧提取失败，退出")
            return None
        frame_count = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        log(f"  提取 {frame_count} 帧，耗时 {time.time()-t0:.1f}s")

        # 2. 提取音频（可选）
        audio_file = ""
        if with_audio:
            t0 = time.time()
            if extract_audio(video_path, audio_wav):
                audio_file = audio_wav
                log(f"  音频提取完成，耗时 {time.time()-t0:.1f}s")
            else:
                log("  音频提取失败，通道 5 将跳过")

        # 3. 运行 channel_test
        t0 = time.time()
        ok = run_channel_test(
            ai_engine_dir=ai_engine_dir,
            frames_dir=frames_dir,
            video_duration=duration,
            channels=channels,
            gt_path=gt_path,
            output_json=result_json,
            audio_file=audio_file,
            sample_fps=sample_fps,
            confidence_threshold=confidence_threshold,
            yolo_confidence=yolo_confidence,
        )
        log(f"  channel_test 完成，耗时 {time.time()-t0:.1f}s")

        if not ok:
            log("channel_test 执行失败")
            return None

        # 4. 渲染报告
        rows = render_report(result_json, video_name)

        if output_json and result_json != output_json:
            shutil.copy(result_json, output_json)
            log(f"详细结果已保存: {output_json}")

        return rows

    finally:
        if cleanup and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
# 批量评估
# ============================================================

def eval_batch(
    gt_dir: str,
    video_dir: str,
    ai_engine_dir: str,
    channels: str,
    with_audio: bool,
    sample_fps: float,
    confidence_threshold: float,
    yolo_confidence: float,
    output_dir: str = "",
):
    gt_files = sorted(Path(gt_dir).glob("*.json"))
    if not gt_files:
        log(f"未找到 GT 文件: {gt_dir}/*.json")
        return

    log(f"批量评估: {len(gt_files)} 个视频")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for gt_file in gt_files:
        # 从 GT 文件名推断视频文件名
        stem = gt_file.stem
        video_path = find_video_file(video_dir, stem)
        if not video_path:
            log(f"找不到对应视频: {stem}（跳过）")
            continue

        output_json = os.path.join(output_dir, f"{stem}_result.json") if output_dir else ""

        rows = eval_single(
            video_path=video_path,
            gt_path=str(gt_file),
            ai_engine_dir=ai_engine_dir,
            channels=channels,
            with_audio=with_audio,
            sample_fps=sample_fps,
            confidence_threshold=confidence_threshold,
            yolo_confidence=yolo_confidence,
            output_json=output_json,
        )
        if rows:
            all_results.append({"video": Path(video_path).name, "rows": rows})

    # 批量汇总
    if len(all_results) > 1:
        print(f"\n{'='*65}")
        print(f"  批量汇总  ({len(all_results)} 个视频)")
        print(f"{'='*65}")

        # 按通道聚合
        channel_agg: Dict[str, Dict] = {}
        for item in all_results:
            for row in item["rows"]:
                ch = row["channel"]
                if row.get("tp") is None:
                    continue
                if ch not in channel_agg:
                    channel_agg[ch] = {"name": row["name"], "tp": 0, "fp": 0, "fn": 0}
                channel_agg[ch]["tp"] += row["tp"]
                channel_agg[ch]["fp"] += row["fp"]
                channel_agg[ch]["fn"] += row["fn"]

        print(f"  {'通道':<6} {'名称':<22} {'TP':>5} {'FP':>5} {'FN':>5} "
              f"{'Prec':>7} {'Rec':>7} {'F1':>7}")
        print(f"  {'-'*60}")
        for ch, agg in sorted(channel_agg.items()):
            tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            print(f"  {ch:<6} {agg['name']:<22} {tp:>5} {fp:>5} {fn:>5} "
                  f"{prec:>7.1%} {rec:>7.1%} {f1:>7.1%}")
        print()


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="GoalCut 通道准确率评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单视频全通道评估
  python test/scripts/eval.py \\
      --video test/素材/formal_game_01.mp4 \\
      --gt test/ground_truth/formal_game_01.json

  # 只评估通道 1a 和 3
  python test/scripts/eval.py \\
      --video test/素材/formal_game_01.mp4 \\
      --gt test/ground_truth/formal_game_01.json \\
      --channels 1a,3

  # 含音频通道（5）
  python test/scripts/eval.py \\
      --video test/素材/formal_game_01.mp4 \\
      --gt test/ground_truth/formal_game_01.json \\
      --channels 1a,5 --with-audio

  # 批量评估（GT 目录下所有 JSON，视频在另一目录）
  python test/scripts/eval.py \\
      --batch test/ground_truth/ \\
      --video-dir test/素材/

  # 保存结果到 JSON
  python test/scripts/eval.py \\
      --video test/素材/formal_game_01.mp4 \\
      --gt test/ground_truth/formal_game_01.json \\
      --output /tmp/eval_result.json
        """)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--video", help="单视频模式：视频文件路径")
    mode.add_argument("--batch", metavar="GT_DIR",
                      help="批量模式：Ground Truth JSON 所在目录")

    parser.add_argument("--gt", default="",
                        help="Ground Truth JSON 路径（单视频模式必填）")
    parser.add_argument("--video-dir", default="",
                        help="批量模式下视频文件所在目录")

    parser.add_argument("--channels", default="all",
                        help="要评估的通道，逗号分隔，如 1a,1b,3（默认: all）")
    parser.add_argument("--with-audio", action="store_true",
                        help="提取音频并评估通道 5")

    parser.add_argument("--sample-fps", type=float, default=3.0,
                        help="采样帧率（默认: 3.0）")
    parser.add_argument("--confidence-threshold", type=float, default=0.55,
                        help="置信度阈值（默认: 0.55）")
    parser.add_argument("--yolo-confidence", type=float, default=0.25,
                        help="YOLO 检测置信度（默认: 0.25）")

    parser.add_argument("--ai-engine-dir", default="",
                        help="ai-engine 目录路径（不填则自动推断）")
    parser.add_argument("--output", default="",
                        help="单视频结果保存路径（JSON）")
    parser.add_argument("--output-dir", default="",
                        help="批量结果保存目录")
    parser.add_argument("--work-dir", default="",
                        help="指定临时工作目录（帧/音频存放，不填则用系统 tmp）")
    parser.add_argument("--keep-frames", action="store_true",
                        help="评估完成后保留提取的帧（默认删除）")

    args = parser.parse_args()

    # 定位 ai-engine
    ai_engine_dir = args.ai_engine_dir or find_ai_engine_dir()
    if not ai_engine_dir:
        log("错误: 找不到 ai-engine 目录，请用 --ai-engine-dir 手动指定")
        sys.exit(1)
    log(f"ai-engine 目录: {ai_engine_dir}")

    # 音频模式时自动包含通道 5
    channels = args.channels
    if args.with_audio and "5" not in channels and channels != "all":
        channels = channels + ",5"

    if args.video:
        # 单视频模式
        if not args.gt:
            log("错误: 单视频模式必须提供 --gt 参数")
            sys.exit(1)
        if not os.path.exists(args.video):
            log(f"错误: 视频文件不存在: {args.video}")
            sys.exit(1)
        if not os.path.exists(args.gt):
            log(f"错误: GT 文件不存在: {args.gt}")
            sys.exit(1)

        eval_single(
            video_path=args.video,
            gt_path=args.gt,
            ai_engine_dir=ai_engine_dir,
            channels=channels,
            with_audio=args.with_audio,
            sample_fps=args.sample_fps,
            confidence_threshold=args.confidence_threshold,
            yolo_confidence=args.yolo_confidence,
            output_json=args.output,
            keep_frames=args.keep_frames,
            work_dir=args.work_dir,
        )

    else:
        # 批量模式
        if not args.video_dir:
            log("错误: 批量模式必须提供 --video-dir 参数")
            sys.exit(1)
        if not os.path.isdir(args.batch):
            log(f"错误: GT 目录不存在: {args.batch}")
            sys.exit(1)
        if not os.path.isdir(args.video_dir):
            log(f"错误: 视频目录不存在: {args.video_dir}")
            sys.exit(1)

        eval_batch(
            gt_dir=args.batch,
            video_dir=args.video_dir,
            ai_engine_dir=ai_engine_dir,
            channels=channels,
            with_audio=args.with_audio,
            sample_fps=args.sample_fps,
            confidence_threshold=args.confidence_threshold,
            yolo_confidence=args.yolo_confidence,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
