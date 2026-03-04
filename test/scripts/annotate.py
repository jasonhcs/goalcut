#!/usr/bin/env python3
"""
GoalCut 视频进球时间戳标注工具

用于手动标注视频中每个进球的精确时间戳，输出标准 Ground Truth JSON 文件，
供 eval.py / channel_test.py 做准确率评估使用。

用法:
    python test/scripts/annotate.py <video_file>
    python test/scripts/annotate.py <video_file> -o test/ground_truth/game01.json
    python test/scripts/annotate.py <video_file> --load test/ground_truth/game01.json

交互命令:
    add <时间> [备注]   添加进球标注（支持 ss、ss.s、mm:ss、hh:mm:ss 格式）
    list               列出所有标注
    del  <编号>         删除指定编号的标注
    edit <编号> <时间>  修改指定编号的时间戳
    preview [编号]      用 ffmpeg 在标注点前后各截取 3 帧缩略图，保存到 --preview-dir
    info               重新显示视频信息
    save               保存到 JSON 并退出
    quit / q           不保存，直接退出
    help / h           显示帮助
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# ============================================================
# 视频元信息
# ============================================================

def get_video_info(video_path: str) -> Dict:
    """用 ffprobe 读取视频元信息"""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", video_path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        data = json.loads(out)
    except Exception as e:
        print(f"[错误] ffprobe 失败: {e}")
        sys.exit(1)

    info = {
        "path": video_path,
        "filename": os.path.basename(video_path),
        "duration": 0.0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "codec": "",
        "size_mb": 0.0,
    }

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            info["width"] = stream.get("width", 0)
            info["height"] = stream.get("height", 0)
            info["codec"] = stream.get("codec_name", "")
            fps_str = stream.get("r_frame_rate", "0/1")
            try:
                num, den = fps_str.split("/")
                info["fps"] = round(float(num) / float(den), 2) if float(den) else 0.0
            except Exception:
                info["fps"] = 0.0
            dur = stream.get("duration") or data.get("format", {}).get("duration", "0")
            info["duration"] = float(dur)
            break

    if info["duration"] == 0:
        fmt = data.get("format", {})
        info["duration"] = float(fmt.get("duration", 0))

    size_bytes = int(data.get("format", {}).get("size", 0))
    info["size_mb"] = round(size_bytes / 1024 / 1024, 1)

    return info


def print_video_info(info: Dict):
    dur = info["duration"]
    m, s = divmod(int(dur), 60)
    h, m = divmod(m, 60)
    dur_str = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    print("\n" + "=" * 55)
    print(f"  视频文件: {info['filename']}")
    print(f"  时长:     {dur_str}  ({dur:.1f} 秒)")
    print(f"  分辨率:   {info['width']}x{info['height']}  @{info['fps']} fps")
    print(f"  编码:     {info['codec']}  ({info['size_mb']} MB)")
    print("=" * 55)


# ============================================================
# 时间格式解析
# ============================================================

def parse_time(time_str: str) -> Optional[float]:
    """
    解析时间字符串为秒数。

    支持格式：
        12         → 12.0s
        12.5       → 12.5s
        1:23       → 83.0s  (mm:ss)
        1:23.5     → 83.5s
        1:02:34    → 3754.0s  (hh:mm:ss)
    """
    time_str = time_str.strip()

    # hh:mm:ss[.ms]
    m = re.fullmatch(r"(\d+):(\d{2}):(\d{2})(?:\.(\d+))?", time_str)
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        frac = float("0." + m.group(4)) if m.group(4) else 0.0
        return h * 3600 + mi * 60 + s + frac

    # mm:ss[.ms]
    m = re.fullmatch(r"(\d+):(\d{2})(?:\.(\d+))?", time_str)
    if m:
        mi, s = int(m.group(1)), int(m.group(2))
        frac = float("0." + m.group(3)) if m.group(3) else 0.0
        return mi * 60 + s + frac

    # 纯秒数
    m = re.fullmatch(r"\d+(?:\.\d+)?", time_str)
    if m:
        return float(time_str)

    return None


def fmt_time(seconds: float) -> str:
    """秒数转 mm:ss.s 字符串"""
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    if h:
        return f"{h:d}:{int(m):02d}:{s:05.2f}"
    return f"{int(m):02d}:{s:05.2f}"


# ============================================================
# 截图预览
# ============================================================

def extract_preview_frames(
    video_path: str,
    timestamp: float,
    output_dir: str,
    label: str,
    duration: float,
    offsets: Tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0),
) -> List[str]:
    """在 timestamp 附近截取若干帧，保存为 JPEG"""
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for offset in offsets:
        t = max(0.0, min(duration - 0.1, timestamp + offset))
        sign = "+" if offset >= 0 else ""
        fname = f"{label}_{sign}{int(offset)}s.jpg"
        out_path = os.path.join(output_dir, fname)
        cmd = [
            "ffmpeg", "-y", "-ss", str(t),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "3",
            out_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            saved.append(out_path)
        except subprocess.CalledProcessError:
            pass
    return saved


# ============================================================
# 标注数据管理
# ============================================================

class AnnotationSession:
    def __init__(self, video_info: Dict):
        self.video_info = video_info
        self.goals: List[Dict] = []  # [{"timestamp": float, "note": str}]
        self._next_id = 1

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            # 旧格式：纯时间戳列表
            for t in data:
                self.goals.append({"timestamp": float(t), "note": ""})
        elif isinstance(data, dict):
            goals_raw = data.get("goals", [])
            for g in goals_raw:
                if isinstance(g, (int, float)):
                    self.goals.append({"timestamp": float(g), "note": ""})
                elif isinstance(g, dict):
                    self.goals.append({
                        "timestamp": float(g.get("timestamp", 0)),
                        "note": g.get("note", ""),
                    })
        self.goals.sort(key=lambda g: g["timestamp"])
        print(f"  已加载 {len(self.goals)} 条标注")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "video": self.video_info["filename"],
            "duration": self.video_info["duration"],
            "total_goals": len(self.goals),
            "goals": [
                {"timestamp": round(g["timestamp"], 2), "note": g["note"]}
                for g in sorted(self.goals, key=lambda x: x["timestamp"])
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  已保存 {len(self.goals)} 条标注 → {path}")

    def add(self, timestamp: float, note: str = "") -> int:
        idx = len(self.goals) + 1
        self.goals.append({"timestamp": timestamp, "note": note})
        self.goals.sort(key=lambda g: g["timestamp"])
        return idx

    def delete(self, display_idx: int) -> bool:
        """按显示编号（1-based）删除"""
        goals_sorted = sorted(self.goals, key=lambda g: g["timestamp"])
        if display_idx < 1 or display_idx > len(goals_sorted):
            return False
        self.goals.remove(goals_sorted[display_idx - 1])
        return True

    def edit(self, display_idx: int, new_ts: float) -> bool:
        goals_sorted = sorted(self.goals, key=lambda g: g["timestamp"])
        if display_idx < 1 or display_idx > len(goals_sorted):
            return False
        target = goals_sorted[display_idx - 1]
        for g in self.goals:
            if g is target:
                g["timestamp"] = new_ts
                break
        self.goals.sort(key=lambda g: g["timestamp"])
        return True

    def list_goals(self):
        if not self.goals:
            print("  (暂无标注)")
            return
        sorted_goals = sorted(self.goals, key=lambda g: g["timestamp"])
        print(f"\n  共 {len(sorted_goals)} 个进球:")
        for i, g in enumerate(sorted_goals, 1):
            note_str = f"  # {g['note']}" if g["note"] else ""
            print(f"    [{i:2d}]  {fmt_time(g['timestamp']):12s}  "
                  f"({g['timestamp']:.2f}s){note_str}")


# ============================================================
# 帮助文本
# ============================================================

HELP_TEXT = """
可用命令:
  add  <时间> [备注]    添加进球时间戳
                         时间格式: 12  12.5  1:23  1:23.5  1:02:34
  list                   列出所有标注
  del  <编号>            删除指定编号（见 list 中的编号）
  edit <编号> <时间>     修改指定编号的时间戳
  preview [编号]         截图预览（全部或指定编号），保存到 --preview-dir
  info                   重新显示视频信息
  save                   保存 JSON 并退出
  quit / q               不保存直接退出
  help / h               显示此帮助

时间格式示例:
  12         →  0:12.00  (12 秒)
  1:23       →  1:23.00  (1 分 23 秒)
  1:23.5     →  1:23.50  (1 分 23.5 秒)
  1:02:34    →  1:02:34  (1 小时 2 分 34 秒)
"""


# ============================================================
# 主交互循环
# ============================================================

def run_interactive(session: AnnotationSession, output_path: str, preview_dir: str):
    duration = session.video_info["duration"]
    print("\n输入 help 查看命令，输入 save 保存退出。")
    print(f"视频时长: {fmt_time(duration)} ({duration:.1f}s)\n")

    while True:
        try:
            raw = input("annotate> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[中断] 未保存，退出。")
            sys.exit(0)

        if not raw:
            continue

        parts = raw.split(maxsplit=2)
        cmd = parts[0].lower()

        # ---- add ----
        if cmd == "add":
            if len(parts) < 2:
                print("  用法: add <时间> [备注]")
                continue
            ts = parse_time(parts[1])
            if ts is None:
                print(f"  [错误] 无法解析时间: {parts[1]}")
                continue
            if ts < 0 or ts > duration:
                print(f"  [错误] 时间超出视频范围 [0, {duration:.1f}s]")
                continue
            note = parts[2] if len(parts) > 2 else ""
            session.add(ts, note)
            idx = next(
                i + 1 for i, g in enumerate(
                    sorted(session.goals, key=lambda x: x["timestamp"]))
                if abs(g["timestamp"] - ts) < 0.01
            )
            print(f"  + 已添加 [{idx}] {fmt_time(ts)}  ({ts:.2f}s)"
                  + (f"  # {note}" if note else ""))

        # ---- list ----
        elif cmd == "list":
            session.list_goals()

        # ---- del ----
        elif cmd in ("del", "delete", "rm"):
            if len(parts) < 2 or not parts[1].isdigit():
                print("  用法: del <编号>")
                continue
            idx = int(parts[1])
            if session.delete(idx):
                print(f"  - 已删除编号 {idx}")
            else:
                print(f"  [错误] 编号 {idx} 不存在")

        # ---- edit ----
        elif cmd == "edit":
            if len(parts) < 3:
                print("  用法: edit <编号> <新时间>")
                continue
            if not parts[1].isdigit():
                print("  [错误] 编号必须是整数")
                continue
            idx = int(parts[1])
            ts = parse_time(parts[2])
            if ts is None:
                print(f"  [错误] 无法解析时间: {parts[2]}")
                continue
            if session.edit(idx, ts):
                print(f"  ~ 编号 {idx} 已修改为 {fmt_time(ts)}")
            else:
                print(f"  [错误] 编号 {idx} 不存在")

        # ---- preview ----
        elif cmd == "preview":
            if not preview_dir:
                print("  [提示] 请用 --preview-dir 指定截图保存目录")
                continue
            sorted_goals = sorted(session.goals, key=lambda g: g["timestamp"])
            if len(parts) >= 2 and parts[1].isdigit():
                indices = [int(parts[1]) - 1]
            else:
                indices = list(range(len(sorted_goals)))

            if not sorted_goals:
                print("  (暂无标注，请先 add)")
                continue

            for i in indices:
                if i < 0 or i >= len(sorted_goals):
                    print(f"  [错误] 编号 {i+1} 不存在")
                    continue
                g = sorted_goals[i]
                label = f"goal_{i+1:02d}_{int(g['timestamp'])}s"
                print(f"  截图 [{i+1}] {fmt_time(g['timestamp'])} → {preview_dir}/{label}_*.jpg")
                frames = extract_preview_frames(
                    session.video_info["path"],
                    g["timestamp"],
                    preview_dir,
                    label,
                    duration,
                )
                print(f"    已保存 {len(frames)} 张截图")

        # ---- info ----
        elif cmd == "info":
            print_video_info(session.video_info)

        # ---- save ----
        elif cmd == "save":
            session.save(output_path)
            sys.exit(0)

        # ---- quit ----
        elif cmd in ("quit", "q", "exit"):
            if session.goals:
                confirm = input(f"  共 {len(session.goals)} 条标注未保存，确认退出？[y/N] ").strip().lower()
                if confirm != "y":
                    continue
            print("  已退出（未保存）")
            sys.exit(0)

        # ---- help ----
        elif cmd in ("help", "h", "?"):
            print(HELP_TEXT)

        else:
            print(f"  未知命令: {cmd}，输入 help 查看帮助")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="GoalCut 视频进球时间戳标注工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 标注新视频
  python test/scripts/annotate.py test/素材/formal_game_01.mp4

  # 指定输出路径
  python test/scripts/annotate.py test/素材/formal_game_01.mp4 \\
      -o test/ground_truth/formal_game_01.json

  # 加载已有标注继续编辑
  python test/scripts/annotate.py test/素材/formal_game_01.mp4 \\
      --load test/ground_truth/formal_game_01.json

  # 启用截图预览（标注后 preview 命令生效）
  python test/scripts/annotate.py test/素材/formal_game_01.mp4 \\
      --preview-dir /tmp/goalcut_preview
        """)

    parser.add_argument("video", help="视频文件路径")
    parser.add_argument("-o", "--output", default="",
                        help="输出 JSON 路径（默认: test/ground_truth/<视频名>.json）")
    parser.add_argument("--load", default="",
                        help="加载已有的 Ground Truth JSON 文件（续标模式）")
    parser.add_argument("--preview-dir", default="",
                        help="截图预览保存目录（preview 命令使用）")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[错误] 视频文件不存在: {args.video}")
        sys.exit(1)

    # 默认输出路径
    if not args.output:
        gt_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "ground_truth",
        )
        stem = Path(args.video).stem
        args.output = os.path.join(gt_dir, f"{stem}.json")

    # 默认截图目录
    if not args.preview_dir:
        args.preview_dir = os.path.join("/tmp", "goalcut_preview")

    # 读取视频信息
    print("正在读取视频信息...")
    video_info = get_video_info(args.video)
    print_video_info(video_info)

    # 初始化标注会话
    session = AnnotationSession(video_info)

    # 加载已有标注
    if args.load:
        if not os.path.exists(args.load):
            print(f"[错误] 文件不存在: {args.load}")
            sys.exit(1)
        print(f"\n加载已有标注: {args.load}")
        session.load(args.load)
        session.list_goals()
    elif os.path.exists(args.output):
        print(f"\n检测到已有标注文件: {args.output}")
        ans = input("  是否加载续标？[Y/n] ").strip().lower()
        if ans != "n":
            session.load(args.output)
            session.list_goals()

    print(f"\n输出路径: {args.output}")
    print(f"截图目录: {args.preview_dir}  (preview 命令使用)")

    run_interactive(session, args.output, args.preview_dir)


if __name__ == "__main__":
    main()
