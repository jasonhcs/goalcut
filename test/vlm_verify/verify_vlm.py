#!/usr/bin/env python3
"""
GoalCut VLM 独立验证器 v2.0
=============================
使用视觉语言大模型 (Vision-Language Model) 对 GoalCut 剪辑出的视频片段
进行完全独立的进球验证。

v2.0 新增"全视频扫描"模式：VLM 独立扫描整段视频找出所有进球时间点，
再与算法检测结果做交叉比对，精确定位漏检(FN)和误检(FP)。

技术原理与原检测算法完全正交：
  - 原算法: YOLO检测框 + IOU追踪 + 几何规则 + 光流 + 音频频谱
  - VLM验证: 端到端视觉语义理解，zero-shot 推理

用法:
  # 全视频扫描模式（VLM 独立找出所有进球，再与算法比对）
  python3 verify_vlm.py \\
    --scan /path/to/source.mp4 \\
    --detection /path/to/detection.json \\
    --output /path/to/vlm_scan_report.json

  # 验证单个视频（需要原视频 + 检测结果）
  python3 verify_vlm.py \\
    --input /path/to/source.mp4 \\
    --detection /path/to/detection.json \\
    --output /path/to/vlm_report.json

  # 直接验证已剪辑的集锦视频
  python3 verify_vlm.py \\
    --highlight /path/to/highlight_goalcut.mp4 \\
    --detection /path/to/detection.json \\
    --output /path/to/vlm_report.json

  # 验证单个 clip 目录
  python3 verify_vlm.py \\
    --clips-dir /path/to/clips/ \\
    --output /path/to/vlm_report.json

环境变量:
  VLM_PROVIDER   - 模型提供商: openai / gemini / ollama / dashscope (默认 openai)
  VLM_MODEL      - 模型名称 (默认按 provider 自动选择)
  VLM_API_KEY    - API Key (openai/gemini/dashscope 必填)
  VLM_BASE_URL   - 自定义 API 端点 (可选，用于代理或自部署)
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ClipInfo:
    """单个片段信息"""
    index: int
    clip_path: str
    start_time: float = 0.0
    end_time: float = 0.0
    source_event_timestamp: float = 0.0
    source_event_confidence: float = 0.0
    source_event_detail: str = ""


@dataclass
class VLMJudgment:
    """VLM 对单个片段的判定"""
    is_goal: bool = False
    confidence: float = 0.0
    reasoning: str = ""
    goal_frame_index: Optional[int] = None
    error: Optional[str] = None
    latency_ms: int = 0


@dataclass
class ClipVerification:
    """单个片段的完整验证结果"""
    clip: dict = field(default_factory=dict)
    algorithm_says_goal: bool = True
    algorithm_confidence: float = 0.0
    vlm_judgment: dict = field(default_factory=dict)
    agreement: bool = False
    classification: str = ""  # TP_AGREE / FP_SUSPECT / FN_SUSPECT / TN_AGREE


@dataclass
class VerificationReport:
    """完整验证报告"""
    version: str = "2.0"
    timestamp: str = ""
    provider: str = ""
    model: str = ""
    source_video: str = ""
    total_clips: int = 0
    results: List[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    tuning_suggestions: List[str] = field(default_factory=list)


@dataclass
class ScanGoalEvent:
    """全视频扫描发现的进球事件"""
    timestamp: float  # 进球时间戳（秒）
    confidence: float  # VLM 置信度
    reasoning: str  # VLM 理由
    window_start: float  # 扫描窗口起始时间
    window_end: float  # 扫描窗口结束时间
    goal_frame_index: Optional[int] = None


# ============================================================================
# 帧采样器
# ============================================================================

class FrameSampler:
    """从视频片段中均匀采样关键帧"""

    def __init__(self, ffmpeg_path: str = "ffmpeg", num_frames: int = 8,
                 max_dimension: int = 768):
        self.ffmpeg_path = ffmpeg_path
        self.num_frames = num_frames
        self.max_dimension = max_dimension

    def sample_frames(self, video_path: str, output_dir: str) -> List[str]:
        """从视频中均匀采样帧，返回帧图片路径列表"""
        duration = self._get_duration(video_path)
        if duration <= 0:
            raise ValueError(f"无法获取视频时长: {video_path}")

        margin = min(0.1, duration * 0.05)
        timestamps = []
        for i in range(self.num_frames):
            t = margin + (duration - 2 * margin) * i / max(1, self.num_frames - 1)
            timestamps.append(min(t, duration - 0.01))

        frame_paths = []
        for i, ts in enumerate(timestamps):
            frame_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
            scale_filter = (f"scale=min({self.max_dimension}\\,iw):"
                           f"min({self.max_dimension}\\,ih):"
                           f"force_original_aspect_ratio=decrease")
            cmd = [
                self.ffmpeg_path,
                "-ss", f"{ts:.3f}",
                "-i", video_path,
                "-vframes", "1",
                "-vf", scale_filter,
                "-q:v", "2",
                "-y",
                frame_path
            ]
            try:
                subprocess.run(cmd, capture_output=True, check=True, timeout=10)
                if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                    frame_paths.append(frame_path)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"  [warn] 帧提取失败 t={ts:.2f}s: {e}", file=sys.stderr)

        return frame_paths

    def sample_frames_from_source(self, video_path: str, start: float,
                                   end: float, output_dir: str) -> List[str]:
        """从源视频的指定时间区间采样帧"""
        duration = end - start
        if duration <= 0:
            return []

        margin = min(0.05, duration * 0.03)
        timestamps = []
        for i in range(self.num_frames):
            t = start + margin + (duration - 2 * margin) * i / max(1, self.num_frames - 1)
            timestamps.append(min(t, end - 0.01))

        frame_paths = []
        for i, ts in enumerate(timestamps):
            frame_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
            scale_filter = (f"scale=min({self.max_dimension}\\,iw):"
                           f"min({self.max_dimension}\\,ih):"
                           f"force_original_aspect_ratio=decrease")
            cmd = [
                self.ffmpeg_path,
                "-ss", f"{ts:.3f}",
                "-i", video_path,
                "-vframes", "1",
                "-vf", scale_filter,
                "-q:v", "2",
                "-y",
                frame_path
            ]
            try:
                subprocess.run(cmd, capture_output=True, check=True, timeout=10)
                if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                    frame_paths.append(frame_path)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass

        return frame_paths

    def _get_duration(self, video_path: str) -> float:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            video_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    check=True, timeout=10)
            info = json.loads(result.stdout)
            return float(info.get("format", {}).get("duration", 0))
        except Exception:
            return 0.0


# ============================================================================
# VLM 客户端（支持多个 Provider）
# ============================================================================

# 审查 Prompt：让 VLM 以"篮球裁判/审片员"视角判断
REVIEW_PROMPT = """你是一位专业的篮球视频审查员。以下是从一段篮球视频中按时间顺序截取的连续帧画面（从第1帧到最后一帧按时间先后排列）。

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
  "confidence": 0.0到1.0之间的浮点数（你对判断的确信程度）,
  "reasoning": "用1-2句话简述你观察到的关键画面证据",
  "goal_frame_index": null或0到N的整数（如果是进球，进球大约发生在第几帧附近，从0开始计数；非进球时填null）
}"""

# 负样本审查 Prompt：用于验证原算法未检测区间
NEGATIVE_REVIEW_PROMPT = """你是一位专业的篮球视频审查员。以下是从一段篮球视频中按时间顺序截取的连续帧画面。

这段画面来自原始比赛视频中 **未被标记为进球** 的区间。请判断其中是否 **遗漏了** 进球事件。

请以严格的 JSON 格式回答（不要添加 markdown 代码块标记）:
{
  "is_goal": true或false,
  "confidence": 0.0到1.0之间的浮点数,
  "reasoning": "简述观察到的画面内容",
  "goal_frame_index": null或整数
}"""

# 全视频扫描 Prompt：用于滑动窗口逐段扫描
SCAN_PROMPT = """你是一位专业的篮球视频审查员。以下是从一段篮球比赛视频中按时间顺序截取的连续帧画面（时间跨度约 {window_seconds} 秒）。

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

注意：一个窗口内可能包含 0 个或 1 个进球，不会有多个。
请以严格的 JSON 格式回答（不要添加 markdown 代码块标记）:
{{
  "is_goal": true或false,
  "confidence": 0.0到1.0之间的浮点数（你对判断的确信程度，不确定时给低值）,
  "reasoning": "用1-2句话简述你观察到的关键画面证据",
  "goal_frame_index": null或0到N的整数（如果是进球，进球大约发生在第几帧附近，从0开始计数；非进球时填null）
}}"""


def encode_image_base64(image_path: str) -> str:
    """将图片编码为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class VLMClient:
    """统一的 VLM 调用接口，支持 OpenAI / Gemini / Ollama / DashScope"""

    def __init__(self, provider: str = "openai", model: str = "",
                 api_key: str = "", base_url: str = ""):
        self.provider = provider.lower()
        self.api_key = api_key or os.environ.get("VLM_API_KEY", "")
        self.base_url = base_url or os.environ.get("VLM_BASE_URL", "")

        if model:
            self.model = model
        elif os.environ.get("VLM_MODEL"):
            self.model = os.environ["VLM_MODEL"]
        else:
            defaults = {
                "openai": "gpt-4o",
                "gemini": "gemini-1.5-pro",
                "ollama": "llava:13b",
                "dashscope": "qwen-vl-max",
            }
            self.model = defaults.get(self.provider, "gpt-4o")

    def judge_clip(self, frame_paths: List[str],
                   prompt: str = REVIEW_PROMPT) -> VLMJudgment:
        """发送帧图片给 VLM，获取进球判定"""
        start_ts = time.time()
        try:
            if self.provider == "openai":
                result = self._call_openai(frame_paths, prompt)
            elif self.provider == "gemini":
                result = self._call_gemini(frame_paths, prompt)
            elif self.provider == "ollama":
                result = self._call_ollama(frame_paths, prompt)
            elif self.provider == "dashscope":
                result = self._call_dashscope(frame_paths, prompt)
            else:
                return VLMJudgment(error=f"不支持的 provider: {self.provider}")

            result.latency_ms = int((time.time() - start_ts) * 1000)
            return result
        except Exception as e:
            return VLMJudgment(
                error=str(e),
                latency_ms=int((time.time() - start_ts) * 1000)
            )

    def _call_openai(self, frame_paths: List[str], prompt: str) -> VLMJudgment:
        """调用 OpenAI GPT-4o 兼容接口"""
        import urllib.request
        import urllib.error

        base = self.base_url or "https://api.openai.com/v1"
        url = f"{base}/chat/completions"

        content = [{"type": "text", "text": prompt}]
        for fp in frame_paths:
            b64 = encode_image_base64(fp)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low"
                }
            })

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 300,
            "temperature": 0.1,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API error {e.code}: {body}")

        text = resp_data["choices"][0]["message"]["content"].strip()
        return self._parse_response(text)

    def _call_gemini(self, frame_paths: List[str], prompt: str) -> VLMJudgment:
        """调用 Google Gemini API"""
        import urllib.request
        import urllib.error

        base = self.base_url or "https://generativelanguage.googleapis.com/v1beta"
        url = f"{base}/models/{self.model}:generateContent?key={self.api_key}"

        parts = [{"text": prompt}]
        for fp in frame_paths:
            b64 = encode_image_base64(fp)
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": b64
                }
            })

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 300,
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API error {e.code}: {body}")

        text = resp_data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return self._parse_response(text)

    def _call_ollama(self, frame_paths: List[str], prompt: str) -> VLMJudgment:
        """调用本地 Ollama 服务"""
        import urllib.request

        base = self.base_url or "http://localhost:11434"
        url = f"{base}/api/chat"

        images = [encode_image_base64(fp) for fp in frame_paths]

        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": images
            }],
            "stream": False,
            "options": {"temperature": 0.1}
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))

        text = resp_data["message"]["content"].strip()
        return self._parse_response(text)

    def _call_dashscope(self, frame_paths: List[str], prompt: str) -> VLMJudgment:
        """调用阿里 DashScope (通义千问 VL) API"""
        import urllib.request
        import urllib.error

        base = self.base_url or "https://dashscope.aliyuncs.com/api/v1"
        url = f"{base}/services/aigc/multimodal-generation/generation"

        content = [{"text": prompt}]
        for fp in frame_paths:
            b64 = encode_image_base64(fp)
            content.append({"image": f"data:image/jpeg;base64,{b64}"})

        payload = {
            "model": self.model,
            "input": {
                "messages": [{
                    "role": "user",
                    "content": content
                }]
            },
            "parameters": {
                "temperature": 0.1,
                "max_tokens": 300,
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DashScope API error {e.code}: {body}")

        text = resp_data["output"]["choices"][0]["message"]["content"][0]["text"].strip()
        return self._parse_response(text)

    def _parse_response(self, text: str) -> VLMJudgment:
        """解析 VLM 的 JSON 响应"""
        cleaned = text
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1]
        if "```" in cleaned:
            cleaned = cleaned.split("```", 1)[0]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            return VLMJudgment(
                is_goal=bool(data.get("is_goal", False)),
                confidence=float(data.get("confidence", 0.0)),
                reasoning=str(data.get("reasoning", "")),
                goal_frame_index=data.get("goal_frame_index"),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            is_goal = any(kw in text.lower() for kw in
                          ["is_goal\": true", "\"is_goal\":true", "is goal: yes"])
            return VLMJudgment(
                is_goal=is_goal,
                confidence=0.5,
                reasoning=f"[解析异常，原始回复] {text[:200]}",
                error=f"JSON解析失败: {e}"
            )


# ============================================================================
# 片段提取器
# ============================================================================

class ClipExtractor:
    """从源视频中按检测结果提取各个片段"""

    def __init__(self, ffmpeg_path: str = "ffmpeg",
                 before_seconds: float = 3.0, after_seconds: float = 3.0):
        self.ffmpeg_path = ffmpeg_path
        self.before_seconds = before_seconds
        self.after_seconds = after_seconds

    def extract_clips_from_detection(
        self, video_path: str, detection_json_path: str,
        output_dir: str
    ) -> List[ClipInfo]:
        """根据检测结果 JSON，从源视频中切出各个进球片段"""
        with open(detection_json_path, "r") as f:
            events = json.load(f)

        clips = []
        for i, event in enumerate(events):
            ts = event.get("timestamp", 0)
            conf = event.get("confidence", 0)
            detail = event.get("detail", "")
            start = max(0, ts - self.before_seconds)
            end = ts + self.after_seconds

            clip_path = os.path.join(output_dir, f"clip_{i:03d}.mp4")
            cmd = [
                self.ffmpeg_path,
                "-ss", f"{start:.3f}",
                "-i", video_path,
                "-t", f"{end - start:.3f}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                "-y", clip_path
            ]
            try:
                subprocess.run(cmd, capture_output=True, check=True, timeout=30)
                if os.path.exists(clip_path):
                    clips.append(ClipInfo(
                        index=i,
                        clip_path=clip_path,
                        start_time=start,
                        end_time=end,
                        source_event_timestamp=ts,
                        source_event_confidence=conf,
                        source_event_detail=detail,
                    ))
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"  [warn] clip {i} 提取失败: {e}", file=sys.stderr)

        return clips

    def split_highlight_into_clips(
        self, highlight_path: str, detection_json_path: str,
        output_dir: str
    ) -> List[ClipInfo]:
        """从已剪辑的集锦视频中按场景切换点拆分为独立片段"""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "frame=pts_time",
            "-of", "json",
            "-f", "lavfi",
            f"movie={highlight_path},select='gt(scene\\,0.3)'"
        ]
        scene_times = [0.0]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for frame in data.get("frames", []):
                    t = float(frame.get("pts_time", 0))
                    if t > 0:
                        scene_times.append(t)
        except Exception:
            pass

        duration = self._get_duration(highlight_path)
        if duration > 0:
            scene_times.append(duration)

        events = []
        if detection_json_path and os.path.exists(detection_json_path):
            with open(detection_json_path, "r") as f:
                events = json.load(f)

        clips = []
        for i in range(len(scene_times) - 1):
            start = scene_times[i]
            end = scene_times[i + 1]
            if end - start < 0.5:
                continue

            clip_path = os.path.join(output_dir, f"clip_{i:03d}.mp4")
            cmd = [
                self.ffmpeg_path,
                "-ss", f"{start:.3f}",
                "-i", highlight_path,
                "-t", f"{end - start:.3f}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                "-y", clip_path
            ]
            try:
                subprocess.run(cmd, capture_output=True, check=True, timeout=30)
                if os.path.exists(clip_path):
                    event = events[i] if i < len(events) else {}
                    clips.append(ClipInfo(
                        index=i,
                        clip_path=clip_path,
                        start_time=start,
                        end_time=end,
                        source_event_timestamp=event.get("timestamp", 0),
                        source_event_confidence=event.get("confidence", 0),
                        source_event_detail=event.get("detail", ""),
                    ))
            except Exception:
                pass

        if not clips:
            clips.append(ClipInfo(
                index=0,
                clip_path=highlight_path,
                start_time=0,
                end_time=duration,
            ))

        return clips

    def _get_duration(self, path: str) -> float:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
               "-show_format", path]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
            return float(json.loads(r.stdout).get("format", {}).get("duration", 0))
        except Exception:
            return 0.0


# ============================================================================
# 全视频扫描引擎（v2.0 新增）
# ============================================================================

class VLMScanner:
    """
    全视频扫描引擎：用滑动窗口逐段扫描整段视频，让 VLM 独立找出所有进球时间点。

    工作流程:
      1. 将视频按滑动窗口（默认 6s 窗口，2s 步长）切片
      2. 每个窗口采样 6~8 帧交给 VLM 判断
      3. 对 VLM 判定为进球的窗口做去重合并
      4. 输出 VLM 发现的进球列表（作为"VLM Ground Truth"）
      5. 与算法检测结果交叉比对，输出 FP/FN 分析
    """

    def __init__(self, vlm_client: VLMClient, frame_sampler: FrameSampler,
                 window_seconds: float = 6.0, step_seconds: float = 2.0,
                 goal_confidence_threshold: float = 0.5,
                 merge_window: float = 4.0):
        self.vlm = vlm_client
        self.sampler = frame_sampler
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.goal_confidence_threshold = goal_confidence_threshold
        self.merge_window = merge_window

    def scan_video(self, video_path: str) -> List[ScanGoalEvent]:
        """
        全视频滑动窗口扫描，返回 VLM 发现的所有进球事件（去重后）。
        """
        duration = self.sampler._get_duration(video_path)
        if duration <= 0:
            print("  [error] 无法获取视频时长", file=sys.stderr)
            return []

        # 计算所有窗口
        windows = []
        t = 0.0
        while t + self.window_seconds <= duration + 0.5:
            w_end = min(t + self.window_seconds, duration)
            windows.append((t, w_end))
            t += self.step_seconds
        # 确保最后一段被覆盖
        if windows and windows[-1][1] < duration - 1.0:
            windows.append((max(0, duration - self.window_seconds), duration))

        total_windows = len(windows)
        print(f"\n{'='*60}")
        print(f"  VLM 全视频扫描模式")
        print(f"  视频时长: {duration:.1f}s | 窗口: {self.window_seconds}s | 步长: {self.step_seconds}s")
        print(f"  总计 {total_windows} 个扫描窗口")
        print(f"  模型: {self.vlm.provider}/{self.vlm.model}")
        print(f"{'='*60}")

        raw_goals = []
        errors = 0

        # Rate limiting: Gemini 免费层每分钟 15 次，留一些余量
        rate_limit_interval = 5.0  # 每次调用间隔 5 秒
        if self.vlm.provider == "gemini":
            rate_limit_interval = 5.0
        elif self.vlm.provider == "dashscope":
            rate_limit_interval = 1.0
        else:
            rate_limit_interval = 0.5
        max_retries = 3  # 429 错误最多重试次数

        for idx, (w_start, w_end) in enumerate(windows):
            progress = f"[{idx+1}/{total_windows}]"
            print(f"  {progress} 扫描 {w_start:.1f}~{w_end:.1f}s ...", end=" ", flush=True)

            with tempfile.TemporaryDirectory(prefix="vlm_scan_") as tmpdir:
                frame_paths = self.sampler.sample_frames_from_source(
                    video_path, w_start, w_end, tmpdir
                )

                if len(frame_paths) < 3:
                    print("(帧不足，跳过)")
                    continue

                prompt = SCAN_PROMPT.format(window_seconds=self.window_seconds)

                # 带重试的 VLM 调用
                judgment = None
                for retry in range(max_retries + 1):
                    judgment = self.vlm.judge_clip(frame_paths, prompt)
                    if judgment.error and "429" in str(judgment.error):
                        wait = rate_limit_interval * (2 ** retry)  # 指数退避
                        if retry < max_retries:
                            print(f"(限速，等待{wait:.0f}s后重试...)", end=" ", flush=True)
                            time.sleep(wait)
                        continue
                    break

                if judgment.error:
                    print(f"(错误: {judgment.error[:50]})")
                    errors += 1
                    # Rate limit: 即使出错也等待一下，避免连续触发限速
                    time.sleep(rate_limit_interval)
                    continue

                if judgment.is_goal and judgment.confidence >= self.goal_confidence_threshold:
                    # 计算进球在窗口内的大致时间
                    if judgment.goal_frame_index is not None and self.sampler.num_frames > 0:
                        frac = judgment.goal_frame_index / max(1, self.sampler.num_frames - 1)
                        goal_ts = w_start + frac * (w_end - w_start)
                    else:
                        goal_ts = (w_start + w_end) / 2.0

                    raw_goals.append(ScanGoalEvent(
                        timestamp=round(goal_ts, 2),
                        confidence=judgment.confidence,
                        reasoning=judgment.reasoning,
                        window_start=w_start,
                        window_end=w_end,
                        goal_frame_index=judgment.goal_frame_index,
                    ))
                    print(f"⚽ 进球! conf={judgment.confidence:.2f} "
                          f"t≈{goal_ts:.1f}s | {judgment.reasoning[:50]}")
                else:
                    label = f"conf={judgment.confidence:.2f}" if judgment.is_goal else "无进球"
                    print(f"- {label}")

                # Rate limit: 成功调用后也等待，避免触发频率限制
                time.sleep(rate_limit_interval)

        # 去重合并相近的事件
        merged = self._merge_scan_events(raw_goals)

        print(f"\n{'='*60}")
        print(f"  扫描完成 | 发现 {len(raw_goals)} 个原始命中 → 合并为 {len(merged)} 个进球")
        if errors:
            print(f"  VLM 调用错误: {errors} 个窗口")
        print(f"{'='*60}")

        return merged

    def _merge_scan_events(self, events: List[ScanGoalEvent]) -> List[ScanGoalEvent]:
        """对扫描结果去重合并（相邻窗口可能都检测到同一个进球）"""
        if not events:
            return []

        events.sort(key=lambda e: e.timestamp)
        merged = [events[0]]

        for evt in events[1:]:
            last = merged[-1]
            if evt.timestamp - last.timestamp < self.merge_window:
                # 合并：保留置信度更高的
                if evt.confidence > last.confidence:
                    merged[-1] = evt
            else:
                merged.append(evt)

        return merged

    def cross_compare(
        self, vlm_goals: List[ScanGoalEvent],
        detection_events: List[dict],
        tolerance: float = 3.0
    ) -> dict:
        """
        将 VLM 扫描结果与算法检测结果交叉比对。

        分类逻辑:
          - TP (True Positive):  算法检测到 + VLM 也发现了 → 确认正确
          - FP (False Positive): 算法检测到 + VLM 没发现   → 算法误检
          - FN (False Negative): 算法没检测到 + VLM 发现了  → 算法漏检
        """
        algo_timestamps = [
            {"timestamp": e.get("timestamp", 0), "confidence": e.get("confidence", 0),
             "detail": e.get("detail", "")}
            for e in detection_events
        ]
        vlm_timestamps = [
            {"timestamp": g.timestamp, "confidence": g.confidence,
             "reasoning": g.reasoning, "window": [g.window_start, g.window_end]}
            for g in vlm_goals
        ]

        # 匹配
        algo_matched = set()
        vlm_matched = set()
        tp_pairs = []

        for ai, algo in enumerate(algo_timestamps):
            best_vi = -1
            best_dist = tolerance + 1
            for vi, vlm in enumerate(vlm_timestamps):
                if vi in vlm_matched:
                    continue
                dist = abs(algo["timestamp"] - vlm["timestamp"])
                if dist <= tolerance and dist < best_dist:
                    best_dist = dist
                    best_vi = vi
            if best_vi >= 0:
                algo_matched.add(ai)
                vlm_matched.add(best_vi)
                tp_pairs.append({
                    "algo_timestamp": algo["timestamp"],
                    "vlm_timestamp": vlm_timestamps[best_vi]["timestamp"],
                    "time_diff": round(best_dist, 2),
                    "algo_confidence": algo["confidence"],
                    "vlm_confidence": vlm_timestamps[best_vi]["confidence"],
                    "vlm_reasoning": vlm_timestamps[best_vi]["reasoning"],
                })

        # FP: 算法检测到但 VLM 没发现
        false_positives = []
        for ai, algo in enumerate(algo_timestamps):
            if ai not in algo_matched:
                false_positives.append({
                    "timestamp": algo["timestamp"],
                    "algo_confidence": algo["confidence"],
                    "algo_detail": algo["detail"],
                    "diagnosis": "算法认为进球但 VLM 全视频扫描未发现该进球",
                })

        # FN: VLM 发现了但算法没检测到
        false_negatives = []
        for vi, vlm in enumerate(vlm_timestamps):
            if vi not in vlm_matched:
                false_negatives.append({
                    "timestamp": vlm["timestamp"],
                    "vlm_confidence": vlm["confidence"],
                    "vlm_reasoning": vlm["reasoning"],
                    "scan_window": vlm["window"],
                    "diagnosis": "VLM 发现进球但算法未检测到（漏检）",
                })

        # 计算指标
        tp = len(tp_pairs)
        fp = len(false_positives)
        fn = len(false_negatives)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "vlm_goals_count": len(vlm_goals),
            "algo_detections_count": len(detection_events),
            "metrics": {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
            },
            "matched_pairs": tp_pairs,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "tuning_suggestions": self._generate_scan_suggestions(
                tp_pairs, false_positives, false_negatives, detection_events
            ),
        }

    def _generate_scan_suggestions(
        self, tp_pairs: List[dict], fps: List[dict],
        fns: List[dict], detection_events: List[dict]
    ) -> List[str]:
        """根据扫描比对结果生成调参建议"""
        suggestions = []

        if not fps and not fns:
            suggestions.append(
                "VLM 扫描与算法检测结果完全一致，当前算法表现优秀"
            )
            return suggestions

        # FP 分析
        if fps:
            low_conf_fps = [f for f in fps if f["algo_confidence"] < 0.60]
            high_conf_fps = [f for f in fps if f["algo_confidence"] >= 0.70]

            suggestions.append(
                f"算法存在 {len(fps)} 个误检（VLM 扫描未发现对应进球）"
            )
            if low_conf_fps:
                max_fp_conf = max(f["algo_confidence"] for f in low_conf_fps)
                suggestions.append(
                    f"其中 {len(low_conf_fps)} 个误检的算法置信度 < 0.60，"
                    f"建议将 confidence_threshold 上调至 {max_fp_conf + 0.05:.2f}"
                )
            if high_conf_fps:
                suggestions.append(
                    f"⚠ {len(high_conf_fps)} 个误检的算法置信度 >= 0.70，"
                    f"说明算法逻辑层面可能存在问题，不能仅靠调阈值解决"
                )

            # 分析来源通道
            channel_fps: Dict[str, int] = {}
            for f in fps:
                detail = f.get("algo_detail", "")
                for ch in ["1a", "1b", "2", "3", "4", "5"]:
                    if f"ch{ch}" in detail.lower() or f"channel_{ch}" in detail.lower():
                        channel_fps[ch] = channel_fps.get(ch, 0) + 1
            for ch, count in sorted(channel_fps.items(), key=lambda x: -x[1]):
                if count >= 2:
                    suggestions.append(
                        f"通道 {ch} 产生了 {count} 个误检，建议审查该通道阈值"
                    )

        # FN 分析
        if fns:
            suggestions.append(
                f"算法存在 {len(fns)} 个漏检（VLM 发现了进球但算法未检测到）"
            )
            fn_times = [f["timestamp"] for f in fns]
            suggestions.append(
                f"漏检时间点: {', '.join(f'{t:.1f}s' for t in fn_times)}"
            )
            # 分析漏检区间的 VLM 置信度
            high_conf_fns = [f for f in fns if f["vlm_confidence"] >= 0.8]
            if high_conf_fns:
                suggestions.append(
                    f"其中 {len(high_conf_fns)} 个漏检的 VLM 置信度 >= 0.80，"
                    f"属于高确信漏检，建议重点排查这些时间段的检测通道覆盖"
                )
            suggestions.append(
                "建议: 适当降低 confidence_threshold 或增强弱通道灵敏度以减少漏检"
            )

        # 总体评价
        total = len(tp_pairs) + len(fps) + len(fns)
        if total > 0:
            tp_rate = len(tp_pairs) / total
            if tp_rate >= 0.85:
                suggestions.append(f"总体一致率 {tp_rate:.0%}，算法表现良好")
            elif tp_rate >= 0.70:
                suggestions.append(f"总体一致率 {tp_rate:.0%}，有优化空间")
            else:
                suggestions.append(f"总体一致率 {tp_rate:.0%}，算法需要重点优化")

        return suggestions


# ============================================================================
# 原有验证引擎
# ============================================================================

class VLMVerifier:
    """核心验证引擎：采样帧 → VLM 审查 → 交叉比对 → 生成报告"""

    def __init__(self, vlm_client: VLMClient, frame_sampler: FrameSampler,
                 ffmpeg_path: str = "ffmpeg"):
        self.vlm = vlm_client
        self.sampler = frame_sampler
        self.ffmpeg_path = ffmpeg_path

    def verify_clips(self, clips: List[ClipInfo],
                     source_video: str = "") -> VerificationReport:
        """验证一组片段"""
        report = VerificationReport(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            provider=self.vlm.provider,
            model=self.vlm.model,
            source_video=source_video,
            total_clips=len(clips),
        )

        print(f"\n{'='*60}")
        print(f"  VLM 验证开始 | 模型: {self.vlm.provider}/{self.vlm.model}")
        print(f"  共 {len(clips)} 个片段待验证")
        print(f"{'='*60}")

        for clip in clips:
            print(f"\n  [{clip.index+1}/{len(clips)}] 验证片段 "
                  f"t={clip.source_event_timestamp:.1f}s "
                  f"conf={clip.source_event_confidence:.2f} ...")

            result = self._verify_single_clip(clip)
            report.results.append(result)

            j = result["vlm_judgment"]
            status = "✓ 一致" if result["agreement"] else "✗ 冲突"
            print(f"    VLM: is_goal={j['is_goal']}, conf={j['confidence']:.2f} "
                  f"| {status} [{result['classification']}]")
            if j.get("reasoning"):
                print(f"    理由: {j['reasoning'][:80]}")

        report.summary = self._compute_summary(report.results)
        report.tuning_suggestions = self._generate_suggestions(report.results)

        self._print_summary(report)
        return report

    def verify_negative_samples(
        self, source_video: str, detection_events: List[dict],
        video_duration: float, num_samples: int = 3,
        before_seconds: float = 3.0, after_seconds: float = 1.0
    ) -> List[dict]:
        """随机采样原视频中未被检测为进球的区间，交给 VLM 审查是否有遗漏。"""
        import random

        excluded = []
        for evt in detection_events:
            ts = evt.get("timestamp", 0)
            excluded.append((
                max(0, ts - before_seconds - 2),
                ts + after_seconds + 2
            ))

        available = []
        prev_end = 0
        for start, end in sorted(excluded):
            if start > prev_end + 4:
                available.append((prev_end, start))
            prev_end = max(prev_end, end)
        if video_duration > prev_end + 4:
            available.append((prev_end, video_duration))

        if not available:
            print("  [info] 无可用的负样本区间")
            return []

        results = []
        samples = []
        for _ in range(min(num_samples, len(available) * 2)):
            seg = random.choice(available)
            seg_dur = seg[1] - seg[0]
            if seg_dur < 4:
                continue
            clip_dur = min(4, seg_dur)
            t_start = seg[0] + random.random() * (seg_dur - clip_dur)
            samples.append((t_start, t_start + clip_dur))

        print(f"\n  负样本审查: 随机抽取 {len(samples)} 个未检测区间")

        for i, (t_start, t_end) in enumerate(samples):
            with tempfile.TemporaryDirectory(prefix="vlm_neg_") as tmpdir:
                frame_paths = self.sampler.sample_frames_from_source(
                    source_video, t_start, t_end, tmpdir
                )
                if len(frame_paths) < 3:
                    continue

                judgment = self.vlm.judge_clip(frame_paths, NEGATIVE_REVIEW_PROMPT)
                result = {
                    "sample_index": i,
                    "time_range": {"start": round(t_start, 2), "end": round(t_end, 2)},
                    "vlm_judgment": asdict(judgment),
                }
                results.append(result)

                status = "⚠ 发现遗漏!" if judgment.is_goal else "✓ 确认无进球"
                print(f"    [{i+1}] {t_start:.1f}~{t_end:.1f}s: "
                      f"{status} (conf={judgment.confidence:.2f})")

        return results

    def _verify_single_clip(self, clip: ClipInfo) -> dict:
        """验证单个片段"""
        with tempfile.TemporaryDirectory(prefix="vlm_verify_") as tmpdir:
            frame_paths = self.sampler.sample_frames(clip.clip_path, tmpdir)

            if len(frame_paths) < 3:
                judgment = VLMJudgment(error="帧采样不足", confidence=0.0)
            else:
                judgment = self.vlm.judge_clip(frame_paths)

        algo_says_goal = True
        vlm_says_goal = judgment.is_goal and judgment.confidence >= 0.4

        if algo_says_goal and vlm_says_goal:
            classification = "TP_AGREE"
            agreement = True
        elif algo_says_goal and not vlm_says_goal:
            classification = "FP_SUSPECT"
            agreement = False
        elif not algo_says_goal and vlm_says_goal:
            classification = "FN_SUSPECT"
            agreement = False
        else:
            classification = "TN_AGREE"
            agreement = True

        return {
            "clip": asdict(clip),
            "algorithm_says_goal": algo_says_goal,
            "algorithm_confidence": clip.source_event_confidence,
            "vlm_judgment": asdict(judgment),
            "agreement": agreement,
            "classification": classification,
        }

    def _compute_summary(self, results: List[dict]) -> dict:
        """计算验证汇总统计"""
        total = len(results)
        if total == 0:
            return {"total": 0, "agreement_rate": 0}

        agree = sum(1 for r in results if r["agreement"])
        tp_agree = sum(1 for r in results if r["classification"] == "TP_AGREE")
        fp_suspect = sum(1 for r in results if r["classification"] == "FP_SUSPECT")
        fn_suspect = sum(1 for r in results if r["classification"] == "FN_SUSPECT")
        errors = sum(1 for r in results if r["vlm_judgment"].get("error"))

        valid_confs = [
            r["vlm_judgment"]["confidence"]
            for r in results if not r["vlm_judgment"].get("error")
        ]
        avg_vlm_conf = sum(valid_confs) / len(valid_confs) if valid_confs else 0

        algo_confs = [r["algorithm_confidence"] for r in results]
        avg_algo_conf = sum(algo_confs) / len(algo_confs) if algo_confs else 0

        fp_details = [r for r in results if r["classification"] == "FP_SUSPECT"]

        return {
            "total_clips": total,
            "agreement_count": agree,
            "agreement_rate": round(agree / total, 3) if total else 0,
            "tp_agree": tp_agree,
            "fp_suspect": fp_suspect,
            "fn_suspect": fn_suspect,
            "vlm_errors": errors,
            "avg_vlm_confidence": round(avg_vlm_conf, 3),
            "avg_algorithm_confidence": round(avg_algo_conf, 3),
            "suspected_false_positives": [
                {
                    "clip_index": r["clip"]["index"],
                    "timestamp": r["clip"]["source_event_timestamp"],
                    "algo_confidence": r["algorithm_confidence"],
                    "vlm_confidence": r["vlm_judgment"]["confidence"],
                    "vlm_reasoning": r["vlm_judgment"]["reasoning"],
                    "algo_detail": r["clip"]["source_event_detail"],
                }
                for r in fp_details
            ],
        }

    def _generate_suggestions(self, results: List[dict]) -> List[str]:
        """根据验证结果生成调参建议"""
        suggestions = []

        fp_results = [r for r in results if r["classification"] == "FP_SUSPECT"]
        fn_results = [r for r in results if r["classification"] == "FN_SUSPECT"]
        total = len(results)

        if not total:
            return suggestions

        fp_rate = len(fp_results) / total
        fn_rate = len(fn_results) / total

        if fp_rate > 0.2:
            fp_confs = [r["algorithm_confidence"] for r in fp_results]
            max_fp_conf = max(fp_confs) if fp_confs else 0
            suggestions.append(
                f"误检率偏高({fp_rate:.0%})，建议将 confidence_threshold "
                f"从当前值上调至 {max_fp_conf + 0.05:.2f} 以过滤低置信度误检"
            )

        channel_fp_count: Dict[str, int] = {}
        for r in fp_results:
            detail = r["clip"].get("source_event_detail", "")
            for ch in ["1a", "1b", "2", "3", "4", "5"]:
                if f"ch{ch}" in detail.lower() or f"channel_{ch}" in detail.lower():
                    channel_fp_count[ch] = channel_fp_count.get(ch, 0) + 1

        for ch, count in sorted(channel_fp_count.items(), key=lambda x: -x[1]):
            if count >= 2:
                suggestions.append(
                    f"通道 {ch} 产生了 {count} 个疑似误检，"
                    f"建议审查该通道的阈值或降低其跨通道权重"
                )

        low_conf_fp = [r for r in fp_results if r["algorithm_confidence"] < 0.60]
        if low_conf_fp:
            suggestions.append(
                f"{len(low_conf_fp)} 个疑似误检的原算法置信度低于 0.60，"
                f"提高 confidence_threshold 到 0.60 可直接消除"
            )

        if fn_rate > 0.1:
            suggestions.append(
                f"在负样本审查中发现 {len(fn_results)} 个疑似漏检，"
                f"建议降低 confidence_threshold 或增强弱通道灵敏度"
            )

        agree_rate = sum(1 for r in results if r["agreement"]) / total
        if agree_rate >= 0.9:
            suggestions.append(
                f"VLM 与原算法一致性达 {agree_rate:.0%}，当前算法表现良好"
            )

        return suggestions

    def _print_summary(self, report: VerificationReport):
        """打印验证摘要"""
        s = report.summary
        print(f"\n{'='*60}")
        print(f"  VLM 验证报告摘要")
        print(f"{'='*60}")
        print(f"  总片段数:       {s.get('total_clips', 0)}")
        print(f"  一致确认:       {s.get('agreement_count', 0)} "
              f"({s.get('agreement_rate', 0):.1%})")
        print(f"  疑似误检(FP):   {s.get('fp_suspect', 0)}")
        print(f"  疑似漏检(FN):   {s.get('fn_suspect', 0)}")
        print(f"  VLM 调用错误:   {s.get('vlm_errors', 0)}")
        print(f"  VLM 平均置信度: {s.get('avg_vlm_confidence', 0):.2f}")
        print(f"  算法平均置信度: {s.get('avg_algorithm_confidence', 0):.2f}")

        if report.tuning_suggestions:
            print(f"\n  调参建议:")
            for i, sug in enumerate(report.tuning_suggestions, 1):
                print(f"    {i}. {sug}")

        fps = s.get("suspected_false_positives", [])
        if fps:
            print(f"\n  疑似误检详情:")
            for fp in fps:
                print(f"    - clip#{fp['clip_index']} t={fp['timestamp']:.1f}s "
                      f"algo_conf={fp['algo_confidence']:.2f} "
                      f"vlm_conf={fp['vlm_confidence']:.2f}")
                print(f"      VLM: {fp['vlm_reasoning'][:80]}")

        print(f"{'='*60}\n")


# ============================================================================
# Ground Truth 对比（可选，有 GT 时输出精确指标）
# ============================================================================

def compare_with_ground_truth(
    report: VerificationReport,
    ground_truth_path: str,
    tolerance: float = 3.0
) -> dict:
    """将 VLM 验证结果与 Ground Truth 对比，计算精确的 P/R/F1"""
    with open(ground_truth_path, "r") as f:
        gt = json.load(f)

    gt_timestamps = [g["timestamp"] for g in gt.get("goals", [])]
    det_timestamps = [
        r["clip"]["source_event_timestamp"]
        for r in report.results
    ]
    vlm_confirmed = [
        r["clip"]["source_event_timestamp"]
        for r in report.results
        if r["vlm_judgment"]["is_goal"] and r["vlm_judgment"]["confidence"] >= 0.5
    ]

    def match(dets, gts, tol):
        tp, matched = 0, set()
        for d in sorted(dets):
            for i, g in enumerate(gts):
                if i not in matched and abs(d - g) <= tol:
                    tp += 1
                    matched.add(i)
                    break
        fp = len(dets) - tp
        fn = len(gts) - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return {"tp": tp, "fp": fp, "fn": fn,
                "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}

    return {
        "ground_truth_goals": len(gt_timestamps),
        "algorithm_metrics": match(det_timestamps, gt_timestamps, tolerance),
        "vlm_filtered_metrics": match(vlm_confirmed, gt_timestamps, tolerance),
        "note": "vlm_filtered_metrics 表示仅保留 VLM 确认的检测后的指标"
    }


# ============================================================================
# CLI 主入口
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="GoalCut VLM 独立验证器 v2.0 - 支持全视频扫描和片段验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 全视频扫描模式（VLM 独立发现所有进球，与算法比对）
  python3 verify_vlm.py --scan video.mp4 --detection detection.json

  # 验证源视频 + 检测结果
  python3 verify_vlm.py --input video.mp4 --detection detection.json

  # 直接验证已剪辑的集锦
  python3 verify_vlm.py --highlight video_goalcut.mp4 --detection detection.json

  # 使用 Ollama 本地模型
  VLM_PROVIDER=ollama VLM_MODEL=llava:13b python3 verify_vlm.py --scan video.mp4

  # 指定 API Key
  VLM_API_KEY=sk-xxx python3 verify_vlm.py --scan video.mp4 --detection detection.json

环境变量:
  VLM_PROVIDER   openai / gemini / ollama / dashscope (默认 openai)
  VLM_MODEL      模型名称 (默认按 provider 自动选择)
  VLM_API_KEY    API Key
  VLM_BASE_URL   自定义 API 端点
        """
    )

    # 输入模式（四选一）
    input_group = parser.add_argument_group("输入（四选一）")
    input_group.add_argument("--scan", "-s", metavar="VIDEO",
                             help="全视频扫描模式: VLM 独立扫描整段视频找出所有进球，"
                                  "再与算法检测结果比对")
    input_group.add_argument("--input", "-i", metavar="VIDEO",
                             help="源视频路径（将根据 detection.json 自动切出片段）")
    input_group.add_argument("--highlight", metavar="VIDEO",
                             help="已剪辑的集锦视频路径（自动按场景拆分）")
    input_group.add_argument("--clips-dir", metavar="DIR",
                             help="已切好的片段目录（直接验证）")

    # 检测结果
    parser.add_argument("--detection", "-d", metavar="JSON",
                        help="AI 检测结果 JSON 文件路径")

    # 输出
    parser.add_argument("--output", "-o", metavar="JSON",
                        default="vlm_report.json",
                        help="验证报告输出路径 (默认 vlm_report.json)")

    # Ground Truth（可选）
    parser.add_argument("--ground-truth", "--gt", metavar="JSON",
                        help="Ground Truth JSON 路径（可选，用于计算精确 P/R/F1）")

    # VLM 配置
    vlm_group = parser.add_argument_group("VLM 配置（也可通过环境变量设置）")
    vlm_group.add_argument("--provider", default="",
                           help="VLM 提供商 (覆盖 VLM_PROVIDER 环境变量)")
    vlm_group.add_argument("--model", default="",
                           help="模型名称 (覆盖 VLM_MODEL 环境变量)")
    vlm_group.add_argument("--api-key", default="",
                           help="API Key (覆盖 VLM_API_KEY 环境变量)")
    vlm_group.add_argument("--base-url", default="",
                           help="自定义 API 端点 (覆盖 VLM_BASE_URL 环境变量)")

    # 采样配置
    sample_group = parser.add_argument_group("采样配置")
    sample_group.add_argument("--num-frames", type=int, default=8,
                              help="每个片段/窗口采样帧数 (默认 8)")
    sample_group.add_argument("--max-dimension", type=int, default=768,
                              help="帧图片最大边长 (默认 768)")

    # 扫描配置
    scan_group = parser.add_argument_group("扫描配置（仅 --scan 模式）")
    scan_group.add_argument("--scan-window", type=float, default=6.0,
                            help="扫描窗口大小（秒，默认 6.0）")
    scan_group.add_argument("--scan-step", type=float, default=2.0,
                            help="扫描步长（秒，默认 2.0）")
    scan_group.add_argument("--scan-threshold", type=float, default=0.5,
                            help="扫描模式 VLM 进球判定最低置信度 (默认 0.5)")
    scan_group.add_argument("--scan-merge-window", type=float, default=4.0,
                            help="扫描结果去重合并窗口（秒，默认 4.0）")

    # 高级选项
    adv_group = parser.add_argument_group("高级选项")
    adv_group.add_argument("--before-seconds", type=float, default=3.0,
                           help="进球前截取秒数 (默认 3.0)")
    adv_group.add_argument("--after-seconds", type=float, default=1.0,
                           help="进球后截取秒数 (默认 1.0)")
    adv_group.add_argument("--check-negatives", action="store_true",
                           help="启用负样本审查（检测漏检，需要 --input）")
    adv_group.add_argument("--negative-samples", type=int, default=3,
                           help="负样本抽检数量 (默认 3)")
    adv_group.add_argument("--ffmpeg-path", default="ffmpeg",
                           help="ffmpeg 路径 (默认 ffmpeg)")
    adv_group.add_argument("--tolerance", type=float, default=3.0,
                           help="事件匹配容差秒数 (默认 3.0)")

    return parser.parse_args()


def _print_scan_report(comparison: dict):
    """打印扫描比对报告"""
    m = comparison["metrics"]
    print(f"\n{'='*60}")
    print(f"  VLM 全视频扫描比对报告")
    print(f"{'='*60}")
    print(f"  VLM 发现进球:   {comparison['vlm_goals_count']}")
    print(f"  算法检测进球:   {comparison['algo_detections_count']}")
    print(f"  匹配(TP):       {m['tp']}")
    print(f"  算法误检(FP):   {m['fp']}")
    print(f"  算法漏检(FN):   {m['fn']}")
    print(f"  算法 Precision: {m['precision']:.1%}")
    print(f"  算法 Recall:    {m['recall']:.1%}")
    print(f"  算法 F1:        {m['f1']:.1%}")

    if comparison["matched_pairs"]:
        print(f"\n  匹配详情:")
        for p in comparison["matched_pairs"]:
            print(f"    ✓ algo={p['algo_timestamp']:.1f}s ↔ vlm={p['vlm_timestamp']:.1f}s "
                  f"(Δ={p['time_diff']:.1f}s) "
                  f"algo_conf={p['algo_confidence']:.2f} vlm_conf={p['vlm_confidence']:.2f}")

    if comparison["false_positives"]:
        print(f"\n  算法误检(FP):")
        for f in comparison["false_positives"]:
            print(f"    ✗ t={f['timestamp']:.1f}s algo_conf={f['algo_confidence']:.2f}")
            print(f"      {f['diagnosis']}")

    if comparison["false_negatives"]:
        print(f"\n  算法漏检(FN):")
        for f in comparison["false_negatives"]:
            print(f"    ⚠ t={f['timestamp']:.1f}s vlm_conf={f['vlm_confidence']:.2f} "
                  f"window=[{f['scan_window'][0]:.1f}~{f['scan_window'][1]:.1f}s]")
            print(f"      VLM: {f['vlm_reasoning'][:80]}")

    if comparison["tuning_suggestions"]:
        print(f"\n  调参建议:")
        for i, sug in enumerate(comparison["tuning_suggestions"], 1):
            print(f"    {i}. {sug}")

    print(f"{'='*60}\n")


def main():
    args = parse_args()

    # 验证输入参数
    if not args.scan and not args.input and not args.highlight and not args.clips_dir:
        print("错误: 必须指定 --scan、--input、--highlight 或 --clips-dir 之一",
              file=sys.stderr)
        sys.exit(1)

    # 初始化 VLM 客户端
    provider = args.provider or os.environ.get("VLM_PROVIDER", "openai")
    vlm_client = VLMClient(
        provider=provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    # 初始化帧采样器
    frame_sampler = FrameSampler(
        ffmpeg_path=args.ffmpeg_path,
        num_frames=args.num_frames,
        max_dimension=args.max_dimension,
    )

    # ================================================================
    # 全视频扫描模式（v2.0 新增）
    # ================================================================
    if args.scan:
        scanner = VLMScanner(
            vlm_client=vlm_client,
            frame_sampler=frame_sampler,
            window_seconds=args.scan_window,
            step_seconds=args.scan_step,
            goal_confidence_threshold=args.scan_threshold,
            merge_window=args.scan_merge_window,
        )

        # 第一步：VLM 全视频扫描
        vlm_goals = scanner.scan_video(args.scan)

        # 第二步：与算法检测结果交叉比对
        comparison = None
        if args.detection and os.path.exists(args.detection):
            with open(args.detection, "r") as f:
                detection_events = json.load(f)

            comparison = scanner.cross_compare(
                vlm_goals, detection_events, tolerance=args.tolerance
            )
            _print_scan_report(comparison)
        else:
            print(f"\n  VLM 独立发现 {len(vlm_goals)} 个进球:")
            for i, g in enumerate(vlm_goals, 1):
                print(f"    {i}. t={g.timestamp:.1f}s conf={g.confidence:.2f} "
                      f"| {g.reasoning[:60]}")
            if not args.detection:
                print(f"\n  [提示] 未提供 --detection，无法进行算法比对。")
                print(f"  如需比对，请加: --detection /path/to/detection.json")

        # 组装报告
        scan_report = {
            "version": "2.0",
            "mode": "scan",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provider": vlm_client.provider,
            "model": vlm_client.model,
            "source_video": args.scan,
            "scan_config": {
                "window_seconds": args.scan_window,
                "step_seconds": args.scan_step,
                "goal_confidence_threshold": args.scan_threshold,
                "merge_window": args.scan_merge_window,
                "num_frames": args.num_frames,
            },
            "vlm_goals": [
                {
                    "index": i,
                    "timestamp": g.timestamp,
                    "confidence": g.confidence,
                    "reasoning": g.reasoning,
                    "window": [g.window_start, g.window_end],
                    "goal_frame_index": g.goal_frame_index,
                }
                for i, g in enumerate(vlm_goals)
            ],
        }
        if comparison:
            scan_report["comparison"] = comparison

        # Ground Truth 比对（如果提供）
        if args.ground_truth and os.path.exists(args.ground_truth):
            with open(args.ground_truth, "r") as f:
                gt = json.load(f)
            gt_timestamps = [g["timestamp"] for g in gt.get("goals", [])]
            vlm_ts = [g.timestamp for g in vlm_goals]

            def match_ts(dets, gts, tol):
                tp_count, matched = 0, set()
                for d in sorted(dets):
                    for gi, g in enumerate(gts):
                        if gi not in matched and abs(d - g) <= tol:
                            tp_count += 1
                            matched.add(gi)
                            break
                fp_count = len(dets) - tp_count
                fn_count = len(gts) - tp_count
                p = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
                r = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
                f1_score = 2 * p * r / (p + r) if (p + r) > 0 else 0
                return {"tp": tp_count, "fp": fp_count, "fn": fn_count,
                        "precision": round(p, 3), "recall": round(r, 3),
                        "f1": round(f1_score, 3)}

            gt_comp = {
                "ground_truth_goals": len(gt_timestamps),
                "vlm_scan_metrics": match_ts(vlm_ts, gt_timestamps, args.tolerance),
            }
            if comparison:
                algo_ts = [e.get("timestamp", 0) for e in detection_events]
                gt_comp["algorithm_metrics"] = match_ts(
                    algo_ts, gt_timestamps, args.tolerance)

            scan_report["ground_truth_comparison"] = gt_comp
            print(f"\n  Ground Truth 对比:")
            vlm_m = gt_comp["vlm_scan_metrics"]
            print(f"    VLM 扫描:  P={vlm_m['precision']:.1%} "
                  f"R={vlm_m['recall']:.1%} F1={vlm_m['f1']:.1%}")
            if "algorithm_metrics" in gt_comp:
                algo_m = gt_comp["algorithm_metrics"]
                print(f"    算法检测:  P={algo_m['precision']:.1%} "
                      f"R={algo_m['recall']:.1%} F1={algo_m['f1']:.1%}")

        # 写入报告
        output_path = args.output
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(scan_report, f, ensure_ascii=False, indent=2)
        print(f"\n  报告已保存: {output_path}")

        # 退出码
        if comparison:
            if comparison["metrics"]["fp"] > 0 or comparison["metrics"]["fn"] > 0:
                sys.exit(1)
        return

    # ================================================================
    # 原有片段验证模式
    # ================================================================
    verifier = VLMVerifier(vlm_client, frame_sampler, args.ffmpeg_path)

    clips = []
    source_video = ""
    temp_dirs = []

    try:
        if args.clips_dir:
            clip_files = sorted([
                f for f in os.listdir(args.clips_dir)
                if f.endswith((".mp4", ".MP4", ".avi", ".mov"))
            ])
            detection_events = []
            if args.detection and os.path.exists(args.detection):
                with open(args.detection, "r") as f:
                    detection_events = json.load(f)

            for i, cf in enumerate(clip_files):
                evt = detection_events[i] if i < len(detection_events) else {}
                clips.append(ClipInfo(
                    index=i,
                    clip_path=os.path.join(args.clips_dir, cf),
                    source_event_timestamp=evt.get("timestamp", 0),
                    source_event_confidence=evt.get("confidence", 0),
                    source_event_detail=evt.get("detail", ""),
                ))
            source_video = args.clips_dir

        elif args.input:
            if not args.detection:
                print("错误: --input 模式需要 --detection 参数", file=sys.stderr)
                sys.exit(1)

            source_video = args.input
            tmpdir = tempfile.mkdtemp(prefix="vlm_clips_")
            temp_dirs.append(tmpdir)

            extractor = ClipExtractor(
                ffmpeg_path=args.ffmpeg_path,
                before_seconds=args.before_seconds,
                after_seconds=args.after_seconds,
            )
            clips = extractor.extract_clips_from_detection(
                args.input, args.detection, tmpdir
            )

        elif args.highlight:
            source_video = args.highlight
            tmpdir = tempfile.mkdtemp(prefix="vlm_clips_")
            temp_dirs.append(tmpdir)

            extractor = ClipExtractor(ffmpeg_path=args.ffmpeg_path)
            clips = extractor.split_highlight_into_clips(
                args.highlight, args.detection, tmpdir
            )

        if not clips:
            print("错误: 未找到任何可验证的片段", file=sys.stderr)
            sys.exit(1)

        report = verifier.verify_clips(clips, source_video)

        # 负样本审查（可选）
        negative_results = []
        if args.check_negatives and args.input and args.detection:
            detection_events = []
            with open(args.detection, "r") as f:
                detection_events = json.load(f)

            duration = frame_sampler._get_duration(args.input)
            negative_results = verifier.verify_negative_samples(
                args.input, detection_events, duration,
                num_samples=args.negative_samples,
                before_seconds=args.before_seconds,
                after_seconds=args.after_seconds,
            )

        # Ground Truth 对比（可选）
        gt_comparison = None
        if args.ground_truth and os.path.exists(args.ground_truth):
            gt_comparison = compare_with_ground_truth(
                report, args.ground_truth, args.tolerance
            )
            print(f"\n  Ground Truth 对比:")
            algo_m = gt_comparison["algorithm_metrics"]
            vlm_m = gt_comparison["vlm_filtered_metrics"]
            print(f"    原算法:     P={algo_m['precision']:.1%} "
                  f"R={algo_m['recall']:.1%} F1={algo_m['f1']:.1%}")
            print(f"    VLM过滤后:  P={vlm_m['precision']:.1%} "
                  f"R={vlm_m['recall']:.1%} F1={vlm_m['f1']:.1%}")

        # 组装最终报告
        final_report = {
            "version": report.version,
            "mode": "verify",
            "timestamp": report.timestamp,
            "provider": report.provider,
            "model": report.model,
            "source_video": report.source_video,
            "total_clips": report.total_clips,
            "results": report.results,
            "summary": report.summary,
            "tuning_suggestions": report.tuning_suggestions,
        }
        if negative_results:
            final_report["negative_sample_checks"] = negative_results
            missed = [r for r in negative_results if r["vlm_judgment"]["is_goal"]]
            if missed:
                final_report["tuning_suggestions"].append(
                    f"负样本审查发现 {len(missed)} 个疑似漏检区间，"
                    f"建议检查这些时间段的检测通道覆盖"
                )
        if gt_comparison:
            final_report["ground_truth_comparison"] = gt_comparison

        # 写入报告
        output_path = args.output
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)

        print(f"\n  报告已保存: {output_path}")

        fp_count = report.summary.get("fp_suspect", 0)
        missed_count = len([r for r in negative_results
                           if r["vlm_judgment"]["is_goal"]]) if negative_results else 0
        if fp_count > 0 or missed_count > 0:
            sys.exit(1)

    finally:
        for d in temp_dirs:
            if os.path.exists(d):
                import shutil
                shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
