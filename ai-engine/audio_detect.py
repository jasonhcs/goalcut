#!/usr/bin/env python3
"""
GoalCut 入网声（Swish）检测模块 (T-1.9.8)

原理：
  篹球入网时会产生一个短促（50~300ms）的高频冲击声（"swish"）。
  野球场安静环境下，这个声音特征鲜明，可作为进球的独立辅助信号。

实现策略（无额外依赖，仅使用 Python stdlib + numpy）：
  1. 使用 wave 模块读取 WAV 文件
  2. 分帧计算短时能量（Short-Time Energy）
  3. 对高频段（2kHz~8kHz）计算频谱能量
  4. 在 "能量突发 + 短促 + 高频" 的位置检测入网声
  5. 可选：使用 librosa 获得更精确的 Mel 频谱特征
"""

import os
import wave
import struct
import subprocess
import tempfile
import numpy as np
from typing import List, Tuple, Optional


def extract_audio(video_path: str, output_wav: str,
                  sample_rate: int = 16000) -> bool:
    """
    使用 FFmpeg 从视频中提取单声道 WAV 音频。

    Returns:
        True 表示提取成功
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-vn",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-f", "wav",
        output_wav,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0 and os.path.exists(output_wav)
    except Exception as e:
        print(f"[audio_detect] 音频提取失败: {e}", flush=True)
        return False


def read_wav(wav_path: str) -> Tuple[Optional[np.ndarray], int]:
    """
    读取 WAV 文件，返回 (samples_float32, sample_rate)。
    samples 归一化到 [-1.0, 1.0]。
    """
    try:
        with wave.open(wav_path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()

            raw = wf.readframes(n_frames)

        if sampwidth == 2:
            fmt = f"{n_frames * n_channels}h"
            samples = np.array(struct.unpack(fmt, raw), dtype=np.float32)
            samples /= 32768.0
        elif sampwidth == 4:
            fmt = f"{n_frames * n_channels}i"
            samples = np.array(struct.unpack(fmt, raw), dtype=np.float32)
            samples /= 2147483648.0
        elif sampwidth == 1:
            fmt = f"{n_frames * n_channels}B"
            samples = np.array(struct.unpack(fmt, raw), dtype=np.float32)
            samples = (samples - 128.0) / 128.0
        else:
            print(f"[audio_detect] 不支持的采样位深: {sampwidth}", flush=True)
            return None, 0

        # 多声道取均值
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        return samples, framerate

    except Exception as e:
        print(f"[audio_detect] 读取 WAV 失败: {e}", flush=True)
        return None, 0


def compute_stft_energy(samples: np.ndarray, sample_rate: int,
                         window_size_ms: int = 20,
                         hop_size_ms: int = 10,
                         freq_low: int = 2000,
                         freq_high: int = 8000) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算短时频谱能量（使用 numpy FFT）。

    Returns:
        (timestamps, band_energy) - 每帧的时间戳和高频段能量
    """
    win_samples = int(sample_rate * window_size_ms / 1000)
    hop_samples = int(sample_rate * hop_size_ms / 1000)

    # 计算 FFT 的频率分辨率
    freqs = np.fft.rfftfreq(win_samples, d=1.0 / sample_rate)
    freq_mask = (freqs >= freq_low) & (freqs <= freq_high)

    n_frames = (len(samples) - win_samples) // hop_samples + 1
    timestamps = np.array(
        [i * hop_samples / sample_rate for i in range(n_frames)],
        dtype=np.float32
    )
    band_energy = np.zeros(n_frames, dtype=np.float32)

    # 汉宁窗
    window = np.hanning(win_samples)

    for i in range(n_frames):
        start = i * hop_samples
        end = start + win_samples
        if end > len(samples):
            break
        frame = samples[start:end] * window
        spectrum = np.abs(np.fft.rfft(frame))
        # 只取高频段的能量
        band_energy[i] = float(np.sum(spectrum[freq_mask] ** 2))

    return timestamps, band_energy


def detect_swish_events(
    wav_path: str,
    video_duration: float,
    min_duration_ms: float = 30.0,
    max_duration_ms: float = 300.0,
    freq_low: int = 2000,
    freq_high: int = 8000,
    min_burst_ratio: float = 3.0,
    cooldown_s: float = 3.0,
) -> List[dict]:
    """
    从 WAV 文件中检测入网声事件。

    Args:
        wav_path:        WAV 音频文件路径
        video_duration:  视频总时长（秒），用于限制检测范围
        min/max_duration_ms: 入网声持续时长范围
        freq_low/high:   高频段范围（Hz）
        min_burst_ratio: 能量突发阈值 = 均值 + ratio * 标准差
        cooldown_s:      两次事件最小间隔

    Returns:
        [{"frame_index"(近似), "timestamp", "confidence", "detail"}, ...]
    """
    samples, sample_rate = read_wav(wav_path)
    if samples is None or len(samples) == 0:
        print("[audio_detect] 音频数据为空", flush=True)
        return []

    print(f"[audio_detect] 音频: {len(samples)/sample_rate:.1f}s, "
          f"{sample_rate}Hz, {len(samples)} 采样点", flush=True)

    # 计算高频段短时能量
    window_ms = 20  # 20ms 帧
    hop_ms = 10     # 10ms 步进
    timestamps, band_energy = compute_stft_energy(
        samples, sample_rate, window_ms, hop_ms, freq_low, freq_high
    )

    if len(band_energy) < 10:
        return []

    # 计算统计量（排除静音段）
    non_zero = band_energy[band_energy > 0]
    if len(non_zero) < 10:
        print("[audio_detect] 高频能量过低（可能无音轨或静音视频）", flush=True)
        return []

    mean_e = float(np.mean(non_zero))
    std_e = float(np.std(non_zero))
    burst_threshold = mean_e + min_burst_ratio * std_e

    # 标准差过小说明音频非常平稳，无法区分信号
    if std_e < mean_e * 0.1:
        print(f"[audio_detect] 音频能量过于平稳(std/mean={std_e/mean_e:.3f})，跳过", flush=True)
        return []

    print(f"[audio_detect] 高频能量统计: mean={mean_e:.2f}, std={std_e:.2f}, "
          f"threshold={burst_threshold:.2f}", flush=True)

    # 检测突发事件
    hop_s = hop_ms / 1000.0
    min_frames = max(1, int(min_duration_ms / hop_ms))
    max_frames = int(max_duration_ms / hop_ms)
    cooldown_frames = int(cooldown_s / hop_s)

    candidate_events = []
    last_event_frame = -cooldown_frames - 1
    i = 0

    while i < len(band_energy):
        # 跳过低能量帧
        if band_energy[i] <= burst_threshold:
            i += 1
            continue

        # 跳过视频开头 1s（通常有转场噪声）
        if timestamps[i] < 1.0:
            i += 1
            continue

        # 跳过太接近上次事件的帧
        if i - last_event_frame < cooldown_frames:
            i += 1
            continue

        # 找到突发起始点，向后找到突发结束点
        burst_start = i
        burst_end = i
        while burst_end < len(band_energy) and band_energy[burst_end] > burst_threshold:
            burst_end += 1
        burst_duration_frames = burst_end - burst_start

        # 验证突发时长在合理范围内
        if min_frames <= burst_duration_frames <= max_frames:
            peak_idx = burst_start + np.argmax(band_energy[burst_start:burst_end])
            peak_time = float(timestamps[peak_idx])

            # 验证：突发后能量快速衰减（入网声短促特征）
            decay_end = min(len(band_energy), burst_end + max_frames)
            post_energy = band_energy[burst_end:decay_end]
            if len(post_energy) > 0:
                avg_post = float(np.mean(post_energy))
                if avg_post > mean_e * 2.0:
                    # 后续能量仍然很高，不像短促入网声
                    i = burst_end
                    continue

            burst_ratio = (float(band_energy[peak_idx]) - mean_e) / (std_e + 1e-6)
            # 置信度：突发强度 + 时长合理性
            duration_factor = 1.0 - abs(
                burst_duration_frames - (min_frames + max_frames) / 2
            ) / max(1, max_frames)
            confidence = min(0.70, 0.35 + 0.05 * burst_ratio + 0.1 * duration_factor)

            # 限制在视频时长内
            if peak_time <= video_duration + 1.0:
                candidate_events.append({
                    "frame_index": int(peak_time * 3),  # 近似帧索引（3fps）
                    "timestamp": round(peak_time, 2),
                    "confidence": round(confidence, 3),
                    "detail": (
                        f"入网声: peak={float(band_energy[peak_idx]):.1f}, "
                        f"threshold={burst_threshold:.1f}, "
                        f"duration={burst_duration_frames*hop_ms:.0f}ms, "
                        f"ratio={burst_ratio:.1f}x"
                    ),
                })
                last_event_frame = peak_idx

        i = max(i + 1, burst_end)

    print(f"[audio_detect] 检测到 {len(candidate_events)} 个入网声事件", flush=True)
    return candidate_events


def detect_whistle_events(
    wav_path: str,
    video_duration: float,
    freq_low: int = 2000,
    freq_high: int = 4000,
    min_duration_ms: float = 200.0,
    max_duration_ms: float = 1500.0,
    min_burst_ratio: float = 4.0,
    cooldown_s: float = 5.0,
) -> List[dict]:
    """
    检测裁判哨声（正式比赛场景）。

    哨声特征：持续时间较长（200ms~1500ms），频率集中在 2kHz~4kHz。

    Returns:
        [{"timestamp", "confidence", "detail"}, ...]
    """
    samples, sample_rate = read_wav(wav_path)
    if samples is None or len(samples) == 0:
        return []

    hop_ms = 10
    timestamps, band_energy = compute_stft_energy(
        samples, sample_rate,
        window_size_ms=30, hop_size_ms=hop_ms,
        freq_low=freq_low, freq_high=freq_high,
    )

    if len(band_energy) < 10:
        return []

    non_zero = band_energy[band_energy > 0]
    if len(non_zero) < 10:
        return []

    mean_e = float(np.mean(non_zero))
    std_e = float(np.std(non_zero))

    if std_e < mean_e * 0.05:
        return []

    burst_threshold = mean_e + min_burst_ratio * std_e
    min_frames = int(min_duration_ms / hop_ms)
    max_frames = int(max_duration_ms / hop_ms)
    cooldown_frames = int(cooldown_s / 0.01)

    candidate_events = []
    last_event_frame = -cooldown_frames - 1
    i = 0

    while i < len(band_energy):
        if band_energy[i] <= burst_threshold or timestamps[i] < 1.0:
            i += 1
            continue

        if i - last_event_frame < cooldown_frames:
            i += 1
            continue

        burst_start = i
        burst_end = i
        while burst_end < len(band_energy) and band_energy[burst_end] > burst_threshold:
            burst_end += 1

        burst_duration_frames = burst_end - burst_start
        if min_frames <= burst_duration_frames <= max_frames:
            peak_idx = burst_start + np.argmax(band_energy[burst_start:burst_end])
            peak_time = float(timestamps[peak_idx])

            burst_ratio = (float(band_energy[peak_idx]) - mean_e) / (std_e + 1e-6)
            confidence = min(0.65, 0.30 + 0.04 * burst_ratio)

            if peak_time <= video_duration + 1.0:
                candidate_events.append({
                    "frame_index": int(peak_time * 3),
                    "timestamp": round(peak_time, 2),
                    "confidence": round(confidence, 3),
                    "detail": (
                        f"哨声: duration={burst_duration_frames*hop_ms:.0f}ms, "
                        f"ratio={burst_ratio:.1f}x"
                    ),
                })
                last_event_frame = peak_idx

        i = max(i + 1, burst_end)

    return candidate_events


def detect_cheer_events(
    wav_path: str,
    video_duration: float,
    min_duration_ms: float = 300.0,
    max_duration_ms: float = 3000.0,
    min_burst_ratio: float = 2.5,
    cooldown_s: float = 4.0,
) -> List[dict]:
    """
    检测球友叫好声（进球后的欢呼/叫好）。

    叫好声特征：宽频能量突发（不限于高频），持续 300ms~3s，
    比入网声更长、频带更宽。

    Returns:
        [{"frame_index", "timestamp", "confidence", "detail"}, ...]
    """
    samples, sample_rate = read_wav(wav_path)
    if samples is None or len(samples) == 0:
        return []

    # 宽频能量（500Hz~6kHz，涵盖人声基频和谐波）
    hop_ms = 20
    timestamps, band_energy = compute_stft_energy(
        samples, sample_rate,
        window_size_ms=40, hop_size_ms=hop_ms,
        freq_low=500, freq_high=6000,
    )

    if len(band_energy) < 10:
        return []

    non_zero = band_energy[band_energy > 0]
    if len(non_zero) < 10:
        return []

    mean_e = float(np.mean(non_zero))
    std_e = float(np.std(non_zero))
    if std_e < mean_e * 0.1:
        return []

    burst_threshold = mean_e + min_burst_ratio * std_e
    min_frames = max(1, int(min_duration_ms / hop_ms))
    max_frames = int(max_duration_ms / hop_ms)
    cooldown_frames = int(cooldown_s * 1000 / hop_ms)

    candidate_events = []
    last_event_frame = -cooldown_frames - 1
    i = 0

    while i < len(band_energy):
        if band_energy[i] <= burst_threshold or timestamps[i] < 1.0:
            i += 1
            continue

        if i - last_event_frame < cooldown_frames:
            i += 1
            continue

        burst_start = i
        burst_end = i
        while burst_end < len(band_energy) and band_energy[burst_end] > burst_threshold:
            burst_end += 1

        burst_duration_frames = burst_end - burst_start
        if min_frames <= burst_duration_frames <= max_frames:
            peak_idx = burst_start + np.argmax(band_energy[burst_start:burst_end])
            peak_time = float(timestamps[peak_idx])

            burst_ratio = (float(band_energy[peak_idx]) - mean_e) / (std_e + 1e-6)
            confidence = min(0.60, 0.25 + 0.04 * burst_ratio)

            if peak_time <= video_duration + 1.0:
                candidate_events.append({
                    "frame_index": int(peak_time * 3),
                    "timestamp": round(peak_time, 2),
                    "confidence": round(confidence, 3),
                    "detail": (
                        f"叫好声: duration={burst_duration_frames*hop_ms:.0f}ms, "
                        f"ratio={burst_ratio:.1f}x"
                    ),
                })
                last_event_frame = peak_idx

        i = max(i + 1, burst_end)

    print(f"[audio_detect] 检测到 {len(candidate_events)} 个叫好声事件", flush=True)
    return candidate_events


def detect_clap_events(
    wav_path: str,
    video_duration: float,
    min_burst_ratio: float = 4.0,
    cooldown_s: float = 3.0,
) -> List[dict]:
    """
    检测击掌声。

    击掌声特征：极短脉冲（<100ms），宽频冲击，频谱平坦（与入网声的高频集中不同）。

    Returns:
        [{"frame_index", "timestamp", "confidence", "detail"}, ...]
    """
    samples, sample_rate = read_wav(wav_path)
    if samples is None or len(samples) == 0:
        return []

    # 宽频能量（全频段）
    hop_ms = 5
    timestamps, band_energy = compute_stft_energy(
        samples, sample_rate,
        window_size_ms=10, hop_size_ms=hop_ms,
        freq_low=200, freq_high=8000,
    )

    if len(band_energy) < 20:
        return []

    non_zero = band_energy[band_energy > 0]
    if len(non_zero) < 20:
        return []

    mean_e = float(np.mean(non_zero))
    std_e = float(np.std(non_zero))
    if std_e < mean_e * 0.1:
        return []

    burst_threshold = mean_e + min_burst_ratio * std_e
    max_frames = int(100.0 / hop_ms)  # 击掌最长 100ms
    cooldown_frames = int(cooldown_s * 1000 / hop_ms)

    candidate_events = []
    last_event_frame = -cooldown_frames - 1
    i = 0

    while i < len(band_energy):
        if band_energy[i] <= burst_threshold or timestamps[i] < 1.0:
            i += 1
            continue
        if i - last_event_frame < cooldown_frames:
            i += 1
            continue

        burst_start = i
        burst_end = i
        while burst_end < len(band_energy) and band_energy[burst_end] > burst_threshold:
            burst_end += 1

        burst_duration_frames = burst_end - burst_start
        # 击掌声极短（1~max_frames 帧，即 5~100ms）
        if 1 <= burst_duration_frames <= max_frames:
            peak_idx = burst_start + np.argmax(band_energy[burst_start:burst_end])
            peak_time = float(timestamps[peak_idx])

            # 验证：突发后 50ms 内能量回落（击掌是瞬态）
            decay_end = min(len(band_energy), burst_end + int(50 / hop_ms))
            post_energy = band_energy[burst_end:decay_end]
            if len(post_energy) > 0 and float(np.mean(post_energy)) > burst_threshold * 0.5:
                i = burst_end
                continue

            burst_ratio = (float(band_energy[peak_idx]) - mean_e) / (std_e + 1e-6)
            confidence = min(0.55, 0.20 + 0.04 * burst_ratio)

            if peak_time <= video_duration + 1.0:
                candidate_events.append({
                    "frame_index": int(peak_time * 3),
                    "timestamp": round(peak_time, 2),
                    "confidence": round(confidence, 3),
                    "detail": (
                        f"击掌声: duration={burst_duration_frames*hop_ms:.0f}ms, "
                        f"ratio={burst_ratio:.1f}x"
                    ),
                })
                last_event_frame = peak_idx

        i = max(i + 1, burst_end)

    print(f"[audio_detect] 检测到 {len(candidate_events)} 个击掌声事件", flush=True)
    return candidate_events


def detect_audio_events(
    audio_path: str,
    video_duration: float,
) -> List[dict]:
    """
    综合音频事件检测的统一入口。

    检测目标：入网声、哨声、叫好声、击掌声。

    Returns:
        候选事件列表，每个事件已注明来源（detail 字段）
    """
    events = []

    # 入网声检测（野球场安静环境效果最好）
    swish_events = detect_swish_events(audio_path, video_duration)
    events.extend(swish_events)

    # 哨声检测（正式比赛有裁判吹哨）
    whistle_events = detect_whistle_events(audio_path, video_duration)
    events.extend(whistle_events)

    # 叫好声检测（野球场进球后球友欢呼）
    cheer_events = detect_cheer_events(audio_path, video_duration)
    events.extend(cheer_events)

    # 击掌声检测（进球后击掌庆祝）
    clap_events = detect_clap_events(audio_path, video_duration)
    events.extend(clap_events)

    # 按时间排序去重（2s 内认为同一事件，保留置信度最高的）
    if not events:
        return []

    events.sort(key=lambda e: e["timestamp"])
    deduped = [events[0]]
    for ev in events[1:]:
        if ev["timestamp"] - deduped[-1]["timestamp"] > 2.0:
            deduped.append(ev)
        elif ev["confidence"] > deduped[-1]["confidence"]:
            deduped[-1] = ev

    return deduped


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 3:
        print("用法: python audio_detect.py <wav_path> <video_duration_s>")
        sys.exit(1)

    wav_path = sys.argv[1]
    duration = float(sys.argv[2])
    events = detect_audio_events(wav_path, duration)
    print(json.dumps(events, ensure_ascii=False, indent=2))
