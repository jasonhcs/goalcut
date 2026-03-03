# GoalCut 输出结果视频效果验证 — 需求分析文档

> 文档版本：v2.0
>
> 创建日期：2026-03-03
>
> 最后更新：2026-03-03（新增无人工介入自动评估方案）
>
> 文档类型：需求分析

---

## 一、背景与目标

### 1.1 当前状态分析

GoalCut 目前处于 **Phase 1 MVP 已完成** 阶段，核心链路（输入视频 → YOLO 检测进球 → 裁剪拼接 → 输出集锦视频）已跑通。现有验证手段如下：

| 验证方式 | 当前实现 | 局限性 |
|---------|---------|--------|
| `test/test.sh` 集成测试 | 检查输出文件是否存在、文件大小、视频时长 | **只验证"有输出"，不验证"输出内容是否正确"** |
| Pipeline 日志输出 | 打印检测到的进球数量、时间戳、置信度、裁剪区间 | 需要人工阅读日志，无法自动判断质量 |
| 人工肉眼查看 | 打开输出视频手动检查 | 效率低，无法规模化，缺乏量化指标 |

**核心问题**：目前**没有系统化的方法来验证输出视频的效果质量**，即无法自动化回答以下关键问题：

1. 检测到的进球是否为真实进球？（**准确率**）
2. 视频中的真实进球是否都被检测到了？（**召回率**）
3. 裁剪的时间窗口是否合理？（进球瞬间是否在片段中央）
4. 输出视频的技术质量是否达标？（分辨率、帧率、编码、画质）
5. 多个片段拼接后是否连贯？（无黑帧、无卡顿、无音视频不同步）

### 1.2 验证目标

建立一套 **多层次、可量化、可自动化** 的输出视频效果验证体系，覆盖以下维度：

| 验证维度 | 目标 |
|---------|------|
| 进球检测准确性 | 量化准确率(Precision)和召回率(Recall)，目标 P ≥ 85%、R ≥ 80% |
| 时间定位精度 | 进球时间戳与真实时间偏差 ≤ 1 秒 |
| 视频技术质量 | 分辨率/帧率/编码/文件大小均符合预期 |
| 裁剪窗口合理性 | 进球瞬间位于裁剪片段的合理区间内 |
| 拼接完整性 | 无黑帧、无重复帧、片段间过渡正常 |

---

## 二、验证体系架构

### 2.1 四层验证模型（v2.0 升级）

> **v2.0 核心变化**：在原三层架构基础上，新增 **Layer 0（多信号自动进球验证）**，通过计算机视觉、音频分析、OCR 等多模态信号自动验证进球，彻底消除 Layer 2 对人工标注的强依赖。同时将 Layer 1 扩展支持感知质量评分。

```
┌─────────────────────────────────────────────────────┐
│            Layer 3: 端到端效果评估                      │
│   A/B 对比测试 / 自动化基准回归 / (可选) MOS 评分      │
├─────────────────────────────────────────────────────┤
│            Layer 2: 检测质量验证（混合模式）              │
│   P/R/F1 / 时间定位精度 / 支持标注 OR 自动伪标注       │
├─────────────────────────────────────────────────────┤
│            Layer 1: 视频技术质量验证                     │
│   元信息校验 / 画质检测 / 拼接完整性 / VMAF 感知评分    │
├─────────────────────────────────────────────────────┤
│       ★ Layer 0: 多信号自动进球验证（全新）               │
│   球-筐穿越 / 轨迹物理校验 / 音频事件 / 计分板 OCR     │
└─────────────────────────────────────────────────────┘
```

### 2.2 验证流程概览（v2.0）

```
输入: 原始视频 + 输出集锦视频 + AI检测结果JSON
（无需人工标注即可完成 L0~L2 全链路验证）
                    │
                    ▼
         ┌──────────────────────┐
         │  Layer 0 多信号验证    │  ← 全自动，生成自动伪标注
         │  (CV + Audio + OCR)   │    作为 Layer 2 的 Ground Truth
         └──────────┬───────────┘
                    │ 生成 auto_annotations.json
                    ▼
         ┌──────────────────────┐
         │  Layer 1 技术质量验证  │  ← 全自动，含 VMAF 感知评分
         └──────────┬───────────┘
                    │ PASS
                    ▼
         ┌──────────────────────┐
         │  Layer 2 检测质量验证  │  ← 以 L0 自动标注为基准
         │  (或人工标注，优先级高) │    自动计算 P/R/F1
         └──────────┬───────────┘
                    │ PASS
                    ▼
         ┌──────────────────────┐
         │  Layer 3 回归对比评估  │  ← 全自动，仅在参数变更时触发
         └──────────────────────┘
                    │
                    ▼
              验证报告输出
```

### 2.3 人工介入策略（最小化原则）

| 场景 | 是否需要人工 | 说明 |
|------|------------|------|
| 每次构建/CI | **不需要** | L0 自动验证 + L1 技术检查全自动 |
| 每日回归 | **不需要** | L0→L1→L2 全链路自动运行 |
| 新视频接入 | **不需要** | L0 自动生成伪标注 |
| 指标长期退化告警 | **不需要** | 趋势检测自动触发告警 |
| 算法大版本更新 | **可选** | A/B 回归自动判定，人工仅做最终决策 |
| 极端 Hard 样本分析 | **可选** | 仅当 L0 信号置信度全部低于阈值时触发 |

---

## 三（新）、Layer 0 — 多信号自动进球验证（全自动，无需人工标注）

### L0.1 设计思路

**核心思想**：利用多个独立的计算机视觉/音频信号交叉验证，自动判断某一时刻是否真实发生了进球，从而生成 `auto_annotations.json` 替代人工标注。各信号的置信度加权融合，无需任何人工干预。

```
原始视频
    │
    ├──► 信号1: 球-筐穿越检测 (CV)         权重 0.40
    ├──► 信号2: 球体运动轨迹物理校验 (CV)   权重 0.25
    ├──► 信号3: 音频事件检测 (Audio)        权重 0.20
    ├──► 信号4: 计分板 OCR 变化检测 (OCR)   权重 0.15 (仅正式比赛)
    │
    ▼
  多信号融合评分 (0~1.0)
    │
    ├── ≥ 0.70 → 高置信进球 → 自动伪标注 (confidence=high)
    ├── 0.45~0.69 → 待确认 → 标注为 uncertain（仍可用于统计）
    └── < 0.45 → 非进球 → 不标注
```

### L0.2 信号1：球-筐穿越检测

**原理**：真实进球时，球体必然从篮筐平面上方穿越至下方（网内方向），通过追踪球心 Y 轴坐标在 ROI 区域内的变化方向判定。

**实现方案**：
```
输入: 原始视频 + YOLO 已有的篮筐/球体检测框
算法:
  1. 在 AI 检测触发进球的时间窗口 T±2s 内，提取逐帧检测结果
  2. 定位篮筐中心坐标 (hoop_cx, hoop_cy)，建立 ROI 区域 (±1.5×hoop_width)
  3. 在 ROI 内追踪球心 Y 坐标序列：[y_t-n, ..., y_t, ..., y_t+n]
  4. 判定穿越条件:
     a. 球心 Y 存在从 hoop_cy-margin 到 hoop_cy+margin 的单调下行趋势
     b. 过程中球中心 X 坐标始终在 ROI 范围内
     c. 穿越时长 ≤ 0.5s（防止慢动作误判）
输出: crossing_score ∈ [0, 1]，穿越时刻 crossing_ts
```

**置信度分级**：

| 条件 | crossing_score |
|------|---------------|
| 完整穿越轨迹 + 球体连续可见 ≥ 5 帧 | 1.0 |
| 穿越轨迹不完整（部分帧遮挡） | 0.6 |
| 仅检测到球在筐附近（无完整轨迹） | 0.3 |
| 无球/无筐检测 | 0.0 |

### L0.3 信号2：球体运动轨迹物理校验

**原理**：有效投篮的球体运动遵循抛物线物理规律（重力加速度约 9.8m/s²），纯偶发的误触发通常不满足此约束。

**实现方案**：
```
输入: 球体逐帧像素坐标序列 [(x1,y1,t1), (x2,y2,t2), ...]
算法:
  1. 对 Y 轴坐标做二次曲线拟合 y = a*t² + b*t + c
  2. 计算拟合残差 R²（越接近 1.0 表示越符合抛物线）
  3. 检查系数 a > 0（Y 轴向下为正，重力方向）
  4. 估算顶点（出手点）和终点（进筐点）是否在合理区间
  5. 检查水平速度（X 方向）的一致性（投篮期间不应有剧烈变化）
输出: physics_score = R² × direction_check × range_check ∈ [0, 1]
```

**关键参数（可配置）**：

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `min_track_frames` | 8 | 最少追踪帧数，少于此值不做拟合 |
| `min_r2` | 0.85 | R² 最低阈值 |
| `max_arc_duration_s` | 2.0 | 投篮弧线最长时长 |

### L0.4 信号3：音频事件检测

**原理**：进球后通常伴随裁判哨声、现场欢呼声或运动员庆祝声，这些声学事件可作为进球的协同验证信号，且完全独立于视觉检测。

**实现方案**：
```bash
# Step1: 提取音频
ffmpeg -i input.mp4 -vn -ar 16000 -ac 1 audio.wav

# Step2: 能量包络分析 (检测欢呼/鼓掌的突发能量峰值)
python audio_event_detect.py \
  --audio audio.wav \
  --window 0.5s \
  --detection_times "8.5,22.3" \  # 来自 AI 检测的时间戳
  --search_range 3.0              # 检测点前后 3 秒内找音频峰值
```

**音频事件分类**：

| 事件类型 | 检测方法 | 权重 | 适用场景 |
|---------|---------|------|---------|
| 短哨（裁判哨） | 频率 2~4kHz 短促峰值 (<0.5s) | 高 | 正式比赛 |
| 长欢呼声 | 宽频带持续能量上升 (>1s) | 中 | 正式比赛 |
| 球触网声 | 高频短促撞击声 (<0.1s) | 中 | 室内/近景 |
| 能量静默后突升 | 背景噪声基线上的突变 | 低 | 通用 |

```
audio_score 计算:
  base = max(event_scores in T±3s)
  若无音轨或音轨全静音: audio_score = 0.5 (中性，不惩罚也不加分)
```

### L0.5 信号4：计分板 OCR 变化检测（仅正式比赛）

**原理**：正式比赛视频中计分板分数变化是最确定的进球证据，OCR 读取前后帧的分数差即可自动确认。

**实现方案**：
```
1. 计分板区域定位:
   a. 首次运行：YOLO 额外检测 scoreboard ROI，缓存位置
   b. 后续：仅对缓存 ROI 做 OCR，减少计算开销

2. 分数变化检测:
   for each detection_timestamp T:
     score_before = OCR(frame at T-2.0s)
     score_after  = OCR(frame at T+1.0s)
     delta = score_after - score_before
     if delta ∈ {2, 3} (篮球得分规则):
       ocr_score = 1.0
     elif delta == 1 (罚球):
       ocr_score = 0.5  (可能是罚球，不计入进球统计)
     elif delta == 0:
       ocr_score = 0.0  (分数未变，疑似误检)
     else:
       ocr_score = 0.3  (OCR 读取错误，保守评分)

3. 无计分板时: ocr_score = 0.5 (中性)
```

**OCR 工具选型**：
- 优先使用 `tesseract` + 数字字符集限定（速度快）
- 备选：`paddleocr`（精度更高，适合模糊/小字计分板）

### L0.6 多信号融合评分

```
fusion_score = w1 × crossing_score
             + w2 × physics_score
             + w3 × audio_score
             + w4 × ocr_score

其中 (w1=0.40, w2=0.25, w3=0.20, w4=0.15)
当 ocr_score 无效 (无计分板) 时: 重新归一化 w1=0.47, w2=0.29, w3=0.24
```

**置信度等级与后续处理**：

| fusion_score | 等级 | 处理方式 |
|-------------|------|---------|
| ≥ 0.70 | `HIGH` | 写入自动伪标注，作为 L2 基准 |
| 0.45~0.69 | `MEDIUM` | 写入伪标注但标记 `uncertain=true`，参与统计但不计入严格指标 |
| < 0.45 | `LOW` | 若 AI 检测认为是进球，则标记为 **疑似误检**，输出到误检列表 |

### L0.7 自动伪标注输出格式

```json
{
  "video_file": "goalcut_demo_1.MP4",
  "annotation_type": "auto",
  "generated_at": "2026-03-03T10:00:00Z",
  "annotations": [
    {
      "event_id": 1,
      "timestamp": 8.5,
      "type": "goal",
      "confidence": "HIGH",
      "fusion_score": 0.87,
      "signals": {
        "ball_hoop_crossing": { "score": 0.95, "crossing_ts": 8.48 },
        "trajectory_physics": { "score": 0.88, "r2": 0.91 },
        "audio_event": { "score": 0.75, "event_type": "whistle", "event_ts": 8.6 },
        "scoreboard_ocr": { "score": 1.0, "score_delta": 2, "before": "42", "after": "44" }
      }
    }
  ],
  "uncertain_events": [
    {
      "event_id": 2,
      "timestamp": 15.2,
      "fusion_score": 0.52,
      "note": "轨迹遮挡严重，音频信号弱"
    }
  ],
  "suspected_false_positives": [
    {
      "ai_detection_ts": 19.8,
      "fusion_score": 0.31,
      "note": "球体未穿越篮筐，疑似运球动作"
    }
  ]
}
```

### L0.8 与 Layer 2 的协作关系

```
优先级规则（当两种标注都存在时）:
  1. 人工标注 > 自动伪标注 (human overrides auto)
  2. 自动伪标注 (HIGH) 等价于人工标注，参与全部指标计算
  3. 自动伪标注 (MEDIUM) 仅参与召回率计算，不计入准确率分母
  4. 自动伪标注 (LOW) 不参与指标计算，仅输出告警

运行模式:
  --annotation-mode=auto    # 纯自动（无人工介入）
  --annotation-mode=human   # 使用人工标注（优先）
  --annotation-mode=hybrid  # 混合：人工标注优先，缺失的用自动补充
```

---

## 三、Layer 1 — 视频技术质量验证（全自动）

### 3.1 需求概述

对输出视频进行纯技术层面的质量检查，不涉及内容正确性判断，可以完全自动化。

### 3.2 功能需求

#### F1.1 输出文件健康度检查

| 检查项 | 说明 | 判定标准 |
|-------|------|---------|
| 文件存在性 | 输出文件是否生成 | 文件存在且大小 > 0 |
| 文件可播放性 | ffprobe 能否成功解析 | ffprobe 返回码 = 0 |
| 容器格式 | 输出是否为 MP4 | moov atom 存在 |
| 文件完整性 | 文件是否被截断 | ffmpeg 可正常 seek 到文件末尾 |

#### F1.2 视频元信息校验

| 检查项 | 说明 | 判定标准 |
|-------|------|---------|
| 视频编码 | 输出编码格式 | H.264 (libx264) |
| 分辨率一致性 | 输出分辨率与输入是否一致 | 宽高与源视频相同 |
| 帧率合理性 | 输出帧率 | 与源视频帧率一致或在合理范围内 (±1fps) |
| 时长合理性 | 集锦时长 | 大于 0 且小于源视频总时长 |
| 时长精度 | 实际时长与预期裁剪区间之和的偏差 | ≤ 0.5 秒 |
| 码率范围 | 输出码率 | 在合理范围内（不低于源视频 30%，不超过 200%） |

#### F1.3 拼接完整性检查

| 检查项 | 说明 | 判定标准 |
|-------|------|---------|
| 黑帧检测 | 检查输出视频中是否存在非预期黑帧 | 连续黑帧 ≤ 2 帧 |
| 静帧检测 | 检查是否存在画面完全静止 | 连续静帧 ≤ 采样间隔的 2 倍 |
| 片段数量验证 | 实际片段数与检测事件数是否匹配 | 拼接后片段数 ≤ 检测事件数（考虑合并） |

#### F1.4 感知画质评分（v2.0 新增）

> 客观评估输出视频的感知质量，无需参考原始视频即可运行（No-Reference 模式）。

| 指标 | 工具/方法 | 说明 | 判定标准 |
|------|---------|------|---------|
| **VMAF（Full Reference）** | `ffmpeg libvmaf` | 与源视频对比的感知质量分 | ≥ 85（满分100） |
| **BRISQUE（No Reference）** | `ffmpeg` + Python `skimage` | 无参考盲质量评估，检测模糊/噪声/压缩失真 | ≤ 40（越低越好） |
| **SSIM（片段边界）** | `ffmpeg ssim filter` | 仅在裁剪拼接点附近 ±3 帧计算 SSIM | 拼接点 SSIM ≥ 0.80 |
| **压缩失真比** | 基于 PSNR 估算 | 输出 vs 源的质量损失 | PSNR ≥ 35dB |

**VMAF 计算命令**：
```bash
# Full Reference: 对比源视频片段与输出对应片段
ffmpeg -i source_clip.mp4 -i output_clip.mp4 \
  -lavfi "[0:v][1:v]libvmaf=log_fmt=json:log_path=vmaf_result.json" \
  -f null -

# No Reference BRISQUE: 每隔 30 帧采样一次
python3 -c "
import cv2, skimage.metrics
cap = cv2.VideoCapture('output.mp4')
scores = []
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    if frame_idx % 30 == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scores.append(skimage.metrics.structural_similarity(gray, gray))
    frame_idx += 1
print(f'avg_brisque_approx: {sum(scores)/len(scores):.2f}')
"
```

**拼接点 SSIM 检测**（自动定位拼接边界）：
```bash
# 利用帧差检测拼接点（场景切换），在切换点前后计算 SSIM
ffmpeg -i output.mp4 -vf "select='gt(scene,0.4)',showinfo" -vsync vfr \
  scene_cuts.txt 2>&1 | grep "pts_time"
```

#### F1.5 音视频同步检测（v2.0 新增）

| 检查项 | 方法 | 判定标准 |
|-------|------|---------|
| 音视频偏移 | `ffprobe` 对比 audio/video stream start_time | \|A-V offset\| ≤ 40ms |
| 音频连续性 | 检测音频流中的静音跳变（拼接点） | 拼接点静音 ≤ 50ms |
| 采样率一致性 | `ffprobe -select_streams a` | 与源文件一致 |

### 3.3 实现方案

- **工具**：基于 `ffprobe` + `ffmpeg` 的 Shell 脚本或 Go 测试代码
- **VMAF 依赖**：需要 `ffmpeg` 编译时包含 `--enable-libvmaf`（或使用 Docker 镜像）
- **BRISQUE 依赖**：`pip install scikit-image opencv-python`
- **触发方式**：集成到 `test/test.sh` 中，每次测试自动运行
- **输出格式**：结构化 JSON 验证报告

### 3.4 验证报告示例（v2.0）

```json
{
  "video_file": "goalcut_demo_1_goalcut.mp4",
  "source_file": "goalcut_demo_1.MP4",
  "timestamp": "2026-03-03T10:00:00Z",
  "layer1_technical": {
    "status": "PASS",
    "checks": {
      "file_exists": { "status": "PASS", "detail": "size=1618KB" },
      "playable": { "status": "PASS", "detail": "ffprobe OK" },
      "codec": { "status": "PASS", "detail": "h264" },
      "resolution": { "status": "PASS", "detail": "1280x720, matches source" },
      "fps": { "status": "PASS", "detail": "30.0 fps" },
      "duration": { "status": "PASS", "detail": "13.33s, expected=13.33s, diff=0.00s" },
      "bitrate": { "status": "PASS", "detail": "993kbps, source=1500kbps, ratio=66%" },
      "black_frames": { "status": "PASS", "detail": "0 black frames detected" },
      "frozen_frames": { "status": "PASS", "detail": "0 frozen segments" },
      "vmaf": { "status": "PASS", "detail": "score=91.3, threshold=85" },
      "brisque": { "status": "PASS", "detail": "score=28.4, threshold=40" },
      "splice_ssim": { "status": "PASS", "detail": "min=0.87 at 6.33s, threshold=0.80" },
      "av_sync": { "status": "PASS", "detail": "offset=12ms, threshold=40ms" }
    }
  }
}
```

---

## 四、Layer 2 — 检测质量验证（混合模式，v2.0）

### 4.1 需求概述

对 AI 检测结果的内容正确性进行量化评估。**v2.0 支持三种标注来源**，优先级从高到低：

1. **人工标注**（最高精度，`annotation_type=human`）
2. **Layer 0 自动伪标注** HIGH 级别（等价精度，`annotation_type=auto`，`confidence=HIGH`）
3. **Layer 0 自动伪标注** MEDIUM 级别（保守参与，`confidence=MEDIUM`）

> **重要变化**：v2.0 中 Layer 2 **不再强依赖人工标注**。当无人工标注时，自动使用 Layer 0 生成的伪标注作为基准，实现全自动运行。

### 4.2 Ground Truth 标注体系

#### F2.1 标注数据格式（兼容人工/自动双源）

标注文件统一格式（人工标注和自动伪标注共享同一 schema）：

```json
{
  "video_file": "goalcut_demo_1.MP4",
  "duration": 33.0,
  "annotation_type": "human",  // "human" | "auto" | "hybrid"
  "annotator": "human",        // 人工标注时为标注者姓名，自动时为 "auto_layer0"
  "schema_version": "2.0",
  "annotations": [
    {
      "event_id": 1,
      "timestamp": 8.5,
      "type": "goal",
      "confidence": "HIGH",    // 仅 auto 时有效: "HIGH" | "MEDIUM"
      "description": "第一个进球，球从右侧投入",
      "time_window": { "start": 7.0, "end": 10.0 },
      "auto_signals": null     // 人工标注时为 null，auto 时包含 L0 信号详情
    },
    {
      "event_id": 2,
      "timestamp": 22.3,
      "type": "goal",
      "confidence": "HIGH",
      "description": "第二个进球，上篮",
      "time_window": { "start": 20.5, "end": 24.0 },
      "auto_signals": null
    }
  ],
  "non_goal_segments": [
    {
      "start": 0.0,
      "end": 5.0,
      "description": "热身运球，无进球"
    }
  ]
}
```

#### F2.2 标注目录结构（v2.0）

```
test/
├── annotations/
│   ├── human/                      # 人工标注（最高优先级）
│   │   ├── goalcut_demo_1.json
│   │   └── goalcut_demo_2.json
│   ├── auto/                       # Layer 0 自动生成的伪标注（运行时生成）
│   │   ├── goalcut_demo_1.json    # 自动生成，不纳入 git 版本管理
│   │   └── .gitignore
│   └── annotation_spec.md         # 标注规范说明
├── input/                          # 测试输入视频
├── output/                         # 测试输出视频
└── reports/                        # 验证报告输出目录
```

**标注解析优先级逻辑**：
```python
def load_annotations(video_name, mode="hybrid"):
    human_path = f"test/annotations/human/{video_name}.json"
    auto_path  = f"test/annotations/auto/{video_name}.json"

    if mode == "human":
        return load(human_path)  # 无人工标注则报错
    elif mode == "auto":
        return load(auto_path)   # 纯自动，不看人工标注
    else:  # hybrid（默认）
        if exists(human_path):
            return load(human_path)
        elif exists(auto_path):
            return load(auto_path, filter_confidence=["HIGH"])
        else:
            raise Exception("No annotations found, run Layer 0 first")
```

### 4.3 检测指标计算

#### F2.3 核心指标定义

| 指标 | 定义 | 计算方式 | 目标值 |
|------|------|---------|--------|
| **Precision（准确率）** | 检测到的事件中真实进球占比 | TP / (TP + FP) | ≥ 85% |
| **Recall（召回率）** | 真实进球中被检测到的占比 | TP / (TP + FN) | ≥ 80% |
| **F1 Score** | 准确率和召回率的调和平均 | 2PR / (P+R) | ≥ 82% |
| **时间偏差** | 检测时间戳与真实时间戳的平均偏差 | mean(\|t_det - t_gt\|) | ≤ 1.0s |
| **时间偏差 P90** | 90% 的检测事件时间偏差 | P90(\|t_det - t_gt\|) | ≤ 1.5s |

#### F2.4 事件匹配算法

检测事件与标注事件的匹配规则：

```
对于每个检测事件 D:
  1. 找到时间戳最接近的标注事件 G
  2. 若 |D.timestamp - G.timestamp| ≤ tolerance (默认 3.0s):
     → 标记为匹配 (TP), G 被消耗
  3. 若无匹配的标注事件:
     → 标记为误检 (FP)

对于每个未被匹配的标注事件 G:
  → 标记为漏检 (FN)
```

匹配容差 `tolerance` 应可配置，默认为 3.0 秒（与裁剪窗口 before_seconds 一致）。

### 4.4 裁剪窗口验证

#### F2.5 裁剪质量指标

| 指标 | 说明 | 判定标准 |
|------|------|---------|
| 进球可见性 | 进球瞬间是否在裁剪片段内 | 标注时间戳落在裁剪区间 [start, end] 内 |
| 中心偏移度 | 进球瞬间与片段中心的偏移 | \|t_goal - t_center\| / duration ≤ 0.4 |
| 前导时间 | 进球前的缓冲时间 | ≥ 配置的 before_seconds × 0.8 |
| 后续时间 | 进球后的缓冲时间 | ≥ 配置的 after_seconds × 0.8 |

### 4.5 验证报告示例

```json
{
  "video_file": "goalcut_demo_1.MP4",
  "layer2_detection": {
    "status": "PASS",
    "ground_truth_goals": 2,
    "detected_goals": 2,
    "true_positives": 2,
    "false_positives": 0,
    "false_negatives": 0,
    "precision": 1.0,
    "recall": 1.0,
    "f1_score": 1.0,
    "time_deviation_avg": 0.45,
    "time_deviation_p90": 0.62,
    "matches": [
      {
        "detection": { "timestamp": 8.67, "confidence": 0.72 },
        "annotation": { "event_id": 1, "timestamp": 8.5 },
        "time_diff": 0.17,
        "clip_range": { "start": 5.67, "end": 11.67 },
        "goal_visible": true,
        "center_offset": 0.03
      },
      {
        "detection": { "timestamp": 22.0, "confidence": 0.65 },
        "annotation": { "event_id": 2, "timestamp": 22.3 },
        "time_diff": 0.30,
        "clip_range": { "start": 19.0, "end": 25.0 },
        "goal_visible": true,
        "center_offset": 0.05
      }
    ]
  }
}
```

---

## 五、Layer 3 — 端到端自动化回归评估（v2.0 全自动）

### 5.1 需求概述

在关键里程碑节点或参数调优时，进行全自动的批量端到端效果评估和版本对比。**v2.0 移除人工主观评分（MOS）的强依赖，改为自动化指标基准回归**。MOS 降级为可选补充，仅在自动指标显著退化时才触发。

### 5.2 功能需求

#### F3.1 测试数据集管理

| 需求 | 说明 |
|------|------|
| 测试集规模 | 至少 10 段不同场景的篮球视频 |
| 场景覆盖 | 正式比赛（多机位/有记分牌）、野球场（单机位/无记分牌）各 50% |
| 画面多样性 | 横版/竖版、室内/室外、不同分辨率 |
| 难度分级 | Easy（清晰、标准角度）/ Medium（中等遮挡）/ Hard（远景/低质量/快速运动） |
| 标注来源 | 优先人工标注，无人工标注时使用 Layer 0 自动伪标注 |

#### F3.2 全自动批量评估流程

```
评估流程 (全自动，无人工介入):
  for each video in test_dataset:
    1. 运行 Layer 0 多信号验证，生成自动伪标注
    2. 运行 GoalCut 生成集锦
    3. 运行 Layer 1 技术验证（含 VMAF）
    4. 运行 Layer 2 检测质量验证（以 L0 伪标注或人工标注为基准）
    5. 输出单视频 JSON 报告
  合并所有报告，计算汇总统计
  与历史基准对比，检测指标退化
  触发告警（若指标退化超过阈值）
```

#### F3.3 指标基准回归（替代人工 MOS）

将历史最优指标保存为 `baseline.json`，每次评估自动与基准对比：

```json
{
  "baseline_version": "v1.2.0",
  "baseline_date": "2026-02-15",
  "metrics": {
    "precision": 0.88,
    "recall": 0.84,
    "f1": 0.859,
    "time_deviation_avg": 0.68,
    "vmaf_avg": 91.3,
    "brisque_avg": 28.4
  },
  "regression_thresholds": {
    "precision_drop_max": 0.03,
    "recall_drop_max": 0.05,
    "f1_drop_max": 0.04,
    "vmaf_drop_max": 5.0
  }
}
```

**回归判定规则**：
```
for each metric:
  delta = baseline[metric] - current[metric]
  if delta > regression_threshold[metric]:
    → 标记为 REGRESSION，触发告警，阻断 CI（可配置）
  elif delta > 0:
    → 标记为 WARN（轻微退化，记录但不阻断）
  else:
    → 标记为 IMPROVED 或 STABLE
```

#### F3.4 自动化 A/B 对比测试

当调优参数或更新检测算法时，全自动对比新旧版本效果差异：

```
A/B 测试流程 (全自动):
  1. 用同一测试集分别运行版本 A 和版本 B
  2. 各自运行 L0→L1→L2 完整验证链
  3. 计算所有指标的 delta（B - A）
  4. 自动判定: B 是否在关键指标上优于 A（F1↑、VMAF↑、time_deviation↓）
  5. 生成对比报告，用于 code review 参考
  
  自动判定结论:
    - "B 显著优于 A": 所有关键指标均提升且至少一项超过阈值
    - "B 与 A 持平": 指标在统计误差范围内波动
    - "B 部分退化": 某些指标提升但其他下降（需人工判断权衡）
    - "B 明显劣于 A": 关键指标下降超过退化阈值
```

#### F3.5 自动化置信度分布分析（新增）

对 AI 检测结果的置信度分布进行统计，识别算法退化的早期信号：

```
置信度健康度分析:
  1. 绘制所有检测事件的置信度分布直方图
  2. 计算关键统计量:
     - mean_confidence: 平均置信度（目标 ≥ 0.65）
     - p10_confidence:  P10 置信度（目标 ≥ 0.50，过低说明大量弱置信误检）
     - high_conf_ratio: 置信度 ≥ 0.70 的占比（目标 ≥ 60%）
  3. 与历史分布做 KL 散度对比:
     KL(P_current || P_baseline) > 0.15 时触发告警
     （说明置信度分布发生了显著漂移）
```

#### F3.6 可选：MOS 人工评分（仅在自动指标异常时触发）

当以下任一条件成立时，才触发人工 MOS 评分请求：

| 触发条件 | 说明 |
|---------|------|
| F1 退化 > 0.05 | 检测质量显著下降 |
| VMAF 退化 > 8 | 画质显著下降 |
| L0 伪标注 MEDIUM 比例 > 40% | 自动标注可信度低，L2 指标本身不可信 |
| 新场景类型接入 | 超出现有测试集覆盖范围 |

### 5.3 汇总评估报告示例（v2.0 全自动版）

```
====================================
GoalCut 效果评估报告 — 2026-03-03
====================================
运行模式: 全自动 (annotation_mode=hybrid)
标注来源: 2 个人工标注 + 8 个 L0 自动伪标注

测试集: 10 个视频, 总时长 320s, 共 25 个标注进球
  其中: HIGH 置信 23 个, MEDIUM 置信 2 个

一、整体检测指标
  Precision:    88.0%  (22/25 检测为真)     目标: ≥85%  ✅
  Recall:       84.0%  (21/25 进球被检测)    目标: ≥80%  ✅
  F1 Score:     85.9%                        目标: ≥82%  ✅
  时间偏差(均):  0.68s                        目标: ≤1.0s ✅
  时间偏差(P90): 1.23s                        目标: ≤1.5s ✅

二、视频技术质量
  VMAF 均值:    91.3                         目标: ≥85   ✅
  BRISQUE 均值: 28.4                         目标: ≤40   ✅
  拼接点 SSIM:  0.89 (min across all clips)  目标: ≥0.80 ✅
  AV 同步:      12ms (max)                   目标: ≤40ms ✅

三、置信度健康度
  均值置信度:   0.71                          目标: ≥0.65 ✅
  P10 置信度:   0.55                          目标: ≥0.50 ✅
  高置信占比:   67%                           目标: ≥60%  ✅
  KL 散度:      0.08 (vs baseline)            目标: ≤0.15 ✅

四、基准回归对比 (vs v1.2.0 baseline)
  F1:    +0.009 (85.0% → 85.9%)              IMPROVED
  VMAF:  +1.2   (90.1 → 91.3)               IMPROVED
  t_dev: -0.04s (0.72s → 0.68s)             IMPROVED
  结论: ✅ 全面优于基准，建议更新 baseline

五、分场景指标
  正式比赛 (5个视频):  P=92%, R=88%, F1=90%
  野球场 (5个视频):    P=84%, R=80%, F1=82%

六、已知问题（自动检测）
  1. 竖版视频 L0 穿越检测成功率低 (crossing_score 均值 0.51)
  2. Hard 难度 L0 MEDIUM 标注占比 45%，超过阈值 → 建议补充人工标注
  3. demo_7 拼接点 SSIM=0.72，低于阈值 0.80，需排查裁剪边界

⚠️  人工介入建议: Hard 样本自动标注可信度低，建议对 3 个 Hard 视频补充人工标注
```

---

## 六、验证工具需求（v2.0）

### 6.1 工具清单

| 工具 | 功能 | 优先级 | 实现形式 |
|------|------|--------|---------|
| `verify_layer0.py` | **Layer 0 多信号自动进球验证，生成伪标注** | **P0** | Python 脚本 (cv2 + ffmpeg + tesseract) |
| `verify_technical.sh` | Layer 1 视频技术质量验证（含 VMAF/BRISQUE） | P0 | Shell 脚本，集成到 test.sh |
| `verify_detection.py` | Layer 2 检测质量验证（支持自动/人工标注双源） | P1 | Python 脚本 |
| `evaluate_batch.sh` | Layer 3 全自动批量评估编排 | P2 | Shell 编排脚本 |
| `compare_baseline.py` | 指标基准回归对比，生成退化告警 | P2 | Python 脚本 |
| `analyze_confidence.py` | 置信度分布健康度分析（KL 散度告警） | P2 | Python 脚本 |
| `ab_compare.sh` | A/B 版本自动对比测试 | P3 | Shell 编排脚本 |

### 6.2 集成方式（v2.0）

```
test/
├── test.sh                        # 现有集成测试（增强 Layer 1 验证）
├── verify_layer0.py               # ★ NEW: Layer 0 多信号自动进球验证
├── verify_technical.sh            # Layer 1 视频技术质量验证（含 VMAF）
├── verify_detection.py            # Layer 2 检测质量评估（混合标注模式）
├── evaluate_batch.sh              # Layer 3 全自动批量评估
├── compare_baseline.py            # 基准回归对比
├── analyze_confidence.py          # 置信度分布分析
├── annotations/
│   ├── human/                     # 人工标注（优先级最高）
│   │   ├── goalcut_demo_1.json
│   │   └── goalcut_demo_2.json
│   ├── auto/                      # L0 自动生成（运行时写入，不入 git）
│   │   └── .gitignore
│   └── annotation_spec.md
├── reports/
│   ├── baseline.json              # 历史最优指标基准
│   ├── baseline_conf_dist.json    # 历史置信度分布基准
│   └── .gitkeep
├── input/                         # 测试视频
└── output/                        # 输出视频
```

### 6.3 CI/CD 集成（v2.0 全自动）

```
CI 流水线验证策略（所有步骤均无需人工介入）:

  每次提交 (pre-merge):
    ✓ Layer 0 多信号验证 → 生成自动伪标注        < 2min (GPU) / < 5min (CPU)
    ✓ Layer 1 技术验证（含 VMAF/BRISQUE）        < 1min
    ✓ Layer 2 检测验证 → 输出 P/R/F1             < 1min
    ✓ 指标基准回归对比 → REGRESSION 时阻断合并   < 5s

  每日构建 (nightly):
    ✓ 完整测试集 L0→L1→L2 全链路评估
    ✓ 置信度分布 KL 散度趋势分析
    ✓ 生成趋势报告（多日指标变化折线图）
    ✓ 指标持续退化 3 天时触发告警

  参数调优 / 算法更新:
    ✓ A/B 自动对比测试（新版 vs 旧版全量指标对比）
    ✓ 自动判定结论，输出 ab_report.json
    ⚠️ 仅当自动判定为"部分退化"时，才提示人工决策
```

---

## 七、实现优先级与路线图（v2.0）

### 7.1 分阶段实施计划（全自动优先）

| 阶段 | 内容 | 预估工时 | 依赖 | 人工介入 |
|------|------|---------|------|---------|
| **V1** | Layer 1 技术验证基础（文件健康度 + 元信息 + 黑帧/静帧 + AV 同步） | 3h | 无 | **无** |
| **V2** | Layer 0 自动验证（球-筐穿越检测 + 音频事件检测，生成伪标注） | 6h | V1 | **无** |
| **V3** | Layer 2 检测质量验证（P/R/F1 + 裁剪窗口，基于 V2 伪标注） | 4h | V2 | **无** |
| **V4** | Layer 1 感知质量（VMAF + BRISQUE + 拼接点 SSIM） | 3h | V1 | **无** |
| **V5** | 基准回归框架（baseline.json + 退化告警 + KL 散度分析） | 3h | V3+V4 | **无** |
| **V6** | Layer 0 增强（轨迹物理校验 + 计分板 OCR） | 6h | V2 | **无** |
| **V7** | Layer 3 全自动批量评估 + A/B 自动对比 | 3h | V5 | **无** |
| **V8（可选）** | 人工标注补充（为 demo 视频创建高精度人工标注） | 2h | V3 | **需要** |

> V8 为可选优化项。V1~V7 均可在完全无人工介入的情况下运行。

### 7.2 与 task.md 的关系（v2.0 更新）

本验证体系属于 **Phase 1 T-1.7（端到端测试与调优）** 的深化扩展，可标记为：

| ID | 任务 | 优先级 | 状态 | 依赖 | 人工介入 |
|----|------|--------|------|------|---------|
| T-1.8 | 输出视频效果验证体系 v2.0 | P1 | ⬜ | T-1.7 | 无 |
| T-1.8.0 | **Layer 0 多信号自动进球验证（球-筐穿越+音频）** | **P0** | ⬜ | T-1.7.1 | **无** |
| T-1.8.1 | Layer 1 视频技术质量自动验证（含 VMAF/AV同步） | P0 | ⬜ | T-1.7.1 | 无 |
| T-1.8.2 | Layer 2 检测质量验证（基于 L0 自动伪标注） | P1 | ⬜ | T-1.8.0 | 无 |
| T-1.8.3 | Layer 2 裁剪窗口质量验证 | P1 | ⬜ | T-1.8.2 | 无 |
| T-1.8.4 | 感知质量（BRISQUE + 拼接点 SSIM） | P2 | ⬜ | T-1.8.1 | 无 |
| T-1.8.5 | 指标基准回归 + 置信度 KL 散度告警 | P2 | ⬜ | T-1.8.2 | 无 |
| T-1.8.6 | Layer 0 增强（轨迹物理校验 + 计分板 OCR） | P2 | ⬜ | T-1.8.0 | 无 |
| T-1.8.7 | Layer 3 批量评估 + A/B 自动对比 | P2 | ⬜ | T-1.8.5 | 无 |
| T-1.8.8 | 人工标注补充（可选，提升基准质量） | P3 | ⬜ | T-1.8.2 | **需要** |

---

## 八、关键技术方案（v2.0 补充）

### 8.1 黑帧检测方案

```bash
# 使用 ffmpeg blackdetect 滤镜
ffmpeg -i output.mp4 -vf "blackdetect=d=0.05:pix_th=0.10" -an -f null - 2>&1 | grep "black_start"
```

### 8.2 静帧检测方案

```bash
# 使用 ffmpeg freezedetect 滤镜
ffmpeg -i output.mp4 -vf "freezedetect=n=-60dB:d=0.5" -an -f null - 2>&1 | grep "freeze_start"
```

### 8.3 检测结果提取方案

当前 pipeline 将 AI 检测结果写入临时目录的 `detection_result.json`，处理完成后临时目录被清理。**需要增加功能**：将检测结果同时保存到输出目录，供验证脚本使用。

建议在 pipeline 中增加：
```
输出目录/
├── goalcut_demo_1_goalcut.mp4       # 集锦视频
├── goalcut_demo_1_detection.json    # AI 检测结果（进球事件列表）
└── goalcut_demo_1_clips.json        # 裁剪区间信息
```

### 8.5 Layer 0 球-筐穿越检测核心实现

```python
import cv2
import numpy as np

def detect_hoop_crossing(video_path, detection_ts, search_window=2.0):
    """
    在 detection_ts ± search_window 秒内，检测球是否穿越篮筐平面
    返回: (crossing_score, crossing_ts)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int((detection_ts - search_window) * fps)
    end_frame   = int((detection_ts + search_window) * fps)

    ball_positions = []   # [(frame_idx, cx, cy)]
    hoop_center    = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        # 假设已有 YOLO 检测结果 JSON，从中读取当前帧的 ball/hoop bbox
        ball_bbox, hoop_bbox = get_yolo_detections(frame_idx)

        if hoop_bbox:
            hx = (hoop_bbox[0] + hoop_bbox[2]) / 2
            hy = (hoop_bbox[1] + hoop_bbox[3]) / 2
            hoop_center = (hx, hy, hoop_bbox[2] - hoop_bbox[0])  # cx, cy, width

        if ball_bbox and hoop_center:
            bx = (ball_bbox[0] + ball_bbox[2]) / 2
            by = (ball_bbox[1] + ball_bbox[3]) / 2
            ball_positions.append((frame_idx, bx, by))

    cap.release()

    if not hoop_center or len(ball_positions) < 5:
        return 0.0, None

    hcx, hcy, hwidth = hoop_center
    roi_margin = hwidth * 1.5

    # 筛选在篮筐 ROI 内的球轨迹点
    roi_points = [(f, x, y) for f, x, y in ball_positions
                  if abs(x - hcx) < roi_margin]

    if len(roi_points) < 4:
        return 0.3, None

    y_coords = [y for _, _, y in roi_points]

    # 检测从 hcy 上方到下方的单调穿越
    has_above = any(y < hcy - 10 for y in y_coords)
    has_below = any(y > hcy + 10 for y in y_coords)

    if has_above and has_below:
        # 确认穿越方向（从上到下）
        first_above_idx = next(i for i, y in enumerate(y_coords) if y < hcy - 10)
        last_below_idx  = len(y_coords) - 1 - next(
            i for i, y in enumerate(reversed(y_coords)) if y > hcy + 10)

        if first_above_idx < last_below_idx:
            crossing_frame = roi_points[last_below_idx][0]
            crossing_ts = crossing_frame / fps
            score = min(1.0, len(roi_points) / 8.0)  # 轨迹越完整分数越高
            return score, crossing_ts

    return 0.3, None  # 在筐附近但未完整穿越
```

### 8.6 置信度 KL 散度计算

```python
import numpy as np
from scipy.stats import entropy

def compute_kl_divergence(current_confidences, baseline_hist_path):
    """计算当前置信度分布与历史基准的 KL 散度"""
    with open(baseline_hist_path) as f:
        baseline_hist = json.load(f)  # {"bins": [...], "counts": [...]}

    bins = baseline_hist["bins"]
    baseline_dist = np.array(baseline_hist["counts"], dtype=float)
    baseline_dist /= baseline_dist.sum()

    current_counts, _ = np.histogram(current_confidences, bins=bins)
    current_dist = current_counts.astype(float)
    current_dist = (current_dist + 1e-8) / (current_dist.sum() + 1e-8 * len(bins))

    kl = entropy(current_dist, baseline_dist)
    return kl
```

### 8.4 事件匹配算法伪代码

```
function match_events(detections, annotations, tolerance=3.0):
    sort detections by timestamp
    sort annotations by timestamp
    matched_annotations = set()
    results = []

    for det in detections:
        best_match = None
        best_diff = infinity
        for ann in annotations:
            if ann.id in matched_annotations:
                continue
            diff = abs(det.timestamp - ann.timestamp)
            if diff <= tolerance and diff < best_diff:
                best_match = ann
                best_diff = diff

        if best_match:
            matched_annotations.add(best_match.id)
            results.append(TP(det, best_match, best_diff))
        else:
            results.append(FP(det))

    for ann in annotations:
        if ann.id not in matched_annotations:
            results.append(FN(ann))

    return results
```

---

## 九、非功能性需求

| 需求 | 说明 |
|------|------|
| 可扩展性 | 标注格式和验证脚本应支持新增检测维度（如 OCR、音频），无需大改 |
| 可重复性 | 相同输入 + 相同参数的验证结果应完全一致 |
| 可追溯性 | 每份验证报告应记录 GoalCut 版本、配置参数、运行环境等信息 |
| 低侵入性 | 验证工具不应修改现有代码逻辑，仅作为外部观测手段 |
| 增量扩展 | 新增测试视频只需添加视频文件和标注 JSON，无需修改脚本 |

---

## 十、验收标准（v2.0）

### Phase 1 增补验收标准（全自动，无需人工介入）

- [ ] `verify_layer0.py` 实现球-筐穿越检测和音频事件检测，生成 `auto_annotations.json`
- [ ] `test/test.sh` 增加 Layer 1 技术质量验证（含 VMAF/BRISQUE/AV 同步）
- [ ] `verify_detection.py` 支持 `--annotation-mode=hybrid`，Layer 2 可无人工标注运行
- [ ] Layer 0 融合评分 ≥ 0.70 的事件自动标注为 HIGH，写入 `test/annotations/auto/`
- [ ] 验证报告以 JSON 格式输出到 `test/reports/` 目录
- [ ] `./evaluate_batch.sh` 一条命令完成 L0→L1→L2 全链路自动评估
- [ ] `compare_baseline.py` 实现基准回归对比，REGRESSION 时输出非零退出码（可集成 CI）

### 长期目标

- [ ] Layer 0 增加轨迹物理校验和计分板 OCR，融合置信度准确率 ≥ 90%（以人工标注为基准）
- [ ] 建立 ≥ 10 个视频的测试集（允许以 L0 HIGH 级别自动标注替代人工标注）
- [ ] CI 集成全自动验证，指标退化时自动阻断 PR 合并
- [ ] 整体 F1 ≥ 82%，时间偏差 ≤ 1.0s，VMAF ≥ 85
- [ ] L0 自动标注的置信度 HIGH 比例 ≥ 75%（说明视频质量稳定可自动评估）
- [ ] 置信度 KL 散度监控接入告警，分布漂移 > 0.15 时自动通知
