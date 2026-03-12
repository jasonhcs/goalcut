# GoalCut VLM 独立验证方案 v2.0

## 1. 概述

VLM (Vision-Language Model) 验证器是 GoalCut 进球检测的**独立校验工具**，使用视觉语言大模型对视频进行审查。

### 1.1 两种工作模式

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **全视频扫描 (`--scan`)** | VLM 独立扫描整段视频，找出所有进球时间点，再与算法比对 | 评估算法整体 P/R/F1，精确定位漏检和误检 |
| **片段验证 (`--input`)** | 仅审查算法已剪出的片段，判断是否真的进球 | 快速验证算法输出质量 |

### 1.2 设计目标

- **技术正交**：验证方式与原进球检测算法完全独立，避免共同失败
- **全自动化**：无需人工参与，全程自动完成验证和反馈生成
- **闭环反馈**：输出结构化 JSON 报告，Agent 可直接消费并调参
- **全覆盖扫描**：v2.0 新增全视频滑动窗口扫描，不遗漏任何进球

### 1.3 与原算法的技术正交性

| 维度 | 原进球检测算法 | VLM 验证器 |
|------|--------------|-----------|
| 视觉理解 | YOLO 检测框 + 像素级 IOU 追踪 + 几何规则 | 端到端语义理解，无检测框 |
| 决策方式 | 5 通道手工规则 + 阈值融合 | 大模型 zero-shot 推理 |
| 音频利用 | STFT 频谱能量分析 | **不使用音频**（纯视觉） |
| 运动分析 | Farneback 光流 / 帧差法 / 轨迹拟合 | 帧序列高层语义理解 |
| 失败模式 | 检测框丢失、遮挡、阈值敏感 | 语义误解、幻觉 |

两者**失败模式完全不同**，可有效交叉验证。

## 2. 架构

### 2.1 全视频扫描模式（v2.0 核心）

```
输入: 原始比赛视频 + 算法检测结果(detection.json)
                │
                ▼
       ┌─────────────────────┐
       │  滑动窗口切片器      │  将视频按 6s 窗口、2s 步长切片
       │                     │  例: 90min 视频 → ~2700 个窗口
       └──────────┬──────────┘
                  │ 每个窗口
                  ▼
       ┌─────────────────────┐
       │  帧采样器            │  每个窗口均匀抽取 8 帧
       │  (ffmpeg)            │
       └──────────┬──────────┘
                  │ 帧图像序列
                  ▼
       ┌─────────────────────┐
       │  VLM 审查器          │  向大模型发送帧序列 + 扫描 Prompt
       │  (GPT-4o / Gemini / │  询问: "这组帧中是否发生了篮球进球?"
       │   Qwen-VL / Ollama) │  返回: {is_goal, confidence, reasoning}
       └──────────┬──────────┘
                  │ 每个窗口的判定
                  ▼
       ┌─────────────────────┐
       │  去重合并器          │  相邻窗口检测到同一进球时合并（±4s 窗口）
       │                     │  输出: VLM 发现的所有进球列表
       └──────────┬──────────┘
                  │ VLM 进球列表
                  ▼
       ┌─────────────────────┐
       │  交叉比对引擎        │  VLM 进球列表 vs 算法 detection.json
       │                     │  TP: 双方都检测到 → 确认
       │                     │  FP: 仅算法检测到 → 误检
       │                     │  FN: 仅 VLM 检测到 → 漏检
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │  反馈报告生成        │  输出: P/R/F1 + FP/FN 详情 + 调参建议
       │  (vlm_feedback.py)  │  Agent 可直接消费的 feedback.json
       └─────────────────────┘
```

### 2.2 片段验证模式

```
输入: GoalCut 剪辑出的各个 clip（进球前3s + 进球后1s）
                │
                ▼
       ┌─────────────────────┐
       │  帧采样器            │  每个 clip 均匀抽取 8 帧
       └──────────┬──────────┘
                  │ 帧图像序列
                  ▼
       ┌─────────────────────┐
       │  VLM 审查器          │  判断: 这个 clip 中有没有进球?
       └──────────┬──────────┘
                  │ JSON 审查结果
                  ▼
       ┌─────────────────────┐
       │  结果比对引擎        │  VLM vs 原算法 → TP_AGREE / FP_SUSPECT
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │  反馈报告生成        │
       └─────────────────────┘
```

## 3. 文件结构

```
test/vlm_verify/
├── verify_vlm.py           # 核心验证器（含全视频扫描 + 片段验证）
├── vlm_feedback.py          # 闭环反馈引擎
├── run_vlm_verify.sh        # 一键运行/批量验证入口脚本
├── verify_vlm_config.yaml   # 配置文件
└── reports/                 # 报告输出目录（自动生成，已 gitignore）
```

## 4. Agent 闭环工作流

### 4.1 全视频扫描闭环（推荐）

```
1. GoalCut 处理视频 → detection.json
2. verify_vlm.py --scan 扫描整段视频 → vlm_scan_report.json
   （VLM 独立找出所有进球时间点）
3. 自动与 detection.json 交叉比对 → 输出 TP/FP/FN
4. vlm_feedback.py 分析 → feedback.json（含参数修改指令）
5. Agent 读取 feedback.json，调整算法参数/代码
6. 回到步骤 1
```

### 4.2 交叉比对分类

| 算法结果 | VLM 扫描结果 | 分类 | 含义 |
|---------|-------------|------|------|
| 检测到进球 | VLM 也发现 | TP (True Positive) | 算法正确 |
| 检测到进球 | VLM 未发现 | FP (False Positive) | 算法误检 |
| 未检测到 | VLM 发现了 | FN (False Negative) | 算法漏检 |

### 4.3 feedback.json 输出格式

```json
{
  "version": "2.0",
  "mode": "scan",
  "verdict": {
    "status": "NEEDS_TUNING",
    "message": "算法存在 2 个漏检，需要优化召回率",
    "agreement_rate": 0.75,
    "assessment": "good"
  },
  "scan_analysis": {
    "metrics": {
      "tp": 6, "fp": 1, "fn": 2,
      "precision": 0.857, "recall": 0.750, "f1": 0.800
    },
    "fp_count": 1,
    "fn_count": 2,
    "actions": [...]
  },
  "parameter_changes": [
    {
      "parameter": "detection.confidence_threshold",
      "current": 0.55,
      "suggested": 0.62,
      "reason": "消除 1 个低置信度误检"
    }
  ],
  "recommended_actions": [
    {
      "action": "fix_false_negatives_high_priority",
      "description": "2 个高确信漏检，算法在这些时间段完全未检测到进球",
      "risk": "high"
    }
  ]
}
```

`verdict.status` 可能的值：

| 状态 | 含义 |
|------|------|
| `PASS` | 算法与 VLM 扫描完全一致，无需调整 |
| `PASS_WITH_SUGGESTIONS` | 整体良好，有小幅优化空间 |
| `NEEDS_TUNING` | 存在漏检或较多误检，需要调优 |
| `NEEDS_INVESTIGATION` | 存在较严重问题，需深入排查 |

## 5. 使用方式

### 5.1 环境要求

- Python 3.7+（推荐使用 `/opt/miniconda3/bin/python3`）
- ffmpeg / ffprobe
- VLM API Key（使用 ollama 本地模型时不需要）

### 5.2 全视频扫描（推荐）

VLM 独立扫描整段视频，找出所有进球，再与算法结果比对：

```bash
VLM_API_KEY=sk-xxx /opt/miniconda3/bin/python3 test/vlm_verify/verify_vlm.py \
  --scan test/input/video.mp4 \
  --detection output/video_detection.json \
  --output test/vlm_verify/reports/vlm_scan_report.json
```

自定义扫描窗口和步长：

```bash
/opt/miniconda3/bin/python3 test/vlm_verify/verify_vlm.py \
  --scan video.mp4 \
  --detection det.json \
  --scan-window 8.0 \
  --scan-step 3.0
```

### 5.3 仅 VLM 扫描（不需要算法结果）

不提供 `--detection` 时，VLM 独立扫描输出进球列表：

```bash
VLM_API_KEY=sk-xxx /opt/miniconda3/bin/python3 test/vlm_verify/verify_vlm.py \
  --scan test/input/video.mp4 \
  --output test/vlm_verify/reports/vlm_goals.json
```

### 5.4 片段验证模式

```bash
VLM_API_KEY=sk-xxx /opt/miniconda3/bin/python3 test/vlm_verify/verify_vlm.py \
  --input test/input/video.mp4 \
  --detection output/video_detection.json \
  --output test/vlm_verify/reports/vlm_report.json
```

### 5.5 验证已剪辑的集锦

```bash
VLM_API_KEY=sk-xxx /opt/miniconda3/bin/python3 test/vlm_verify/verify_vlm.py \
  --highlight output/video_goalcut.mp4 \
  --detection output/video_detection.json \
  --output test/vlm_verify/reports/vlm_report.json
```

### 5.6 生成闭环反馈

```bash
/opt/miniconda3/bin/python3 test/vlm_verify/vlm_feedback.py \
  --report test/vlm_verify/reports/vlm_scan_report.json \
  --config configs/config.yaml \
  --output test/vlm_verify/reports/feedback.json
```

### 5.7 一键完整流程（扫描 + 反馈）

```bash
VLM_API_KEY=sk-xxx ./test/vlm_verify/run_vlm_verify.sh \
  --scan test/input/video.mp4 \
  --detection output/video_detection.json \
  --feedback
```

### 5.8 批量扫描

```bash
VLM_API_KEY=sk-xxx ./test/vlm_verify/run_vlm_verify.sh --batch --scan-mode --feedback
```

### 5.9 使用本地 Ollama（免费，无需 API Key）

```bash
VLM_PROVIDER=ollama VLM_MODEL=llava:13b \
  ./test/vlm_verify/run_vlm_verify.sh --scan video.mp4 --detection det.json
```

### 5.10 带 Ground Truth 对比

```bash
VLM_API_KEY=sk-xxx /opt/miniconda3/bin/python3 test/vlm_verify/verify_vlm.py \
  --scan video.mp4 --detection det.json \
  --ground-truth test/ground_truth/video.json
```

## 6. 全视频扫描详解

### 6.1 滑动窗口策略

```
视频: |====================================================|
      0s                                              3600s

窗口: |--6s--|
        |--6s--|
          |--6s--|
            ...

步长 2s，窗口 6s → 每个时间点被 3 个窗口覆盖
→ 同一个进球大概率被 2~3 个窗口捕捉到
→ 去重合并后得到准确的进球时间
```

### 6.2 扫描参数选择

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|---------|
| `--scan-window` | 6.0s | 窗口大小 | 增大可覆盖更完整动作，但 API 成本增加 |
| `--scan-step` | 2.0s | 步长 | 减小可减少漏检，但扫描更慢 |
| `--scan-threshold` | 0.5 | VLM 判定阈值 | 降低可减少漏检，但增加误报 |
| `--scan-merge-window` | 4.0s | 去重窗口 | 增大合并更多相邻检测 |

### 6.3 成本估算

| 视频时长 | 窗口数 | 使用 GPT-4o (约$0.01/窗口) | 使用 Ollama (免费) |
|---------|--------|--------------------------|-------------------|
| 5 min | ~150 | ~$1.50 | 免费 (~15min) |
| 30 min | ~900 | ~$9.00 | 免费 (~90min) |
| 90 min | ~2700 | ~$27.00 | 免费 (~4.5h) |

建议：开发阶段用 Ollama 免费扫描，关键评测用 GPT-4o/Gemini 获取更准确结果。

### 6.4 扫描报告示例

```json
{
  "version": "2.0",
  "mode": "scan",
  "vlm_goals": [
    {
      "index": 0,
      "timestamp": 45.3,
      "confidence": 0.92,
      "reasoning": "球从篮筐上方穿过篮网落下，明显的投篮命中",
      "window": [42.0, 48.0]
    },
    {
      "index": 1,
      "timestamp": 127.8,
      "confidence": 0.85,
      "reasoning": "观察到上篮动作，球进入篮筐",
      "window": [124.0, 130.0]
    }
  ],
  "comparison": {
    "metrics": {"tp": 2, "fp": 1, "fn": 0, "precision": 0.667, "recall": 1.0, "f1": 0.800},
    "matched_pairs": [...],
    "false_positives": [...],
    "false_negatives": [],
    "tuning_suggestions": [...]
  }
}
```

## 7. 支持的 VLM 提供商

| Provider | 模型 | 是否需要 API Key | 环境变量 | 适用场景 |
|----------|------|-----------------|---------|---------|
| **openai** | GPT-4o | 是 | `VLM_API_KEY` | 最高精度 |
| **gemini** | Gemini 1.5 Pro | 是 | `VLM_API_KEY` | 高性价比 |
| **dashscope** | Qwen-VL-Max | 是 | `VLM_API_KEY` | 国内直连 |
| **ollama** | LLaVA 等 | 否（本地部署） | - | 免费批量运行 |

通过环境变量切换：

```bash
export VLM_PROVIDER=gemini    # openai / gemini / ollama / dashscope
export VLM_MODEL=gemini-1.5-pro  # 可选，留空使用默认
export VLM_API_KEY=your-key
export VLM_BASE_URL=           # 可选，自定义 API 端点
```

## 8. 配置说明

配置文件位于 `test/vlm_verify/verify_vlm_config.yaml`，主要配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `vlm.provider` | openai | 模型提供商 |
| `vlm.model` | (按 provider) | 模型名称 |
| `sampling.num_frames` | 8 | 每个窗口/片段采样帧数 |
| `sampling.max_dimension` | 768 | 帧图片最大边长(px) |
| `scan.window_seconds` | 6.0 | 扫描窗口大小(秒) |
| `scan.step_seconds` | 2.0 | 扫描步长(秒) |
| `scan.goal_confidence_threshold` | 0.5 | 扫描进球判定最低置信度 |
| `scan.merge_window` | 4.0 | 扫描去重合并窗口(秒) |
| `clip.before_seconds` | 3.0 | 进球前截取秒数 |
| `clip.after_seconds` | 1.0 | 进球后截取秒数 |
| `thresholds.min_agreement_rate` | 0.7 | 最低可接受一致率 |

所有配置均可通过命令行参数或环境变量覆盖。

## 9. VLM Prompt 设计

### 9.1 扫描模式 Prompt

用于全视频滑动窗口扫描，强调精确判断和低误报：

- 明确定义进球（投篮/上篮/扣篮命中）和非进球（弹框、运球经过等）
- 注明窗口时间跨度，帮助 VLM 理解帧间关系
- 要求返回 JSON：`{is_goal, confidence, reasoning, goal_frame_index}`
- `goal_frame_index` 用于反推进球在窗口中的精确时间

### 9.2 片段验证 Prompt

用于审查已剪辑的片段，与 v1.0 一致。

## 10. 退出码

| 退出码 | 含义 |
|--------|------|
| 0 | 验证通过，VLM 与原算法一致（无 FP/FN） |
| 1 | 存在误检或漏检 |

可在 CI/CD 中利用退出码触发后续流程。
