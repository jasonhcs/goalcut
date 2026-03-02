# GoalCut 输出结果视频效果验证 — 需求分析文档

> 文档版本：v1.0
>
> 创建日期：2026-03-03
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

### 2.1 三层验证模型

```
┌─────────────────────────────────────────────────────┐
│            Layer 3: 端到端效果评估                      │
│   人工标注对比 / A/B 测试 / 主观评分(MOS)              │
├─────────────────────────────────────────────────────┤
│            Layer 2: 检测质量验证                        │
│   准确率/召回率/F1 / 时间定位精度 / 置信度分析          │
├─────────────────────────────────────────────────────┤
│            Layer 1: 视频技术质量验证                     │
│   元信息校验 / 画质检测 / 拼接完整性 / 文件健康度        │
└─────────────────────────────────────────────────────┘
```

### 2.2 验证流程概览

```
输入: 原始视频 + 输出集锦视频 + AI检测结果JSON + (可选)人工标注
                    │
                    ▼
         ┌──────────────────┐
         │  Layer 1 技术验证  │  ← 全自动，每次构建必跑
         └────────┬─────────┘
                  │ PASS
                  ▼
         ┌──────────────────┐
         │  Layer 2 检测验证  │  ← 需要人工标注数据集，CI 中运行
         └────────┬─────────┘
                  │ PASS
                  ▼
         ┌──────────────────┐
         │  Layer 3 效果评估  │  ← 里程碑节点 / 参数调优时运行
         └──────────────────┘
                  │
                  ▼
            验证报告输出
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

### 3.3 实现方案

- **工具**：基于 `ffprobe` + `ffmpeg` 的 Shell 脚本或 Go 测试代码
- **触发方式**：集成到 `test/test.sh` 中，每次测试自动运行
- **输出格式**：结构化 JSON 验证报告

### 3.4 验证报告示例

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
      "frozen_frames": { "status": "PASS", "detail": "0 frozen segments" }
    }
  }
}
```

---

## 四、Layer 2 — 检测质量验证（基于标注数据）

### 4.1 需求概述

对 AI 检测结果的内容正确性进行量化评估，需要人工标注的 **Ground Truth（真值标注）** 作为对比基准。

### 4.2 Ground Truth 标注体系

#### F2.1 标注数据格式

为每个测试视频创建对应的标注文件，记录真实进球事件：

```json
{
  "video_file": "goalcut_demo_1.MP4",
  "duration": 33.0,
  "annotator": "human",
  "annotations": [
    {
      "event_id": 1,
      "timestamp": 8.5,
      "type": "goal",
      "description": "第一个进球，球从右侧投入",
      "time_window": { "start": 7.0, "end": 10.0 }
    },
    {
      "event_id": 2,
      "timestamp": 22.3,
      "type": "goal",
      "description": "第二个进球，上篮",
      "time_window": { "start": 20.5, "end": 24.0 }
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

#### F2.2 标注目录结构

```
test/
├── annotations/                    # 人工标注目录
│   ├── goalcut_demo_1.json        # demo_1 的真值标注
│   ├── goalcut_demo_2.json        # demo_2 的真值标注
│   └── README.md                  # 标注规范说明
├── input/                          # 测试输入视频
├── output/                         # 测试输出视频
└── reports/                        # 验证报告输出目录
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

## 五、Layer 3 — 端到端效果评估（人工参与）

### 5.1 需求概述

在关键里程碑节点或参数调优时，进行综合性的端到端效果评估，结合自动化指标与人工主观评价。

### 5.2 功能需求

#### F3.1 测试数据集管理

| 需求 | 说明 |
|------|------|
| 测试集规模 | 至少 10 段不同场景的篮球视频 |
| 场景覆盖 | 正式比赛（多机位/有记分牌）、野球场（单机位/无记分牌）各 50% |
| 画面多样性 | 横版/竖版、室内/室外、不同分辨率 |
| 难度分级 | Easy（清晰、标准角度）/ Medium（中等遮挡）/ Hard（远景/低质量/快速运动） |
| 标注完整性 | 每段视频都需要完整的 Ground Truth 标注 |

#### F3.2 自动化批量评估脚本

- 输入：测试数据集目录（视频 + 标注文件）
- 处理：对每个视频运行 GoalCut 全流程
- 输出：汇总评估报告

```
评估流程:
  for each video in test_dataset:
    1. 运行 GoalCut 生成集锦
    2. 运行 Layer 1 技术验证
    3. 运行 Layer 2 检测质量验证
    4. 收集中间数据（AI 检测结果 JSON、裁剪区间日志）
  生成汇总报告（含各视频/整体的指标统计）
```

#### F3.3 人工主观评分 (MOS)

对输出集锦进行主观质量评分，衡量最终用户感受：

| 评分维度 | 满分 | 评分标准 |
|---------|------|---------|
| 进球捕获完整性 | 5分 | 5=全部捕获, 4=遗漏1个, 3=遗漏2个, 2=遗漏过半, 1=基本未捕获 |
| 误检率感受 | 5分 | 5=无误检, 4=1处误检, 3=2处误检, 2=多处误检, 1=大量误检 |
| 裁剪时机 | 5分 | 5=完美包含进球前后, 4=略有偏移, 3=偏移但可接受, 2=偏移较大, 1=进球不在画面中 |
| 播放流畅度 | 5分 | 5=完全流畅, 4=轻微卡顿, 3=偶尔卡顿, 2=频繁卡顿, 1=无法正常播放 |
| 整体观看体验 | 5分 | 5=优秀, 4=良好, 3=一般, 2=较差, 1=很差 |

#### F3.4 A/B 对比测试

当调优参数或更新检测算法时，对比新旧版本的效果差异：

```
A/B 测试流程:
  1. 用同一测试集分别运行版本 A 和版本 B
  2. 对比两个版本的 Layer 2 指标差异
  3. 人工盲测对比观看体验
  4. 汇总对比报告
```

### 5.3 汇总评估报告示例

```
====================================
GoalCut 效果评估报告 — 2026-03-03
====================================

测试集: 10 个视频, 总时长 320s, 共 25 个标注进球

一、整体指标
  Precision:    88.0%  (22/25 检测为真)     目标: ≥85%  ✅
  Recall:       84.0%  (21/25 进球被检测)    目标: ≥80%  ✅
  F1 Score:     85.9%                        目标: ≥82%  ✅
  时间偏差(均):  0.68s                        目标: ≤1.0s ✅
  时间偏差(P90): 1.23s                        目标: ≤1.5s ✅

二、分场景指标
  正式比赛 (5个视频):  P=92%, R=88%, F1=90%
  野球场 (5个视频):    P=84%, R=80%, F1=82%

三、分难度指标
  Easy (4个):   P=95%, R=92%
  Medium (3个): P=87%, R=83%
  Hard (3个):   P=80%, R=75%

四、主观评分 (MOS)
  进球捕获: 4.2/5
  误检率:   4.0/5
  裁剪时机: 3.8/5
  流畅度:   4.5/5
  整体体验: 4.1/5

五、已知问题
  1. 竖版视频篮筐ROI区域判定偏差较大 (demo_2)
  2. 快速运动场景下球体检测置信度偏低
  3. 远景投篮（三分球）漏检率较高
```

---

## 六、验证工具需求

### 6.1 工具清单

| 工具 | 功能 | 优先级 | 实现形式 |
|------|------|--------|---------|
| `verify_technical.sh` | Layer 1 视频技术质量验证 | P0 | Shell 脚本，集成到 test.sh |
| `verify_detection.py` / `verify_detection.go` | Layer 2 检测质量验证 | P1 | Python 或 Go 脚本 |
| `evaluate_batch.sh` | Layer 3 批量端到端评估 | P2 | Shell 编排脚本 |
| 标注工具/格式规范 | Ground Truth 标注 | P1 | JSON 格式 + 标注规范文档 |
| 报告生成器 | 汇总验证结果 | P2 | 生成 JSON + 可读文本报告 |

### 6.2 集成方式

```
test/
├── test.sh                        # 现有集成测试（增强 Layer 1 验证）
├── verify_technical.sh            # Layer 1 视频技术质量验证脚本
├── verify_detection.py            # Layer 2 检测质量评估脚本
├── evaluate_batch.sh              # Layer 3 批量评估编排脚本
├── annotations/                   # Ground Truth 标注文件
│   ├── annotation_spec.md         # 标注规范
│   ├── goalcut_demo_1.json
│   └── goalcut_demo_2.json
├── reports/                       # 验证报告输出
│   └── .gitkeep
├── input/                         # 测试视频
└── output/                        # 输出视频
```

### 6.3 CI/CD 集成

```
CI 流水线验证策略:
  
  每次提交 (pre-merge):
    ✓ Layer 1 技术验证 — 快速，< 1min
    ✓ Layer 2 检测验证 — 使用 2 个 demo 视频，< 5min
  
  每日构建 (nightly):
    ✓ Layer 1 + Layer 2 — 使用完整测试集
    ✓ 生成趋势报告（指标是否退化）
  
  里程碑节点:
    ✓ Layer 1 + Layer 2 + Layer 3（含人工评分）
    ✓ A/B 对比测试（如有算法更新）
```

---

## 七、实现优先级与路线图

### 7.1 分阶段实施计划

| 阶段 | 内容 | 预估工时 | 依赖 |
|------|------|---------|------|
| **V1: 技术验证基础** | F1.1 + F1.2（输出文件健康度 + 元信息校验） | 3h | 无，可立即开始 |
| **V2: 标注体系建立** | F2.1 + F2.2（标注格式定义 + 为现有 2 个 demo 创建标注） | 2h | 需要人工观看视频标注 |
| **V3: 检测质量验证** | F2.3 + F2.4 + F2.5（指标计算 + 事件匹配 + 裁剪验证） | 5h | V2 |
| **V4: 拼接完整性** | F1.3（黑帧/静帧检测） | 2h | V1 |
| **V5: 批量评估框架** | F3.1 + F3.2（测试集管理 + 批量脚本） | 3h | V3 |
| **V6: 主观评估体系** | F3.3 + F3.4（MOS 评分 + A/B 测试） | 2h | V5，且仅在测试集规模足够时有意义 |

### 7.2 与 task.md 的关系

本验证体系属于 **Phase 1 T-1.7（端到端测试与调优）** 的深化扩展，可标记为：

| ID | 任务 | 优先级 | 状态 | 依赖 |
|----|------|--------|------|------|
| T-1.8 | 输出视频效果验证体系 | P1 | ⬜ | T-1.7 |
| T-1.8.1 | Layer 1 视频技术质量自动验证 | P0 | ⬜ | T-1.7.1 |
| T-1.8.2 | Ground Truth 标注体系与规范 | P1 | ⬜ | T-1.7.1 |
| T-1.8.3 | Layer 2 检测质量量化评估 | P1 | ⬜ | T-1.8.2 |
| T-1.8.4 | Layer 2 裁剪窗口质量验证 | P1 | ⬜ | T-1.8.3 |
| T-1.8.5 | Layer 1 拼接完整性检测 | P2 | ⬜ | T-1.8.1 |
| T-1.8.6 | Layer 3 批量评估框架 | P2 | ⬜ | T-1.8.3 |
| T-1.8.7 | Layer 3 主观评估体系 (MOS/A/B) | P3 | ⬜ | T-1.8.6 |

---

## 八、关键技术方案

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

## 十、验收标准

### Phase 1 增补验收标准

- [ ] `test/test.sh` 增加 Layer 1 视频技术质量自动验证
- [ ] 为现有 2 个 demo 视频创建 Ground Truth 标注文件
- [ ] 实现 Layer 2 检测质量评估脚本，输出 Precision/Recall/F1
- [ ] 验证报告以 JSON 格式输出到 `test/reports/` 目录
- [ ] 所有验证脚本可通过一条命令执行

### 长期目标

- [ ] 建立 ≥ 10 个视频的标注测试集
- [ ] Layer 3 批量评估框架就绪
- [ ] CI 集成自动化验证，指标退化时自动告警
- [ ] 整体 F1 ≥ 82%，时间偏差 ≤ 1.0s
