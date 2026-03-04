# GoalCut 通道准确率评估指南

> 适用版本：ai-engine v2.2（Phase 1.5 全量完成）
>
> 目标：量化评估各检测通道的 Precision / Recall / F1，为参数调优提供数据依据。

---

## 一、整体流程

```
① 标注视频（annotate.py）
        ↓
② 生成 Ground Truth JSON
        ↓
③ 端到端评估（eval.py）
        ↓
④ 查看各通道 P/R/F1 报告
        ↓
⑤ 参数调优 → 重新评估
```

**工具清单：**

| 文件 | 位置 | 功能 |
|---|---|---|
| `annotate.py` | `test/scripts/` | 交互式标注工具，输出 Ground Truth JSON |
| `eval.py` | `test/scripts/` | 端到端评估，自动提取帧 + 调用 channel_test |
| `channel_test.py` | `ai-engine/` | 各通道独立运行 + 评估核心（被 eval.py 调用） |

---

## 二、第一步：标注视频

### 2.1 启动标注工具

```bash
cd /path/to/goalcut

# 标注一个新视频（自动生成 test/ground_truth/<视频名>.json）
python test/scripts/annotate.py test/素材/formal_game_01.mp4

# 指定输出路径
python test/scripts/annotate.py test/素材/formal_game_01.mp4 \
    -o test/ground_truth/formal_game_01.json

# 加载已有标注继续编辑
python test/scripts/annotate.py test/素材/formal_game_01.mp4 \
    --load test/ground_truth/formal_game_01.json

# 开启截图预览功能（在标注后用 preview 命令生成帧截图）
python test/scripts/annotate.py test/素材/formal_game_01.mp4 \
    --preview-dir /tmp/goalcut_preview
```

### 2.2 标注命令参考

启动后进入交互 CLI：

```
annotate>
```

| 命令 | 说明 | 示例 |
|---|---|---|
| `add <时间> [备注]` | 添加进球时间戳 | `add 12.5`  `add 1:23 三分球` |
| `list` | 列出所有标注 | `list` |
| `del <编号>` | 删除指定编号 | `del 3` |
| `edit <编号> <时间>` | 修改时间戳 | `edit 2 1:24.5` |
| `preview [编号]` | 截取该时间点前后 5 帧截图 | `preview 1` 或 `preview`（全部） |
| `info` | 重新显示视频信息 | `info` |
| `save` | 保存 JSON 并退出 | `save` |
| `quit` | 不保存退出 | `quit` |
| `help` | 显示帮助 | `help` |

### 2.3 时间格式

```
12        →  12.0 秒
12.5      →  12.5 秒
1:23      →  1 分 23 秒（83 秒）
1:23.5    →  1 分 23.5 秒
1:02:34   →  1 小时 2 分 34 秒
```

### 2.4 标注技巧

1. **定位进球时刻**：进球时间戳取"球穿越篮筐平面"的那一帧，通常在球飞弧的底部。
2. **用 VLC/ffplay 辅助**：在看视频的同时，记录好时间后在 CLI 中 `add`。
3. **容差说明**：评估时默认容差为 ±3 秒，精确到 1 秒内即可，不必追求帧级精度。
4. **加 `--preview-dir`**：标注完后执行 `preview`，查看截图验证标注是否准确。

### 2.5 输出格式（Ground Truth JSON）

```json
{
  "video": "formal_game_01.mp4",
  "duration": 95.3,
  "total_goals": 5,
  "goals": [
    {"timestamp": 12.50, "note": "左侧三分"},
    {"timestamp": 34.80, "note": ""},
    {"timestamp": 51.20, "note": "快攻上篮"},
    {"timestamp": 67.10, "note": ""},
    {"timestamp": 88.40, "note": "压哨三分"}
  ]
}
```

---

## 三、第二步：运行评估

### 3.1 单视频评估（推荐入门）

```bash
# 全通道评估
python test/scripts/eval.py \
    --video test/素材/formal_game_01.mp4 \
    --gt test/ground_truth/formal_game_01.json

# 只评估视觉核心通道
python test/scripts/eval.py \
    --video test/素材/formal_game_01.mp4 \
    --gt test/ground_truth/formal_game_01.json \
    --channels 1a,1b,3

# 含音频通道（自动提取 WAV）
python test/scripts/eval.py \
    --video test/素材/formal_game_01.mp4 \
    --gt test/ground_truth/formal_game_01.json \
    --channels 1a,5 --with-audio

# 保存详细结果到 JSON
python test/scripts/eval.py \
    --video test/素材/formal_game_01.mp4 \
    --gt test/ground_truth/formal_game_01.json \
    --output /tmp/eval_result.json
```

### 3.2 批量评估（多视频）

先在 `test/ground_truth/` 准备好多个 GT JSON（文件名与视频文件名相同，扩展名不同），然后：

```bash
python test/scripts/eval.py \
    --batch test/ground_truth/ \
    --video-dir test/素材/ \
    --output-dir /tmp/eval_results/
```

批量模式会自动匹配同名视频文件，最终输出各通道在所有视频上的聚合指标。

### 3.3 主要参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--channels` | `all` | 指定通道，逗号分隔：`1a,1b,2,3,4,5` 或 `all` |
| `--with-audio` | 关 | 提取音频并评估通道 5（入网声/哨声） |
| `--sample-fps` | `3.0` | 帧提取采样率（越高越慢越准确） |
| `--confidence-threshold` | `0.55` | 事件最低置信度 |
| `--yolo-confidence` | `0.25` | YOLO 检测框最低置信度 |
| `--work-dir` | 系统 tmp | 帧/音频临时存放目录 |
| `--keep-frames` | 关 | 评估后保留提取的帧（方便调试） |

---

## 四、读懂评估报告

```
=================================================================
  评估报告: formal_game_01.mp4
=================================================================
  通道   名称                    事件数    TP    FP    FN    Prec     Rec      F1    耗时
  --------------------------------------------------------------
  1a     YOLO球体+IOU追踪+几何判定    8    4    4    1  50.0%  80.0%  61.5%   32.1s
  1b     启发式检测（球消失）          3    2    1    3  66.7%  40.0%  50.0%    0.1s
  2      人体运动模式检测              2    1    1    4  50.0%  20.0%  28.6%   18.4s
  3      局部运动突变（帧差分）         6    3    3    2  50.0%  60.0%  54.5%    0.2s
  4      篮网形变（光流法）             2    1    1    4  50.0%  20.0%  28.6%    1.2s
  5      音频事件检测                  1    1    0    4 100.0%  20.0%  33.3%    0.4s
  ==============================================================
```

### 字段含义

| 字段 | 含义 |
|---|---|
| **TP**（True Positive）| 正确检出的进球数 |
| **FP**（False Positive）| 误报数（检出但实际没进球） |
| **FN**（False Negative）| 漏报数（实际进球但未检出） |
| **Precision** | TP / (TP + FP)，检出结果中有多少是真进球 |
| **Recall** | TP / (TP + FN)，真实进球中有多少被检出 |
| **F1** | 2 × P × R / (P + R)，综合指标 |

### 目标指标（Phase 1.5 验收标准）

| 场景 | Precision | Recall | F1 |
|---|---|---|---|
| 野球场 / 业余比赛 | ≥ 80% | ≥ 75% | ≥ 77% |
| 正式比赛 | ≥ 90% | — | ≥ 80% |

---

## 五、单独调用 channel_test.py（高级）

如果已手动提取好帧，可以直接调用 `channel_test.py`，跳过 ffmpeg 步骤：

```bash
cd goalcut/ai-engine

# 全通道测试 + 对比 GT
python channel_test.py \
    --frames-dir /tmp/frames \
    --video-duration 95.3 \
    --ground-truth ../test/ground_truth/formal_game_01.json

# 只测试通道 1a
python channel_test.py \
    --frames-dir /tmp/frames \
    --video-duration 95.3 \
    --channels 1a \
    --ground-truth ../test/ground_truth/formal_game_01.json

# 调参：调整音频通道的突发阈值
python channel_test.py \
    --frames-dir /tmp/frames \
    --video-duration 95.3 \
    --channels 5 \
    --audio-file /tmp/audio.wav \
    --audio-burst-ratio 4.0 \
    --audio-max-duration 200

# 调参：调整篮网形变通道
python channel_test.py \
    --frames-dir /tmp/frames \
    --video-duration 95.3 \
    --channels 4 \
    --net-burst-ratio 3.0 \
    --net-cooldown 2.5

# 保存结果到 JSON
python channel_test.py \
    --frames-dir /tmp/frames \
    --video-duration 95.3 \
    --channels all \
    --ground-truth ../test/ground_truth/formal_game_01.json \
    --output /tmp/channel_result.json
```

### channel_test.py 可调参数

| 参数 | 默认 | 影响通道 | 说明 |
|---|---|---|---|
| `--sample-fps` | 3.0 | 全部 | 采样帧率，越高精度越好但越慢 |
| `--confidence-threshold` | 0.55 | 全部 | 事件输出阈值（降低→提升 Recall，升高→提升 Precision） |
| `--yolo-confidence` | 0.25 | 1a/1b/2 | YOLO 检测框阈值 |
| `--audio-burst-ratio` | 3.0 | 5 | 入网声突发倍数（越高误报越少） |
| `--audio-max-duration` | 300 | 5 | 入网声最长持续时间 ms |
| `--net-burst-ratio` | 2.5 | 4 | 篮网形变突发倍数 |
| `--net-cooldown` | 3.0 | 4 | 篮网形变事件冷静期（秒） |

---

## 六、推荐工作流

### 第一轮（建立基线）

```bash
# 1. 标注 3~5 段视频（野球场 + 正式比赛各几段）
python test/scripts/annotate.py test/素材/amateur_game_01_summer_camp.mp4 \
    -o test/ground_truth/amateur_game_01_summer_camp.json

python test/scripts/annotate.py test/素材/formal_game_01_auburn_highlights.mp4 \
    -o test/ground_truth/formal_game_01_auburn_highlights.json

# 2. 批量评估，查看基线指标
python test/scripts/eval.py \
    --batch test/ground_truth/ \
    --video-dir test/素材/ \
    --output-dir /tmp/eval_baseline/
```

### 第二轮（参数调优）

```bash
# 针对某通道调参，对比 GT
python channel_test.py \
    --frames-dir /tmp/frames \
    --video-duration 95 \
    --channels 1a \
    --confidence-threshold 0.45 \
    --ground-truth test/ground_truth/formal_game_01_auburn_highlights.json
```

### 第三轮（回归测试）

```bash
# 确认调参后整体指标不退化
python test/scripts/eval.py \
    --batch test/ground_truth/ \
    --video-dir test/素材/
```

---

## 七、文件约定

```
goalcut/
  test/
    scripts/
      annotate.py        # 标注工具
      eval.py            # 评估脚本
    ground_truth/        # GT JSON 文件（与视频同名）
      formal_game_01_auburn_highlights.json
      amateur_game_01_summer_camp.json
      ...
    素材/                # 测试视频文件
      formal_game_01_auburn_highlights.mp4
      amateur_game_01_summer_camp.mp4
      ...
  ai-engine/
    channel_test.py      # 通道独立测试工具（核心，被 eval.py 调用）
    detect.py            # 主检测引擎
    ...
```

**GT JSON 命名规则：** 与对应视频文件同名（仅扩展名改为 `.json`），批量模式依赖此约定自动匹配。

---

*参考来源：`test/scripts/annotate.py`、`test/scripts/eval.py`、`ai-engine/channel_test.py`*
