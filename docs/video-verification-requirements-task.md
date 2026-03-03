# GoalCut 输出视频效果验证 — 任务拆分与跟踪

> 最后更新：2026-03-03（同步 requirements v2.0，新增 Layer 0 全自动进球验证）
>
> 需求来源：[video-verification-requirements.md](./video-verification-requirements.md)
>
> 状态说明：⬜ 待开始 | 🔵 进行中 | ✅ 已完成 | ⏸️ 已暂停 | ❌ 已取消
>
> 优先级说明：P0 = 阻塞级 | P1 = 核心 | P2 = 重要 | P3 = 优化增强

---

## 一、任务依赖关系总览

```
V0: 前置准备（Pipeline 改造）
  T-V0.1 Pipeline 输出检测结果持久化 ⬜
  T-V0.2 Pipeline 输出裁剪区间信息 ⬜
         │
         ▼
V-L0: Layer 0 多信号自动进球验证（★ v2.0 新增）    V1: Layer 1 技术质量验证
  T-L0.1 球-筐穿越检测 ⬜                           T-V1.1 文件健康度检查 ⬜
  T-L0.2 球体轨迹物理校验 ⬜         ◄── 并行 ───►  T-V1.2 视频元信息校验 ⬜
  T-L0.3 音频事件检测 ⬜                             T-V1.3 感知画质评分 ⬜
  T-L0.4 计分板 OCR 检测 ⬜                          T-V1.4 音视频同步检测 ⬜
  T-L0.5 融合评分 & 伪标注生成 ⬜                    T-V1.5 黑帧/静帧检测 ⬜
         │                                                   │
         └─────────────────────┬───────────────────────────┘
                               ▼
                  V2: Layer 2 检测质量验证（混合标注模式）
                    T-V2.1 标注格式规范 ⬜
                    T-V2.2 标注加载器（hybrid 模式）⬜
                    T-V2.3 事件匹配算法 ⬜
                    T-V2.4 P/R/F1 + 时间偏差指标 ⬜
                    T-V2.5 裁剪窗口质量验证 ⬜
                    T-V2.6 Layer 2 报告生成 ⬜
                               │
                               ▼
                  V3: Layer 3 全自动回归评估
                    T-V3.1 指标基准回归框架 ⬜
                    T-V3.2 置信度 KL 散度分析 ⬜
                    T-V3.3 批量评估编排脚本 ⬜
                    T-V3.4 A/B 自动对比测试 ⬜

V4（可选）: 人工标注补充
  T-V4.1 demo_1 人工标注 ⬜
  T-V4.2 demo_2 人工标注 ⬜
```

---

## 二、V0 — 前置准备：Pipeline 改造（P0 阻塞级）

> **必须最先完成**。当前 Pipeline 处理完成后清理临时目录，导致 AI 检测结果和裁剪区间信息丢失。Layer 0 和 Layer 2 均依赖这些数据。
>
> 预计工期：0.5 天（2.5h）

### 现状分析

| 问题 | 当前行为 | 影响 |
|------|---------|------|
| 检测结果丢失 | `detection_result.json` 写入临时目录，处理后 `os.RemoveAll` 清理 | Layer 0/2 验证无法获取 AI 检测结果 |
| 裁剪区间无记录 | `mergeClipRanges` 结果仅在内存中使用，未持久化 | 无法验证裁剪窗口合理性 |
| 无结构化输出 | 仅通过 `fmt.Printf` 日志输出，无法被程序解析 | 自动化验证无法读取处理结果 |

### 任务明细

| ID | 任务 | 优先级 | 状态 | 依赖 | 预估工时 | 交付物 | 验收标准 |
|----|------|--------|------|------|---------|--------|---------|
| **T-V0.1** | **Pipeline 输出检测结果持久化** | P0 | ⬜ | 无 | 1.5h | 修改 `pipeline.go` | |
| T-V0.1.1 | 将 AI 检测结果 JSON 复制到输出目录 | P0 | ⬜ | 无 | 0.5h | 输出 `{name}_detection.json` 到集锦视频同级目录 | 文件内容与 `detection_result.json` 一致，含所有 GoalEvent 字段 |
| T-V0.1.2 | 增加 `-save-meta` 命令行参数 | P0 | ⬜ | T-V0.1.1 | 0.5h | `main.go` 增加 flag；默认 false，测试时传 true | 不加参数时行为与现在一致 |
| T-V0.1.3 | 验证检测结果文件可正确解析 | P0 | ⬜ | T-V0.1.1 | 0.5h | 手动验证或单元测试 | JSON 文件可被 `json.Unmarshal` 正确解析 |
| **T-V0.2** | **Pipeline 输出裁剪区间信息** | P0 | ⬜ | T-V0.1 | 1h | 修改 `pipeline.go` | |
| T-V0.2.1 | 将 `mergeClipRanges` 结果序列化为 JSON | P0 | ⬜ | T-V0.1.1 | 0.5h | 输出 `{name}_clips.json`，含每个 ClipRange 的 start/end | JSON 结构包含 `clips` 数组和 `source_events` 数组 |
| T-V0.2.2 | clips.json 包含源视频元信息摘要 | P0 | ⬜ | T-V0.2.1 | 0.5h | JSON 中含 `video_info`（时长/分辨率/帧率） | 验证脚本可从此文件获取源视频信息 |

### V0 验收标准

- [ ] 运行 GoalCut 后，输出目录新增 `{name}_detection.json` 和 `{name}_clips.json`
- [ ] 不加 `-save-meta` 参数时，不输出元数据文件
- [ ] JSON 文件格式正确，可被 Python/Go 正常解析

### clips.json 输出格式规范

```json
{
  "video_info": {
    "source_file": "goalcut_demo_1.MP4",
    "duration": 33.0,
    "width": 1280, "height": 720,
    "fps": 30.0, "codec": "h264"
  },
  "config": {
    "before_seconds": 3, "after_seconds": 3,
    "merge_threshold": 4, "confidence_threshold": 0.3
  },
  "events": [
    { "frame_index": 25, "timestamp": 8.33, "confidence": 0.72 },
    { "frame_index": 66, "timestamp": 22.0, "confidence": 0.65 }
  ],
  "clips": [
    { "index": 0, "start": 5.33, "end": 11.33, "duration": 6.0 },
    { "index": 1, "start": 19.0, "end": 25.0, "duration": 6.0 }
  ]
}
```

---

## 三、V-L0 — Layer 0 多信号自动进球验证（P0 阻塞级，★ v2.0 新增）

> **全新模块**。通过四个独立信号自动验证进球，无需人工标注，生成的伪标注供 Layer 2 使用。
>
> 预计工期：1.5 天（6h）

### 信号权重与设计目标

| 信号 | 权重 | 实现方式 |
|------|------|---------|
| 球-筐穿越检测 | 40% | 追踪球心 Y 轴在篮筐 ROI 内的单调下行轨迹 |
| 球体轨迹物理校验 | 25% | 对球运动轨迹做抛物线拟合，R² ≥ 0.85 |
| 音频事件检测 | 20% | 哨声/欢呼声/球网声频谱分析 |
| 计分板 OCR | 15% | 检测进球前后分数变化（仅正式比赛） |

融合评分 ≥ 0.70 → HIGH（等价人工标注），0.45~0.69 → MEDIUM，< 0.45 → LOW（疑似误检）

### 任务明细

| ID | 任务 | 优先级 | 状态 | 依赖 | 预估工时 | 交付物 | 验收标准 |
|----|------|--------|------|------|---------|--------|---------|
| **T-L0.1** | **球-筐穿越检测** | P0 | ⬜ | T-V0.1 | 2h | `test/verify_layer0.py` 中 `detect_hoop_crossing()` | |
| T-L0.1.1 | 从 detection.json 加载逐帧球/筐检测框 | P0 | ⬜ | T-V0.1 | 0.5h | 函数 `load_frame_detections()` | 正确解析 detection.json，按帧索引索引 |
| T-L0.1.2 | 定位篮筐中心并建立 ROI 区域 | P0 | ⬜ | T-L0.1.1 | 0.5h | 函数 `build_hoop_roi()` | ROI 宽度 = 1.5×hoop_width，稳定跨帧 |
| T-L0.1.3 | 在 ROI 内追踪球心 Y 坐标序列 | P0 | ⬜ | T-L0.1.2 | 0.5h | 返回 `[(frame, cx, cy)]` 轨迹点 | 连续帧间无跳变（遮挡时插值） |
| T-L0.1.4 | 判定从筐上方到下方的单调穿越 | P0 | ⬜ | T-L0.1.3 | 0.5h | 函数 `check_crossing()` 返回 `(score, ts)` | 完整穿越 score=1.0，部分遮挡 score=0.6 |
| **T-L0.2** | **球体轨迹物理校验** | P1 | ⬜ | T-V0.1 | 1h | `verify_layer0.py` 中 `validate_trajectory_physics()` | |
| T-L0.2.1 | 对球心 Y 坐标序列做二次曲线拟合 | P1 | ⬜ | T-L0.1.3 | 0.5h | 函数 `fit_parabola()` 返回 `(a, b, c, r2)` | np.polyfit 或 scipy.optimize.curve_fit |
| T-L0.2.2 | 验证物理合法性（方向 + R² + 时长） | P1 | ⬜ | T-L0.2.1 | 0.5h | 函数 `validate_physics()` 返回 `physics_score` | a > 0（重力方向），R² ≥ 0.85，时长 ≤ 2s |
| **T-L0.3** | **音频事件检测** | P1 | ⬜ | 无 | 1.5h | `verify_layer0.py` 中 `detect_audio_events()` | |
| T-L0.3.1 | 从视频提取音频轨道（16kHz 单声道） | P1 | ⬜ | 无 | 0.25h | 调用 `ffmpeg -vn -ar 16000 -ac 1` | 输出临时 wav，处理完毕后清除 |
| T-L0.3.2 | 计算音频能量包络（短时帧能量） | P1 | ⬜ | T-L0.3.1 | 0.5h | 函数 `compute_energy_envelope()` | 窗口 0.5s，步进 0.1s |
| T-L0.3.3 | 检测哨声（2~4kHz 短促峰值） | P1 | ⬜ | T-L0.3.2 | 0.25h | 函数 `detect_whistle()` | 频率范围内能量峰值且持续 < 0.5s |
| T-L0.3.4 | 检测欢呼/撞击声（宽频带突发能量） | P1 | ⬜ | T-L0.3.2 | 0.25h | 函数 `detect_crowd_noise()` | 背景基线上突增 ≥ 10dB |
| T-L0.3.5 | 在检测时间戳 ±3s 内查找音频事件 | P1 | ⬜ | T-L0.3.3,T-L0.3.4 | 0.25h | 函数 `find_audio_event_near()` 返回 `audio_score` | 无音轨时返回 0.5（中性） |
| **T-L0.4** | **计分板 OCR 变化检测** | P2 | ⬜ | T-V0.1 | 1h | `verify_layer0.py` 中 `detect_score_change()` | |
| T-L0.4.1 | 检测视频中是否存在计分板区域 | P2 | ⬜ | 无 | 0.5h | 函数 `find_scoreboard_roi()` | 正式比赛成功定位，野球场返回 None |
| T-L0.4.2 | 对进球前后帧做 OCR 读取分数 | P2 | ⬜ | T-L0.4.1 | 0.25h | 调用 tesseract，限定数字字符集 | 分数读取成功率 ≥ 80%（正式比赛） |
| T-L0.4.3 | 计算分数差并映射到置信分 | P2 | ⬜ | T-L0.4.2 | 0.25h | delta ∈ {2,3}→1.0；delta=0→0.0；无计分板→0.5 | |
| **T-L0.5** | **多信号融合评分 & 伪标注生成** | P0 | ⬜ | T-L0.1, T-L0.3 | 1h | `verify_layer0.py` 主函数 | |
| T-L0.5.1 | 实现加权融合评分公式 | P0 | ⬜ | T-L0.1~T-L0.4 | 0.5h | 函数 `fuse_signals()` | w1=0.40,w2=0.25,w3=0.20,w4=0.15；无 OCR 时自动重归一化 |
| T-L0.5.2 | 按置信等级分类（HIGH/MEDIUM/LOW） | P0 | ⬜ | T-L0.5.1 | 0.25h | 函数 `classify_confidence()` | ≥0.70→HIGH，0.45~0.69→MEDIUM，<0.45→LOW |
| T-L0.5.3 | 生成 `auto_annotations.json` | P0 | ⬜ | T-L0.5.2 | 0.25h | 输出到 `test/annotations/auto/{name}.json` | 格式符合需求文档 L0.7 节规范；可被 verify_detection.py 加载 |

### V-L0 验收标准

- [ ] `python3 test/verify_layer0.py --video <input.mp4> --detection <detection.json>` 可独立运行
- [ ] 对 demo_1 和 demo_2 运行，输出 `auto_annotations.json` 到 `test/annotations/auto/`
- [ ] HIGH 置信事件时间戳与实际进球偏差 ≤ 1.0s（以人工核查为准）
- [ ] 无计分板的视频（野球场）正常运行，ocr_score 返回 0.5（中性）
- [ ] 音轨缺失时不报错，audio_score 返回 0.5

---

## 四、V1 — Layer 1 视频技术质量验证（P0 阻塞级）

> 全自动验证，不依赖任何标注。v2.0 新增感知画质评分（VMAF/BRISQUE）和 AV 同步检测。
>
> 预计工期：1 天（5h）

### 任务明细

| ID | 任务 | 优先级 | 状态 | 依赖 | 预估工时 | 交付物 | 验收标准 |
|----|------|--------|------|------|---------|--------|---------|
| **T-V1.1** | **输出文件健康度检查** | P0 | ⬜ | 无 | 1h | `test/verify_technical.sh` | |
| T-V1.1.1 | 文件存在性 & 非空检查 | P0 | ⬜ | 无 | 0.25h | `check_file_exists()` | 文件存在且 size > 0 |
| T-V1.1.2 | ffprobe 可播放性检查 | P0 | ⬜ | 无 | 0.25h | `check_playable()` | ffprobe 返回码 = 0 |
| T-V1.1.3 | 文件完整性检查（末尾 seek） | P0 | ⬜ | 无 | 0.25h | `check_integrity()` | `ffmpeg -sseof -1` 返回 0 |
| T-V1.1.4 | 容器格式验证 | P0 | ⬜ | 无 | 0.25h | `check_container()` | format_name 包含 "mov,mp4" |
| **T-V1.2** | **视频元信息校验** | P0 | ⬜ | T-V0.2 | 1.5h | 同 `verify_technical.sh` | |
| T-V1.2.1 | 编码格式验证（H.264） | P0 | ⬜ | 无 | 0.25h | `check_codec()` | codec_name = "h264" |
| T-V1.2.2 | 分辨率一致性验证 | P0 | ⬜ | T-V0.2 | 0.25h | `check_resolution()` | 宽高与源视频一致 |
| T-V1.2.3 | 帧率合理性验证 | P0 | ⬜ | T-V0.2 | 0.25h | `check_fps()` | 与源视频帧率偏差 ≤ 1fps |
| T-V1.2.4 | 时长合理性 + 精度验证 | P0 | ⬜ | T-V0.2 | 0.25h | `check_duration()` | 0 < 时长 < 源时长；与 clips.json 预期偏差 ≤ 0.5s |
| T-V1.2.5 | 码率范围验证 | P1 | ⬜ | T-V0.2 | 0.25h | `check_bitrate()` | 在源码率 30%~200% 之间 |
| T-V1.2.6 | 片段数量匹配验证 | P1 | ⬜ | T-V0.2 | 0.25h | `check_clip_count()` | 拼接片段数 ≤ detection.json 中检测事件数 |
| **T-V1.3** | **感知画质评分（★ v2.0 新增）** | P1 | ⬜ | T-V0.2 | 1.5h | `test/verify_quality.py` | |
| T-V1.3.1 | VMAF 评分（Full Reference，对比源视频片段） | P1 | ⬜ | 无 | 0.5h | 函数 `compute_vmaf()` 调用 ffmpeg libvmaf | VMAF ≥ 85；无 libvmaf 时降级到 BRISQUE |
| T-V1.3.2 | BRISQUE 评分（No Reference，无参考盲评） | P1 | ⬜ | 无 | 0.5h | 函数 `compute_brisque()` 调用 skimage | BRISQUE ≤ 40；每 30 帧采样一次 |
| T-V1.3.3 | 拼接点 SSIM 检测 | P2 | ⬜ | 无 | 0.5h | 函数 `check_splice_ssim()` | 自动定位场景切换点；切换点前后 ±3 帧 SSIM ≥ 0.80 |
| **T-V1.4** | **音视频同步检测（★ v2.0 新增）** | P1 | ⬜ | 无 | 0.5h | 同 `verify_technical.sh` | |
| T-V1.4.1 | 检测 A-V 时间戳偏移 | P1 | ⬜ | 无 | 0.25h | `check_av_sync()` | \|audio_start - video_start\| ≤ 40ms |
| T-V1.4.2 | 检测拼接点音频连续性 | P2 | ⬜ | 无 | 0.25h | `check_audio_continuity()` | 拼接点静音间隔 ≤ 50ms |
| **T-V1.5** | **拼接完整性（黑帧/静帧）** | P2 | ⬜ | T-V1.1 | 0.5h | 同 `verify_technical.sh` | |
| T-V1.5.1 | 使用 ffmpeg blackdetect 检测黑帧 | P2 | ⬜ | 无 | 0.25h | `check_black_frames()` | 连续黑帧 ≤ 2 帧 |
| T-V1.5.2 | 使用 ffmpeg freezedetect 检测静帧 | P2 | ⬜ | 无 | 0.25h | `check_frozen_frames()` | 连续静帧 ≤ 1 秒 |
| **T-V1.6** | **Layer 1 集成与报告** | P0 | ⬜ | T-V1.1~T-V1.5 | 1h | | |
| T-V1.6.1 | JSON 验证报告输出（含 VMAF/BRISQUE） | P0 | ⬜ | T-V1.1~T-V1.5 | 0.5h | 输出到 `test/reports/{name}_layer1.json` | 格式符合需求文档 3.4 节规范 |
| T-V1.6.2 | 集成到 `test/test.sh` | P0 | ⬜ | T-V1.6.1 | 0.5h | 修改 `test.sh` 调用 `verify_technical.sh` + `verify_quality.py` | 原有测试流程不受影响 |

### V1 验收标准

- [ ] `bash test/verify_technical.sh <output_video> <source_video>` 可独立运行
- [ ] `python3 test/verify_quality.py <output_video> <source_video>` 可独立运行
- [ ] 对 2 个 demo 输出视频验证全部 PASS（VMAF ≥ 85，BRISQUE ≤ 40，AV 同步 ≤ 40ms）
- [ ] Layer 1 报告输出到 `test/reports/`，JSON 格式正确

---

## 五、V2 — Layer 2 检测质量验证（P1 核心，混合标注模式）

> v2.0 重大变化：不再强依赖人工标注，默认使用 Layer 0 自动伪标注，支持 `hybrid` 模式自动选择标注源。
>
> 预计工期：1 天（4h）

### 任务明细

| ID | 任务 | 优先级 | 状态 | 依赖 | 预估工时 | 交付物 | 验收标准 |
|----|------|--------|------|------|---------|--------|---------|
| **T-V2.1** | **标注格式规范定义（v2.0）** | P1 | ⬜ | 无 | 0.5h | `test/annotations/annotation_spec.md` | |
| T-V2.1.1 | 定义统一 JSON Schema（兼容 human/auto/hybrid） | P1 | ⬜ | 无 | 0.25h | 含 `annotation_type`、`confidence`、`auto_signals` 字段 | 人工和自动标注共用同一 schema |
| T-V2.1.2 | 编写标注操作指南（人工标注部分） | P2 | ⬜ | T-V2.1.1 | 0.25h | 时间精度说明、进球定义说明 | 新标注者可照此指南独立完成 |
| **T-V2.2** | **标注加载器（hybrid 优先级逻辑）** | P1 | ⬜ | T-L0.5 | 1h | `test/verify_detection.py` 中 `load_annotations()` | |
| T-V2.2.1 | 实现 hybrid 模式优先级逻辑 | P1 | ⬜ | T-L0.5 | 0.5h | 优先加载 `annotations/human/`，缺失时降级到 `auto/` HIGH | human > auto HIGH > auto MEDIUM |
| T-V2.2.2 | 支持 `--annotation-mode=auto/human/hybrid` 参数 | P1 | ⬜ | T-V2.2.1 | 0.25h | CLI 参数解析 | 默认 hybrid；human 模式无标注时报错 |
| T-V2.2.3 | MEDIUM 标注仅参与召回率，不计入准确率分母 | P1 | ⬜ | T-V2.2.1 | 0.25h | 过滤规则实现 | 与需求文档 L0.8 节一致 |
| **T-V2.3** | **事件匹配算法实现** | P1 | ⬜ | T-V2.2 | 1h | 同 `verify_detection.py` | |
| T-V2.3.1 | 实现贪心匹配算法（含可配置 tolerance） | P1 | ⬜ | T-V2.2 | 0.75h | `match_events(detections, annotations, tolerance)` | 默认 tolerance=3.0s；正确区分 TP/FP/FN |
| T-V2.3.2 | 处理 MEDIUM 标注的特殊匹配规则 | P1 | ⬜ | T-V2.3.1 | 0.25h | MEDIUM 匹配到的 TP 不计入准确率分子 | |
| **T-V2.4** | **指标计算** | P1 | ⬜ | T-V2.3 | 0.5h | 同 `verify_detection.py` | |
| T-V2.4.1 | 计算 Precision、Recall、F1 Score | P1 | ⬜ | T-V2.3 | 0.25h | `compute_metrics()` | 公式正确；处理边界情况（0检测/0标注） |
| T-V2.4.2 | 计算时间偏差均值和 P90 | P1 | ⬜ | T-V2.3 | 0.25h | `compute_time_deviation()` | 仅对 TP 事件计算 |
| **T-V2.5** | **裁剪窗口质量验证** | P1 | ⬜ | T-V0.2, T-V2.3 | 0.5h | 同 `verify_detection.py` | |
| T-V2.5.1 | 验证进球可见性（标注时间在裁剪区间内） | P1 | ⬜ | T-V0.2, T-V2.3 | 0.25h | `check_goal_visibility()` | 标注 timestamp 落在某个 clip [start, end] 内 |
| T-V2.5.2 | 计算中心偏移度 + 缓冲时间验证 | P1 | ⬜ | T-V2.5.1 | 0.25h | `compute_clip_quality()` | \|t_goal - t_center\| / duration ≤ 0.4 |
| **T-V2.6** | **Layer 2 验证报告生成** | P1 | ⬜ | T-V2.4, T-V2.5 | 0.5h | | |
| T-V2.6.1 | JSON 报告输出（含标注来源说明） | P1 | ⬜ | T-V2.4, T-V2.5 | 0.25h | 输出到 `test/reports/{name}_layer2.json` | 含 `annotation_source` 字段（human/auto/hybrid） |
| T-V2.6.2 | 终端可读文本摘要 + PASS/FAIL 判定 | P1 | ⬜ | T-V2.6.1 | 0.25h | 彩色终端输出 | 含目标阈值对比（P≥85%，R≥80%，F1≥82%） |

### V2 验收标准

- [ ] `python3 test/verify_detection.py --detection <det.json> --clips <clips.json> --annotation-mode=hybrid` 可运行
- [ ] 无人工标注时自动降级到 Layer 0 伪标注，全程无需人工干预
- [ ] 对 demo_1 和 demo_2 运行，输出包含完整 P/R/F1/时间偏差/裁剪质量的 JSON 报告
- [ ] 边界情况（无检测/无标注/全匹配）处理正确，不崩溃

---

## 六、V3 — Layer 3 全自动回归评估（P2 重要）

> v2.0 核心改造：移除 MOS 人工评分强依赖，全部改为自动化指标回归和 A/B 对比。
>
> 预计工期：1 天（5h）

### 任务明细

| ID | 任务 | 优先级 | 状态 | 依赖 | 预估工时 | 交付物 | 验收标准 |
|----|------|--------|------|------|---------|--------|---------|
| **T-V3.1** | **指标基准回归框架** | P2 | ⬜ | T-V2.6 | 1.5h | `test/compare_baseline.py` | |
| T-V3.1.1 | 设计并创建 `baseline.json` 格式 | P2 | ⬜ | 无 | 0.25h | `test/reports/baseline.json` | 含各指标值和退化阈值（详见需求文档 F3.3） |
| T-V3.1.2 | 实现指标对比和退化判定逻辑 | P2 | ⬜ | T-V3.1.1 | 0.5h | `compare_metrics()` | REGRESSION/WARN/IMPROVED/STABLE 四级判定 |
| T-V3.1.3 | REGRESSION 时返回非零退出码 | P2 | ⬜ | T-V3.1.2 | 0.25h | 退出码 1 | 可被 CI 捕获并阻断合并 |
| T-V3.1.4 | 支持 `--update-baseline` 参数更新基准 | P2 | ⬜ | T-V3.1.2 | 0.5h | 将当前指标写入 baseline.json | 需显式传参，不自动覆盖 |
| **T-V3.2** | **置信度分布健康度分析（★ v2.0 新增）** | P2 | ⬜ | T-V2.4 | 1h | `test/analyze_confidence.py` | |
| T-V3.2.1 | 从 detection.json 提取所有检测置信度 | P2 | ⬜ | T-V0.1 | 0.25h | 函数 `load_confidences()` | 支持批量（多个 detection.json） |
| T-V3.2.2 | 计算置信度统计量（均值/P10/高置信占比） | P2 | ⬜ | T-V3.2.1 | 0.25h | 函数 `compute_confidence_stats()` | 目标：mean≥0.65，p10≥0.50，high_ratio≥60% |
| T-V3.2.3 | 计算与历史基准的 KL 散度 | P2 | ⬜ | T-V3.2.1 | 0.5h | 函数 `compute_kl_divergence()` | KL > 0.15 输出 WARN；> 0.20 触发告警 |
| **T-V3.3** | **全自动批量评估编排脚本** | P2 | ⬜ | T-V1.6, T-V2.6 | 1.5h | `test/evaluate_batch.sh` | |
| T-V3.3.1 | 脚本主循环：遍历测试集所有视频 | P2 | ⬜ | 无 | 0.5h | 自动发现 input 目录中的视频 | 命名规则：`{name}.mp4` 对应 `annotations/[human|auto]/{name}.json` |
| T-V3.3.2 | 对每个视频顺序执行 L0→L1→L2 全链路 | P2 | ⬜ | T-V3.3.1 | 0.5h | 串联调用各验证工具 | 单个视频失败不阻塞后续 |
| T-V3.3.3 | 合并所有报告，运行基准回归对比 | P2 | ⬜ | T-V3.1, T-V3.3.2 | 0.25h | 输出 `test/reports/summary.json` | 含整体/分场景/分难度指标统计 |
| T-V3.3.4 | 生成终端可读汇总报告文本 | P2 | ⬜ | T-V3.3.3 | 0.25h | 格式参照需求文档 5.3 节全自动版 | 含基准回归对比结论和人工介入建议 |
| **T-V3.4** | **A/B 自动对比测试** | P3 | ⬜ | T-V3.3 | 1h | `test/ab_compare.sh` | |
| T-V3.4.1 | 接受两组输出目录，分别计算指标 | P3 | ⬜ | T-V3.3 | 0.5h | 脚本参数：`--version-a <dir> --version-b <dir>` | |
| T-V3.4.2 | 自动判定结论（优于/持平/部分退化/劣于） | P3 | ⬜ | T-V3.4.1 | 0.25h | 函数 `judge_ab_result()` | 判定逻辑见需求文档 F3.4 |
| T-V3.4.3 | 输出 `ab_report.json` | P3 | ⬜ | T-V3.4.2 | 0.25h | 逐视频和整体指标 delta 对比表 | |

### V3 验收标准

- [ ] `bash test/evaluate_batch.sh` 一条命令完成 L0→L1→L2 全链路批量评估，无需人工介入
- [ ] `python3 test/compare_baseline.py` 自动输出 REGRESSION/STABLE/IMPROVED 结论
- [ ] REGRESSION 时退出码 ≠ 0，可被 CI 捕获
- [ ] 置信度 KL 散度 > 0.15 时终端输出 WARN
- [ ] 新增测试视频只需添加文件到 `test/input/`，无需修改脚本

---

## 七、V4 — 人工标注补充（P3 可选，不阻塞主线）

> **完全可选**。V-L0 自动伪标注已可替代人工标注完成 L2 验证。此步骤仅用于提升基准精度或覆盖 Hard 场景。
>
> 触发条件（满足任一时建议执行）：L0 某视频 MEDIUM 标注比例 > 40%；当前 F1 < 80%；接入大量 Hard 样本。

### 任务明细

| ID | 任务 | 优先级 | 状态 | 依赖 | 预估工时 | 交付物 | 验收标准 |
|----|------|--------|------|------|---------|--------|---------|
| **T-V4.1** | **demo_1 视频人工标注** | P3 | ⬜ | T-V2.1 | 1h | `test/annotations/human/goalcut_demo_1.json` | 时间精度 ≤ 0.5s，格式通过 Schema 校验 |
| **T-V4.2** | **demo_2 视频人工标注** | P3 | ⬜ | T-V2.1 | 1h | `test/annotations/human/goalcut_demo_2.json` | 时间精度 ≤ 0.5s，格式通过 Schema 校验 |

---

## 八、任务统计（v2.0）

| 阶段 | 任务数 | 预估总工时 | 优先级 | 当前状态 | 人工介入 |
|------|--------|-----------|--------|---------|---------|
| V0：前置准备（Pipeline 改造） | 6 | 2.5h | P0 | ⬜ 待开始 | **无** |
| V-L0：Layer 0 多信号自动进球验证 | 13 | 6h | P0/P1/P2 | ⬜ 待开始 | **无** |
| V1：Layer 1 技术质量验证（含VMAF） | 18 | 5h | P0/P1/P2 | ⬜ 待开始 | **无** |
| V2：Layer 2 检测质量验证（混合模式） | 14 | 4h | P1 | ⬜ 待开始 | **无** |
| V3：Layer 3 全自动回归评估 | 14 | 5h | P2/P3 | ⬜ 待开始 | **无** |
| V4：人工标注补充（可选） | 2 | 2h | P3 | ⬜ 待开始 | **需要** |
| **合计（V0~V3，全自动）** | **65 个任务** | **22.5h** | | | **无** |

---

## 九、实施路线图（v2.0 全自动优先）

```
Week 1:

  Day 1 上午 (2.5h):
    └── V0: Pipeline 改造 ──────────── P0 阻塞，必须最先完成

  Day 1 下午 (5h，并行):
    ├── V-L0 基础版: T-L0.1（穿越检测）+ T-L0.3（音频）+ T-L0.5（融合输出）
    └── V1  基础版: T-V1.1（健康度）+ T-V1.2（元信息）

  Day 2 (8h):
    ├── V-L0 完整版: T-L0.2（物理校验）+ T-L0.4（OCR）
    └── V1  完整版: T-V1.3（VMAF/BRISQUE）+ T-V1.4（AV同步）+ T-V1.5（黑帧）+ T-V1.6（集成）

  Day 3 (4h):
    └── V2: Layer 2 完整实现（T-V2.1~T-V2.6，依赖 L0 伪标注就绪）

Week 2:

  Day 4~5 (5h):
    ├── V3: Layer 3 回归框架（T-V3.1~T-V3.3）
    └── V3: A/B 对比测试（T-V3.4，可选）

按需（不阻塞主线）:
    └── V4: 人工标注补充（2h）
```

### 关键里程碑

| 里程碑 | 完成条件 | 预计时间 | 人工介入 |
|--------|---------|---------|---------|
| **M1: 数据管道就绪** | V0 完成，Pipeline 输出 detection.json 和 clips.json | Day 1 上午 | 无 |
| **M2: 自动验证核心** | V-L0 基础版 + V1 基础版，可自动生成伪标注并做技术验证 | Day 2 | 无 |
| **M3: 质量指标可量化** | V1 完整 + V2 完成，P/R/F1/VMAF 全部可自动计算 | Day 3 | 无 |
| **M4: 全链路自动评估** | V3 完成，一条命令批量评估 + 基准回归告警 | Day 5 | 无 |
| **M5（可选）** | V4 完成，Hard 样本有高精度人工基准 | 按需 | **需要** |

---

## 十、与主项目 task.md 的关系（v2.0）

| task.md ID | 对应本文档阶段 | 状态 |
|------------|--------------|------|
| T-1.8 | 整个验证体系 v2.0 | ⬜ |
| T-1.8.0 — Layer 0 多信号自动进球验证 | **V-L0** | ⬜ |
| T-1.8.1 — Layer 1 视频技术质量验证（含VMAF） | V1 | ⬜ |
| T-1.8.2 — Layer 2 检测质量验证（混合标注） | V2 | ⬜ |
| T-1.8.3 — Layer 2 裁剪窗口质量验证 | V2 (T-V2.5) | ⬜ |
| T-1.8.4 — 感知质量（BRISQUE + SSIM） | V1 (T-V1.3) | ⬜ |
| T-1.8.5 — 指标基准回归 + KL 散度告警 | V3 (T-V3.1~T-V3.2) | ⬜ |
| T-1.8.6 — Layer 3 批量评估 + A/B 自动对比 | V3 (T-V3.3~T-V3.4) | ⬜ |
| T-1.8.7 — 人工标注补充（可选） | V4 | ⬜ |

---

## 十一、注意事项（v2.0）

1. **V0 必须先行**：Pipeline 改造是 V-L0 和 V2 的前置依赖，范围最小化，不改变现有核心逻辑
2. **V-L0 是核心创新**：用自动伪标注替代人工标注，是 v2.0 最重要的模块，优先级等同于 P0
3. **VMAF 依赖检查**：运行 T-V1.3.1 前需确认 ffmpeg 已编译 libvmaf；无 libvmaf 时降级到 BRISQUE（无参考模式）
4. **低侵入性**：所有验证工具作为独立脚本存在于 `test/` 目录，不修改核心业务代码（V0 除外）
5. **增量可扩展**：新增测试视频只需添加文件到 `test/input/`，Layer 0 自动生成伪标注，无需人工标注
6. **V4 延后执行**：人工标注仅在自动标注可信度不足时触发，不阻塞主线开发
7. **持续更新**：每完成一个子任务，及时更新本文档中对应任务的状态
