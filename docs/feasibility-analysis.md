# GoalCut —— 篮球进球集锦智能剪辑系统 可行性分析报告

## 一、项目概述

### 1.1 项目背景

篮球比赛视频时长通常在 1.5～2.5 小时之间，而观众和球迷往往只关注进球（得分）瞬间的精彩片段。手动从完整比赛视频中剪辑进球集锦耗时耗力，且依赖人工经验。GoalCut 旨在利用计算机视觉、音频分析和深度学习技术，自动从篮球比赛视频中识别进球事件，并智能生成进球集锦视频。

除正式比赛外，日常野球场（街头篮球、社区球场、业余球友活动等）场景同样有强烈的进球集锦需求。与正式比赛不同，野球场通常**没有记分牌、没有解说员、没有多机位和慢动作回放**，拍摄多为单机位固定视角。GoalCut 需要同时适配这两种差异显著的场景，通过**场景模式自适应**机制动态调整检测策略和权重分配。

### 1.2 项目目标

- **自动检测进球事件**：从完整篮球比赛视频中自动识别每一次得分（包括两分球、三分球、罚球）
- **智能集锦生成**：自动裁剪进球前后的精彩片段，拼接生成集锦视频
- **支持多种输入源**：本地视频文件、在线视频 URL
- **提供 Web 管理界面**：用户可上传视频、查看处理进度、预览/下载集锦
- **高性能处理**：利用 Go 的并发能力实现高效的视频处理流水线

### 1.3 目标用户

| 用户类型 | 使用场景 |
|---------|---------|
| 篮球自媒体博主 | 快速生成比赛集锦用于内容创作 |
| 球队教练/分析师 | 提取得分片段进行战术分析 |
| 球迷个人用户 | 收藏喜欢球队的进球精彩瞬间 |
| 体育媒体平台 | 批量处理比赛视频生成集锦内容 |
| 野球爱好者/球友群体 | 从日常打球视频中自动提取进球精彩瞬间，分享社交平台 |

---

## 二、可行性分析

### 2.1 技术可行性

#### 2.1.1 进球检测技术路线

篮球进球检测可通过以下多模态信号融合实现：

| 检测维度 | 技术方案 | 可行性评估 |
|---------|---------|-----------|
| **视觉检测** | 基于目标检测模型（YOLOv8/YOLOv9）检测篮球、篮筐，通过轨迹分析判断进球 | ✅ 成熟，YOLO 系列模型在体育场景已有广泛应用 |
| **篮网形变检测** | 基于光流法/帧差分检测篮网区域晃动幅度，进球后篮网有明显形变 | ✅ 可行，计算量低，野球场景下是关键辅助信号 |
| **记分牌 OCR** | 利用 OCR 技术实时读取比赛画面中的记分牌，分数变化即进球 | ✅ 成熟，PaddleOCR/EasyOCR 精度高（仅正式比赛可用） |
| **音频分析** | 检测解说员激动语调、观众欢呼声、篮球入网声等音频特征 | ✅ 可行，音频事件检测技术成熟 |
| **场景变化检测** | 进球后通常有回放/慢动作切换，检测画面转场 | ✅ 可行，基于帧差分/直方图分析（仅正式比赛可用） |
| **多模态融合** | 综合以上信号，根据场景模式动态调整权重，加权投票/模型融合判断进球事件 | ✅ 可行，提升准确率和召回率 |

> **场景模式说明**：系统支持「正式比赛模式」和「野球场模式」两种场景。野球场场景下记分牌 OCR 和场景变化检测不可用，系统将自动提升视觉轨迹检测和篮网形变检测的权重，确保检测准确率。详见 8.2 节置信度融合算法。

#### 2.1.2 视频处理技术

| 技术环节 | 方案 | 说明 |
|---------|------|------|
| 视频解码/编码 | FFmpeg (通过 Go 调用) | 工业级标准，支持所有主流格式 |
| 帧提取与分析 | GoCV (OpenCV Go binding) / FFmpeg 抽帧 | 支持高效帧级处理 |
| 视频裁剪拼接 | FFmpeg 命令行 / libav | 无损剪辑，速度快 |
| AI 推理 | Python 微服务 (PyTorch/ONNX Runtime) | GPU 加速推理，模型生态丰富 |

#### 2.1.3 技术成熟度总结

- **目标检测（YOLO）**：大量开源预训练模型和篮球场景数据集（如 NBA 数据集），训练门槛低
- **篮网形变检测**：基于光流法/帧差分的运动检测技术成熟，计算开销低，适合作为野球场景的核心辅助信号
- **OCR 技术**：PaddleOCR 对数字识别准确率 >99%，记分牌检测可靠（正式比赛场景）
- **音频分类**：基于 Mel 频谱 + CNN/Transformer 的音频事件检测已广泛应用，可扩展至入网声检测
- **FFmpeg**：视频处理工业标准，Go 生态中有成熟封装
- **结论：技术上完全可行，核心难点在于多模态信号的准确融合、进球事件的精确定位，以及不同场景模式（正式比赛 vs 野球场）下的自适应策略**

### 2.2 经济可行性

| 成本项 | 说明 | 预估 |
|-------|------|------|
| 开发人力 | 后端(Go) + AI算法 + 前端，预计 2-3 人 | 中等 |
| GPU 服务器 | AI 推理需 GPU（开发阶段可用消费级显卡） | 低～中 |
| 存储成本 | 视频文件较大，需对象存储 | 按量付费，可控 |
| 开源依赖 | FFmpeg、YOLO、PaddleOCR 等均开源免费 | 零成本 |
| 数据标注 | 需要一定量的篮球进球标注数据用于模型微调 | 低（可利用公开数据集） |

### 2.3 市场可行性

- 短视频/自媒体赛道持续增长，体育内容创作需求旺盛
- 目前市面上针对篮球进球集锦的自动化工具较少，存在明确的市场空白
- NBA、CBA 等联赛的球迷基数庞大，潜在用户群体广泛

### 2.4 风险评估

| 风险 | 等级 | 应对策略 |
|------|------|---------|
| 进球检测准确率不足 | 中 | 多模态融合 + 人工校准机制 |
| 不同比赛视频画面差异大 | 中 | 数据增强 + 场景自适应策略 |
| 野球场场景缺少 OCR/场景信号 | 中 | 场景模式自适应 + 提升视觉轨迹和篮网形变权重 |
| 视频处理性能瓶颈 | 低 | Go 并发流水线 + GPU 加速 |
| 视频版权问题 | 高 | 仅作为工具使用，用户自行负责版权 |
| 大文件上传/存储压力 | 低 | 分片上传 + 对象存储 |

---

## 三、功能模块设计

### 3.1 系统架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web 前端 (React)                         │
├─────────────────────────────────────────────────────────────────┤
│                      API Gateway (Go/Gin)                       │
├──────────┬──────────┬──────────┬──────────┬─────────────────────┤
│  用户模块  │  视频管理  │  任务调度  │  集锦生成  │    系统管理        │
├──────────┴──────────┴──────────┴──────────┴─────────────────────┤
│                   AI 分析引擎 (Python gRPC)                      │
├──────────┬──────────┬──────────┬──────────┬─────────────────────┤
│ 目标检测  │  OCR识别  │ 音频分析  │ 场景检测  │   多模态融合         │
├──────────┴──────────┴──────────┴──────────┴─────────────────────┤
│              基础设施层 (FFmpeg / 存储 / 消息队列)                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 模块详细说明

#### 模块一：用户管理模块

| 功能 | 说明 |
|------|------|
| 用户注册/登录 | 支持邮箱、手机号注册，JWT Token 认证 |
| 权限管理 | 基于 RBAC 的角色权限（普通用户/VIP/管理员） |
| 用量配额 | 不同等级用户的视频处理时长/次数配额管理 |
| 个人中心 | 查看历史任务、集锦视频、账户信息 |

#### 模块二：视频上传与管理模块

| 功能 | 说明 |
|------|------|
| 视频上传 | 支持分片上传（大文件），断点续传，支持 MP4/AVI/MKV/MOV 等格式 |
| URL 导入 | 支持通过视频链接导入（下载到服务端） |
| 视频列表 | 查看已上传视频，支持搜索、筛选、分页 |
| 视频预览 | 在线预览已上传的原始视频 |
| 视频元信息 | 自动提取视频时长、分辨率、帧率、编码格式等元数据 |
| 存储管理 | 对接对象存储（MinIO/S3），管理视频文件生命周期 |

#### 模块三：AI 分析引擎模块（核心）

##### 3.3.1 视觉检测子模块

| 功能 | 说明 |
|------|------|
| 关键帧提取 | 按固定间隔或自适应策略从视频中提取关键帧 |
| 篮球检测 | 使用 YOLOv8 检测画面中的篮球位置 |
| 篮筐检测 | 检测篮筐/篮网位置，建立投篮区域 ROI |
| 球体轨迹追踪 | 基于 DeepSORT/ByteTrack 追踪篮球运动轨迹 |
| 进球判定 | 分析球体轨迹与篮筐区域的交叉关系，判断是否进球 |
| 球员检测 | 检测球员位置，辅助判断投篮类型（两分/三分/罚球） |
| 篮网形变检测 | 基于光流法/帧差分检测篮网区域的晃动幅度，进球后篮网有明显形变，是野球场模式下的关键辅助信号 |

##### 3.3.2 记分牌 OCR 子模块

| 功能 | 说明 |
|------|------|
| 记分牌区域检测 | 自动定位比赛画面中记分牌的位置 |
| 分数识别 | OCR 识别当前比分数字 |
| 分数变化检测 | 对比相邻帧的分数变化，检测得分事件 |
| 得分归属判断 | 根据分数变化判断哪一方得分及得分分值 |

##### 3.3.3 音频分析子模块

| 功能 | 说明 |
|------|------|
| 音频提取 | 从视频中分离音频轨道 |
| 音频特征提取 | 提取 Mel 频谱、MFCC 等音频特征 |
| 欢呼声检测 | 检测观众欢呼/呐喊的音频事件（正式比赛模式） |
| 哨声检测 | 检测裁判哨声（可辅助判断暂停/犯规/得分，正式比赛模式） |
| 解说激动度分析 | 分析解说员语调变化，高激动度通常对应精彩进球（正式比赛模式） |
| 入网声检测 | 检测篮球入网的"刷"声特征，野球场模式下的重要音频信号 |
| 球友叫好声检测 | 检测现场球友击掌/叫好声，野球场模式下的辅助信号 |

##### 3.3.4 场景变化检测子模块

| 功能 | 说明 |
|------|------|
| 镜头切换检测 | 检测视频中的镜头转场（硬切/渐变） |
| 回放检测 | 检测慢动作回放片段（进球后通常有回放） |
| 特写镜头检测 | 检测球员特写（进球后常切到投篮球员特写） |

##### 3.3.5 多模态融合子模块

| 功能 | 说明 |
|------|------|
| 场景模式识别 | 自动检测视频前几秒是否存在记分牌、多机位等特征，判断场景类型（正式比赛/野球场），也支持用户手动指定 |
| 动态权重分配 | 根据场景模式动态调整各检测维度的权重（详见 8.2 节） |
| 时间轴对齐 | 将视觉、篮网形变、OCR、音频、场景各维度的检测结果对齐到统一时间轴 |
| 置信度融合 | 加权融合各维度的进球置信度分数 |
| 事件去重 | 对同一进球的多次检测进行去重合并 |
| 进球事件输出 | 输出最终的进球事件列表（时间戳、得分类型、置信度） |

**场景模式权重对照表**：

| 检测维度 | 正式比赛模式 | 野球场模式 | 说明 |
|---------|------------|-----------|------|
| 记分牌 OCR | 0.40 | 0.00（禁用） | 野球场无记分牌 |
| 视觉轨迹检测 | 0.30 | **0.70** | 野球场主力检测手段 |
| 篮网形变检测 | — | **0.20** | 进球后篮网晃动是可靠信号 |
| 音频分析 | 0.15 | **0.10** | 野球场仅检测入网声/叫好声 |
| 场景变化检测 | 0.15 | 0.00（禁用） | 野球场无回放/慢动作 |

#### 模块四：集锦生成模块

| 功能 | 说明 |
|------|------|
| 片段裁剪 | 根据进球事件时间戳，裁剪进球前 N 秒到进球后 M 秒的片段 |
| 时间窗口配置 | 用户可自定义进球片段的前后时间窗口（默认：前 8 秒，后 5 秒） |
| 片段排序 | 支持按时间顺序、精彩程度排序 |
| 转场效果 | 片段间添加转场效果（淡入淡出/黑场过渡等） |
| 背景音乐 | 可选添加背景音乐（淡入淡出处理） |
| 字幕叠加 | 在进球片段上叠加得分信息字幕（比分、时间、球员等） |
| 片头片尾 | 可选添加自定义片头片尾 |
| 多码率输出 | 支持输出不同分辨率/码率的集锦视频（1080p/720p/480p） |
| 视频导出 | 导出为 MP4 格式，支持在线预览和下载 |

#### 模块五：任务调度与处理模块

| 功能 | 说明 |
|------|------|
| 任务队列 | 基于消息队列（Redis/NATS）的异步任务队列 |
| 任务状态管理 | 任务生命周期管理（排队中/处理中/已完成/失败） |
| 进度跟踪 | 实时追踪视频处理进度（百分比 + 当前阶段） |
| WebSocket 推送 | 通过 WebSocket 实时推送处理进度到前端 |
| 任务重试 | 失败任务自动重试机制（指数退避） |
| 并发控制 | 根据服务器资源动态控制并发处理任务数 |
| 优先级调度 | 支持任务优先级（VIP 用户优先处理） |

#### 模块六：前端 Web 界面模块

| 功能 | 说明 |
|------|------|
| 首页/仪表盘 | 展示系统概览、最近任务、使用统计 |
| 视频上传页 | 拖拽上传、进度展示、格式校验 |
| 任务管理页 | 查看所有处理任务，实时进度展示 |
| 集锦预览页 | 在线视频播放器预览集锦，支持逐片段浏览 |
| 集锦编辑页 | 可视化时间轴编辑器，调整片段顺序/时长、人工校准 |
| 结果下载页 | 选择输出参数，下载最终集锦视频 |
| 个人中心 | 账户管理、历史记录、配额查看 |

#### 模块七：系统管理模块

| 功能 | 说明 |
|------|------|
| 模型管理 | AI 模型版本管理、热更新 |
| 系统监控 | CPU/GPU/内存/磁盘使用率监控 |
| 日志管理 | 结构化日志收集与查询 |
| 配置管理 | 系统参数在线配置（检测阈值、时间窗口等） |
| 数据统计 | 使用量统计、处理成功率统计 |

---

## 四、技术栈选型

### 4.1 后端技术栈（Go 为核心）

| 类别 | 技术选型 | 版本 | 选型理由 |
|------|---------|------|---------|
| **编程语言** | Go | 1.21+ | 高并发、高性能、编译型语言，适合视频处理流水线 |
| **Web 框架** | Gin | v1.9+ | 高性能 HTTP 框架，生态成熟，中间件丰富 |
| **ORM** | GORM | v2.0+ | Go 生态最流行的 ORM，功能完善 |
| **数据库** | PostgreSQL | 15+ | 功能强大的关系型数据库，JSON 支持好 |
| **缓存** | Redis | 7.0+ | 任务队列、缓存、Session 存储 |
| **消息队列** | NATS | 2.10+ | 轻量级高性能消息系统，Go 原生开发，适合任务调度 |
| **对象存储** | MinIO | 最新 | S3 兼容的开源对象存储，用于视频文件存储 |
| **认证** | JWT (golang-jwt) | v5 | 无状态 Token 认证 |
| **配置管理** | Viper | v1.18+ | 支持多格式配置文件和环境变量 |
| **日志** | Zap | v1.26+ | 高性能结构化日志 |
| **API 文档** | Swagger (swag) | 最新 | 自动生成 API 文档 |
| **gRPC** | grpc-go | v1.60+ | 与 Python AI 服务通信 |
| **WebSocket** | gorilla/websocket | v1.5+ | 实时进度推送 |
| **视频处理** | FFmpeg (命令行调用) | 6.0+ | 视频解码/编码/剪辑/拼接 |
| **视频信息** | ffprobe (Go 封装) | - | 提取视频元信息 |

### 4.2 AI 分析引擎技术栈（Python 微服务）

| 类别 | 技术选型 | 版本 | 选型理由 |
|------|---------|------|---------|
| **编程语言** | Python | 3.10+ | AI/ML 生态最丰富 |
| **深度学习框架** | PyTorch | 2.1+ | 灵活易用，社区活跃 |
| **目标检测** | Ultralytics YOLOv8 | 8.1+ | SOTA 目标检测，开箱即用，易于训练自定义模型 |
| **目标追踪** | ByteTrack / BoT-SORT | 最新 | 高性能多目标追踪 |
| **OCR** | PaddleOCR | 2.7+ | 高精度中英文 OCR，轻量部署 |
| **音频分析** | Librosa + torchaudio | 最新 | 音频特征提取和分析 |
| **推理加速** | ONNX Runtime | 1.16+ | 模型推理加速，支持 GPU/CPU |
| **gRPC 服务** | grpcio | 1.60+ | 与 Go 后端通信 |
| **视频处理** | OpenCV (cv2) | 4.8+ | 帧级图像处理 |
| **数据处理** | NumPy + Pandas | 最新 | 数据处理和分析 |
| **模型服务** | Triton Inference Server (可选) | 最新 | 高性能模型服务部署 |

### 4.3 前端技术栈

| 类别 | 技术选型 | 版本 | 选型理由 |
|------|---------|------|---------|
| **框架** | React | 18+ | 组件化开发，生态丰富 |
| **构建工具** | Vite | 5.0+ | 快速的开发构建工具 |
| **UI 组件库** | Ant Design | 5.0+ | 企业级 UI 组件库，组件丰富 |
| **状态管理** | Zustand | 4.0+ | 轻量灵活的状态管理 |
| **路由** | React Router | 6.0+ | SPA 路由管理 |
| **HTTP 客户端** | Axios | 1.6+ | HTTP 请求库 |
| **视频播放器** | Video.js / xgplayer | 最新 | 功能强大的 Web 视频播放器 |
| **文件上传** | tus-js-client | 最新 | 支持分片/断点续传的上传协议 |
| **WebSocket** | 原生 WebSocket API | - | 实时进度推送 |
| **时间轴编辑** | 自研 Canvas 组件 | - | 集锦片段可视化编辑 |
| **TypeScript** | TypeScript | 5.0+ | 类型安全，提升开发体验 |
| **样式方案** | TailwindCSS | 3.0+ | 原子化 CSS，快速开发 |

### 4.4 DevOps 与基础设施

| 类别 | 技术选型 | 说明 |
|------|---------|------|
| **容器化** | Docker + Docker Compose | 服务容器化部署 |
| **编排（可选）** | Kubernetes | 生产环境弹性伸缩 |
| **CI/CD** | GitHub Actions / GitLab CI | 自动化构建和部署 |
| **监控** | Prometheus + Grafana | 系统和业务指标监控 |
| **日志收集** | Loki / ELK | 日志聚合和查询 |
| **反向代理** | Nginx / Traefik | 负载均衡和反向代理 |
| **GPU 调度** | NVIDIA Container Toolkit | Docker 容器 GPU 支持 |

### 4.5 技术栈全景图

```
┌───────────────────────────────────────────────────────────┐
│                   前端 (React + TypeScript)                 │
│         Ant Design · Vite · TailwindCSS · Video.js         │
├───────────────────────────────────────────────────────────┤
│                    Nginx 反向代理                           │
├────────────────────────┬──────────────────────────────────┤
│   Go 后端 (Gin)         │       Python AI 引擎              │
│   ├─ REST API          │       ├─ YOLOv8 目标检测           │
│   ├─ gRPC Client       │◄─────►│  ├─ PaddleOCR 分数识别     │
│   ├─ WebSocket         │ gRPC  │  ├─ 音频分析引擎            │
│   ├─ 任务调度           │       │  ├─ 场景变化检测            │
│   └─ 视频处理(FFmpeg)   │       │  └─ 多模态融合             │
├────────────────────────┴──────────────────────────────────┤
│              基础设施层                                      │
│   PostgreSQL · Redis · NATS · MinIO · FFmpeg               │
├───────────────────────────────────────────────────────────┤
│              运维层                                         │
│   Docker · Prometheus · Grafana · Loki                     │
└───────────────────────────────────────────────────────────┘
```

---

## 五、Go 后端项目结构设计

遵循 Go 社区推荐的项目布局：

```
goalcut/
├── cmd/
│   └── server/
│       └── main.go                 # 应用入口
├── internal/
│   ├── config/                     # 配置管理
│   │   └── config.go
│   ├── handler/                    # HTTP 处理器 (Controller 层)
│   │   ├── video.go                # 视频相关接口
│   │   ├── task.go                 # 任务相关接口
│   │   ├── highlight.go            # 集锦相关接口
│   │   └── user.go                 # 用户相关接口
│   ├── service/                    # 业务逻辑层
│   │   ├── video.go                # 视频业务逻辑
│   │   ├── task.go                 # 任务调度逻辑
│   │   ├── highlight.go            # 集锦生成逻辑
│   │   ├── analyzer.go             # AI 分析调用逻辑
│   │   └── user.go                 # 用户业务逻辑
│   ├── repository/                 # 数据访问层
│   │   ├── video.go
│   │   ├── task.go
│   │   ├── highlight.go
│   │   └── user.go
│   ├── model/                      # 数据模型
│   │   ├── video.go
│   │   ├── task.go
│   │   ├── highlight.go
│   │   ├── user.go
│   │   └── event.go                # 进球事件模型
│   ├── middleware/                  # 中间件
│   │   ├── auth.go                 # JWT 认证
│   │   ├── cors.go                 # 跨域
│   │   └── ratelimit.go            # 限流
│   ├── worker/                     # 后台工作器
│   │   ├── dispatcher.go           # 任务分发器
│   │   ├── processor.go            # 视频处理器
│   │   └── pipeline.go             # 处理流水线
│   ├── ffmpeg/                     # FFmpeg 封装
│   │   ├── probe.go                # 视频信息探测
│   │   ├── clip.go                 # 视频裁剪
│   │   ├── concat.go               # 视频拼接
│   │   └── transcode.go            # 转码
│   ├── grpc/                       # gRPC 客户端
│   │   └── analyzer_client.go      # AI 分析引擎客户端
│   └── pkg/                        # 内部公共包
│       ├── response/               # 统一响应格式
│       ├── errors/                 # 错误定义
│       └── utils/                  # 工具函数
├── api/
│   └── proto/                      # gRPC Proto 定义
│       └── analyzer.proto
├── configs/
│   ├── config.yaml                 # 配置文件
│   └── config.example.yaml         # 配置示例
├── migrations/                     # 数据库迁移
│   └── 001_init.sql
├── scripts/                        # 辅助脚本
├── deployments/                    # 部署相关
│   ├── docker/
│   │   ├── Dockerfile.server       # Go 服务镜像
│   │   ├── Dockerfile.ai           # Python AI 服务镜像
│   │   └── Dockerfile.frontend     # 前端镜像
│   └── docker-compose.yaml
├── docs/                           # 文档目录
├── go.mod
├── go.sum
└── Makefile
```

---

## 六、核心数据模型设计

### 6.1 数据库 ER 关系

```
users(用户) ──1:N──> videos(视频) ──1:N──> tasks(任务)
                                    └──1:N──> highlights(集锦)
tasks ──1:N──> goal_events(进球事件)
highlights ──N:N──> goal_events (通过 highlight_events 关联表)
```

### 6.2 核心表结构

```sql
-- 用户表
CREATE TABLE users (
    id          BIGSERIAL PRIMARY KEY,
    username    VARCHAR(64) UNIQUE NOT NULL,
    email       VARCHAR(128) UNIQUE NOT NULL,
    password    VARCHAR(256) NOT NULL,
    role        VARCHAR(16) DEFAULT 'user',
    quota_used  INT DEFAULT 0,
    quota_limit INT DEFAULT 100,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 视频表
CREATE TABLE videos (
    id           BIGSERIAL PRIMARY KEY,
    user_id      BIGINT REFERENCES users(id),
    title        VARCHAR(256) NOT NULL,
    file_path    VARCHAR(512) NOT NULL,       -- 对象存储路径
    file_size    BIGINT NOT NULL,             -- 文件大小(字节)
    duration     FLOAT NOT NULL,              -- 时长(秒)
    width        INT,                         -- 分辨率宽
    height       INT,                         -- 分辨率高
    fps          FLOAT,                       -- 帧率
    codec        VARCHAR(32),                 -- 编码格式
    scene_mode   VARCHAR(16) DEFAULT 'auto',  -- 场景模式: auto/official/pickup
    status       VARCHAR(16) DEFAULT 'uploaded', -- uploaded/processing/ready
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW()
);

-- 任务表
CREATE TABLE tasks (
    id           BIGSERIAL PRIMARY KEY,
    user_id      BIGINT REFERENCES users(id),
    video_id     BIGINT REFERENCES videos(id),
    type         VARCHAR(32) NOT NULL,        -- analyze/generate
    status       VARCHAR(16) DEFAULT 'pending', -- pending/running/completed/failed
    progress     FLOAT DEFAULT 0,             -- 进度百分比 0-100
    stage        VARCHAR(64),                 -- 当前阶段描述
    config       JSONB,                       -- 任务配置参数
    result       JSONB,                       -- 处理结果
    error_msg    TEXT,                        -- 错误信息
    started_at   TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW()
);

-- 进球事件表
CREATE TABLE goal_events (
    id           BIGSERIAL PRIMARY KEY,
    task_id      BIGINT REFERENCES tasks(id),
    video_id     BIGINT REFERENCES videos(id),
    timestamp    FLOAT NOT NULL,              -- 进球时间点(秒)
    duration     FLOAT,                       -- 事件持续时长
    score_type   VARCHAR(16),                 -- two_pointer/three_pointer/free_throw
    team         VARCHAR(64),                 -- 得分球队
    score_before VARCHAR(16),                 -- 进球前比分 "98:96"
    score_after  VARCHAR(16),                 -- 进球后比分 "100:96"
    confidence   FLOAT,                       -- 置信度 0-1
    detections   JSONB,                       -- 各维度检测详情
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- 集锦表
CREATE TABLE highlights (
    id           BIGSERIAL PRIMARY KEY,
    user_id      BIGINT REFERENCES users(id),
    video_id     BIGINT REFERENCES videos(id),
    task_id      BIGINT REFERENCES tasks(id),
    title        VARCHAR(256),
    file_path    VARCHAR(512),                -- 集锦视频存储路径
    file_size    BIGINT,
    duration     FLOAT,
    resolution   VARCHAR(16),                 -- 1080p/720p/480p
    config       JSONB,                       -- 生成配置(前后时间窗口等)
    status       VARCHAR(16) DEFAULT 'pending',
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW()
);

-- 集锦-事件关联表
CREATE TABLE highlight_events (
    highlight_id BIGINT REFERENCES highlights(id),
    event_id     BIGINT REFERENCES goal_events(id),
    seq_order    INT,                         -- 在集锦中的顺序
    clip_start   FLOAT,                       -- 裁剪起始时间
    clip_end     FLOAT,                       -- 裁剪结束时间
    PRIMARY KEY (highlight_id, event_id)
);
```

---

## 七、核心 API 设计

### 7.1 视频相关

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/videos/upload/init` | 初始化分片上传 |
| POST | `/api/v1/videos/upload/chunk` | 上传分片 |
| POST | `/api/v1/videos/upload/complete` | 完成上传 |
| GET | `/api/v1/videos` | 获取视频列表 |
| GET | `/api/v1/videos/:id` | 获取视频详情 |
| DELETE | `/api/v1/videos/:id` | 删除视频 |

### 7.2 任务相关

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/tasks` | 创建分析/生成任务 |
| GET | `/api/v1/tasks` | 获取任务列表 |
| GET | `/api/v1/tasks/:id` | 获取任务详情（含进度） |
| DELETE | `/api/v1/tasks/:id` | 取消任务 |
| WS | `/api/v1/tasks/:id/ws` | WebSocket 实时进度 |

### 7.3 进球事件相关

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/videos/:id/events` | 获取视频的进球事件列表 |
| PUT | `/api/v1/events/:id` | 修改进球事件（人工校准） |
| DELETE | `/api/v1/events/:id` | 删除误检事件 |
| POST | `/api/v1/events` | 手动添加进球事件 |

### 7.4 集锦相关

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/highlights` | 创建集锦（选择事件 + 配置参数） |
| GET | `/api/v1/highlights` | 获取集锦列表 |
| GET | `/api/v1/highlights/:id` | 获取集锦详情 |
| GET | `/api/v1/highlights/:id/download` | 下载集锦视频 |
| GET | `/api/v1/highlights/:id/preview` | 预览集锦视频（流式） |

---

## 八、AI 分析引擎处理流程

### 8.1 整体处理流水线

```
原始视频输入
    │
    ├─── [阶段1] 预处理 ──────────────────────────────────┐
    │    ├─ 视频解码                                       │
    │    ├─ 关键帧提取 (2-5 fps 采样)                       │
    │    └─ 音频轨道分离                                    │
    │                                                      │
    ├─── [阶段2] 并行多模态分析 ──────────── (并行执行) ──────┤
    │    ├─ [Pipeline A] 视觉检测                           │
    │    │   ├─ YOLO 目标检测 (篮球/篮筐/球员)                │
    │    │   ├─ 目标追踪 (ByteTrack)                        │
    │    │   ├─ 轨迹分析 → 进球判定                          │
    │    │   └─ 篮网形变检测 (光流法/帧差分)                   │
    │    │                                                  │
    │    ├─ [Pipeline B] 记分牌 OCR (正式比赛模式)            │
    │    │   ├─ 记分牌区域检测                                │
    │    │   ├─ 分数 OCR 识别                                │
    │    │   └─ 分数变化检测 → 得分事件                       │
    │    │                                                  │
    │    ├─ [Pipeline C] 音频分析                            │
    │    │   ├─ 音频特征提取 (Mel/MFCC)                      │
    │    │   ├─ 欢呼声/哨声检测 (正式比赛模式)                  │
    │    │   ├─ 入网声/叫好声检测 (野球场模式)                   │
    │    │   └─ 激动度评分                                   │
    │    │                                                  │
    │    └─ [Pipeline D] 场景分析 (正式比赛模式)               │
    │        ├─ 镜头切换检测                                  │
    │        └─ 回放片段检测                                  │
    │                                                      │
    ├─── [阶段3] 多模态融合 ──────────────────────────────────┤
    │    ├─ 场景模式识别 (自动/手动)                            │
    │    ├─ 动态权重分配                                       │
    │    ├─ 时间轴对齐                                       │
    │    ├─ 加权融合评分                                      │
    │    ├─ 阈值过滤                                         │
    │    └─ 事件去重与合并                                    │
    │                                                      │
    └─── [阶段4] 输出 ────────────────────────────────────────┘
         └─ 进球事件列表 [{timestamp, score_type, confidence, ...}]
```

### 8.2 置信度融合算法

```python
# 多模态融合置信度计算（伪代码）—— 支持场景模式自适应

# 场景模式权重预设
WEIGHT_PRESETS = {
    'official': {  # 正式比赛模式
        'ocr':        0.40,
        'visual':     0.30,
        'net_deform': 0.00,  # 正式比赛中不启用（信号被 OCR 覆盖）
        'audio':      0.15,
        'scene':      0.15,
    },
    'pickup': {  # 野球场模式
        'ocr':        0.00,  # 无记分牌，禁用
        'visual':     0.70,  # 主力检测手段
        'net_deform': 0.20,  # 篮网形变是关键辅助信号
        'audio':      0.10,  # 仅入网声/叫好声
        'scene':      0.00,  # 无回放/慢动作，禁用
    },
}

def detect_scene_mode(video_path):
    """
    自动场景模式识别：分析视频前 10 秒，检测是否存在记分牌、
    多机位切换等正式比赛特征，返回场景模式。
    """
    has_scoreboard = detect_scoreboard_region(video_path, duration=10)
    has_multi_camera = detect_camera_switches(video_path, duration=10)
    if has_scoreboard or has_multi_camera:
        return 'official'
    return 'pickup'

def fuse_confidence(visual_score, ocr_score, audio_score, scene_score,
                    net_deform_score=None, scene_mode='official'):
    """
    根据场景模式动态选择权重配置，融合各维度置信度。

    场景模式权重分配：
    ┌──────────────┬──────────────┬──────────────┐
    │  检测维度      │ 正式比赛模式  │  野球场模式   │
    ├──────────────┼──────────────┼──────────────┤
    │ 记分牌 OCR    │    0.40      │    0.00      │
    │ 视觉轨迹检测   │    0.30      │    0.70      │
    │ 篮网形变检测   │    0.00      │    0.20      │
    │ 音频分析       │    0.15      │    0.10      │
    │ 场景变化检测   │    0.15      │    0.00      │
    └──────────────┴──────────────┴──────────────┘
    """
    weights = WEIGHT_PRESETS.get(scene_mode, WEIGHT_PRESETS['official'])

    scores = {
        'ocr':        ocr_score,
        'visual':     visual_score,
        'net_deform': net_deform_score,
        'audio':      audio_score,
        'scene':      scene_score,
    }

    # 仅对有效检测结果进行加权
    total_weight = 0
    total_score = 0
    for key, score in scores.items():
        if score is not None and weights[key] > 0:
            total_weight += weights[key]
            total_score += weights[key] * score

    if total_weight == 0:
        return 0

    fused = total_score / total_weight

    # 正式比赛模式：OCR 检测到分数变化，给予额外置信度加成
    if scene_mode == 'official' and ocr_score is not None and ocr_score > 0.8:
        fused = min(1.0, fused * 1.2)

    # 野球场模式：视觉轨迹 + 篮网形变同时高置信度，给予额外加成
    if scene_mode == 'pickup':
        if (visual_score is not None and visual_score > 0.7 and
                net_deform_score is not None and net_deform_score > 0.7):
            fused = min(1.0, fused * 1.15)

    return fused
```

---

## 九、性能优化策略

### 9.1 视频处理优化

| 策略 | 说明 |
|------|------|
| **采样帧分析** | 非逐帧分析，按 2-5 fps 采样关键帧，大幅减少计算量 |
| **ROI 区域检测** | 锁定篮筐区域后只对 ROI 区域做精细分析 |
| **GPU 批量推理** | 批量送帧到 GPU 推理，提升吞吐量 |
| **流水线并行** | 视频解码、AI 推理、后处理三级流水线并行 |
| **模型量化** | 使用 INT8/FP16 量化加速推理 |
| **FFmpeg 硬件加速** | 利用 NVENC/VAAPI 硬件编解码 |

### 9.2 系统架构优化

| 策略 | 说明 |
|------|------|
| **异步任务队列** | 视频处理任务异步执行，不阻塞用户请求 |
| **Go 并发调度** | 利用 goroutine 池管理并发任务 |
| **分片上传** | 大视频文件分片上传，支持断点续传 |
| **CDN 加速** | 集锦视频通过 CDN 分发，降低带宽压力 |
| **结果缓存** | Redis 缓存热点数据和中间结果 |

---

## 十、项目开发计划（建议）

### 第一阶段：MVP（最小可行产品）— 4-6 周

- [x] 项目架构搭建（Go 后端 + Python AI 服务）
- [ ] 视频上传和管理基本功能
- [ ] 场景模式识别与选择（自动检测/用户手动指定正式比赛或野球场模式）
- [ ] 基于记分牌 OCR 的进球检测（正式比赛模式，最可靠的单一信号）
- [ ] 基于视觉轨迹 + 篮网形变的进球检测（野球场模式，核心检测手段）
- [ ] 基础集锦生成（裁剪 + 拼接）
- [ ] 简单 Web 前端（上传 + 进度 + 下载）

### 第二阶段：核心能力增强 — 4-6 周

- [ ] YOLO 视觉检测 + 球体追踪
- [ ] 音频分析子模块
- [ ] 多模态融合引擎
- [ ] 集锦编辑功能（调整片段、转场效果）
- [ ] WebSocket 实时进度推送

### 第三阶段：体验优化 — 3-4 周

- [ ] 前端可视化时间轴编辑器
- [ ] 背景音乐 / 字幕叠加
- [ ] 多码率输出
- [ ] 用户体系和权限管理
- [ ] 性能优化和 GPU 加速

### 第四阶段：生产就绪 — 2-3 周

- [ ] Docker 容器化部署
- [ ] 监控告警体系
- [ ] 日志收集和分析
- [ ] 压力测试和性能调优
- [ ] 文档完善

---

## 十一、结论

GoalCut 篮球进球集锦智能剪辑系统在技术上完全可行，核心技术栈（YOLO 目标检测、PaddleOCR、音频分析、FFmpeg）均已成熟且开源。以 Go 为后端核心语言能够充分发挥其在并发处理和性能方面的优势，配合 Python AI 微服务实现高效的视频分析流水线。

**关键成功因素**：
1. **场景模式自适应**是系统的核心差异化能力，需同时适配正式比赛和野球场两种截然不同的场景
2. 正式比赛模式下，记分牌 OCR 是最可靠的检测信号，应优先实现
3. 野球场模式下，视觉轨迹检测（权重 0.70）+ 篮网形变检测（权重 0.20）是核心依赖，需重点优化准确率
4. 多模态融合策略的动态权重分配直接决定不同场景下的进球检测精度
5. 视频处理性能需通过流水线并行 + GPU 加速来保障
6. 人工校准机制是保证输出质量的重要兜底手段

建议从 MVP 阶段快速起步：正式比赛场景以记分牌 OCR + 基础剪辑为核心，野球场场景以视觉轨迹检测 + 篮网形变检测为核心，逐步迭代增加多模态分析能力，持续优化检测准确率和用户体验。
