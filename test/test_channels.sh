#!/bin/bash
#
# GoalCut 通道独立测试脚本
#
# 用法:
#   ./test/test_channels.sh                          # 对默认视频测试全部通道
#   ./test/test_channels.sh 5                        # 只测试通道 5（音频）
#   ./test/test_channels.sh 1a,2                     # 测试通道 1a 和 2
#   ./test/test_channels.sh all input.mp4            # 对指定视频测试全部通道
#   ./test/test_channels.sh 5 input.mp4 gt.json      # 测试通道 5 + ground truth 评测
#

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# 项目路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AI_ENGINE="$PROJECT_ROOT/ai-engine"
PYTHON="${PYTHON:-/opt/miniconda3/bin/python3}"

# 参数
CHANNELS="${1:-all}"
INPUT_VIDEO="${2:-$SCRIPT_DIR/input/amateur_game_01_summer_camp.mp4}"
GROUND_TRUTH="${3:-}"

# 配置参数（可通过环境变量覆盖）
SAMPLE_FPS="${SAMPLE_FPS:-3}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.55}"
YOLO_CONF="${YOLO_CONF:-0.25}"

# 音频通道参数
AUDIO_BURST_RATIO="${AUDIO_BURST_RATIO:-3.0}"
AUDIO_MAX_DURATION="${AUDIO_MAX_DURATION:-300}"

# 篮网形变参数
NET_BURST_RATIO="${NET_BURST_RATIO:-2.5}"
NET_COOLDOWN="${NET_COOLDOWN:-3.0}"

echo -e "${CYAN}========================================"
echo -e "  GoalCut 通道独立测试"
echo -e "========================================${NC}"
echo -e "通道:     ${YELLOW}${CHANNELS}${NC}"
echo -e "视频:     ${INPUT_VIDEO}"
echo -e "FPS:      ${SAMPLE_FPS}"
echo -e "阈值:     ${CONF_THRESHOLD}"

# 检查输入
if [ ! -f "$INPUT_VIDEO" ]; then
    echo -e "${RED}错误: 视频文件不存在: $INPUT_VIDEO${NC}"
    exit 1
fi

if [ ! -f "$AI_ENGINE/channel_test.py" ]; then
    echo -e "${RED}错误: channel_test.py 不存在${NC}"
    exit 1
fi

# 创建临时目录
TMPDIR=$(mktemp -d /tmp/goalcut-channel-test-XXXXXX)
FRAMES_DIR="$TMPDIR/frames"
AUDIO_FILE="$TMPDIR/audio.wav"
OUTPUT_DIR="$SCRIPT_DIR/output/channel_results"
mkdir -p "$FRAMES_DIR" "$OUTPUT_DIR"

echo -e "\n${CYAN}临时目录: $TMPDIR${NC}"

# 获取视频时长
echo -e "\n${CYAN}[步骤 1] 获取视频信息...${NC}"
DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$INPUT_VIDEO")
echo -e "视频时长: ${DURATION}s"

# 提取帧
echo -e "\n${CYAN}[步骤 2] 提取帧 (fps=$SAMPLE_FPS)...${NC}"
ffmpeg -y -loglevel error -i "$INPUT_VIDEO" \
    -vf "fps=$SAMPLE_FPS" -q:v 2 \
    "$FRAMES_DIR/frame_%06d.jpg"
FRAME_COUNT=$(ls "$FRAMES_DIR"/frame_*.jpg 2>/dev/null | wc -l)
echo -e "提取完成: ${GREEN}${FRAME_COUNT} 帧${NC}"

# 提取音频（如果需要通道 5）
AUDIO_ARG=""
if echo "$CHANNELS" | grep -qE '(all|5)'; then
    echo -e "\n${CYAN}[步骤 3] 提取音频...${NC}"
    ffmpeg -y -loglevel error -i "$INPUT_VIDEO" \
        -vn -ar 16000 -ac 1 -f wav "$AUDIO_FILE" 2>/dev/null || true
    if [ -f "$AUDIO_FILE" ]; then
        AUDIO_SIZE=$(du -h "$AUDIO_FILE" | cut -f1)
        echo -e "音频提取完成: ${GREEN}${AUDIO_SIZE}${NC}"
        AUDIO_ARG="--audio-file $AUDIO_FILE"
    else
        echo -e "${YELLOW}警告: 无音轨，通道 5 将跳过${NC}"
    fi
fi

# Ground Truth 参数
GT_ARG=""
if [ -n "$GROUND_TRUTH" ] && [ -f "$GROUND_TRUTH" ]; then
    GT_ARG="--ground-truth $GROUND_TRUTH"
    echo -e "\nGround Truth: ${GREEN}${GROUND_TRUTH}${NC}"
fi

# 生成输出文件名
VIDEO_NAME=$(basename "$INPUT_VIDEO" | sed 's/\.[^.]*$//')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/${VIDEO_NAME}_channels_${TIMESTAMP}.json"

# 运行通道测试
echo -e "\n${CYAN}[步骤 4] 运行通道测试...${NC}"
echo ""

cd "$AI_ENGINE"
$PYTHON channel_test.py \
    --frames-dir "$FRAMES_DIR" \
    --video-duration "$DURATION" \
    --channels "$CHANNELS" \
    --sample-fps "$SAMPLE_FPS" \
    --confidence-threshold "$CONF_THRESHOLD" \
    --yolo-confidence "$YOLO_CONF" \
    --audio-burst-ratio "$AUDIO_BURST_RATIO" \
    --audio-max-duration "$AUDIO_MAX_DURATION" \
    --net-burst-ratio "$NET_BURST_RATIO" \
    --net-cooldown "$NET_COOLDOWN" \
    --output "$OUTPUT_FILE" \
    $AUDIO_ARG \
    $GT_ARG

# 清理
echo -e "\n${CYAN}清理临时目录: $TMPDIR${NC}"
rm -rf "$TMPDIR"

echo -e "\n${GREEN}结果已保存: $OUTPUT_FILE${NC}"
echo -e "${CYAN}========================================"
echo -e "  测试完成"
echo -e "========================================${NC}"
