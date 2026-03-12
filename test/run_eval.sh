#!/bin/bash
# GoalCut 快速迭代验证脚本
# 用法: ./run_eval.sh <RUN_NAME> [额外detect.py参数...]
#
# 示例:
#   ./run_eval.sh baseline
#   ./run_eval.sh round1 --confidence-threshold 0.50

set -e

PYTHON="/opt/miniconda3/bin/python3"
AI_DIR="/root/project/github.com/goalcut/ai-engine"
TEST_DIR="/root/project/github.com/goalcut/test"
FRAMES_DIR="/tmp/causal_test"
AUDIO_FILE="/tmp/causal_test/audio.wav"
VIDEO_DURATION="120.09"
SOURCE_VIDEO="/root/project/github.com/goalcut/test/野球场素材/IMG_7225_1.mp4"
GT_FILE="${TEST_DIR}/ground_truth/IMG_7225_1_gt.json"

RUN_NAME="${1:-test}"
shift 2>/dev/null || true

OUT_DIR="/tmp/goalcut_eval/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

echo "=========================================="
echo "运行: ${RUN_NAME}"
echo "额外参数: $@"
echo "输出目录: ${OUT_DIR}"
echo "=========================================="

# 运行检测
cd "${AI_DIR}"
$PYTHON detect.py \
    --frames-dir "${FRAMES_DIR}" \
    --sample-fps 3.0 \
    --confidence-threshold 0.55 \
    --yolo-confidence 0.25 \
    --video-duration "${VIDEO_DURATION}" \
    --audio-file "${AUDIO_FILE}" \
    --enable-net-deform \
    --two-stage \
    --source-video "${SOURCE_VIDEO}" \
    --output "${OUT_DIR}/result.json" \
    "$@" \
    2>&1 | tee "${OUT_DIR}/detect.log"

echo ""
echo "--- 评估结果 ---"
$PYTHON "${TEST_DIR}/eval_gt.py" "${OUT_DIR}/result.json" "${GT_FILE}" 3.0
