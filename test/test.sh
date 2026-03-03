#!/bin/bash
#
# GoalCut 集成测试脚本
# 用法: cd goalcut && bash test/test.sh [输入目录或文件]
# 默认输入源: test/input 目录
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INPUT_DIR="${1:-$SCRIPT_DIR/input}"
OUTPUT_DIR="$SCRIPT_DIR/output"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  GoalCut 集成测试"
echo "=========================================="
echo "项目目录: $PROJECT_DIR"
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 检查环境
echo -e "${YELLOW}[检查] 环境依赖...${NC}"

if ! command -v go &> /dev/null; then
    echo -e "${RED}[错误] Go 未安装${NC}"
    exit 1
fi
echo "  Go: $(go version | awk '{print $3}')"

PYTHON_PATH="${PYTHON_PATH:-/opt/miniconda3/bin/python3}"
if [ ! -f "$PYTHON_PATH" ]; then
    PYTHON_PATH="$(which python3 2>/dev/null || echo "")"
fi
if [ -z "$PYTHON_PATH" ]; then
    echo -e "${RED}[错误] Python3 未找到${NC}"
    exit 1
fi
echo "  Python: $($PYTHON_PATH --version 2>&1)"

if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}[错误] FFmpeg 未安装${NC}"
    exit 1
fi
echo "  FFmpeg: $(ffmpeg -version 2>&1 | head -1)"

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}[错误] 输入目录不存在: $INPUT_DIR${NC}"
    exit 1
fi

VIDEO_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" \) | wc -l)
if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo -e "${RED}[错误] 输入目录中无视频文件${NC}"
    exit 1
fi
echo "  输入视频: ${VIDEO_COUNT} 个"
echo ""

# 清理旧输出
echo -e "${YELLOW}[准备] 清理旧输出...${NC}"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# ============================================
# 测试 1: 批量处理（文件夹输入）
# ============================================
echo ""
echo -e "${YELLOW}=========================================="
echo "  测试 1: 批量处理（文件夹输入 → 输出目录）"
echo -e "==========================================${NC}"

cd "$PROJECT_DIR"
export PATH="/opt/miniconda3/bin:$PATH"

START_TIME=$(date +%s)

go run cmd/goalcut/main.go \
    -input "$INPUT_DIR" \
    -output "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/test_batch.log"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo -e "${YELLOW}[结果] 批量处理耗时: ${ELAPSED}s${NC}"

# ============================================
# 验证输出
# ============================================
echo ""
echo -e "${YELLOW}=========================================="
echo "  验证输出结果"
echo -e "==========================================${NC}"

PASS_COUNT=0
FAIL_COUNT=0

for input_file in "$INPUT_DIR"/*.MP4 "$INPUT_DIR"/*.mp4; do
    [ -f "$input_file" ] || continue
    
    base_name=$(basename "$input_file")
    name_no_ext="${base_name%.*}"
    expected_output="$OUTPUT_DIR/${name_no_ext}_goalcut.mp4"

    if [ -f "$expected_output" ]; then
        # 获取输出文件信息
        file_size=$(stat -c%s "$expected_output" 2>/dev/null || stat -f%z "$expected_output" 2>/dev/null)
        file_size_kb=$((file_size / 1024))
        duration=$(ffprobe -v quiet -show_format "$expected_output" 2>/dev/null | grep duration | head -1 | cut -d= -f2)
        
        echo -e "  ${GREEN}✅ $base_name → ${name_no_ext}_goalcut.mp4 (${file_size_kb}KB, ${duration}s)${NC}"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo -e "  ${RED}❌ $base_name → 输出文件不存在: ${name_no_ext}_goalcut.mp4${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

# ============================================
# 测试 2: 单文件处理（可选，传 --single 启用）
# ============================================
if [ "${SINGLE_TEST:-0}" = "1" ]; then
    FIRST_VIDEO=$(find "$INPUT_DIR" -maxdepth 1 -type f -iname "*.mp4" | head -1)
    if [ -n "$FIRST_VIDEO" ]; then
        echo ""
        echo -e "${YELLOW}=========================================="
        echo "  测试 2: 单文件处理（验证默认输出路径）"
        echo -e "==========================================${NC}"

        SINGLE_OUTPUT="$OUTPUT_DIR/single_test"
        mkdir -p "$SINGLE_OUTPUT"

        go run cmd/goalcut/main.go \
            -input "$FIRST_VIDEO" \
            -output "$SINGLE_OUTPUT/" \
            2>&1 | tail -5

        base_name=$(basename "$FIRST_VIDEO")
        name_no_ext="${base_name%.*}"
        single_expected="$SINGLE_OUTPUT/${name_no_ext}_goalcut.mp4"

        if [ -f "$single_expected" ]; then
            echo -e "  ${GREEN}✅ 单文件输出正确: $single_expected${NC}"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            echo -e "  ${RED}❌ 单文件输出缺失: $single_expected${NC}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    fi
fi

# ============================================
# 汇总
# ============================================
echo ""
echo "=========================================="
TOTAL=$((PASS_COUNT + FAIL_COUNT))
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo -e "  ${GREEN}全部通过: $PASS_COUNT/$TOTAL${NC}"
else
    echo -e "  ${RED}通过: $PASS_COUNT/$TOTAL, 失败: $FAIL_COUNT${NC}"
fi
echo "  输出目录: $OUTPUT_DIR"
echo "=========================================="

# 列出输出文件
echo ""
echo "输出文件列表:"
ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null || echo "  (无输出文件)"

exit $FAIL_COUNT
