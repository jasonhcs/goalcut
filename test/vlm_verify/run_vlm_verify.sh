#!/bin/bash
# ==============================================================================
# GoalCut VLM 验证 — 一键运行脚本 v2.0
# ==============================================================================
#
# 用法:
#   # 全视频扫描模式（VLM 独立发现所有进球，与算法比对）
#   ./run_vlm_verify.sh --scan video.mp4 --detection detection.json --feedback
#
#   # 验证单个视频（需要源视频 + 检测结果）
#   ./run_vlm_verify.sh --input video.mp4 --detection detection.json
#
#   # 批量验证 test/input 下的所有视频
#   ./run_vlm_verify.sh --batch
#
#   # 批量扫描模式
#   ./run_vlm_verify.sh --batch --scan-mode
#
#   # 完整闭环（验证 + 生成反馈）
#   ./run_vlm_verify.sh --input video.mp4 --detection detection.json --feedback
#
# 环境变量:
#   VLM_PROVIDER   - openai / gemini / ollama / dashscope (默认 openai)
#   VLM_MODEL      - 模型名称
#   VLM_API_KEY    - API Key
#   VLM_BASE_URL   - 自定义端点
#   PYTHON         - Python 路径 (默认 python3)
#   GOALCUT_CONFIG - GoalCut 配置文件路径
# ==============================================================================

set -euo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 脚本所在目录（test/vlm_verify/）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$TEST_DIR/.." && pwd)"

# 默认配置
PYTHON="${PYTHON:-/opt/miniconda3/bin/python3}"
GOALCUT_CONFIG="${GOALCUT_CONFIG:-$PROJECT_ROOT/configs/config.yaml}"
REPORTS_DIR="$SCRIPT_DIR/reports"
VERIFY_SCRIPT="$SCRIPT_DIR/verify_vlm.py"
FEEDBACK_SCRIPT="$SCRIPT_DIR/vlm_feedback.py"

# 参数
SCAN_VIDEO=""
INPUT_VIDEO=""
HIGHLIGHT_VIDEO=""
DETECTION_JSON=""
GROUND_TRUTH=""
BATCH_MODE=false
BATCH_SCAN_MODE=false
GENERATE_FEEDBACK=false
CHECK_NEGATIVES=false
OUTPUT_DIR=""
SCAN_WINDOW=""
SCAN_STEP=""

usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --scan VIDEO            全视频扫描模式（VLM 独立找出所有进球，再与算法比对）"
    echo "  --input, -i VIDEO       源视频路径（片段验证模式）"
    echo "  --highlight VIDEO       集锦视频路径"
    echo "  --detection, -d JSON    检测结果 JSON"
    echo "  --ground-truth, --gt    Ground Truth JSON"
    echo "  --batch                 批量模式（验证/扫描 test/input 下所有视频）"
    echo "  --scan-mode             批量模式下使用扫描模式（配合 --batch）"
    echo "  --feedback              生成闭环反馈报告"
    echo "  --check-negatives       启用负样本审查"
    echo "  --scan-window SEC       扫描窗口大小（秒，默认 6.0）"
    echo "  --scan-step SEC         扫描步长（秒，默认 2.0）"
    echo "  --output-dir DIR        报告输出目录 (默认 test/vlm_verify/reports)"
    echo "  --help, -h              显示帮助"
    echo ""
    echo "环境变量:"
    echo "  VLM_PROVIDER   模型提供商 (默认 openai)"
    echo "  VLM_API_KEY    API Key"
    echo "  VLM_MODEL      模型名称"
}

log() { echo -e "${BLUE}[vlm-verify]${NC} $*"; }
ok()  { echo -e "${GREEN}[vlm-verify]${NC} $*"; }
warn(){ echo -e "${YELLOW}[vlm-verify]${NC} $*"; }
err() { echo -e "${RED}[vlm-verify]${NC} $*" >&2; }

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --scan) SCAN_VIDEO="$2"; shift 2;;
        --input|-i) INPUT_VIDEO="$2"; shift 2;;
        --highlight) HIGHLIGHT_VIDEO="$2"; shift 2;;
        --detection|-d) DETECTION_JSON="$2"; shift 2;;
        --ground-truth|--gt) GROUND_TRUTH="$2"; shift 2;;
        --batch) BATCH_MODE=true; shift;;
        --scan-mode) BATCH_SCAN_MODE=true; shift;;
        --feedback) GENERATE_FEEDBACK=true; shift;;
        --check-negatives) CHECK_NEGATIVES=true; shift;;
        --scan-window) SCAN_WINDOW="$2"; shift 2;;
        --scan-step) SCAN_STEP="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --help|-h) usage; exit 0;;
        *) err "未知参数: $1"; usage; exit 1;;
    esac
done

# 环境检查
check_env() {
    log "检查环境..."

    if ! command -v "$PYTHON" &>/dev/null; then
        err "Python 不可用: $PYTHON"
        exit 1
    fi

    if ! command -v ffmpeg &>/dev/null; then
        err "ffmpeg 不可用"
        exit 1
    fi

    if ! command -v ffprobe &>/dev/null; then
        err "ffprobe 不可用"
        exit 1
    fi

    if [[ ! -f "$VERIFY_SCRIPT" ]]; then
        err "验证脚本不存在: $VERIFY_SCRIPT"
        exit 1
    fi

    local provider="${VLM_PROVIDER:-openai}"
    if [[ "$provider" != "ollama" && -z "${VLM_API_KEY:-}" ]]; then
        warn "VLM_API_KEY 未设置，$provider 模式可能无法工作"
        warn "请设置: export VLM_API_KEY=your-api-key"
    fi

    ok "环境检查通过"
}

# 全视频扫描
scan_video() {
    local video="$1"
    local detection="$2"
    local gt="${3:-}"
    local report_dir="${4:-$REPORTS_DIR}"

    local video_name
    video_name="$(basename "$video" | sed 's/\.[^.]*$//')"
    local report_path="$report_dir/${video_name}_vlm_scan_report.json"
    local feedback_path="$report_dir/${video_name}_scan_feedback.json"

    mkdir -p "$report_dir"

    log "全视频扫描: $video_name"
    log "  源视频: $video"
    [[ -n "$detection" ]] && log "  检测结果: $detection"

    local scan_args=(
        "$PYTHON" "$VERIFY_SCRIPT"
        "--scan" "$video"
        "--output" "$report_path"
    )

    if [[ -n "$detection" ]]; then
        scan_args+=("--detection" "$detection")
    fi

    if [[ -n "$gt" && -f "$gt" ]]; then
        scan_args+=("--ground-truth" "$gt")
    fi

    if [[ -n "$SCAN_WINDOW" ]]; then
        scan_args+=("--scan-window" "$SCAN_WINDOW")
    fi

    if [[ -n "$SCAN_STEP" ]]; then
        scan_args+=("--scan-step" "$SCAN_STEP")
    fi

    local scan_exit=0
    "${scan_args[@]}" || scan_exit=$?

    if [[ -f "$report_path" ]]; then
        ok "扫描报告: $report_path"
    else
        err "扫描失败，未生成报告"
        return 1
    fi

    # 生成反馈（可选）
    if [[ "$GENERATE_FEEDBACK" == true && -f "$FEEDBACK_SCRIPT" ]]; then
        log "生成闭环反馈..."

        local feedback_args=(
            "$PYTHON" "$FEEDBACK_SCRIPT"
            "--report" "$report_path"
            "--output" "$feedback_path"
        )

        if [[ -f "$GOALCUT_CONFIG" ]]; then
            feedback_args+=("--config" "$GOALCUT_CONFIG")
        fi

        local feedback_exit=0
        "${feedback_args[@]}" || feedback_exit=$?

        if [[ -f "$feedback_path" ]]; then
            ok "反馈报告: $feedback_path"
        fi
    fi

    return $scan_exit
}

# 验证单个视频（片段模式）
verify_single() {
    local input="$1"
    local detection="$2"
    local gt="${3:-}"
    local report_dir="${4:-$REPORTS_DIR}"

    local video_name
    video_name="$(basename "$input" | sed 's/\.[^.]*$//')"
    local report_path="$report_dir/${video_name}_vlm_report.json"
    local feedback_path="$report_dir/${video_name}_feedback.json"

    mkdir -p "$report_dir"

    log "验证: $video_name"
    log "  源视频: $input"
    log "  检测结果: $detection"

    local verify_args=(
        "$PYTHON" "$VERIFY_SCRIPT"
        "--input" "$input"
        "--detection" "$detection"
        "--output" "$report_path"
    )

    if [[ -n "$gt" && -f "$gt" ]]; then
        verify_args+=("--ground-truth" "$gt")
    fi

    if [[ "$CHECK_NEGATIVES" == true ]]; then
        verify_args+=("--check-negatives")
    fi

    local verify_exit=0
    "${verify_args[@]}" || verify_exit=$?

    if [[ -f "$report_path" ]]; then
        ok "验证报告: $report_path"
    else
        err "验证失败，未生成报告"
        return 1
    fi

    # 生成反馈（可选）
    if [[ "$GENERATE_FEEDBACK" == true && -f "$FEEDBACK_SCRIPT" ]]; then
        log "生成闭环反馈..."

        local feedback_args=(
            "$PYTHON" "$FEEDBACK_SCRIPT"
            "--report" "$report_path"
            "--output" "$feedback_path"
        )

        if [[ -f "$GOALCUT_CONFIG" ]]; then
            feedback_args+=("--config" "$GOALCUT_CONFIG")
        fi

        local feedback_exit=0
        "${feedback_args[@]}" || feedback_exit=$?

        if [[ -f "$feedback_path" ]]; then
            ok "反馈报告: $feedback_path"
        fi
    fi

    return $verify_exit
}

# 验证集锦视频
verify_highlight() {
    local highlight="$1"
    local detection="$2"
    local report_dir="${3:-$REPORTS_DIR}"

    local video_name
    video_name="$(basename "$highlight" | sed 's/\.[^.]*$//')"
    local report_path="$report_dir/${video_name}_vlm_report.json"

    mkdir -p "$report_dir"

    log "验证集锦: $video_name"

    local verify_args=(
        "$PYTHON" "$VERIFY_SCRIPT"
        "--highlight" "$highlight"
        "--output" "$report_path"
    )

    if [[ -n "$detection" ]]; then
        verify_args+=("--detection" "$detection")
    fi

    "${verify_args[@]}" || true

    if [[ -f "$report_path" ]]; then
        ok "验证报告: $report_path"
    fi
}

# 批量验证/扫描
batch_verify() {
    local input_dir="$TEST_DIR/input"
    local output_dir="$TEST_DIR/output"
    local report_dir="${OUTPUT_DIR:-$REPORTS_DIR}"

    if [[ ! -d "$input_dir" ]]; then
        err "测试输入目录不存在: $input_dir"
        exit 1
    fi

    mkdir -p "$report_dir"

    local mode_label="验证"
    [[ "$BATCH_SCAN_MODE" == true ]] && mode_label="全视频扫描"

    log "=========================================="
    log "  批量 VLM $mode_label"
    log "  输入目录: $input_dir"
    log "  报告目录: $report_dir"
    log "=========================================="

    local total=0
    local passed=0
    local failed=0
    local skipped=0

    for video in "$input_dir"/*.mp4 "$input_dir"/*.MP4; do
        [[ -f "$video" ]] || continue
        total=$((total + 1))

        local name
        name="$(basename "$video" | sed 's/\.[^.]*$//')"

        # 查找对应的检测结果
        local detection=""
        for det_path in \
            "$output_dir/${name}_detection.json" \
            "$output_dir/${name}_goalcut_detection.json" \
            "$TEST_DIR/${name}_detection.json"; do
            if [[ -f "$det_path" ]]; then
                detection="$det_path"
                break
            fi
        done

        if [[ -z "$detection" && "$BATCH_SCAN_MODE" != true ]]; then
            warn "跳过 $name: 未找到检测结果 JSON"
            warn "  提示: 请先运行 GoalCut 处理该视频，或使用 -save-meta 参数"
            skipped=$((skipped + 1))
            continue
        fi

        # 查找 Ground Truth
        local gt=""
        for gt_path in \
            "$TEST_DIR/ground_truth/${name}.json" \
            "$TEST_DIR/annotations/human/${name}.json"; do
            if [[ -f "$gt_path" ]]; then
                gt="$gt_path"
                break
            fi
        done

        # 运行
        local exit_code=0
        if [[ "$BATCH_SCAN_MODE" == true ]]; then
            scan_video "$video" "$detection" "$gt" "$report_dir" || exit_code=$?
        else
            verify_single "$video" "$detection" "$gt" "$report_dir" || exit_code=$?
        fi

        if [[ $exit_code -eq 0 ]]; then
            passed=$((passed + 1))
        else
            failed=$((failed + 1))
        fi

        echo ""
    done

    # 汇总
    log "=========================================="
    log "  批量${mode_label}完成"
    log "  总计: $total | 通过: $passed | 异常: $failed | 跳过: $skipped"
    log "  报告目录: $report_dir"
    log "=========================================="

    if [[ $total -gt 0 ]]; then
        local summary_path="$report_dir/vlm_batch_summary.json"
        cat > "$summary_path" <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "mode": "$([ "$BATCH_SCAN_MODE" == true ] && echo "scan" || echo "verify")",
  "total_videos": $total,
  "passed": $passed,
  "failed": $failed,
  "skipped": $skipped,
  "provider": "${VLM_PROVIDER:-openai}",
  "model": "${VLM_MODEL:-default}",
  "report_dir": "$report_dir"
}
EOF
        ok "汇总报告: $summary_path"
    fi

    [[ $failed -eq 0 ]] || exit 1
}

# ============================================================================
# 主逻辑
# ============================================================================

check_env

if [[ "$BATCH_MODE" == true ]]; then
    batch_verify
elif [[ -n "$SCAN_VIDEO" ]]; then
    scan_video "$SCAN_VIDEO" "$DETECTION_JSON" "$GROUND_TRUTH" "${OUTPUT_DIR:-$REPORTS_DIR}"
elif [[ -n "$INPUT_VIDEO" ]]; then
    if [[ -z "$DETECTION_JSON" ]]; then
        err "--input 模式需要 --detection 参数"
        exit 1
    fi
    verify_single "$INPUT_VIDEO" "$DETECTION_JSON" "$GROUND_TRUTH" "${OUTPUT_DIR:-$REPORTS_DIR}"
elif [[ -n "$HIGHLIGHT_VIDEO" ]]; then
    verify_highlight "$HIGHLIGHT_VIDEO" "$DETECTION_JSON" "${OUTPUT_DIR:-$REPORTS_DIR}"
else
    err "请指定 --scan、--input、--highlight 或 --batch"
    usage
    exit 1
fi
