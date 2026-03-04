package pipeline

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"time"

	"github.com/nicefan/goalcut/internal/config"
	"github.com/nicefan/goalcut/internal/ffmpeg"
)

// GoalEvent 进球事件
type GoalEvent struct {
	FrameIndex int     `json:"frame_index"`
	Timestamp  float64 `json:"timestamp"`
	Confidence float64 `json:"confidence"`
	Detail     string  `json:"detail"`
}

// ClipRange 裁剪区间
type ClipRange struct {
	Start float64
	End   float64
}

// Pipeline 处理流水线
type Pipeline struct {
	cfg     *config.Config
	ff      *ffmpeg.FFmpeg
	tempDir string
}

func New(cfg *config.Config) *Pipeline {
	return &Pipeline{
		cfg: cfg,
		ff:  ffmpeg.New(&cfg.FFmpeg),
	}
}

// Run 执行完整处理流程
func (p *Pipeline) Run(inputPath, outputPath string) error {
	startTime := time.Now()
	fmt.Println("========================================")
	fmt.Printf("[pipeline] 开始处理: %s\n", inputPath)
	fmt.Println("========================================")

	// 1. 探测视频信息
	fmt.Println("\n[pipeline] === 步骤 1/5: 探测视频信息 ===")
	info, err := p.ff.Probe(inputPath)
	if err != nil {
		return fmt.Errorf("探测视频失败: %w", err)
	}
	if info.Duration <= 0 {
		return fmt.Errorf("无法获取视频时长")
	}

	// 2. 提取帧 + 音频
	fmt.Println("\n[pipeline] === 步骤 2/5: 提取视频帧和音频 ===")
	p.tempDir, err = os.MkdirTemp("", "goalcut-*")
	if err != nil {
		return fmt.Errorf("创建临时目录失败: %w", err)
	}
	defer func() {
		fmt.Printf("[pipeline] 清理临时目录: %s\n", p.tempDir)
		os.RemoveAll(p.tempDir)
	}()

	framesDir := filepath.Join(p.tempDir, "frames")
	frameCount, err := p.ff.ExtractFrames(inputPath, framesDir, p.cfg.Detection.SampleFPS)
	if err != nil {
		return fmt.Errorf("提取帧失败: %w", err)
	}
	if frameCount == 0 {
		return fmt.Errorf("未提取到任何帧")
	}

	// 提取音频（失败不阻断主流程）
	audioPath := filepath.Join(p.tempDir, "audio.wav")
	if err := p.ff.ExtractAudio(inputPath, audioPath); err != nil {
		fmt.Printf("[pipeline] 音频提取出错（继续）: %v\n", err)
		audioPath = ""
	}
	// 若提取后文件不存在，置空路径
	if audioPath != "" {
		if _, statErr := os.Stat(audioPath); os.IsNotExist(statErr) {
			audioPath = ""
		}
	}

	// 3. AI 检测进球
	fmt.Println("\n[pipeline] === 步骤 3/5: AI 视觉检测进球 ===")
	events, err := p.detectGoals(framesDir, frameCount, info.Duration, audioPath)
	if err != nil {
		return fmt.Errorf("AI 检测失败: %w", err)
	}
	fmt.Printf("[pipeline] 检测到 %d 个进球事件\n", len(events))
	for i, e := range events {
		fmt.Printf("[pipeline]   进球 #%d: 时间=%.2fs, 置信度=%.2f, %s\n", i+1, e.Timestamp, e.Confidence, e.Detail)
	}

	if len(events) == 0 {
		fmt.Println("[pipeline] 未检测到进球事件，跳过集锦生成")
		// 即使没有进球也生成一个空的提示
		return fmt.Errorf("未检测到任何进球事件")
	}

	// 4. 计算裁剪区间并合并重叠
	fmt.Println("\n[pipeline] === 步骤 4/5: 计算裁剪区间 ===")
	clips := p.mergeClipRanges(events, info.Duration)
	fmt.Printf("[pipeline] 合并后 %d 个片段:\n", len(clips))
	for i, c := range clips {
		fmt.Printf("[pipeline]   片段 #%d: %.2f ~ %.2f (%.2fs)\n", i+1, c.Start, c.End, c.End-c.Start)
	}

	// 5. 裁剪并拼接
	fmt.Println("\n[pipeline] === 步骤 5/5: 裁剪拼接集锦 ===")
	clipsDir := filepath.Join(p.tempDir, "clips")
	var clipPaths []string
	for i, c := range clips {
		clipPath := filepath.Join(clipsDir, fmt.Sprintf("clip_%03d.mp4", i))
		if err := p.ff.CutClip(inputPath, clipPath, c.Start, c.End); err != nil {
			return fmt.Errorf("裁剪片段 #%d 失败: %w", i+1, err)
		}
		clipPaths = append(clipPaths, clipPath)
	}

	if len(clipPaths) == 1 {
		// 只有一个片段，直接复制
		data, err := os.ReadFile(clipPaths[0])
		if err != nil {
			return fmt.Errorf("读取片段失败: %w", err)
		}
		if err := os.MkdirAll(filepath.Dir(outputPath), 0755); err != nil {
			return fmt.Errorf("创建输出目录失败: %w", err)
		}
		if err := os.WriteFile(outputPath, data, 0644); err != nil {
			return fmt.Errorf("写入输出失败: %w", err)
		}
	} else {
		if err := p.ff.ConcatClips(clipPaths, outputPath); err != nil {
			return fmt.Errorf("拼接集锦失败: %w", err)
		}
	}

	elapsed := time.Since(startTime)
	fmt.Println("\n========================================")
	fmt.Printf("[pipeline] 处理完成！耗时: %s\n", elapsed.Round(time.Millisecond))
	fmt.Printf("[pipeline] 输出文件: %s\n", outputPath)
	fmt.Println("========================================")
	return nil
}

// detectGoals 调用 Python AI 脚本检测进球
// audioPath 为空字符串时不启用音频检测通道
func (p *Pipeline) detectGoals(framesDir string, frameCount int, duration float64, audioPath string) ([]GoalEvent, error) {
	// 获取 ai-engine 脚本路径
	exePath, err := os.Executable()
	if err != nil {
		// fallback: 使用当前工作目录
		exePath, _ = os.Getwd()
	}
	// 在开发模式下，从项目根目录查找 ai-engine
	projectRoot := findProjectRoot(exePath)
	scriptPath := filepath.Join(projectRoot, "ai-engine", "detect.py")

	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("AI 检测脚本不存在: %s", scriptPath)
	}

	resultPath := filepath.Join(p.tempDir, "detection_result.json")

	fmt.Printf("[pipeline] 调用 Python AI 检测: %s\n", scriptPath)
	fmt.Printf("[pipeline] Python: %s\n", p.cfg.Detection.PythonPath)
	fmt.Printf("[pipeline] 帧目录: %s, 帧数: %d\n", framesDir, frameCount)
	if audioPath != "" {
		fmt.Printf("[pipeline] 音频文件: %s (音频检测通道已启用)\n", audioPath)
	}

	args := []string{
		scriptPath,
		"--frames-dir", framesDir,
		"--output", resultPath,
		"--sample-fps", fmt.Sprintf("%.2f", p.cfg.Detection.SampleFPS),
		"--confidence-threshold", fmt.Sprintf("%.2f", p.cfg.Detection.ConfidenceThreshold),
		"--yolo-confidence", fmt.Sprintf("%.2f", p.cfg.Detection.YOLOConfidence),
		"--video-duration", fmt.Sprintf("%.2f", duration),
		"--two-stage",        // 默认启用两阶段采样
		"--enable-net-deform", // 默认启用篮网形变检测
	}
	if audioPath != "" {
		args = append(args, "--audio-file", audioPath)
	}

	cmd := exec.Command(p.cfg.Detection.PythonPath, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("Python AI 检测脚本执行失败: %w", err)
	}

	// 读取检测结果
	data, err := os.ReadFile(resultPath)
	if err != nil {
		return nil, fmt.Errorf("读取检测结果失败: %w", err)
	}

	var events []GoalEvent
	if err := json.Unmarshal(data, &events); err != nil {
		return nil, fmt.Errorf("解析检测结果失败: %w", err)
	}

	return events, nil
}

// mergeClipRanges 合并重叠的裁剪区间
func (p *Pipeline) mergeClipRanges(events []GoalEvent, duration float64) []ClipRange {
	if len(events) == 0 {
		return nil
	}

	// 按时间戳排序
	sort.Slice(events, func(i, j int) bool {
		return events[i].Timestamp < events[j].Timestamp
	})

	var ranges []ClipRange
	for _, e := range events {
		start := math.Max(0, e.Timestamp-p.cfg.Highlight.BeforeSeconds)
		end := math.Min(duration, e.Timestamp+p.cfg.Highlight.AfterSeconds)
		ranges = append(ranges, ClipRange{Start: start, End: end})
	}

	// 合并重叠区间
	merged := []ClipRange{ranges[0]}
	for i := 1; i < len(ranges); i++ {
		last := &merged[len(merged)-1]
		if ranges[i].Start <= last.End+p.cfg.Highlight.MergeThreshold {
			// 重叠或相邻，合并
			if ranges[i].End > last.End {
				last.End = ranges[i].End
			}
		} else {
			merged = append(merged, ranges[i])
		}
	}

	return merged
}

// findProjectRoot 查找项目根目录
func findProjectRoot(startPath string) string {
	// 首先尝试从工作目录查找
	cwd, err := os.Getwd()
	if err == nil {
		if _, err := os.Stat(filepath.Join(cwd, "ai-engine", "detect.py")); err == nil {
			return cwd
		}
	}

	// 向上查找包含 go.mod 的目录
	dir := startPath
	for i := 0; i < 10; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}

	return cwd
}
