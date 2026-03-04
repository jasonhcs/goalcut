package ffmpeg

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/nicefan/goalcut/internal/config"
)

type VideoInfo struct {
	Duration   float64 `json:"duration"`
	Width      int     `json:"width"`
	Height     int     `json:"height"`
	FPS        float64 `json:"fps"`
	CodecName  string  `json:"codec_name"`
	BitRate    int64   `json:"bit_rate"`
}

type FFmpeg struct {
	ffmpegPath  string
	ffprobePath string
}

func New(cfg *config.FFmpegConfig) *FFmpeg {
	return &FFmpeg{
		ffmpegPath:  cfg.FFmpegPath,
		ffprobePath: cfg.FFprobePath,
	}
}

// Probe 获取视频元信息
func (f *FFmpeg) Probe(videoPath string) (*VideoInfo, error) {
	fmt.Printf("[ffmpeg] 探测视频信息: %s\n", videoPath)
	cmd := exec.Command(f.ffprobePath,
		"-v", "quiet",
		"-print_format", "json",
		"-show_format",
		"-show_streams",
		videoPath,
	)
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("ffprobe 执行失败: %w", err)
	}

	var result struct {
		Streams []struct {
			CodecType string `json:"codec_type"`
			CodecName string `json:"codec_name"`
			Width     int    `json:"width"`
			Height    int    `json:"height"`
			RFrameRate string `json:"r_frame_rate"`
			Duration  string `json:"duration"`
			BitRate   string `json:"bit_rate"`
		} `json:"streams"`
		Format struct {
			Duration string `json:"duration"`
			BitRate  string `json:"bit_rate"`
		} `json:"format"`
	}
	if err := json.Unmarshal(output, &result); err != nil {
		return nil, fmt.Errorf("解析 ffprobe 输出失败: %w", err)
	}

	info := &VideoInfo{}
	for _, s := range result.Streams {
		if s.CodecType == "video" {
			info.Width = s.Width
			info.Height = s.Height
			info.CodecName = s.CodecName
			if s.Duration != "" {
				info.Duration, _ = strconv.ParseFloat(s.Duration, 64)
			}
			if s.BitRate != "" {
				info.BitRate, _ = strconv.ParseInt(s.BitRate, 10, 64)
			}
			// 解析帧率 "30/1" 或 "30000/1001"
			if parts := strings.Split(s.RFrameRate, "/"); len(parts) == 2 {
				num, _ := strconv.ParseFloat(parts[0], 64)
				den, _ := strconv.ParseFloat(parts[1], 64)
				if den > 0 {
					info.FPS = num / den
				}
			}
			break
		}
	}

	// 如果 stream 没有 duration，从 format 获取
	if info.Duration == 0 && result.Format.Duration != "" {
		info.Duration, _ = strconv.ParseFloat(result.Format.Duration, 64)
	}
	if info.BitRate == 0 && result.Format.BitRate != "" {
		info.BitRate, _ = strconv.ParseInt(result.Format.BitRate, 10, 64)
	}

	fmt.Printf("[ffmpeg] 视频信息: 时长=%.2fs, 分辨率=%dx%d, 帧率=%.2f, 编码=%s\n",
		info.Duration, info.Width, info.Height, info.FPS, info.CodecName)
	return info, nil
}

// ExtractFrames 按指定 fps 提取帧到目录
func (f *FFmpeg) ExtractFrames(videoPath string, outputDir string, fps float64) (int, error) {
	fmt.Printf("[ffmpeg] 提取帧: fps=%.1f, 输出目录=%s\n", fps, outputDir)
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return 0, fmt.Errorf("创建帧输出目录失败: %w", err)
	}

	outputPattern := filepath.Join(outputDir, "frame_%06d.jpg")
	cmd := exec.Command(f.ffmpegPath,
		"-i", videoPath,
		"-vf", fmt.Sprintf("fps=%.2f", fps),
		"-q:v", "2",
		"-y",
		outputPattern,
	)
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return 0, fmt.Errorf("提取帧失败: %w", err)
	}

	// 统计帧数
	entries, err := os.ReadDir(outputDir)
	if err != nil {
		return 0, err
	}
	count := 0
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".jpg") {
			count++
		}
	}
	fmt.Printf("[ffmpeg] 提取完成: 共 %d 帧\n", count)
	return count, nil
}

// CutClip 裁剪视频片段
func (f *FFmpeg) CutClip(inputPath string, outputPath string, startSec, endSec float64) error {
	duration := endSec - startSec
	fmt.Printf("[ffmpeg] 裁剪片段: %.2f ~ %.2f (%.2fs) → %s\n", startSec, endSec, duration, outputPath)

	if err := os.MkdirAll(filepath.Dir(outputPath), 0755); err != nil {
		return fmt.Errorf("创建输出目录失败: %w", err)
	}

	cmd := exec.Command(f.ffmpegPath,
		"-i", inputPath,
		"-ss", fmt.Sprintf("%.3f", startSec),
		"-t", fmt.Sprintf("%.3f", duration),
		"-c:v", "libx264",
		"-preset", "fast",
		"-crf", "23",
		"-c:a", "aac",
		"-b:a", "128k",
		"-y",
		outputPath,
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("裁剪失败: %w\n%s", err, string(out))
	}
	return nil
}

// ConcatClips 拼接多个视频片段
func (f *FFmpeg) ConcatClips(clipPaths []string, outputPath string) error {
	fmt.Printf("[ffmpeg] 拼接 %d 个片段 → %s\n", len(clipPaths), outputPath)

	if err := os.MkdirAll(filepath.Dir(outputPath), 0755); err != nil {
		return fmt.Errorf("创建输出目录失败: %w", err)
	}

	// 创建 concat list 文件
	listPath := outputPath + ".list.txt"
	var lines []string
	for _, p := range clipPaths {
		absPath, _ := filepath.Abs(p)
		lines = append(lines, fmt.Sprintf("file '%s'", absPath))
	}
	if err := os.WriteFile(listPath, []byte(strings.Join(lines, "\n")), 0644); err != nil {
		return fmt.Errorf("创建 concat list 失败: %w", err)
	}
	defer os.Remove(listPath)

	cmd := exec.Command(f.ffmpegPath,
		"-f", "concat",
		"-safe", "0",
		"-i", listPath,
		"-c:v", "libx264",
		"-preset", "fast",
		"-crf", "23",
		"-c:a", "aac",
		"-b:a", "128k",
		"-y",
		outputPath,
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("拼接失败: %w\n%s", err, string(out))
	}
	fmt.Printf("[ffmpeg] 拼接完成: %s\n", outputPath)
	return nil
}

// ExtractAudio 从视频中提取单声道 WAV 音频（用于音频检测通道）。
// 提取 16kHz 单声道，如果视频无音轨则静默返回 nil。
func (f *FFmpeg) ExtractAudio(videoPath string, outputWav string) error {
	fmt.Printf("[ffmpeg] 提取音频: %s → %s\n", videoPath, outputWav)

	if err := os.MkdirAll(filepath.Dir(outputWav), 0755); err != nil {
		return fmt.Errorf("创建音频输出目录失败: %w", err)
	}

	cmd := exec.Command(f.ffmpegPath,
		"-i", videoPath,
		"-vn",
		"-ar", "16000",
		"-ac", "1",
		"-f", "wav",
		"-y",
		outputWav,
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		// 视频无音轨时 ffmpeg 返回错误，不阻断主流程
		fmt.Printf("[ffmpeg] 音频提取失败（视频可能无音轨，跳过音频检测）: %s\n", string(out))
		return nil
	}

	// 检查输出文件是否存在且非空
	info, err := os.Stat(outputWav)
	if err != nil || info.Size() < 44 { // WAV header 至少 44 字节
		fmt.Printf("[ffmpeg] 音频文件无效或为空，跳过音频检测\n")
		os.Remove(outputWav)
		return nil
	}

	fmt.Printf("[ffmpeg] 音频提取完成: %.1f KB\n", float64(info.Size())/1024)
	return nil
}
