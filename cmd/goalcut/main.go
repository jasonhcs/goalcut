package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/nicefan/goalcut/internal/config"
	"github.com/nicefan/goalcut/internal/pipeline"
)

// 支持的视频扩展名
var videoExtensions = map[string]bool{
	".mp4": true, ".MP4": true,
	".avi": true, ".AVI": true,
	".mov": true, ".MOV": true,
	".mkv": true, ".MKV": true,
	".flv": true, ".FLV": true,
	".wmv": true, ".WMV": true,
	".webm": true, ".WEBM": true,
}

func main() {
	input := flag.String("input", "", "输入视频文件或文件夹路径（必填）")
	output := flag.String("output", "", "输出路径：输入为文件时可指定输出文件路径或目录；输入为文件夹时指定输出目录（默认: 与输入同目录）")
	configPath := flag.String("config", "", "配置文件路径（默认: configs/config.yaml）")
	before := flag.Float64("before", 0, "进球前截取秒数（覆盖配置文件）")
	after := flag.Float64("after", 0, "进球后截取秒数（覆盖配置文件）")
	fps := flag.Float64("fps", 0, "采样帧率（覆盖配置文件）")
	threshold := flag.Float64("threshold", 0, "置信度阈值（覆盖配置文件）")
	algorithmProfile := flag.String("algorithm", "", "算法配置方案: default, pickup_with_net, pickup_no_net, minimal, vlm_primary")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "GoalCut - 篮球进球集锦自动生成工具\n\n")
		fmt.Fprintf(os.Stderr, "用法:\n")
		fmt.Fprintf(os.Stderr, "  goalcut -input <video_path|dir> [-output <output_path|dir>] [options]\n\n")
		fmt.Fprintf(os.Stderr, "输出文件名规则: 原文件名_goalcut.mp4\n\n")
		fmt.Fprintf(os.Stderr, "选项:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\n示例:\n")
		fmt.Fprintf(os.Stderr, "  goalcut -input basketball.mp4                        # 输出 basketball_goalcut.mp4 (同目录)\n")
		fmt.Fprintf(os.Stderr, "  goalcut -input basketball.mp4 -output ./out/          # 输出到 ./out/basketball_goalcut.mp4\n")
		fmt.Fprintf(os.Stderr, "  goalcut -input basketball.mp4 -output my_clip.mp4     # 输出为指定文件名\n")
		fmt.Fprintf(os.Stderr, "  goalcut -input ./videos/                              # 处理目录下所有视频\n")
		fmt.Fprintf(os.Stderr, "  goalcut -input ./videos/ -output ./highlights/         # 批量处理，输出到指定目录\n")
	}

	flag.Parse()

	if *input == "" {
		fmt.Fprintln(os.Stderr, "错误: 必须指定 -input 参数")
		flag.Usage()
		os.Exit(1)
	}

	// 检查输入路径
	inputInfo, err := os.Stat(*input)
	if os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "错误: 输入路径不存在: %s\n", *input)
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "错误: 无法访问输入路径: %v\n", err)
		os.Exit(1)
	}

	// 加载配置
	cfg := loadConfig(*configPath)

	// 命令行参数覆盖配置
	if *before > 0 {
		cfg.Highlight.BeforeSeconds = *before
	}
	if *after > 0 {
		cfg.Highlight.AfterSeconds = *after
	}
	if *fps > 0 {
		cfg.Detection.SampleFPS = *fps
	}
	if *threshold > 0 {
		cfg.Detection.ConfidenceThreshold = *threshold
	}
	if *algorithmProfile != "" {
		cfg.Detection.AlgorithmProfile = *algorithmProfile
	}

	// 收集待处理的视频文件
	var videoFiles []string
	if inputInfo.IsDir() {
		videoFiles = collectVideos(*input)
		if len(videoFiles) == 0 {
			fmt.Fprintf(os.Stderr, "错误: 目录中未找到视频文件: %s\n", *input)
			os.Exit(1)
		}
		fmt.Printf("[goalcut] 在目录中找到 %d 个视频文件\n", len(videoFiles))
	} else {
		videoFiles = []string{*input}
	}

	// 处理每个视频
	p := pipeline.New(cfg)
	successCount := 0
	failCount := 0

	for i, videoFile := range videoFiles {
		if len(videoFiles) > 1 {
			fmt.Printf("\n\n████████ [%d/%d] 处理视频: %s ████████\n", i+1, len(videoFiles), filepath.Base(videoFile))
		}

		outputPath := resolveOutputPath(videoFile, *output, inputInfo.IsDir())
		if err := p.Run(videoFile, outputPath); err != nil {
			fmt.Fprintf(os.Stderr, "\n[goalcut] 处理失败 [%s]: %v\n", filepath.Base(videoFile), err)
			failCount++
		} else {
			successCount++
		}
	}

	// 批量处理汇总
	if len(videoFiles) > 1 {
		fmt.Printf("\n\n========================================\n")
		fmt.Printf("[goalcut] 批量处理完成: 成功 %d, 失败 %d, 共 %d\n", successCount, failCount, len(videoFiles))
		fmt.Printf("========================================\n")
	}

	if failCount > 0 && successCount == 0 {
		os.Exit(1)
	}
}

// loadConfig 加载配置文件
func loadConfig(cfgPath string) *config.Config {
	if cfgPath == "" {
		cwd, _ := os.Getwd()
		candidates := []string{
			filepath.Join(cwd, "configs", "config.yaml"),
			filepath.Join(cwd, "config.yaml"),
		}
		for _, c := range candidates {
			if _, err := os.Stat(c); err == nil {
				cfgPath = c
				break
			}
		}
	}

	cfg, err := config.Load(cfgPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "错误: 加载配置失败: %v\n", err)
		os.Exit(1)
	}
	return cfg
}

// collectVideos 收集目录下所有视频文件
func collectVideos(dir string) []string {
	var videos []string
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		ext := filepath.Ext(e.Name())
		if videoExtensions[ext] {
			videos = append(videos, filepath.Join(dir, e.Name()))
		}
	}
	return videos
}

// resolveOutputPath 计算输出文件路径
//
// 规则：
//  1. 输出文件名 = 原文件名（不含扩展名）+ "_goalcut.mp4"
//  2. -output 未指定 → 输出到与输入文件同目录
//  3. -output 为目录（以 / 结尾或已存在的目录） → 输出到该目录，文件名自动生成
//  4. -output 为具体文件路径（仅单文件输入有效） → 使用指定路径
func resolveOutputPath(inputFile, outputFlag string, isBatchMode bool) string {
	// 生成默认文件名: 原文件名_goalcut.mp4
	baseName := filepath.Base(inputFile)
	ext := filepath.Ext(baseName)
	nameWithoutExt := baseName[:len(baseName)-len(ext)]
	defaultName := nameWithoutExt + "_goalcut.mp4"

	if outputFlag == "" {
		// 未指定 -output，输出到输入文件同目录
		return filepath.Join(filepath.Dir(inputFile), defaultName)
	}

	// 判断 -output 是否为目录
	if isOutputDir(outputFlag) {
		return filepath.Join(outputFlag, defaultName)
	}

	// 批量模式下 -output 一律当目录用
	if isBatchMode {
		return filepath.Join(outputFlag, defaultName)
	}

	// 单文件模式，-output 为具体文件路径
	return outputFlag
}

// isOutputDir 判断输出路径是否为目录
func isOutputDir(path string) bool {
	// 以 / 结尾视为目录
	if strings.HasSuffix(path, "/") || strings.HasSuffix(path, string(os.PathSeparator)) {
		return true
	}
	// 已存在且是目录
	info, err := os.Stat(path)
	if err == nil && info.IsDir() {
		return true
	}
	// 没有扩展名视为目录
	if filepath.Ext(path) == "" {
		return true
	}
	return false
}
