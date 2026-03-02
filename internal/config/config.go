package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

type FFmpegConfig struct {
	FFmpegPath  string `yaml:"ffmpeg_path"`
	FFprobePath string `yaml:"ffprobe_path"`
}

type DetectionConfig struct {
	SampleFPS           float64 `yaml:"sample_fps"`
	ConfidenceThreshold float64 `yaml:"confidence_threshold"`
	YOLOConfidence      float64 `yaml:"yolo_confidence"`
	PythonPath          string  `yaml:"python_path"`
}

type HighlightConfig struct {
	BeforeSeconds  float64 `yaml:"before_seconds"`
	AfterSeconds   float64 `yaml:"after_seconds"`
	MergeThreshold float64 `yaml:"merge_threshold"`
}

type Config struct {
	FFmpeg    FFmpegConfig    `yaml:"ffmpeg"`
	Detection DetectionConfig `yaml:"detection"`
	Highlight HighlightConfig `yaml:"highlight"`
}

func DefaultConfig() *Config {
	return &Config{
		FFmpeg: FFmpegConfig{
			FFmpegPath:  "ffmpeg",
			FFprobePath: "ffprobe",
		},
		Detection: DetectionConfig{
			SampleFPS:           3,
			ConfidenceThreshold: 0.3,
			YOLOConfidence:      0.3,
			PythonPath:          "python3",
		},
		Highlight: HighlightConfig{
			BeforeSeconds:  3,
			AfterSeconds:   3,
			MergeThreshold: 4,
		},
	}
}

func Load(path string) (*Config, error) {
	cfg := DefaultConfig()
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			fmt.Printf("[config] 配置文件 %s 不存在，使用默认配置\n", path)
			return cfg, nil
		}
		return nil, fmt.Errorf("读取配置文件失败: %w", err)
	}
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("解析配置文件失败: %w", err)
	}
	fmt.Printf("[config] 已加载配置: sample_fps=%.1f, threshold=%.2f, before=%.0fs, after=%.0fs\n",
		cfg.Detection.SampleFPS, cfg.Detection.ConfidenceThreshold,
		cfg.Highlight.BeforeSeconds, cfg.Highlight.AfterSeconds)
	return cfg, nil
}
