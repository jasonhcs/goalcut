#!/usr/bin/env python3
"""下载 Roboflow 篮球检测数据集（v2）"""
import os
os.chdir("/root/project/github.com/goalcut/ai-engine/datasets")

from roboflow import Roboflow

API_KEY = "BKVzc3o5LyeRgNBiWQIh"
rf = Roboflow(api_key=API_KEY)

print("加载项目...")
project = rf.workspace("ownprojects").project("basketball-w2xcw")
print(f"类别: {project.classes}")

version = project.version(2)
print(f"版本信息: splits={version.splits if hasattr(version, 'splits') else 'N/A'}")

# 下载到当前目录下的 basketball 子文件夹
print("下载数据集 (YOLOv8 格式)...")
dataset = version.download("yolov8", location="basketball")

print(f"\n下载完成!")
print(f"dataset.location = {dataset.location}")

# 检查实际下载的文件
import glob
for pattern in ["**/*.yaml", "**/train/images/*.jpg", "**/valid/images/*.jpg", "**/test/images/*.jpg"]:
    files = glob.glob(os.path.join("basketball", pattern), recursive=True)
    print(f"  {pattern}: {len(files)} files")
    if files and "yaml" in pattern:
        for f in files:
            print(f"    {f}")
            with open(f) as fp:
                print(fp.read())
