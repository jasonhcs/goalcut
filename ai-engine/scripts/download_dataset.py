#!/usr/bin/env python3
"""
使用 Roboflow API 搜索和下载篮球检测数据集
"""
import os
import sys
import json
import requests

API_KEY = "BKVzc3o5LyeRgNBiWQIh"

# 搜索 Roboflow Universe 上的篮球数据集
print("搜索 Roboflow Universe 篮球检测数据集...")

# 方式1: 直接尝试下载已知的篮球检测数据集
# 来源: https://universe.roboflow.com/ownprojects/basketball-w2xcw/dataset/2
print("\n尝试下载已知数据集: ownprojects/basketball-w2xcw v2")
try:
    from roboflow import Roboflow
    rf = Roboflow(api_key=API_KEY)
    
    # 尝试搜索公开的篮球数据集
    # 首先列出工作空间
    print(f"API Key 验证中...")
    
    # 直接尝试访问公开数据集
    datasets_to_try = [
        ("ownprojects", "basketball-w2xcw", 2),
        # 其他可能的公开数据集
    ]
    
    for workspace, project_name, version_num in datasets_to_try:
        try:
            print(f"\n尝试: {workspace}/{project_name} v{version_num}")
            project = rf.workspace(workspace).project(project_name)
            print(f"  项目类型: {project.type}")
            print(f"  类别: {project.classes}")
            version = project.version(version_num)
            print(f"  版本 {version_num} 信息: {version}")
            
            # 下载为 YOLOv8 格式
            target_dir = "/root/project/github.com/goalcut/ai-engine/datasets/basketball"
            os.makedirs(target_dir, exist_ok=True)
            dataset = version.download("yolov8", location=target_dir)
            print(f"  下载成功! 位置: {target_dir}")
            break
        except Exception as e:
            print(f"  失败: {e}")
            continue

except Exception as e:
    print(f"Roboflow 错误: {e}")
    import traceback
    traceback.print_exc()
