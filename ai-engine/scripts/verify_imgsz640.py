#!/usr/bin/env python3
"""验证 imgsz=640 在野球场视频上的检测效果"""
import sys, os, time, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ultralytics import YOLO

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'basketball_v1', 'weights', 'best.pt')
model = YOLO(model_path)

ball_cls = rim_cls = None
for cid, name in model.names.items():
    if name.lower() in ('ball', 'basketball'): ball_cls = cid
    if name.lower() in ('rim', 'hoop'): rim_cls = cid
print(f'ball_cls={ball_cls}, rim_cls={rim_cls}')

frames = sorted(glob.glob('/tmp/frames_7225_1/frame_*.jpg'))
sample = frames[::max(1, len(frames)//36)][:36]
print(f'采样 {len(sample)} 帧')

for imgsz in [416, 640]:
    t0 = time.time()
    ball_hit = rim_hit = 0
    for f in sample:
        results = model(f, conf=0.25, imgsz=imgsz, verbose=False)
        found_ball = found_rim = False
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                c = int(box.cls[0])
                if c == ball_cls: found_ball = True
                if c == rim_cls: found_rim = True
        if found_ball: ball_hit += 1
        if found_rim: rim_hit += 1
    elapsed = time.time() - t0
    print(f'\n=== imgsz={imgsz} conf=0.25 ===')
    print(f'ball: {ball_hit}/{len(sample)} ({ball_hit*100//len(sample)}%)')
    print(f'rim:  {rim_hit}/{len(sample)} ({rim_hit*100//len(sample)}%)')
    print(f'耗时: {elapsed:.1f}s ({elapsed/len(sample):.2f}s/帧)')

print('\n✅ 验证完成')
