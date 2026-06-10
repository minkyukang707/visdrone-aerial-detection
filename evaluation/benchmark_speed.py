"""추론 속도 벤치마크 (PyTorch / ONNX / TensorRT 공통).

워밍업 후 N회 반복 추론하여 평균 지연(ms)과 FPS를 측정한다.

사용 예:
    python evaluation/benchmark_speed.py --weights best.pt --device 2
    python evaluation/benchmark_speed.py --weights best.engine --device 2 --runs 200
"""
import argparse
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    model = YOLO(args.weights)
    dummy = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    for _ in range(args.warmup):
        model.predict(dummy, imgsz=args.imgsz, device=args.device, verbose=False)

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        model.predict(dummy, imgsz=args.imgsz, device=args.device, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    print(f"\n모델   : {Path(args.weights).name}")
    print(f"입력   : {args.imgsz}x{args.imgsz}, device={args.device}, runs={args.runs}")
    print(f"평균   : {arr.mean():.2f} ms  (±{arr.std():.2f})")
    print(f"중앙값 : {np.median(arr):.2f} ms")
    print(f"FPS    : {1000 / arr.mean():.1f}")


if __name__ == "__main__":
    main()
