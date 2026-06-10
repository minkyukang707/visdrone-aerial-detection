"""YOLO 학습 스크립트 (VisDrone 항공뷰 객체 탐지).

Ultralytics YOLO로 n/s/m/l 모델을 학습한다. 멀티 GPU(DDP) 지원.

사용 예:
    # 단일 GPU
    python training/train.py --model yolo11s --epochs 100 --device 2
    # 멀티 GPU (DDP)
    python training/train.py --model yolo11m --epochs 100 --device 2,3
    # 전처리/증강 데이터로 학습 (data yaml의 train 경로만 바꿔서)
    python training/train.py --data training/visdrone_pp.yaml --model yolo11s
"""
import argparse
from pathlib import Path

from ultralytics import YOLO, settings

ROOT = Path(__file__).resolve().parent.parent
# YOLO data yaml의 상대 path를 프로젝트 루트 기준으로 해석하도록 고정
settings.update({"datasets_dir": str(ROOT)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s",
                    help="yolo11n/s/m/l 또는 .pt 경로")
    ap.add_argument("--data", default=str(ROOT / "training" / "visdrone.yaml"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="-1=자동(GPU 메모리 기반)")
    ap.add_argument("--device", default="0", help="'0' | '2,3' | 'cpu'")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--name", default=None, help="run 이름 (기본: 모델명)")
    ap.add_argument("--project", default=str(ROOT / "runs" / "detect"))
    args = ap.parse_args()

    weights = args.model if args.model.endswith(".pt") else f"{args.model}.pt"
    model = YOLO(weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name or Path(weights).stem,
        # 항공뷰 소형객체에 유리한 옵션
        mosaic=1.0,
        close_mosaic=10,
        cos_lr=True,
    )


if __name__ == "__main__":
    main()
