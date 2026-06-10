"""다중 객체 추적 (ByteTrack / BoT-SORT) — 항공뷰 영상.

Ultralytics 내장 트래커로 영상에 대해 탐지+추적을 수행하고 결과 영상을 저장한다.
TensorRT 엔진(.engine)을 detector로 쓰면 실시간 추적 데모가 가능하다.

사용 예:
    python tracking/run_tracking.py --weights best.engine --source aerial.mp4 --tracker bytetrack
    python tracking/run_tracking.py --weights best.pt --source aerial.mp4 --tracker botsort --device 2
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="입력 영상 경로")
    ap.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"])
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.track(
        source=args.source,
        tracker=f"{args.tracker}.yaml",
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        persist=True,
        project=str(ROOT / "runs" / "track"),
        name=args.tracker,
    )
    print(f"추적 결과 저장 → {ROOT / 'runs' / 'track' / args.tracker}")


if __name__ == "__main__":
    main()
