"""학습된 모델의 mAP 평가 (VisDrone val/test).

사용 예:
    python evaluation/eval_map.py --weights runs/detect/yolo11s/weights/best.pt
    python evaluation/eval_map.py --weights best.onnx --device 2
"""
import argparse
from pathlib import Path

from ultralytics import YOLO, settings

ROOT = Path(__file__).resolve().parent.parent
settings.update({"datasets_dir": str(ROOT)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help=".pt / .onnx / .engine")
    ap.add_argument("--data", default=str(ROOT / "training" / "visdrone.yaml"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()

    model = YOLO(args.weights)
    m = model.val(data=args.data, imgsz=args.imgsz, device=args.device, split=args.split)

    print("\n===== 결과 =====")
    print(f"mAP@0.5      : {m.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {m.box.map:.4f}")
    print(f"Precision    : {m.box.mp:.4f}")
    print(f"Recall       : {m.box.mr:.4f}")
    print("\n클래스별 mAP@0.5:0.95")
    for i, name in model.names.items():
        if i < len(m.box.maps):
            print(f"  {name:18s}: {m.box.maps[i]:.4f}")


if __name__ == "__main__":
    main()
