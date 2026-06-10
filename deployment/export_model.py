"""학습 모델을 ONNX / TensorRT로 export하고 정확도를 검증.

엣지/실시간 배포를 가정해 FP16 TensorRT 엔진까지 생성한다.

사용 예:
    python deployment/export_model.py --weights runs/detect/yolo11s/weights/best.pt --format onnx
    python deployment/export_model.py --weights ...best.pt --format engine --half --device 2
"""
import argparse
from pathlib import Path

from ultralytics import YOLO, settings

ROOT = Path(__file__).resolve().parent.parent
settings.update({"datasets_dir": str(ROOT)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--format", default="onnx", choices=["onnx", "engine"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16 (TensorRT 권장)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--data", default=str(ROOT / "training" / "visdrone.yaml"))
    ap.add_argument("--eval", action="store_true", help="export 후 mAP 검증")
    args = ap.parse_args()

    model = YOLO(args.weights)
    out = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        device=args.device,
        simplify=True,
    )
    print(f"\nexport 완료 → {out}")

    if args.eval:
        em = YOLO(out)
        m = em.val(data=args.data, imgsz=args.imgsz, device=args.device)
        print(f"[{args.format}] mAP@0.5={m.box.map50:.4f}  mAP@0.5:0.95={m.box.map:.4f}")


if __name__ == "__main__":
    main()
