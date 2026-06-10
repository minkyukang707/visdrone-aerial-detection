"""VisDrone 어노테이션을 YOLO 포맷으로 변환.

VisDrone annotation(.txt) 한 줄 형식:
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>

- score==0 : ignored region → 제외
- category : 0=ignored, 1..10=유효 클래스, 11=others → 1..10만 사용(0-indexed로 0..9)

각 split 폴더(VisDrone2019-DET-train 등) 아래 images/, annotations/ 를 읽어
labels/ 에 YOLO 포맷 라벨을 생성합니다.

사용 예:
    python data/visdrone_to_yolo.py --root datasets/VisDrone
"""
import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

SPLITS = ["VisDrone2019-DET-train", "VisDrone2019-DET-val", "VisDrone2019-DET-test-dev"]

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def convert_box(size, box):
    """(left, top, w, h) → 정규화된 (xc, yc, w, h)."""
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x = (box[0] + box[2] / 2) * dw
    y = (box[1] + box[3] / 2) * dh
    return x, y, box[2] * dw, box[3] * dh


def convert_split(split_dir: Path) -> int:
    img_dir = split_dir / "images"
    ann_dir = split_dir / "annotations"
    lbl_dir = split_dir / "labels"
    if not ann_dir.exists():
        print(f"[skip] {split_dir.name}: annotations 없음")
        return 0
    lbl_dir.mkdir(exist_ok=True)

    n = 0
    for ann in tqdm(sorted(ann_dir.glob("*.txt")), desc=split_dir.name):
        img_path = (img_dir / ann.name).with_suffix(".jpg")
        if not img_path.exists():
            continue
        w, h = Image.open(img_path).size
        lines = []
        for row in ann.read_text().strip().splitlines():
            p = row.split(",")
            if len(p) < 6 or p[4] == "0":  # ignored region
                continue
            cat = int(p[5])
            if cat < 1 or cat > 10:  # 0=ignored, 11=others 제외
                continue
            cls = cat - 1
            box = convert_box((w, h), tuple(map(int, p[:4])))
            lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in box))
        (lbl_dir / ann.name).write_text("\n".join(lines))
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/VisDrone", help="split 폴더들이 있는 루트")
    args = ap.parse_args()

    root = Path(args.root)
    total = 0
    for s in SPLITS:
        d = root / s
        if d.exists():
            total += convert_split(d)
    print(f"\n변환 완료: {total} 이미지. 클래스 {len(CLASS_NAMES)}종 → {CLASS_NAMES}")


if __name__ == "__main__":
    main()
