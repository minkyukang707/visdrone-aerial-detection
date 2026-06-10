"""YOLO 데이터 증강 — 항공뷰 소형객체 탐지 특화.

항공/드론 영상은 객체가 매우 작고 밀집해 있어, 일반 증강보다 다음이 효과적이다.
  - Case 1 (tiling)   : 이미지를 NxN 타일로 분할 → 소형객체를 상대적으로 크게 학습
  - Case 2 (rotation) : ±각도 회전 → 다양한 비행 자세/시점에 robust
  - Case 3 (rot+tiling): 회전 후 타일링 (복합)

각 case는 원본 images/labels 를 읽어 증강본을 out_dir/images, out_dir/labels 로 저장.

사용 예:
    python augmentation/augment_yolo.py --src datasets/VisDrone/VisDrone2019-DET-train \
           --dst datasets/VisDrone_aug/case1 --case tiling --tiles 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------- YOLO 라벨 입출력 ----------
def load_labels(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().strip().splitlines():
        if not line:
            continue
        c, x, y, w, h = line.split()
        out.append([int(c), float(x), float(y), float(w), float(h)])
    return out


def save_labels(path: Path, labels) -> None:
    path.write_text("\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in labels))


# ---------- Case 1: 타일링 ----------
def tile_image(img, labels, n: int):
    """이미지를 n x n 타일로 분할하고 각 타일에 속한 bbox를 재계산."""
    H, W = img.shape[:2]
    th, tw = H // n, W // n
    results = []
    # 정규화 bbox → 절대 좌표 (x1,y1,x2,y2)
    boxes = [(c, (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H)
             for c, x, y, w, h in labels]
    for r in range(n):
        for col in range(n):
            x0, y0 = col * tw, r * th
            crop = img[y0:y0 + th, x0:x0 + tw]
            ch, cw = crop.shape[:2]
            new_labels = []
            for c, bx1, by1, bx2, by2 in boxes:
                ix1, iy1 = max(bx1, x0), max(by1, y0)
                ix2, iy2 = min(bx2, x0 + tw), min(by2, y0 + th)
                if ix2 - ix1 < 2 or iy2 - iy1 < 2:
                    continue
                # 원 박스의 50% 이상이 타일 안에 있어야 유효
                inter = (ix2 - ix1) * (iy2 - iy1)
                area = (bx2 - bx1) * (by2 - by1)
                if area <= 0 or inter / area < 0.5:
                    continue
                nx = ((ix1 + ix2) / 2 - x0) / cw
                ny = ((iy1 + iy2) / 2 - y0) / ch
                nw, nh = (ix2 - ix1) / cw, (iy2 - iy1) / ch
                new_labels.append([c, nx, ny, nw, nh])
            if new_labels:  # 객체 있는 타일만 저장
                results.append((f"_t{r}{col}", crop, new_labels))
    return results


# ---------- Case 2: 회전 ----------
def rotate_image(img, labels, angle: float):
    """이미지를 angle도 회전하고 bbox를 회전된 코너 기준으로 재계산."""
    H, W = img.shape[:2]
    cx, cy = W / 2, H / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rot = cv2.warpAffine(img, M, (W, H), borderValue=(0, 0, 0))

    new_labels = []
    for c, x, y, w, h in labels:
        ax, ay, aw, ah = x * W, y * H, w * W, h * H
        corners = np.array([
            [ax - aw / 2, ay - ah / 2], [ax + aw / 2, ay - ah / 2],
            [ax + aw / 2, ay + ah / 2], [ax - aw / 2, ay + ah / 2],
        ])
        ones = np.ones((4, 1))
        rc = (M @ np.hstack([corners, ones]).T).T
        nx1, ny1 = rc[:, 0].min(), rc[:, 1].min()
        nx2, ny2 = rc[:, 0].max(), rc[:, 1].max()
        nx1, nx2 = np.clip([nx1, nx2], 0, W)
        ny1, ny2 = np.clip([ny1, ny2], 0, H)
        if nx2 - nx1 < 2 or ny2 - ny1 < 2:
            continue
        new_labels.append([c, (nx1 + nx2) / 2 / W, (ny1 + ny2) / 2 / H,
                           (nx2 - nx1) / W, (ny2 - ny1) / H])
    return rot, new_labels


def process(src: Path, dst: Path, case: str, tiles: int, angle: float) -> None:
    img_src, lbl_src = src / "images", src / "labels"
    img_dst, lbl_dst = dst / "images", dst / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(img_src.glob("*")) if p.suffix.lower() in IMG_EXTS]
    n_out = 0
    for p in tqdm(files, desc=f"{case}"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        labels = load_labels(lbl_src / f"{p.stem}.txt")
        if not labels:
            continue

        variants = []
        if case == "tiling":
            variants = tile_image(img, labels, tiles)
        elif case == "rotation":
            rimg, rlab = rotate_image(img, labels, angle)
            if rlab:
                variants = [(f"_r{int(angle)}", rimg, rlab)]
        elif case == "rot_tiling":
            rimg, rlab = rotate_image(img, labels, angle)
            if rlab:
                variants = tile_image(rimg, rlab, tiles)

        for suffix, vimg, vlab in variants:
            name = f"{p.stem}{suffix}"
            cv2.imwrite(str(img_dst / f"{name}.jpg"), vimg)
            save_labels(lbl_dst / f"{name}.txt", vlab)
            n_out += 1
    print(f"완료: {len(files)} 원본 → {n_out} 증강본 ({dst})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--case", choices=["tiling", "rotation", "rot_tiling"], required=True)
    ap.add_argument("--tiles", type=int, default=3, help="NxN 타일 분할 수")
    ap.add_argument("--angle", type=float, default=15.0, help="회전 각도(도)")
    args = ap.parse_args()
    process(Path(args.src), Path(args.dst), args.case, args.tiles, args.angle)


if __name__ == "__main__":
    main()
