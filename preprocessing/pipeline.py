"""3-Stage 적응형 영상 복원 파이프라인.

회사에서 다룬 "조건부 복원 → 대비 향상 → 디테일 강화" 흐름을 공개 알고리즘으로 재구성.

    Stage 1 (조건부): 영상 상태를 자동 판별해 저조도(Retinex/감마) 또는 안개(DCP) 복원
    Stage 2          : CLAHE 대비 향상
    Stage 3          : Wavelet 디테일 강화

폴더 단위 일괄 처리 + (YOLO) 라벨 동기 복사를 지원한다.

사용 예:
    python preprocessing/pipeline.py --src datasets/VisDrone/VisDrone2019-DET-train \
                                     --dst datasets/VisDrone_pp/VisDrone2019-DET-train
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from clahe import apply_clahe
from dehaze import dehaze
from lowlight import enhance_lowlight
from wavelet import enhance_detail


def diagnose(img: np.ndarray) -> str:
    """영상 상태 자동 판별: 'lowlight' | 'haze' | 'normal'."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean() / 255.0
    # dark channel 평균이 높으면 안개(전역적으로 밝고 대비 낮음)
    dark = np.min(img, axis=2).mean() / 255.0
    contrast = gray.std() / 255.0

    if brightness < 0.35:
        return "lowlight"
    if dark > 0.55 and contrast < 0.18:
        return "haze"
    return "normal"


def restore(img: np.ndarray, force: str | None = None) -> tuple[np.ndarray, str]:
    """3-Stage 복원 적용. force로 stage1 모드를 강제할 수 있다."""
    mode = force or diagnose(img)

    if mode == "lowlight":
        img = enhance_lowlight(img)
    elif mode == "haze":
        img = dehaze(img)

    img = apply_clahe(img, clip_limit=2.0, tile_grid=8)        # Stage 2
    img = enhance_detail(img, gain=1.25, denoise=2.0)           # Stage 3
    return img, mode


def process_folder(src: Path, dst: Path, force: str | None, copy_labels: bool) -> None:
    img_src = src / "images"
    img_dst = dst / "images"
    img_dst.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in sorted(img_src.glob("*")) if p.suffix.lower() in exts] \
        if img_src.exists() else \
        [p for p in sorted(src.glob("*")) if p.suffix.lower() in exts]

    stats: dict[str, int] = {}
    for i, p in enumerate(files, 1):
        img = cv2.imread(str(p))
        if img is None:
            continue
        out, mode = restore(img, force)
        stats[mode] = stats.get(mode, 0) + 1
        cv2.imwrite(str(img_dst / p.name), out)
        if i % 200 == 0:
            print(f"  {i}/{len(files)} 처리…")

    if copy_labels and (src / "labels").exists():
        shutil.copytree(src / "labels", dst / "labels", dirs_exist_ok=True)

    print(f"완료: {len(files)}장 처리, 모드 분포={stats}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="원본 split 폴더 (images/ 포함)")
    ap.add_argument("--dst", required=True, help="결과 저장 폴더")
    ap.add_argument("--force", choices=["lowlight", "haze", "normal"], default=None,
                    help="Stage1 모드 강제 (미지정 시 자동 판별)")
    ap.add_argument("--no-labels", action="store_true", help="labels 복사 생략")
    args = ap.parse_args()

    process_folder(Path(args.src), Path(args.dst), args.force, not args.no_labels)


if __name__ == "__main__":
    main()
