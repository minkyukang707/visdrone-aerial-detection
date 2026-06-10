"""전처리 파이프라인 before/after 데모 패널 생성 (README용).

VisDrone 영상은 대부분 주간이므로, 저조도/안개 악조건을 인위적으로 시뮬레이션한 뒤
복원 알고리즘을 적용해 효과를 시각적으로 비교한다.
"""
from pathlib import Path

import cv2
import numpy as np

from clahe import apply_clahe
from dehaze import dehaze
from lowlight import enhance_lowlight
from wavelet import enhance_detail

ROOT = Path(__file__).resolve().parent.parent


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def synth_lowlight(img, gamma=2.6):
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def synth_haze(img, beta=0.9):
    h, w = img.shape[:2]
    A = 230
    d = np.tile(np.linspace(0.3, 1.0, h)[:, None], (1, w)).astype(np.float32)
    t = np.exp(-beta * d)[:, :, None]
    return np.clip(img * t + A * (1 - t), 0, 255).astype(np.uint8)


def main():
    src = ROOT / "datasets/VisDrone/VisDrone2019-DET-val/images"
    imgs = sorted(src.glob("*.jpg"))
    img = cv2.imread(str(imgs[10]))
    img = cv2.resize(img, (640, 480))

    # 행 1: 저조도 → 복원
    low = synth_lowlight(img)
    low_r = enhance_lowlight(low)
    row1 = cv2.hconcat([label(img, "Original"),
                        label(low, "Low-light (synthetic)"),
                        label(low_r, "Restored (gamma+Retinex)")])

    # 행 2: 안개 → 복원 → 디테일 강화
    haze = synth_haze(img)
    haze_r = dehaze(haze)
    final = enhance_detail(apply_clahe(haze_r), gain=1.2, denoise=2.0)
    row2 = cv2.hconcat([label(haze, "Haze (synthetic)"),
                        label(haze_r, "Dehazed (DCP)"),
                        label(final, "Dehaze+CLAHE+Wavelet")])

    panel = cv2.vconcat([row1, row2])
    out = ROOT / "assets" / "preprocess_demo.jpg"
    cv2.imwrite(str(out), panel)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
