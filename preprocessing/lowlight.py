"""저조도(야간) 영상 개선.

- 적응형 감마 보정: 영상 평균 밝기에 따라 감마를 자동 조절
- Single-Scale Retinex(SSR): 조명 성분을 추정·제거해 디테일 복원
"""
from __future__ import annotations

import cv2
import numpy as np


def auto_gamma(img: np.ndarray, target_mean: float = 0.5) -> np.ndarray:
    """영상 평균 밝기를 target_mean에 맞추도록 감마를 자동 산출해 보정."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mean = float(np.clip(gray.mean(), 1e-3, 1.0))
    gamma = float(np.log(target_mean) / np.log(mean)) if mean > 0 else 1.0
    gamma = float(np.clip(gamma, 0.4, 2.5))
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def single_scale_retinex(img: np.ndarray, sigma: float = 80.0) -> np.ndarray:
    """SSR: log(I) - log(Gaussian(I)) 후 정규화."""
    out = np.zeros_like(img, dtype=np.float32)
    img_f = img.astype(np.float32) + 1.0
    for c in range(3):
        blur = cv2.GaussianBlur(img_f[:, :, c], (0, 0), sigma)
        retinex = np.log(img_f[:, :, c]) - np.log(blur + 1.0)
        lo, hi = np.percentile(retinex, 1), np.percentile(retinex, 99)
        retinex = np.clip((retinex - lo) / (hi - lo + 1e-6), 0, 1)
        out[:, :, c] = retinex * 255.0
    return out.astype(np.uint8)


def enhance_lowlight(img: np.ndarray, use_retinex: bool = True) -> np.ndarray:
    """저조도 개선 파이프라인: 적응형 감마 (+ 선택적 Retinex)."""
    out = auto_gamma(img)
    if use_retinex:
        out = cv2.addWeighted(out, 0.6, single_scale_retinex(out), 0.4, 0)
    return out
