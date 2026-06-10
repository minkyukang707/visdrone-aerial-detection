"""안개 제거(Dehazing) — Dark Channel Prior (He et al., 2009).

대기 산란 모델 I(x) = J(x)t(x) + A(1 - t(x)) 를 가정하고,
dark channel로 전달량 t(x)와 대기광 A를 추정해 안개 없는 J(x)를 복원한다.
"""
from __future__ import annotations

import cv2
import numpy as np


def _dark_channel(img: np.ndarray, patch: int = 15) -> np.ndarray:
    min_c = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(min_c, kernel)


def _atmospheric_light(img: np.ndarray, dark: np.ndarray, top_ratio: float = 0.001) -> np.ndarray:
    h, w = dark.shape
    n = max(int(h * w * top_ratio), 1)
    idx = np.argsort(dark.ravel())[-n:]
    flat = img.reshape(-1, 3)
    return flat[idx].max(axis=0)


def dehaze(img: np.ndarray, patch: int = 15, omega: float = 0.95,
           t0: float = 0.1, refine: bool = True) -> np.ndarray:
    """Dark Channel Prior 기반 안개 제거.

    Args:
        img: BGR uint8.
        omega: 안개 제거 강도 (0~1).
        t0: 최소 전달량 (하한).
        refine: guided filter로 transmission map 정제 여부.
    """
    I = img.astype(np.float32) / 255.0
    dark = _dark_channel(I, patch)
    A = _atmospheric_light(I, dark)
    A = np.clip(A, 1e-3, None)

    t = 1.0 - omega * _dark_channel(I / A, patch)
    if refine:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t = cv2.ximgproc.guidedFilter(gray, t.astype(np.float32), 40, 1e-3) \
            if hasattr(cv2, "ximgproc") else cv2.GaussianBlur(t, (0, 0), 20)
    t = np.clip(t, t0, 1.0)[:, :, None]

    J = (I - A) / t + A
    return np.clip(J * 255.0, 0, 255).astype(np.uint8)
