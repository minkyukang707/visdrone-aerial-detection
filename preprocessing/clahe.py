"""CLAHE 기반 대비 향상 (LAB 색공간 L 채널)."""
from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """LAB 색공간의 L 채널에만 CLAHE를 적용해 색 왜곡 없이 대비를 높인다.

    Args:
        img: BGR 이미지 (uint8).
        clip_limit: 대비 제한 (클수록 강하게).
        tile_grid: 타일 그리드 크기 (NxN).
    Returns:
        대비가 향상된 BGR 이미지.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
