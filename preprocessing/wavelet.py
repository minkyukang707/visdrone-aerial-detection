"""Wavelet 기반 디테일 강화.

YCrCb의 Y(휘도) 채널에 2D DWT를 적용해 고주파(디테일) 성분을 증폭한 뒤
역변환하여 소형 객체의 윤곽을 또렷하게 만든다. (항공뷰 소형객체 탐지에 유리)
"""
from __future__ import annotations

import cv2
import numpy as np
import pywt


def enhance_detail(img: np.ndarray, wavelet: str = "haar",
                   gain: float = 1.3, denoise: float = 0.0) -> np.ndarray:
    """Wavelet 고주파 성분 증폭으로 디테일 강화.

    Args:
        img: BGR uint8.
        wavelet: 사용할 wavelet 종류.
        gain: 고주파(LH/HL/HH) 증폭 계수.
        denoise: >0이면 soft-thresholding으로 노이즈 억제.
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)

    cA, (cH, cV, cD) = pywt.dwt2(y, wavelet)
    if denoise > 0:
        cH, cV, cD = (pywt.threshold(c, denoise, mode="soft") for c in (cH, cV, cD))
    cH, cV, cD = cH * gain, cV * gain, cD * gain

    y_rec = pywt.idwt2((cA, (cH, cV, cD)), wavelet)
    y_rec = np.clip(y_rec[: y.shape[0], : y.shape[1]], 0, 255).astype(np.uint8)

    ycrcb[:, :, 0] = y_rec
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
