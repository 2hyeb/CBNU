"""
features.py
===========
4가지 '고전 CV' 특징 추출기. 모두 수업에서 배운 알고리즘에 뿌리를 둠.

  1) feat_dft   : 이산 푸리에 변환(DFT) 기반 방사형 파워 스펙트럼   (L4)
  2) feat_hist  : 색공간 변환 + 채널별 히스토그램/통계             (L3)
  3) feat_hog   : HoG (Histogram of Oriented Gradients)            (L12)
  4) feat_sobel : Sobel gradient 기반 엣지 통계                    (L4/L5)

각 함수: 입력 RGB 이미지(32x32x3 uint8) -> 1차원 float 특징벡터.
extract_all(): 한 장에 대해 {이름: 벡터} dict 반환.
build_matrix(): 이미지 배열 전체를 특징행렬로 변환.
"""
import numpy as np
import cv2
from skimage.feature import hog


# ---- 공통 ----------------------------------------------------------
def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# ---- 1) DFT 방사형 파워 스펙트럼  (Lecture 4) -----------------------
def feat_dft(img, n_bins=24):
    """
    그레이 영상 -> 2D FFT -> 중심정렬 -> log 파워 스펙트럼 ->
    중심으로부터 거리(주파수)별로 평균(방사형 평균)을 내어 n_bins 특징.
    생성 영상의 고주파 격자 아티팩트가 이 곡선의 모양에 드러남.
    """
    g = _gray(img).astype(np.float32) / 255.0
    g = g - g.mean()
    F = np.fft.fftshift(np.fft.fft2(g))
    power = np.log1p(np.abs(F) ** 2)
    h, w = power.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = r.max()
    edges = np.linspace(0, r_max + 1e-6, n_bins + 1)
    feats = []
    for i in range(n_bins):
        m = (r >= edges[i]) & (r < edges[i + 1])
        feats.append(power[m].mean() if m.any() else 0.0)
    return np.asarray(feats, dtype=np.float32)


# ---- 2) 색/히스토그램 통계  (Lecture 3) ----------------------------
def feat_hist(img, bins=16):
    """
    RGB + HSV 각 채널 히스토그램(정규화) + 채널별 평균/표준편차.
    생성 영상은 색 분포·채도 통계가 실제와 미묘히 다른 경향.
    """
    feats = []
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    for space in (img, hsv):
        for c in range(3):
            ch = space[:, :, c]
            h = cv2.calcHist([ch], [0], None, [bins], [0, 256]).flatten()
            h = h / (h.sum() + 1e-8)
            feats.extend(h.tolist())
            feats.append(ch.mean() / 255.0)
            feats.append(ch.std() / 255.0)
    return np.asarray(feats, dtype=np.float32)


# ---- 3) HoG  (Lecture 12) ------------------------------------------
def feat_hog(img):
    """그레이 영상의 기울기 방향 히스토그램. 32x32에 맞춘 작은 셀."""
    g = _gray(img)
    return hog(g, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys",
               feature_vector=True).astype(np.float32)


# ---- 4) Sobel 엣지 통계  (Lecture 4/5) -----------------------------
def feat_sobel(img, bins=16):
    """
    Sobel로 gradient 크기/방향 계산 -> 크기 통계 + 크기 히스토그램 +
    엣지 밀도(임계 초과 비율). 생성 영상은 엣지 선명도/분포가 다른 경향.
    """
    g = _gray(img).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    feats = [mag.mean(), mag.std(), mag.max(),
             np.median(mag), float((mag > 50).mean())]
    h, _ = np.histogram(mag, bins=bins, range=(0, 400), density=True)
    feats.extend(h.tolist())
    return np.asarray(feats, dtype=np.float32)


# ---- 묶음 ----------------------------------------------------------
EXTRACTORS = {
    "DFT":   feat_dft,
    "HIST":  feat_hist,
    "HOG":   feat_hog,
    "SOBEL": feat_sobel,
}


def build_matrix(X, which):
    """X[N,32,32,3] -> 특징행렬[N,D]. which: EXTRACTORS의 키."""
    fn = EXTRACTORS[which]
    return np.stack([fn(im) for im in X])
