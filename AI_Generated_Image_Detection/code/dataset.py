"""
dataset.py
==========
CIFAKE 데이터 로더 + (데이터 없이도 코드가 돌도록) 합성 데이터 폴백.

CIFAKE 폴더 구조 (Kaggle/HuggingFace에서 받으면 이 형태):
    data/
      train/REAL/*.jpg
      train/FAKE/*.jpg
      test/REAL/*.jpg
      test/FAKE/*.jpg

라벨 규약: REAL=0, FAKE(=AI생성)=1
"""
import os, glob
import numpy as np
import cv2


def load_cifake(root, split="train", limit_per_class=None):
    """root/<split>/REAL, .../FAKE 에서 읽어 (X[N,32,32,3] uint8 RGB, y[N]) 반환. REAL=0, FAKE=1."""
    mapping = {"REAL": 0, "FAKE": 1}
    imgs, labels = [], []
    for cls_name, cls_label in mapping.items():
        files = sorted(glob.glob(os.path.join(root, split, cls_name, "*")))
        if limit_per_class:
            files = files[:limit_per_class]
        for f in files:
            bgr = cv2.imread(f)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (32, 32):
                rgb = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
            imgs.append(rgb); labels.append(cls_label)
    if not imgs:
        raise FileNotFoundError(f"'{os.path.join(root, split)}' 에서 이미지를 못 찾음. CIFAKE 경로 확인.")
    return np.stack(imgs), np.array(labels)


def _smooth_base(rng, size=32):
    """저주파 위주의 자연스러운 배경 한 장 (RGB float[0,1])."""
    small = rng.random((4, 4, 3)).astype(np.float32)
    base = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
    base += 0.04 * rng.standard_normal((size, size, 3)).astype(np.float32)
    return np.clip(base, 0, 1)


def make_synthetic(n_per_class=400, size=32, seed=0):
    """합성 (X, y). REAL=자연영상 흉내, FAKE=주기적 격자(주파수 아티팩트) 주입."""
    rng = np.random.default_rng(seed)
    imgs, labels = [], []
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(n_per_class):
        img = _smooth_base(rng, size)
        imgs.append((img * 255).astype(np.uint8)); labels.append(0)
    for _ in range(n_per_class):
        img = _smooth_base(rng, size)
        freq = rng.uniform(0.6, 1.2); phase = rng.uniform(0, np.pi)
        grid = 0.06 * np.sin(2*np.pi*freq*xx + phase)[..., None]
        grid = grid + 0.06 * np.sin(2*np.pi*freq*yy + phase)[..., None]
        img = np.clip(img + grid.astype(np.float32), 0, 1)
        imgs.append((img * 255).astype(np.uint8)); labels.append(1)
    X = np.stack(imgs); y = np.array(labels)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def get_data(root=None, limit_per_class=None, seed=0):
    """root 유효시 CIFAKE, 아니면 합성. 반환 (Xtr,ytr,Xte,yte,source)."""
    if root and os.path.isdir(os.path.join(root, "train")):
        Xtr, ytr = load_cifake(root, "train", limit_per_class)
        Xte, yte = load_cifake(root, "test", limit_per_class)
        return Xtr, ytr, Xte, yte, "cifake"
    X, y = make_synthetic(n_per_class=(limit_per_class or 400), seed=seed)
    n = len(y); cut = int(n * 0.75)
    return X[:cut], y[:cut], X[cut:], y[cut:], "synthetic"
