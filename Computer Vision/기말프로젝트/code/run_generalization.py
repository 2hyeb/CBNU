"""
run_generalization.py
=====================
일반화 실험: CIFAKE(Stable Diffusion 1.4)로 학습한 탐지기가
'못 보던' 다른 생성기(Gemini) 이미지에 얼마나 통하는가?

  - 학습:  CIFAKE train (REAL + FAKE)
  - 평가1(in-distribution):   CIFAKE test
  - 평가2(generalization):    Gemini 가짜 + CIFAKE 진짜(같은 수)

Gemini 이미지 전처리: 중앙 정사각 크롭 -> 워터마크 제거용 중앙 85% 크롭 -> 32x32.
(눈에 보이는 구석 워터마크 제거 목적)

산출물(results/):
  - generalization_table.csv
  - generalization_bar.png
"""
import os, glob, csv
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import dataset, features

FEATS = ["DFT", "HIST", "HOG", "SOBEL"]
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
GEN_DIR = os.path.join(HERE, "..", "generalization_test", "FAKE_gemini")
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)


def preprocess_external(bgr, keep=0.85):
    """다른 생성기 이미지 -> CIFAKE 형식(32x32 RGB).
    중앙 정사각 크롭 후, 중앙 keep 비율만 남겨 구석 워터마크 제거 -> 32x32."""
    h, w = bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    sq = bgr[y0:y0 + s, x0:x0 + s]          # 중앙 정사각
    m = int(s * (1 - keep) / 2)
    sq = sq[m:s - m, m:s - m]                # 중앙 85%
    sq = cv2.resize(sq, (32, 32), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)


def load_external_fakes(folder):
    imgs = []
    for f in sorted(glob.glob(os.path.join(folder, "*"))):
        bgr = cv2.imread(f)
        if bgr is None:
            continue
        imgs.append(preprocess_external(bgr))
    return np.stack(imgs)


def ci95(p, n):
    """이항 비율 95% 신뢰구간 반치폭 (정규근사)."""
    return 1.96 * np.sqrt(max(p * (1 - p), 1e-9) / n)


def feat_matrix(X, which):
    if which == "FUSION":
        return np.hstack([features.build_matrix(X, f) for f in FEATS])
    return features.build_matrix(X, which)


def main():
    # 1) 데이터 로드
    Xtr, ytr, Xte, yte, source = dataset.get_data(DATA)
    assert source == "cifake", "CIFAKE 데이터가 필요합니다 (data 폴더 확인)."
    print(f"[학습] CIFAKE train={len(ytr)}  in-dist test={len(yte)}")

    # 2) Gemini 가짜 로드 + 같은 수의 CIFAKE 진짜로 일반화 테스트셋 구성
    Xfake = load_external_fakes(GEN_DIR)
    n_fake = len(Xfake)
    real_idx = np.where(yte == 0)[0]
    rng = np.random.default_rng(0)
    sel = rng.choice(real_idx, size=min(n_fake, len(real_idx)), replace=False)
    Xreal = Xte[sel]
    Xgen = np.concatenate([Xreal, Xfake])
    ygen = np.concatenate([np.zeros(len(Xreal)), np.ones(len(Xfake))]).astype(int)
    print(f"[일반화 테스트] Gemini 가짜={n_fake}, CIFAKE 진짜={len(Xreal)}, 합계={len(ygen)}")

    rows = []
    for feat in FEATS + ["FUSION"]:
        Ftr = feat_matrix(Xtr, feat)
        clf = make_pipeline(StandardScaler(),
                            SVC(kernel="rbf", C=10, gamma="scale", random_state=0))
        clf.fit(Ftr, ytr)

        # in-distribution (CIFAKE test)
        acc_in = accuracy_score(yte, clf.predict(feat_matrix(Xte, feat)))

        # generalization (Gemini + CIFAKE real)
        pred_gen = clf.predict(feat_matrix(Xgen, feat))
        acc_gen = accuracy_score(ygen, pred_gen)
        # Gemini 가짜를 가짜로 잡은 비율(recall on fake)
        fake_mask = ygen == 1
        recall_fake = accuracy_score(ygen[fake_mask], pred_gen[fake_mask])

        rows.append({
            "feature": feat,
            "acc_in_distribution": round(acc_in, 3),
            "acc_generalization": round(acc_gen, 3),
            "gemini_fake_recall": round(recall_fake, 3),
            "fake_recall_ci95": round(ci95(recall_fake, n_fake), 3),
            "drop": round(acc_in - acc_gen, 3),
        })
        print(f"  {feat:7s} in={acc_in:.3f}  gen={acc_gen:.3f}  "
              f"fake_recall={recall_fake:.3f}±{ci95(recall_fake,n_fake):.3f}  "
              f"drop={acc_in-acc_gen:+.3f}")

    # CSV
    cols = ["feature", "acc_in_distribution", "acc_generalization",
            "gemini_fake_recall", "fake_recall_ci95", "drop"]
    with open(os.path.join(OUT, "generalization_table.csv"), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)

    # 막대그래프: 특징별 in-dist vs generalization
    labels = [r["feature"] for r in rows]
    a_in = [r["acc_in_distribution"] for r in rows]
    a_gen = [r["acc_generalization"] for r in rows]
    x = np.arange(len(labels)); wbar = 0.38
    plt.figure(figsize=(7, 4.2))
    plt.bar(x - wbar/2, a_in, wbar, label="In-distribution (CIFAKE)")
    plt.bar(x + wbar/2, a_gen, wbar, label="Generalization (Gemini)")
    plt.axhline(0.5, color="k", ls="--", alpha=0.4, label="chance(0.5)")
    plt.xticks(x, labels); plt.ylim(0, 1)
    plt.ylabel("Accuracy"); plt.title("CIFAKE-trained detector: in-dist vs Gemini")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "generalization_bar.png"), dpi=120)
    plt.close()

    print(f"\n[완료] 결과: {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
