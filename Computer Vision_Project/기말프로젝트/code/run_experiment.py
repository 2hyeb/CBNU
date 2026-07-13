"""
run_experiment.py
=================
4개 특징(DFT/HIST/HOG/SOBEL) x 2개 분류기(SVM/KNN) 비교 실험.
+ K-means 비지도 베이스라인.

산출물(results/ 폴더):
  - results_table.csv         : 특징x분류기별 정확도/정밀도/재현율/F1/AUC
  - roc_<clf>.png             : 분류기별 ROC 곡선(특징 4개 한 그림)
  - confmat_<feat>_<clf>.png  : 혼동행렬
  - misclassified_<feat>.png  : 대표 오분류 사례(고찰용)

사용법:
  python run_experiment.py                 # 데이터 없으면 합성으로 자동 실행
  python run_experiment.py --data ../data --limit 2000   # CIFAKE 사용
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve, confusion_matrix,
                             ConfusionMatrixDisplay)

import dataset, features

FEATS = ["DFT", "HIST", "HOG", "SOBEL"]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)


def get_classifiers():
    return {
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10,
                             gamma="scale", random_state=0)),
        "KNN": make_pipeline(StandardScaler(),
                             KNeighborsClassifier(n_neighbors=7)),
    }


def get_score(clf, X):
    """ROC/AUC용 점수. SVM은 decision_function, KNN은 predict_proba 사용."""
    if hasattr(clf, "predict_proba"):
        try:
            return clf.predict_proba(X)[:, 1]
        except Exception:
            pass
    return clf.decision_function(X)


def scores(y_true, y_pred, y_score):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": p, "recall": r, "f1": f1,
        "auc": roc_auc_score(y_true, y_score),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="CIFAKE 루트 (train/test 포함)")
    ap.add_argument("--limit", type=int, default=None, help="클래스당 최대 장수")
    args = ap.parse_args()

    Xtr, ytr, Xte, yte, source = dataset.get_data(args.data, args.limit)
    print(f"[데이터] source={source}  train={len(ytr)}  test={len(yte)}")

    rows = []
    roc_data = {clf: [] for clf in ["SVM", "KNN"]}

    for feat in FEATS:
        Ftr = features.build_matrix(Xtr, feat)
        Fte = features.build_matrix(Xte, feat)
        print(f"[특징] {feat:5s} dim={Ftr.shape[1]}")
        for clf_name, clf in get_classifiers().items():
            clf.fit(Ftr, ytr)
            y_pred = clf.predict(Fte)
            y_score = get_score(clf, Fte)
            s = scores(yte, y_pred, y_score)
            s.update(feature=feat, classifier=clf_name)
            rows.append(s)
            print(f"    {clf_name}: acc={s['accuracy']:.3f} "
                  f"f1={s['f1']:.3f} auc={s['auc']:.3f}")
            cm = confusion_matrix(yte, y_pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=["REAL", "FAKE"])
            disp.plot(cmap="Blues", colorbar=False)
            plt.title(f"{feat} + {clf_name}")
            plt.savefig(os.path.join(OUT, f"confmat_{feat}_{clf_name}.png"),
                        dpi=120, bbox_inches="tight"); plt.close()
            fpr, tpr, _ = roc_curve(yte, y_score)
            roc_data[clf_name].append((feat, fpr, tpr, s["auc"]))

    for clf_name, items in roc_data.items():
        plt.figure(figsize=(5, 5))
        for feat, fpr, tpr, auc in items:
            plt.plot(fpr, tpr, label=f"{feat} (AUC={auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC - {clf_name}"); plt.legend(loc="lower right")
        plt.savefig(os.path.join(OUT, f"roc_{clf_name}.png"),
                    dpi=120, bbox_inches="tight"); plt.close()

    # K-means 비지도 베이스라인 (4개 특징 결합)
    Ftr_all = np.hstack([features.build_matrix(Xtr, f) for f in FEATS])
    Fte_all = np.hstack([features.build_matrix(Xte, f) for f in FEATS])
    scaler = StandardScaler().fit(Ftr_all)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(scaler.transform(Ftr_all))
    cl = km.predict(scaler.transform(Fte_all))
    acc_km = max(accuracy_score(yte, cl), accuracy_score(yte, 1 - cl))
    rows.append({"feature": "ALL", "classifier": "KMeans(unsup)",
                 "accuracy": acc_km, "precision": np.nan, "recall": np.nan,
                 "f1": np.nan, "auc": np.nan})
    print(f"[비지도] KMeans acc={acc_km:.3f}")

    import csv
    cols = ["feature", "classifier", "accuracy", "precision", "recall", "f1", "auc"]
    with open(os.path.join(OUT, "results_table.csv"), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    # 오분류 사례 (SVM+DFT 기준 8장)
    Fte_dft = features.build_matrix(Xte, "DFT")
    clf = get_classifiers()["SVM"].fit(features.build_matrix(Xtr, "DFT"), ytr)
    pred = clf.predict(Fte_dft)
    wrong = np.where(pred != yte)[0][:8]
    if len(wrong):
        plt.figure(figsize=(8, 2))
        for i, idx in enumerate(wrong):
            plt.subplot(1, len(wrong), i + 1)
            plt.imshow(Xte[idx]); plt.axis("off")
            plt.title(f"T{yte[idx]}/P{pred[idx]}", fontsize=7)
        plt.suptitle("Misclassified (SVM+DFT)  T=true,P=pred  0=REAL 1=FAKE")
        plt.savefig(os.path.join(OUT, "misclassified_DFT.png"),
                    dpi=120, bbox_inches="tight"); plt.close()

    print(f"\n[완료] 결과 저장 위치: {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
