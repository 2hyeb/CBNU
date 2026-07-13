# 실행 안내

## 1. 패키지 설치
```bash
pip install numpy opencv-python scikit-learn scikit-image matplotlib scipy
```

## 2. 바로 실행 (데이터 없이, 합성 데이터로 파이프라인 검증)
```bash
cd code
python run_experiment.py
```
→ `../results/` 에 결과표(CSV), 혼동행렬, ROC, 오분류 사례가 생성된다.

## 3. 실제 CIFAKE로 실행
1) 데이터 다운로드 (둘 중 하나)
   - Kaggle: "CIFAKE: Real and AI-Generated Synthetic Images"
   - HuggingFace: `dragonintelligence/CIFAKE-image-dataset`
2) 아래 구조로 배치
   ```
   data/
     train/REAL/*.jpg   train/FAKE/*.jpg
     test/REAL/*.jpg    test/FAKE/*.jpg
   ```
3) 실행 (처음엔 --limit 로 빠르게 확인 권장)
   ```bash
   python run_experiment.py --data ../data --limit 2000   # 클래스당 2000장
   python run_experiment.py --data ../data                # 전체
   ```

## 파일 구성
- `dataset.py`        : CIFAKE 로더 + 합성 폴백
- `features.py`       : 4개 특징추출기 (DFT / HIST / HOG / SOBEL)
- `run_experiment.py` : 학습·평가·시각화 (SVM/KNN + KMeans 베이스라인)

## 확장 아이디어 (심화 점수용)
- `features.py` 에 LBP, co-occurrence(GLCM) 특징 추가
- 다른 생성기 이미지로 test 만 교체해 "일반화 한계" 실험
- 4개 특징을 결합(concatenate)한 fusion 성능 추가
