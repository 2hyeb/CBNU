# 고전 컴퓨터비전 특징 기반 AI 생성 이미지 판별

딥러닝 탐지기 없이, 수업에서 배운 **고전 컴퓨터비전 특징(handcrafted feature)** 만으로
실제 이미지와 AI 생성 이미지를 어디까지 구분할 수 있는지, 그리고 학습에 쓰지 않은
**다른 생성기(Gemini)** 에도 통하는지를 실험한 프로젝트입니다.

> 산업 컴퓨터비전 실제 · 기말 프로젝트 · 산업인공지능학과 이혜빈

---

## 문제 정의

- 최근 생성 AI(Stable Diffusion, GPT Image 2, Gemini Nano Banana 2 등) 이미지는 육안 구분이 어려운 수준이고, 실제로 2026년 4월 대전 오월드 늑대 탈출 사건에서 AI 조작 사진이 수색에 혼선을 준 사례가 있었다.
- 기존 탐지는 무거운 CNN/ViT에 의존한다. 본 프로젝트는 가볍고 해석 가능한 고전 특징의 탐지 한계를 정량적으로 규명한다.

## 연구 질문

1. 네 가지 고전 특징(DFT · 색 · HoG · Sobel) 중 AI 생성 흔적을 가장 잘 포착하는 특징은?
2. 분류기(SVM · K-NN)와 비지도(K-means)의 성능 차이는?
3. 한 생성기(Stable Diffusion 1.4)로 학습한 탐지기가 다른 생성기(Gemini)에도 일반화되는가?

## 적용 알고리즘

| 구분 | 알고리즘 |
|------|----------|
| 특징 추출 | **DFT**(2D 푸리에 변환 방사형 파워 스펙트럼), **HIST**(RGB·HSV 히스토그램/통계), **HoG**, **Sobel** 엣지 통계 |
| 분류기 | **SVM**(RBF), **K-NN** |
| 비지도 베이스라인 | **K-means** |

## 데이터

- **CIFAKE** (공개): 진짜 = CIFAR-10 실사진, 가짜 = Stable Diffusion 1.4 생성 이미지, 32×32 RGB
  - HuggingFace: `dragonintelligence/CIFAKE-image-dataset` 또는 Kaggle "CIFAKE"
- **일반화 테스트셋**: Gemini Nano Banana 2로 직접 생성한 50장 (`generalization_test/FAKE_gemini/`)

---

## 폴더 구조

```
.
├── code/
│   ├── dataset.py            # CIFAKE 로더 + 합성 데이터 폴백
│   ├── features.py           # 4개 특징 추출기 (DFT/HIST/HoG/Sobel)
│   ├── run_experiment.py     # 특징×분류기 비교 (in-distribution)
│   ├── run_generalization.py # 다른 생성기(Gemini) 일반화 실험
│   └── README.md             # 코드 실행 상세 안내
├── generalization_test/
│   └── FAKE_gemini/          # Gemini 생성 이미지 50장
├── results/                  # 결과 그래프·표 (ROC, 혼동행렬 등)
├── requirements.txt
└── README.md
```

> `data/`(CIFAKE)와 `venv/`는 용량이 커서 `.gitignore`로 제외되어 있습니다. 아래 안내대로 받으세요.

---

## 설치 및 실행

```bash
# 1) 가상환경
python -m venv venv
# Windows: .\venv\Scripts\Activate.ps1   |   macOS/Linux: source venv/bin/activate

# 2) 패키지 설치
pip install -r requirements.txt

# 3) CIFAKE 다운로드 후 아래 구조로 배치
#    data/train/REAL, data/train/FAKE, data/test/REAL, data/test/FAKE

# 4) 실험 실행
cd code
python run_experiment.py --data ../data --limit 2000   # 특징×분류기 비교
python run_generalization.py                            # Gemini 일반화 실험
```

> 데이터가 없으면 `run_experiment.py`는 파이프라인 검증용 **합성 데이터**로 자동 실행됩니다.
> 결과 그래프·표는 `results/`에 저장됩니다.

---

## 주요 결과

### ① 학습 분포 내(CIFAKE) 판별 성능 — SVM 기준

| 특징 | 정확도 | AUC |
|------|--------|-----|
| **DFT** | **0.796** | **0.865** |
| HIST | 0.770 | 0.858 |
| HoG | 0.753 | 0.819 |
| Sobel | 0.710 | 0.750 |

- 주파수 특징(DFT)이 가장 높은 판별력, 모든 특징에서 SVM > K-NN, 비지도 K-means는 0.61(네 특징 결합).

### ② 다른 생성기(Gemini) 일반화 성능

| 특징 | CIFAKE | Gemini | 가짜검출 |
|------|--------|--------|----------|
| **DFT** | 0.80 | 0.86 | **0.94** |
| HIST | 0.77 | 0.52 | 0.26 |
| HoG | 0.75 | 0.62 | 0.42 |
| Sobel | 0.71 | 0.57 | 0.44 |
| **FUSION** | **0.90** | 0.83 | 0.72 |

- 주파수 특징(DFT)은 못 보던 최신 생성기에도 잘 통하지만, 색·질감 특징은 학습 생성기에 과적합되어 무너진다.

---

## 한계

- 정확도 80~90%대로 딥러닝(93~96%)보다 낮음 — 가볍고 해석 가능한 대신 표현력 한계.
- Gemini 일반화 테스트는 50장으로 신뢰구간이 큼(약 ±14%).
- 일반화 실험에서 진짜는 원본 32×32, 가짜는 고해상도 축소본이라 해상도 교란이 섞임(메인 CIFAKE 실험은 무관).

## 출처

- CIFAKE: Bird & Lotfi, "CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images", IEEE Access, 2024.
- 늑구 사건 이미지: 중앙일보 김성태 객원기자 (joongang.co.kr/article/25419220)
