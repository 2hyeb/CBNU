# 문서 이미지 전처리 파이프라인 (LLM/OCR 입력용) — 중간 프로젝트

사진·스캔 문서를 OCR과 LLM이 잘 읽도록 **고전 컴퓨터비전 알고리즘**으로 전처리하고,
전처리 전/후 OCR 인식률을 비교한 프로젝트입니다.

> 산업 컴퓨터비전 실제 · 중간 프로젝트 · CBNU 2025 · 이혜빈

## 개요

- 문서 이미지 → 화질 개선 → 텍스트 영역 검출 → 기하 보정(deskew) → 영역 분할 순으로 전처리
- 전처리 전 vs 후를 **EasyOCR**로 인식시켜 성능 차이를 정량 비교

## 적용 알고리즘 (강의 1~7 기반)

| 강의 | 알고리즘 |
|------|----------|
| Lec 1–3 | 색공간 변환 (BGR→Grayscale, LAB) |
| Lec 4 | Unsharp Masking, Gaussian LPF, Otsu Thresholding, Morphological filters |
| Lec 5 | Sobel gradient, Canny edge, Hough Transform(기울기 보정), CCL(연결요소 라벨링) |
| Lec 6 | K-means 영역 분할 |
| Lec 7 | Harris corner, SIFT keypoint (문서 코너 검출) |

## 파일 구성

```
├── doc_cv_pipeline.py   # 전처리 파이프라인 (step1~4)
├── ocr_comparison.py    # 전처리 전/후 EasyOCR 인식률 비교
├── test_img.jpg         # 테스트 입력 이미지
├── test_img_label.json  # 정답 텍스트 라벨
├── demo_document.png    # 데모용 문서 이미지
├── results/             # 단계별 결과 이미지 + OCR 비교 결과(json)
└── requirements.txt
```

## 실행

```bash
pip install -r requirements.txt

python doc_cv_pipeline.py    # 전처리 파이프라인 실행 → results/에 단계별 이미지 저장
python ocr_comparison.py     # 전처리 전/후 OCR 인식률 비교 → results/ocr_comparison.png, ocr_results.json
```

> 한글 렌더링에 Windows 기본 폰트(malgun.ttf)를 사용합니다.
