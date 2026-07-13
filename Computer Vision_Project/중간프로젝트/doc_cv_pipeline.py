"""
=============================================================
 Document Image Preprocessing Pipeline for LLM-Based OCR
 산업 컴퓨터비전 실제 - 중간 프로젝트
 CBNU 2025 · 발표일: 2025-04-21
=============================================================

[적용 알고리즘 목록 - 강의 1~7 기반]
 Lec 4: Unsharp Masking, Gaussian LPF, Thresholding (Otsu), Morphological filters
 Lec 5: Sobel gradient, Canny edge detection, Hough Transform (deskewing), CCL
 Lec 6: K-means segmentation
 Lec 7: Harris corner, SIFT keypoint (문서 코너 검출)
 Lec 1-3: 색 공간 변환 (BGR→Grayscale, LAB)
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os

# Matplotlib 폰트/마이너스 설정 (한글 및 특수문자 깨짐 방지)
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 한글 폰트 경로 (Windows)
KOREAN_FONT_PATH = None
for font_name in ['malgunbd.ttf', 'gulim.ttc', 'batang.ttc']:
    font_path = os.path.join('C:\\Windows\\Fonts', font_name)
    if os.path.exists(font_path):
        KOREAN_FONT_PATH = font_path
        break
if KOREAN_FONT_PATH is None:
    KOREAN_FONT_PATH = 'C:\\Windows\\Fonts\\malgun.ttf'

# ─────────────────────────────────────────────────────────────
# 0. 유틸리티
# ─────────────────────────────────────────────────────────────

def show_comparison(images: list, titles: list, filename: str, cols: int = 3):
    """여러 이미지를 한 그림에 나란히 저장 (PIL로 한글 타이틀 추가)"""
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).flatten()
    for ax, img, title in zip(axes, images, titles):
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray', interpolation='nearest')
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation='nearest')
        # matplotlib 타이틀은 비우기 (PIL로 나중에 그릴 예정)
        ax.set_title('')
        ax.axis('off')
    for ax in axes[len(images):]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=220, bbox_inches='tight')
    plt.close()
    
    # PIL로 이미지를 읽어서 한글 타이틀 추가
    try:
        pil_img = Image.open(filename).convert('RGB')
        draw = ImageDraw.Draw(pil_img)
        
        # 폰트 로드 (한글)
        try:
            font = ImageFont.truetype(KOREAN_FONT_PATH, size=28)
        except:
            font = ImageFont.load_default()
        
        # 각 서브플롯의 타이틀 위치 계산 및 그리기
        img_w, img_h = pil_img.size
        # tight_layout 적용 후 여백 보정
        margin_left = int(img_w * 0.02)
        margin_top = int(img_h * 0.02)
        usable_w = img_w - 2 * margin_left
        usable_h = img_h - 2 * margin_top
        subplot_w = usable_w // cols
        subplot_h = usable_h // rows
        
        for idx, title in enumerate(titles):
            if title == '':
                continue
            row_idx = idx // cols
            col_idx = idx % cols
            # 서브플롯 중앙 상단에 배치
            cx = margin_left + col_idx * subplot_w + subplot_w // 2
            cy = margin_top + row_idx * subplot_h + 8
            
            # 텍스트 크기 측정
            try:
                bbox = font.getbbox(title)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except:
                tw, th = len(title) * 14, 20
            
            tx = cx - tw // 2
            ty = cy
            
            # 배경 박스 그리기 (가독성)
            pad = 4
            draw.rectangle(
                [tx - pad, ty - pad, tx + tw + pad, ty + th + pad],
                fill=(0, 0, 0, 200)
            )
            # 타이틀을 흰색으로 그리기
            draw.text((tx, ty), title, fill=(255, 255, 255), font=font)
        
        pil_img.save(filename)
    except Exception as e:
        print(f"  ⚠ PIL 타이틀 추가 실패: {e}")
    
    print(f"  → 저장: {filename}")


# ─────────────────────────────────────────────────────────────
# STEP 1: 이미지 품질 개선  (Lec 3, 4)
# ─────────────────────────────────────────────────────────────

def step1_enhance(image_bgr: np.ndarray) -> dict:
    """
    1-a. Gaussian LPF로 노이즈 제거
    1-b. Unsharp Masking으로 선명화
    1-c. LAB 색 공간 → L채널 CLAHE로 조명 균일화
    반환: {'enhanced': ..., 'gray': ..., 'clahe_l': ...}
    """
    results = {}

    # 1-a. Gaussian smoothing (Lec 4 – lowpass filter)
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 1.0)

    # 1-b. Unsharp Masking  f_high = f - f_low,  enhanced = f + k·f_high  (Lec 4)
    k = 1.5
    high_freq = cv2.subtract(image_bgr.astype(np.int16),
                             blurred.astype(np.int16))
    enhanced = np.clip(image_bgr.astype(np.int16) + k * high_freq,
                       0, 255).astype(np.uint8)
    results['enhanced'] = enhanced

    # 1-c. LAB 변환 → L채널 CLAHE  (Lec 2-3 색 공간)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    enhanced_clahe = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    results['enhanced_clahe'] = enhanced_clahe

    # Grayscale 변환 (이후 단계 공통 사용)
    gray = cv2.cvtColor(enhanced_clahe, cv2.COLOR_BGR2GRAY)
    results['gray'] = gray

    return results


# ─────────────────────────────────────────────────────────────
# STEP 2: 텍스트 영역 탐지  (Lec 4, 5)
# ─────────────────────────────────────────────────────────────

def step2_detect_text_regions(gray: np.ndarray) -> dict:
    """
    2-a. Otsu Thresholding  (Lec 4)
    2-b. Morphological dilation – 텍스트 블록 연결  (Lec 4)
    2-c. Canny Edge Detection  (Lec 5)
    2-d. Connected Component Labeling (CCL)  (Lec 5)
    반환: {'binary', 'dilated', 'edges', 'ccl_vis', 'bboxes'}
    """
    results = {}

    # 2-a. Otsu Binarization
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    results['binary'] = binary

    # 2-b. Morphological dilation → opening으로 노이즈 정리 후 dilation으로 글자 연결
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    dilated = cv2.dilate(opened, kernel_dilate, iterations=2)
    results['dilated'] = dilated

    # 2-c. Canny Edge Detection
    edges = cv2.Canny(gray, 50, 150)
    results['edges'] = edges

    # 2-d. Connected Component Labeling
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dilated, connectivity=8)

    # 의미 있는 컴포넌트만 필터 (너무 작거나 전체 화면 크기인 것 제외)
    h, w = gray.shape
    min_area = (h * w) * 0.0005   # 전체의 0.05% 이상
    max_area = (h * w) * 0.95

    bboxes = []
    for i in range(1, num_labels):   # 0 = 배경
        x, y, bw, bh, area = stats[i]
        if min_area < area < max_area:
            bboxes.append((x, y, bw, bh))

    # CCL 시각화
    ccl_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for (x, y, bw, bh) in bboxes:
        cv2.rectangle(ccl_vis, (x, y), (x + bw, y + bh), (0, 200, 0), 2)
    results['ccl_vis'] = ccl_vis
    results['bboxes'] = bboxes

    print(f"  → 검출된 텍스트 블록: {len(bboxes)}개")
    return results


# ─────────────────────────────────────────────────────────────
# STEP 3: 기하학적 보정  (Lec 5, 7)
# ─────────────────────────────────────────────────────────────

def step3_geometric_correction(image_bgr: np.ndarray,
                               gray: np.ndarray) -> dict:
    """
    3-a. Hough Line Transform → 기울기 검출 → Deskewing  (Lec 5)
    3-b. Harris Corner Detection  (Lec 7)
    3-c. SIFT Keypoint Detection  (Lec 7)
    반환: {'deskewed', 'harris_vis', 'sift_vis', 'angle'}
    """
    results = {}

    # 3-a. Hough Line Transform – 텍스트 라인 기울기 추출
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    angle = 0.0
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            a = np.degrees(theta) - 90   # 수평 기준 각도
            if abs(a) < 45:              # ±45° 이내만 사용
                angles.append(a)
        if angles:
            angle = float(np.median(angles))

    # 미세 각도에서는 오히려 과보정될 수 있어 임계값 이상일 때만 회전한다.
    h, w = gray.shape
    center = (w // 2, h // 2)
    min_deskew_deg = 1.5
    if abs(angle) >= min_deskew_deg:
        applied_rotation = angle
        M = cv2.getRotationMatrix2D(center, applied_rotation, 1.0)
        deskewed = cv2.warpAffine(image_bgr, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        deskew_applied = True
    else:
        applied_rotation = 0.0
        deskewed = image_bgr.copy()
        deskew_applied = False

    results['deskewed'] = deskewed
    results['angle'] = angle
    results['applied_rotation'] = applied_rotation
    results['deskew_applied'] = deskew_applied
    if deskew_applied:
        print(f"  → 검출 기울기: {angle:.2f}° / 보정 회전: {applied_rotation:.2f}°")
    else:
        print(f"  → 검출 기울기: {angle:.2f}° (|angle| < {min_deskew_deg:.1f}° 이므로 회전 생략)")

    # 3-b. Harris Corner Detection
    gray_f = np.float32(gray)
    harris = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)
    harris_norm = cv2.dilate(harris, None)
    harris_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    harris_vis[harris_norm > 0.01 * harris_norm.max()] = [0, 0, 255]
    results['harris_vis'] = harris_vis

    # 3-c. SIFT Keypoint Detection (또는 ORB 대체)
    try:
        sift = cv2.SIFT_create(nfeatures=300)
    except AttributeError:
        # OpenCV contrib 없는 경우 ORB 사용
        sift = cv2.ORB_create(nfeatures=300)
    kps, _ = sift.detectAndCompute(gray, None)
    sift_vis = cv2.drawKeypoints(
        image_bgr, kps, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    results['sift_vis'] = sift_vis
    print(f"  → 특징점 키포인트: {len(kps)}개")

    return results


# ─────────────────────────────────────────────────────────────
# STEP 4: K-means 색상 세그멘테이션  (Lec 6)
# ─────────────────────────────────────────────────────────────

def step4_kmeans_segment(image_bgr: np.ndarray, k: int = 3) -> dict:
    """
    K-means clustering으로 배경 / 텍스트 / 이미지 영역을 분리  (Lec 6)
    k=3: 배경(흰색), 텍스트(어두운 픽셀), 이미지/표 영역
    """
    Z = image_bgr.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                10, 1.0)
    _, labels, centers = cv2.kmeans(
        Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)

    # 레이블 → 대표 색상으로 재구성
    segmented = centers[labels.flatten()].reshape(image_bgr.shape)

    # 텍스트 레이블 (가장 어두운 클러스터)
    brightness = np.mean(centers, axis=1)
    text_label = int(np.argmin(brightness))
    text_mask = (labels.flatten() == text_label).reshape(
        image_bgr.shape[:2]).astype(np.uint8) * 255

    return {
        'segmented': segmented,
        'text_mask': text_mask,
        'k': k,
        'centers': centers
    }


# ─────────────────────────────────────────────────────────────
# 전체 파이프라인 실행
# ─────────────────────────────────────────────────────────────

def run_pipeline(image_path: str, output_dir: str = ".") -> None:
    """
    입력 이미지 한 장에 대해 Step 1~4를 순서대로 실행하고
    단계별 결과 이미지를 output_dir에 저장한다.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 입력 로드
    src = cv2.imread(image_path)
    if src is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")
    print(f"\n{'='*60}")
    print(f"입력 이미지: {image_path}  ({src.shape[1]}×{src.shape[0]})")
    print(f"{'='*60}")

    # ── Step 1 ──────────────────────────────────────────────
    print("\n[Step 1] 이미지 품질 개선 (Unsharp Masking + CLAHE)")
    s1 = step1_enhance(src)
    show_comparison(
        [src, s1['enhanced'], s1['enhanced_clahe']],
        ["원본", "Unsharp Masking", "CLAHE (LAB 색공간)"],
        str(out / "step1_enhance.png")
    )

    # ── Step 2 ──────────────────────────────────────────────
    print("\n[Step 2] 텍스트 영역 탐지 (Otsu + Morphology + CCL)")
    s2 = step2_detect_text_regions(s1['gray'])
    show_comparison(
        [s1['gray'], s2['binary'], s2['dilated'],
         s2['edges'], s2['ccl_vis']],
        ["Grayscale", "Otsu 이진화", "Morphological Dilation",
         "Canny Edge", f"CCL ({len(s2['bboxes'])}개 블록)"],
        str(out / "step2_text_detect.png")
    )

    # ── Step 3 ──────────────────────────────────────────────
    print("\n[Step 3] 기하학적 보정 (Hough Deskewing + Harris + SIFT)")
    s3 = step3_geometric_correction(s1['enhanced_clahe'], s1['gray'])
    deskew_title = (
        f"Deskewing (검출 {s3['angle']:.1f}° / 보정 {s3['applied_rotation']:.1f}°)"
        if s3['deskew_applied'] else
        f"Deskewing 생략 (검출 {s3['angle']:.1f}°)"
    )
    show_comparison(
        [src, s3['deskewed'], s3['harris_vis'], s3['sift_vis']],
        ["원본", deskew_title,
         "Harris Corner", "SIFT Keypoints"],
        str(out / "step3_geometry.png")
    )

    # ── Step 4 ──────────────────────────────────────────────
    print("\n[Step 4] K-means 세그멘테이션 (k=3)")
    s4 = step4_kmeans_segment(s3['deskewed'], k=3)
    show_comparison(
        [s3['deskewed'], s4['segmented'], s4['text_mask']],
        ["보정된 이미지", f"K-means (k={s4['k']})", "텍스트 마스크"],
        str(out / "step4_segmentation.png")
    )

    # ── 전체 요약 ───────────────────────────────────────────
    print("\n[요약] 전체 파이프라인 결과")
    show_comparison(
        [src, s1['enhanced_clahe'], s2['ccl_vis'],
         s3['deskewed'], s4['segmented'], s4['text_mask']],
        ["① 원본", "② 품질 개선", "③ 텍스트 블록 CCL",
         "④ Deskewing", "⑤ K-means", "⑥ 텍스트 마스크"],
        str(out / "pipeline_summary.png"),
        cols=3
    )
    print(f"\n✅ 완료! 결과 이미지는 '{output_dir}' 폴더에 저장되었습니다.")


# ─────────────────────────────────────────────────────────────
# 데모용 테스트 이미지 생성 (실제 문서 이미지가 없을 때)
# ─────────────────────────────────────────────────────────────

def create_demo_image(save_path: str = "demo_document.png") -> str:
    """
    기울어진 가짜 문서 이미지를 합성하여 저장.
    실제 문서 스캔 이미지가 있으면 그걸 사용하면 됩니다.
    """
    h, w = 960, 720
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 240  # 연한 배경

    # 데모 재현성 고정
    np.random.seed(42)

    # 노이즈 추가
    noise = np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)
    canvas = cv2.add(canvas, noise)

    # 텍스트 라인 그리기 (다양한 굵기)
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines_text = [
        "Document Image Preprocessing Pipeline",
        "Using Classical Computer Vision Algorithms",
        "",
        "Step 1: Unsharp Masking + CLAHE",
        "Step 2: Otsu Threshold + CCL",
        "Step 3: Hough Deskewing + SIFT",
        "Step 4: K-means Segmentation",
        "",
        "Lec 4: Frequency-domain Filtering",
        "Lec 5: Edge & Boundary Detection",
        "Lec 6: Image Segmentation",
        "Lec 7: Feature Detection (Harris, SIFT)",
    ]
    for i, txt in enumerate(lines_text):
        y = 80 + i * 52
        if txt == "":
            continue
        fs = 0.6 if i > 2 else 0.75
        thick = 1 if i > 2 else 2
        cv2.putText(canvas, txt, (50, y), font, fs, (30, 30, 30), thick, lineType=cv2.LINE_AA)

    # 수평선 (표 모양)
    for y in [300, 350, 400, 450]:
        cv2.line(canvas, (40, y), (560, y), (100, 100, 100), 1)
    cv2.line(canvas, (40, 300), (40, 450), (100, 100, 100), 1)
    cv2.line(canvas, (200, 300), (200, 450), (100, 100, 100), 1)
    cv2.line(canvas, (560, 300), (560, 450), (100, 100, 100), 1)

    # 기울임 적용 (-4도)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -4.0, 1.0)
    canvas = cv2.warpAffine(canvas, M, (w, h),
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(220, 220, 220))

    cv2.imwrite(save_path, canvas)
    print(f"  → 데모 이미지 생성: {save_path}  (기울기 -4°)")
    return save_path


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    OUTPUT_DIR = "results"

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"사용자 제공 이미지: {image_path}")
    else:
        # 실제 테스트 이미지 사용
        image_path = "test_img.jpg"
        if not os.path.exists(image_path):
            print("test_img.jpg가 없으므로 데모 이미지를 생성합니다.")
            image_path = create_demo_image("demo_document.png")
        else:
            print(f"실제 테스트 이미지 사용: {image_path}")

    run_pipeline(image_path, output_dir=OUTPUT_DIR)
