"""
=============================================================
 OCR 성능 비교 스크립트
 전처리 전 vs 전처리 후 OCR 인식률 비교
 산업 컴퓨터비전 실제 - 중간 프로젝트
=============================================================
"""

import cv2
import numpy as np
import json
import os
import easyocr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from doc_cv_pipeline import step1_enhance, step2_detect_text_regions, step3_geometric_correction, step4_kmeans_segment

# ─────────────────────────────────────────────────────────────
# 한글 폰트 설정
# ─────────────────────────────────────────────────────────────
KOREAN_FONT_PATH = None
for font_name in ['malgunbd.ttf', 'gulim.ttc', 'batang.ttc', 'malgun.ttf']:
    font_path = os.path.join('C:\\Windows\\Fonts', font_name)
    if os.path.exists(font_path):
        KOREAN_FONT_PATH = font_path
        break
if KOREAN_FONT_PATH is None:
    KOREAN_FONT_PATH = 'C:\\Windows\\Fonts\\malgun.ttf'


def load_ground_truth(json_path):
    """test_img_label.json에서 정답 텍스트 추출"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gt_texts = []
    for bbox in data.get('Bbox', []):
        text = bbox.get('data', '').strip()
        if text:
            gt_texts.append(text)
    
    return gt_texts


def run_ocr(image, reader):
    """EasyOCR로 이미지에서 텍스트 추출"""
    results = reader.readtext(image)
    detected_texts = []
    for (bbox, text, confidence) in results:
        text = text.strip()
        if text:
            detected_texts.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox
            })
    return detected_texts


def calculate_accuracy(gt_texts, detected_texts):
    """정답과 OCR 결과 비교하여 정확도 계산"""
    detected_strs = [d['text'] for d in detected_texts]
    
    # 1. 단어 단위 정확도 (정답 중 OCR로 찾은 비율)
    matched = 0
    partial_matched = 0
    for gt in gt_texts:
        # 정확 매칭
        if gt in detected_strs:
            matched += 1
        else:
            # 부분 매칭 (정답이 OCR 결과에 포함되어 있는 경우)
            for det in detected_strs:
                if gt in det or det in gt:
                    partial_matched += 1
                    break
    
    exact_accuracy = matched / len(gt_texts) * 100 if gt_texts else 0
    partial_accuracy = (matched + partial_matched) / len(gt_texts) * 100 if gt_texts else 0
    
    # 2. 문자 단위 정확도
    gt_all_chars = ''.join(gt_texts)
    det_all_chars = ''.join(detected_strs)
    
    # 공통 문자 수 계산
    gt_char_set = {}
    for c in gt_all_chars:
        gt_char_set[c] = gt_char_set.get(c, 0) + 1
    
    det_char_set = {}
    for c in det_all_chars:
        det_char_set[c] = det_char_set.get(c, 0) + 1
    
    common_chars = 0
    for c, count in gt_char_set.items():
        if c in det_char_set:
            common_chars += min(count, det_char_set[c])
    
    char_precision = common_chars / len(det_all_chars) * 100 if det_all_chars else 0
    char_recall = common_chars / len(gt_all_chars) * 100 if gt_all_chars else 0
    char_f1 = 2 * char_precision * char_recall / (char_precision + char_recall) if (char_precision + char_recall) > 0 else 0
    
    # 3. 평균 신뢰도
    avg_confidence = np.mean([d['confidence'] for d in detected_texts]) * 100 if detected_texts else 0
    
    return {
        'exact_word_accuracy': round(exact_accuracy, 2),
        'partial_word_accuracy': round(partial_accuracy, 2),
        'char_precision': round(char_precision, 2),
        'char_recall': round(char_recall, 2),
        'char_f1': round(char_f1, 2),
        'avg_confidence': round(avg_confidence, 2),
        'total_gt_words': len(gt_texts),
        'total_detected_words': len(detected_strs),
        'exact_matched': matched,
        'partial_matched': partial_matched
    }


def apply_preprocessing(image_bgr):
    """전체 파이프라인 적용 (Step 1~4)"""
    # Step 1: 이미지 품질 개선
    s1 = step1_enhance(image_bgr)
    
    # Step 2: 텍스트 영역 탐지
    s2 = step2_detect_text_regions(s1['gray'])
    
    # Step 3: 기하학적 보정
    s3 = step3_geometric_correction(s1['enhanced_clahe'], s1['gray'])
    
    # Step 4: K-means 세그멘테이션
    s4 = step4_kmeans_segment(s3['deskewed'], k=3)
    
    return s3['deskewed'], s4, s1, s2, s3


def create_comparison_chart(before_metrics, after_metrics, output_path):
    """전처리 전/후 OCR 성능 비교 차트 생성"""
    categories = [
        'char_recall',
        'char_precision', 
        'char_f1',
        'exact_word_accuracy',
        'partial_word_accuracy',
        'avg_confidence'
    ]
    labels_kr = [
        'Char Recall',
        'Char Precision',
        'Char F1',
        'Exact Word\nAccuracy',
        'Partial Word\nAccuracy',
        'Avg Confidence'
    ]
    
    before_vals = [before_metrics[c] for c in categories]
    after_vals = [after_metrics[c] for c in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0e1c2e')
    ax.set_facecolor('#0e1c2e')
    
    bars1 = ax.bar(x - width/2, before_vals, width, label='Before (Raw)', 
                    color='#FF6B6B', alpha=0.9, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, after_vals, width, label='After (Preprocessed)',
                    color='#4FC3F7', alpha=0.9, edgecolor='white', linewidth=0.5)
    
    ax.set_ylabel('Score (%)', fontsize=12, color='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_kr, fontsize=10, color='#b0c4de')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11, facecolor='#1E2761', edgecolor='#4FC3F7', labelcolor='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('#4FC3F7')
    ax.spines['left'].set_color('#4FC3F7')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 값 라벨 추가
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color='#FF6B6B', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color='#4FC3F7', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    # PIL로 한글 제목 추가
    try:
        pil_img = Image.open(output_path).convert('RGB')
        draw = ImageDraw.Draw(pil_img)
        try:
            font_title = ImageFont.truetype(KOREAN_FONT_PATH, size=30)
        except:
            font_title = ImageFont.load_default()
        
        img_w, img_h = pil_img.size
        title_text = "OCR Performance : Before vs After Preprocessing"
        try:
            bbox = font_title.getbbox(title_text)
            tw = bbox[2] - bbox[0]
        except:
            tw = len(title_text) * 15
        
        tx = (img_w - tw) // 2
        # 상단에 타이틀 배경 + 텍스트
        draw.rectangle([0, 0, img_w, 45], fill=(14, 28, 46))
        draw.text((tx, 8), title_text, fill=(79, 195, 247), font=font_title)
        pil_img.save(output_path)
    except Exception as e:
        print(f"  Warning: PIL title failed: {e}")
    
    print(f"  -> Chart saved: {output_path}")


def main():
    print("=" * 60)
    print("  OCR Performance Comparison")
    print("  Before vs After Preprocessing Pipeline")
    print("=" * 60)
    
    # 파일 경로
    image_path = "test_img.jpg"
    label_path = "test_img_label.json"
    output_dir = "results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 정답 데이터 로드
    print("\n[1] Loading ground truth...")
    gt_texts = load_ground_truth(label_path)
    print(f"    Ground truth words: {len(gt_texts)}")
    print(f"    Sample: {gt_texts[:5]}...")
    
    # 2. EasyOCR 리더 초기화
    print("\n[2] Initializing EasyOCR (Korean)...")
    reader = easyocr.Reader(['ko'], gpu=False)
    print("    EasyOCR ready.")
    
    # 3. 원본 이미지 OCR (전처리 전)
    print("\n[3] Running OCR on ORIGINAL image...")
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error: Cannot read {image_path}")
        return
    
    before_results = run_ocr(original_img, reader)
    before_metrics = calculate_accuracy(gt_texts, before_results)
    print(f"    Detected words: {before_metrics['total_detected_words']}")
    print(f"    Exact match: {before_metrics['exact_matched']}/{before_metrics['total_gt_words']}")
    print(f"    Char Recall: {before_metrics['char_recall']}%")
    print(f"    Char F1: {before_metrics['char_f1']}%")
    print(f"    Avg Confidence: {before_metrics['avg_confidence']}%")
    
    # 4. 전처리 파이프라인 적용
    print("\n[4] Applying preprocessing pipeline...")
    preprocessed_img, s4, s1, s2, s3 = apply_preprocessing(original_img)
    
    # 5. 전처리 후 이미지 OCR
    print("\n[5] Running OCR on PREPROCESSED image...")
    after_results = run_ocr(preprocessed_img, reader)
    after_metrics = calculate_accuracy(gt_texts, after_results)
    print(f"    Detected words: {after_metrics['total_detected_words']}")
    print(f"    Exact match: {after_metrics['exact_matched']}/{after_metrics['total_gt_words']}")
    print(f"    Char Recall: {after_metrics['char_recall']}%")
    print(f"    Char F1: {after_metrics['char_f1']}%")
    print(f"    Avg Confidence: {after_metrics['avg_confidence']}%")
    
    # 6. 비교 차트 생성
    print("\n[6] Creating comparison chart...")
    chart_path = os.path.join(output_dir, "ocr_comparison.png")
    create_comparison_chart(before_metrics, after_metrics, chart_path)
    
    # 7. 결과 JSON 저장
    results_data = {
        'before_preprocessing': before_metrics,
        'after_preprocessing': after_metrics,
        'improvement': {
            'char_recall_gain': round(after_metrics['char_recall'] - before_metrics['char_recall'], 2),
            'char_f1_gain': round(after_metrics['char_f1'] - before_metrics['char_f1'], 2),
            'exact_word_gain': round(after_metrics['exact_word_accuracy'] - before_metrics['exact_word_accuracy'], 2),
            'confidence_gain': round(after_metrics['avg_confidence'] - before_metrics['avg_confidence'], 2),
        },
        'pipeline_stats': {
            'deskew_angle': float(s3.get('angle', 0)),
            'ccl_blocks': len(s2.get('bboxes', [])),
            'kmeans_k': s4.get('k', 3),
        }
    }
    
    results_json_path = os.path.join(output_dir, "ocr_results.json")
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    print(f"\n[7] Results saved: {results_json_path}")
    
    # 8. 결과 요약 출력
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Before':>10} {'After':>10} {'Gain':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Char Recall (%)':<25} {before_metrics['char_recall']:>10.1f} {after_metrics['char_recall']:>10.1f} {results_data['improvement']['char_recall_gain']:>+10.1f}")
    print(f"  {'Char F1 (%)':<25} {before_metrics['char_f1']:>10.1f} {after_metrics['char_f1']:>10.1f} {results_data['improvement']['char_f1_gain']:>+10.1f}")
    print(f"  {'Exact Word Acc (%)':<25} {before_metrics['exact_word_accuracy']:>10.1f} {after_metrics['exact_word_accuracy']:>10.1f} {results_data['improvement']['exact_word_gain']:>+10.1f}")
    print(f"  {'Avg Confidence (%)':<25} {before_metrics['avg_confidence']:>10.1f} {after_metrics['avg_confidence']:>10.1f} {results_data['improvement']['confidence_gain']:>+10.1f}")
    print(f"  {'Deskew Angle':<25} {results_data['pipeline_stats']['deskew_angle']:>10.1f}")
    print(f"  {'CCL Blocks':<25} {results_data['pipeline_stats']['ccl_blocks']:>10d}")
    print("=" * 60)
    print("  Done!")


if __name__ == "__main__":
    main()
