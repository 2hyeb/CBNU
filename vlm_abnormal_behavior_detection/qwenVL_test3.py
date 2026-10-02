from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import torch
import json
import time
import ast
import os
import re
from roi_test import load_roi_config

def extract_assistant_response(text):
    if "assistant" in text:
        text = text.split("assistant")[-1].strip()
    for key in ["{", "["]:
        if key in text:
            text = text[text.index(key):]
            break
    return text.strip()

def extract_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    return text[start:end+1]

def is_collapse(output_text):
    pure = output_text.replace("!", "").strip()
    if pure == "":
        return True
    if output_text.count("!") > 10:
        return True
    return False

def extract_bool_after_key(text, key):
    # key 이후에 처음 등장하는 true/false 를 대소문자/따옴표/구두점과 무관하게 추출
    pattern = rf"{re.escape(key)}[^a-zA-Z]*(true|false)"
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).lower() == "true"

# 모델 로드
t_load_start = time.time()

model_id = "Qwen/Qwen2.5-VL-72B-Instruct"

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    use_fast=False,
)

# 8bit 양자화 설정 (BitsAndBytesConfig 사용)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

# 멀티 GPU에 자동 분산 + 8bit 양자화 로딩
max_mem = {i: "22GiB" for i in range(torch.cuda.device_count())}

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory=max_mem,
    trust_remote_code=True,
)

t_load_end = time.time()
print(f"\n모델 로드 완료 (소요 시간: {t_load_end - t_load_start:.2f}초)\n")

# 이미지 폴더 설정
IMAGE_DIR = "./test_data"
valid_ext = [".jpg", ".jpeg", ".png"]

# ROI 설정 파일 (roi_test.py 에서 생성한 좌표 파일 사용)
ROI_CONFIG_PATH = "roi_config.txt"
roi_points = load_roi_config(ROI_CONFIG_PATH) if os.path.exists(ROI_CONFIG_PATH) else None

query_prompt = """ 
[역할 정의]
당신은 AI 기반 CCTV 영상 분석 시스템입니다.
입력된 단일 이미지 또는 영상의 개별 프레임을 기반으로 흡연, 음주, 군집 여부를 판단합니다.

[탐지 규칙]  
아래 행동들은 이미지에서 시각적으로 확실히 관찰 가능한 기준만 사용하여 판별합니다.
탐지 기준에 하나라도 부합하면 true를 반환합니다.

1. 흡연 Smoking 탐지 기준
- 손에 길쭉한 막대형 흡연 기기가 명확히 보이면 흡연으로 판단한다. 흡연기기는 일반 연초, 궐련형 전자담배, 액상형 전자담배 등.
- 흡연 기기가 얼굴 중심에서 근접한 위치 입/코 주변에 있으면 흡연으로 판단한다.
- 손가락이 흡연 특유의 그립 형태인 엄지 검지로 집거나 두 손가락 사이에 끼운 형태를 보이면 흡연으로 판단한다.
- 입 주변에 담배 연기 또는 전자담배의 수증기 형태의 연기가 존재하면 흡연으로 판단한다.
- 손가락으로 흡연 기기를 집은 채 얼굴 높이에서 유지하는 사람이 있으면 흡연으로 판단한다.
1-1.비흡연 조건
- 얼굴에 손이 가까이 있으나 손에 물체가 없는 경우는 흡연이 아님.
- 휴대폰을 입 또는 턱 근처에 가져가는 경우는 흡연이 아님.
- 컵, 빨대, 립스틱 등 흡연기기와 형태가 다른 물체는 흡연이 아님.
- 두꺼운 옷을 입은 사람이 있는 추운 환경에서 발생하는 입김으로 인해 연기로 오인될 수 있는 경우는 흡연이 아님.

2. 음주 Drinking Alcohol 탐지 기준
- 프레임에 길쭉한 병 형태 또는 짧은 원통형 캔 형태의 물체가 보이면 음주로 판단한다.
- 사람 손에 병이나 캔처럼 보이는 물체가 들려 있으면 음주로 판단한다.
- 초록색, 갈색, 은색, 파란색의 병이나 캔이 사람 주변에 보이면 음주로 판단한다.
- 라벨이나 글씨가 보이지 않아도 전체 형태가 병이나 캔과 일치하면 음주로 판단한다.
- 사람 근처에 병이나 캔 형태의 물체가 존재하면 음주로 판단한다.
- 이미지 내에 유리 재질의 병이 존재하며, 병 상단에 명확한 목(neck) 구조가 보이면 음주로 판단한다.
- 병 전체가 길쭉하고 슬림하며, 상단이 좁고 하단이 상대적으로 넓은 전형적인 주류 병 형태라면 음주로 판단한다.
- 갈색 또는 초록색과 같은 어두운 색상의 유리병이 사람 손에 들려 있거나 사람 가까이에 위치하면 음주로 판단한다.
- 짧고 굵은 원통형 금속 캔 형태의 물체가 사람 주변에 존재하면 음주로 판단한다.
- 라벨, 글씨, 브랜드가 보이지 않더라도 전체 형태가 주류 전용 용기와 일치하면 음주로 판단한다.

3. 군집 Gathering 탐지 기준  
- 동일 공간에 5명 이상이 명확히 존재하며 사람들 사이 거리가 상대적으로 가깝고 밀집된 형태라면 군집으로 판단한다.
3-1. 비군집 조건
- 이미지 내에 5명 이상이 존재하나 서로 1m 이상씩 흩어져 있는 경우는 군집이 아님.

[지시사항]
- 탐지 행동 조건과 비탐지 조건 명확히 구분.
- evidence는 한 줄로 간결하게 작성.
- 아래 JSON 형식에 따라 탐지 결과 출력.

[출력 형식]
- 결과를 아래 JSON 파싱 가능한 형태로 출력.
{
"smoking_detected": true or false,
"smoking_evidence": "string",
    
"drinking_detected": true or false,
"drinking_evidence": "string",
    
"gathering_detected": true or false,
"gathering_evidence": "string"
}
    
[최종 요청]
입력된 이미지에서 흡연, 음주, 군집 행동이 존재하는지 탐지하고 위 JSON 형식으로 출력하세요.
"""

files = sorted([f for f in os.listdir(IMAGE_DIR) if os.path.splitext(f)[1].lower() in valid_ext])

print(f"이미지 분석 시작 — 총 {len(files)}장\n")

results = []

# collapse 방지용 generate 함수
def safe_generate(model, processor, inputs, image):
    # collapse 발생 시 이미지 축소 후 1회 재시도
    # 1차 시도
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
        do_sample=False
    )

    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    clean = extract_assistant_response(text)

    # collapse 확인
    if not is_collapse(clean):
        return clean

    print("collapse 감지됨 — 이미지 축소 후 재시도")

    # 2차 시도: 이미지 축소 후 다시 processor 입력 생성
    small_img = image.resize((1024, 1024))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": small_img},
                {"type": "text", "text": query_prompt}
            ]
        }
    ]

    prompt_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    new_inputs = processor(
        text=prompt_text,
        images=[small_img],
        return_tensors="pt"
    )
    new_inputs = {k: v.to(model.device) for k, v in new_inputs.items()}

    # 재시도 generate
    output_ids = model.generate(
        **new_inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
        do_sample=False
    )

    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    clean = extract_assistant_response(text)

    # 그래도 collapse면 실패 처리
    if is_collapse(clean):
        print("재시도 후에도 collapse — 탐지 실패 처리")
        return None

    return clean

# 이미지 반복 분석
for idx, filename in enumerate(files, 1):

    path = os.path.join(IMAGE_DIR, filename)

    # 한 이미지 분석 시작 시각
    t_infer_start = time.time()

    image = Image.open(path).convert("RGB")

    # 파일명이 숫자라면 해당 번호를 기준으로 1~100번에만 ROI 마스킹 적용
    image_num = None
    base_name, _ = os.path.splitext(filename)
    try:
        image_num = int(base_name)
    except ValueError:
        image_num = None

    if roi_points and image_num is not None and 1 <= image_num <= 100:
        mask = Image.new("L", image.size, 0)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.polygon(roi_points, fill=255)

        black_bg = Image.new("RGB", image.size, (0, 0, 0))
        image = Image.composite(image, black_bg, mask)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query_prompt}
            ]
        }
    ]

    prompt_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    inputs = processor(
        text=prompt_text,
        images=[image],
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # collapse 방지 + 자동 재시도 적용된 generate
    clean_output = safe_generate(model, processor, inputs, image)

    t_infer_end = time.time()

    print(f"[{idx}/{len(files)}] {filename} 결과:\n{clean_output}\n")
    print(f"→ {filename} 처리 시간: {t_infer_end - t_infer_start:.2f}초")

    # 실패(=collapse 지속) 처리
    if clean_output is None:
        results.append({
            "image_name": filename,
            "smoking_pred": 0,
            "drinking_pred": 0,
            "gathering_pred": 0
        })
        continue

    # JSON 추출
    json_str = extract_json(clean_output)
    if json_str is None:
        print(f"JSON 블록 없음 — {filename} 탐지 실패")
        # JSON 블록 자체가 없을 때도 키워드 기반 fallback 시도
        smoking_flag = extract_bool_after_key(clean_output, "smoking_detected")
        drinking_flag = extract_bool_after_key(clean_output, "drinking_detected")
        gathering_flag = extract_bool_after_key(clean_output, "gathering_detected")

        if any(f is not None for f in (smoking_flag, drinking_flag, gathering_flag)):
            print(f"키워드 기반 fallback 파싱 사용 — {filename}")
            smoking_pred = int(bool(smoking_flag)) if smoking_flag is not None else 0
            drinking_pred = int(bool(drinking_flag)) if drinking_flag is not None else 0
            gathering_pred = int(bool(gathering_flag)) if gathering_flag is not None else 0

            results.append({
                "image_name": filename,
                "smoking_pred": smoking_pred,
                "drinking_pred": drinking_pred,
                "gathering_pred": gathering_pred
            })
        else:
            results.append({
                "image_name": filename,
                "smoking_pred": 0,
                "drinking_pred": 0,
                "gathering_pred": 0
            })
        continue
    # 1) 흔한 실수 보정: 마지막 요소 뒤 콤마 제거
    json_str = re.sub(r',\s*}', '}', json_str)
    try:
        # 2) 표준 JSON 먼저 시도
        parsed = json.loads(json_str)
        print(f"JSON 파싱 성공(JSON) — {filename}")
    except Exception as e:
        print(f"표준 JSON 파싱 실패 — {filename}, ast.literal_eval로 재시도")
        # 3) Python literal 형태로 변환
        py_like = re.sub(r'\btrue\b', 'True', json_str)
        py_like = re.sub(r'\bfalse\b', 'False', py_like)
        py_like = re.sub(r'\bnull\b', 'None', py_like)
        try:
            parsed = ast.literal_eval(py_like)
            print(f"JSON 파싱 성공(ast.literal_eval) — {filename}")
        except Exception as e2:
            print(f"JSON 파싱 완전 실패 — {filename} 탐지 실패")
            print("1차 에러:", e)
            print("2차 에러:", e2)
            print("문자열 repr:", repr(json_str))

            # 최후 수단: 전체 텍스트에서 키워드 기반 true/false 추출
            smoking_flag = extract_bool_after_key(clean_output, "smoking_detected")
            drinking_flag = extract_bool_after_key(clean_output, "drinking_detected")
            gathering_flag = extract_bool_after_key(clean_output, "gathering_detected")

            if any(f is not None for f in (smoking_flag, drinking_flag, gathering_flag)):
                print(f"키워드 기반 fallback 파싱 사용 — {filename}")
                smoking_pred = int(bool(smoking_flag)) if smoking_flag is not None else 0
                drinking_pred = int(bool(drinking_flag)) if drinking_flag is not None else 0
                gathering_pred = int(bool(gathering_flag)) if gathering_flag is not None else 0

                results.append({
                    "image_name": filename,
                    "smoking_pred": smoking_pred,
                    "drinking_pred": drinking_pred,
                    "gathering_pred": gathering_pred
                })
                continue
            else:
                results.append({
                    "image_name": filename,
                    "smoking_pred": 0,
                    "drinking_pred": 0,
                    "gathering_pred": 0
                })
                continue

    smoking_pred = int(parsed.get("smoking_detected", False))
    drinking_pred = int(parsed.get("drinking_detected", False))
    gathering_pred = int(parsed.get("gathering_detected", False))

    results.append({
        "image_name": filename,
        "smoking_pred": smoking_pred,
        "drinking_pred": drinking_pred,
        "gathering_pred": gathering_pred
    })

    # 탐지 시 이미지에 표시
    if smoking_pred or drinking_pred or gathering_pred:

        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 100)
        except:
            font = ImageFont.load_default()

        x, y = 20, 30
        line_spacing = 110

        if smoking_pred:
            draw.text((x, y), "SMOKING DETECTED", fill="red", font=font)
            y += line_spacing

        if drinking_pred:
            draw.text((x, y), "DRINKING DETECTED", fill="red", font=font)
            y += line_spacing

        if gathering_pred:
            draw.text((x, y), "GATHERING DETECTED", fill="red", font=font)

        os.makedirs("./annotated_image/annotated_images", exist_ok=True)
        save_path = os.path.join("./annotated_image/annotated_images", filename)
        annotated_image.save(save_path)

# 결과 저장
df_pred = pd.DataFrame(results)
df_pred.to_excel("model_output9.xlsx", index=False)

print("\nmodel_output9.xlsx 저장 완료!")

# https://github.com/QwenLM/Qwen3-VL/issues/810 -> VLM 답변 붕괴 현상 이슈 (같은 모델, 8개의 GPU에 나눠서 실행)