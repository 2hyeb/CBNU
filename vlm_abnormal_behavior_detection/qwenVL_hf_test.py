from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch

model_id = "Qwen/Qwen2.5-VL-32B-Instruct"

processor = AutoProcessor.from_pretrained(
    model_id, 
    trust_remote_code=True,
    use_fast=False                   
)

# 권장 모델 클래스: Vision2Seq → ImageTextToText
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("모델 로드 완료")

# ------------ 이미지 로드 ------------
image_path = "./test_image/KakaoTalk_20251201_134229604_02.jpg"
image = Image.open(image_path).convert("RGB")

query = """ 
[역할 정의]
당신은 AI 기반 CCTV 영상 분석 시스템입니다.  
입력된 이미지 또는 영상 프레임에서 아래 행동들을 탐지하고, 각 행동의 여부와 근거를 명확히 판단하세요.

[탐지 대상 행동 및 규칙]

1. 흡연(Smoking) 탐지 규칙
   - 손에 담배 또는 전자담배 형태의 물체를 들고 있는 경우
   - 입에 담배를 가져다 대거나, 들고-입에 대는 반복 제스처가 보이는 경우
   - 흡연 고유의 손 모양(엄지·검지 사이로 들고 있는 자세 등) 패턴이 보이는 경우

2. 음주(Drinking Alcohol) 탐지 규칙
   - 주변에 술병(소주병, 맥주병, 캔맥주, 막걸리병 등)이 식별되는 경우
   - 사람이 병/캔/컵을 입에 가져다 대거나 들고 마시는 행동을 수행하는 경우
   - 테이블 등 주변 환경에서 술자리를 구성하는 패턴이 존재하는 경우

3. 군집(Gathering) 탐지 규칙
   - 동일 공간 안에 5명 이상의 사람이 모여 있는 경우
   - 한 장소에서 가까운 거리 간격으로 사람이 밀집한 패턴이 확인되는 경우

[출력 형식]
결과를 아래 JSON 형태로 명확하게 출력하세요.

- "smoking_detected": true/false  
- "smoking_evidence": 흡연으로 판단한 근거 (없으면 "none")

- "drinking_detected": true/false  
- "drinking_evidence": 음주로 판단한 근거 (없으면 "none")

- "gathering_detected": true/false  
- "people_count": 숫자  
- "gathering_evidence": 군집 판단 근거

- "overall_summary": 한 문단으로 핵심만 요약

[지시사항]
- 판단이 불확실한 경우 "불확실함"이라고 명시하세요.  
- 사물과 사람의 위치, 행동 근거를 가능한 한 구체적으로 설명하세요.  
- 없는 행동을 있다고 추정하지 않습니다.
- 이미지 내 정보만 기반으로 판단하세요.

[최종 요청]  
이 이미지 내에서 흡연, 음주, 군집 행동이 존재하는지 탐지하고 위 출력 형식으로 분석하세요.
"""

# ------------ 메시지 구성 ------------
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": query}
        ]
    }
]

# ------------ 템플릿 문자열 생성 (string) ------------
prompt_text = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False                  
)

# ------------ tokenize (input_ids 생성) ------------
inputs = processor(
    text=prompt_text,
    images=[image],
    return_tensors="pt"
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

# ------------ Generate ------------
output_ids = model.generate(
    **inputs,
    max_new_tokens=256
)

# ------------ Decode ------------
result = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
print("\n결과:\n", result)