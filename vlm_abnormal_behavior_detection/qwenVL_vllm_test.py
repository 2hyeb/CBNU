import base64
from vllm import LLM

# 1) 모델 로드
llm = LLM(
    model="Qwen/Qwen2.5-VL-32B-Instruct",
    trust_remote_code=True,
    dtype="float16",
    max_model_len=8192,
    gpu_memory_utilization=0.95,
    enforce_eager=True,
)

# 2) JPG 파일을 base64 + data URL로 변환
def load_image_as_data_url(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{b64}"

img = load_image_as_data_url("./test_image/KakaoTalk_20251201_134229604_01.jpg")

# 3) 메시지 포맷 (vLLM 멀티모달 공식)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "이 이미지 설명해줘."},
            {
                "type": "image_url",
                "image_url": {"url": img}   # ★ 중요: 여기로 넣어야 정상 동작
            }
        ]
    }
]

# 4) 모델 추론
out = llm.chat(messages)
print(out[0].outputs[0].text)