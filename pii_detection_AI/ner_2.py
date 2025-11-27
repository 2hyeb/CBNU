import requests, json, traceback, re
from typing import List, Dict

SENSITIVE_PROMPT = """
당신은 주어진 텍스트에서 '키'와 '몸무게' 정보를 정확히 추출하는 AI 어시스턴트입니다.

[추출 규칙]
- 반드시 키와 몸무게 정보만 추출합니다.
- 숫자와 함께 사용된 다양한 단위(cm, kg 등)와 구분자(, / - 띄어쓰기 등)를 모두 처리해야 합니다.
- 예시:
  - "170cm이고 70kg인데" → "170cm, 70kg"
  - "162/48 인데" → "162, 48"
  - "키 몸무게 160/42입니다." → "160, 42"

[출력 규칙]
- 아래 JSON 형식으로만 출력
- 탐지 항목이 없으면 빈 문자열("") 사용
- 설명이나 추가 텍스트 없이 JSON만 출력
{
    "sensitive": "신체정보1, 신체정보2, ..."
}
"""

class LlamaVerifier:
    def __init__(self, ollama_url, ollama_model):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    def _make_payload(self, system_prompt, text, detected_list, key):
        inst = system_prompt.strip() + "\n\n"
        inst += f"원본 텍스트:\n{text}\n"
        if detected_list:
            inst += f"\n1차 탐지된 {key}: {', '.join(detected_list)}\n"
        inst += "\n### Response:\n"
        return {
            "model": self.ollama_model,
            "prompt": inst,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.5,
                "top_k": 20,
                "num_predict": 512
            }
        }

    def _call_ollama(self, payload: dict) -> str:
        try:
            r = requests.post(self.ollama_url, json=payload, timeout=120)
            if r.status_code == 200:
                # Ollama 응답에서 "response" 필드의 텍스트 추출
                return r.json().get("response", "")
            else:
                print(f"API 호출 실패: {r.status_code} - {r.text}")
                return ""
        except Exception as e:
            print(f"오류 발생: {e}")
            traceback.print_exc()
            return ""

    def verify_all(
        self,
        text: str,
        use_names: bool,
        use_addresses: bool,
        use_sensitive: bool,
        detected_names: List[str],
        detected_addresses: List[Dict[str, any]],
        process_prompts: dict | None = None,
    ) -> Dict[str, List[str]]:
        """
        선택된 항목별로 전용 프롬프트 블록을 합친 뒤
        한 번만 Ollama API를 호출하여 JSON 결과를 받아옵니다.
        """
        pp = process_prompts or {}

        name_prompt = pp.get("names") if isinstance(pp, dict) else None
        address_prompt = pp.get("addresses") if isinstance(pp, dict) else None
        sensitive_prompt = pp.get("sensitive") if isinstance(pp, dict) else None

        if isinstance(name_prompt, str):
            name_prompt = name_prompt.strip()
        if isinstance(address_prompt, str):
            address_prompt = address_prompt.strip()
        if isinstance(sensitive_prompt, str):
            sensitive_prompt = sensitive_prompt.strip()

        # 1) 프롬프트 블록 구성 
        blocks = []
        if use_names:
            blocks.append(
                (name_prompt if name_prompt else NAME_PROMPT).strip()
                + "\n\n1차 탐지된 names: "
                + (", ".join(detected_names) if detected_names else '""')
                + "\n※ 동일한 이름이 여러 번 등장하더라도 중복을 포함해서 모두 출력하세요."
            )
        if use_addresses:
            address_strings = []
            if detected_addresses:
                for a in detected_addresses:
                    if isinstance(a, dict):
                        address_strings.append(str(a.get("address") or a.get("value") or a))
                    else:
                        address_strings.append(str(a))
            blocks.append(
                (address_prompt if address_prompt else ADDRESS_PROMPT).strip()
                + "\n\n1차 탐지된 addresses: "
                + (", ".join(address_strings) if address_strings else '""')
                + "\n※ 동일한 주소가 여러 번 등장하더라도 중복을 포함해서 모두 출력하세요."
            )
        if use_sensitive:
            blocks.append((sensitive_prompt if sensitive_prompt else SENSITIVE_PROMPT).strip())

        # 2) 시스템 프롬프트 조합
        system_prompt = "\n\n".join([
            *blocks,
            "응답에는 JSON 키(names, addresses, sensitive)만 포함하고, 다른 설명은 생략하세요."
            "동일한 항목이 여러 번 등장하더라도 중복 포함 모두 출력하세요."
        ])

        # 3) 전체 프롬프트 생성
        prompt = (
            system_prompt
            + "\n\n원본 텍스트:\n"
            + text
            + "\n### Response:\n"
        )
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.5,
                "top_k": 20,
                "num_predict": 512
            }
        }

        # 4) 단일 API 호출
        raw = self._call_ollama(payload)

        # 5) JSON 블록 파싱
        json_str = raw
        if "```json" in raw:
            json_str = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            json_str = raw.split("```")[1].split("```")[0].strip()

        json_str = json_str.replace("'", '"')
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*\]', ']', json_str)

        try:
            data = json.loads(json_str)
        except Exception:
            data = {}

        # 6) 결과 리스트화
        return {
            "names": [
                s.strip() for s in data.get("names", "").split(",") if s.strip()
            ] if use_names else [],
            "addresses": [
                s.strip() for s in data.get("addresses", "").split(",") if s.strip()
            ] if use_addresses else [],
            "sensitive": [
                s.strip() for s in data.get("sensitive", "").split(",") if s.strip()
            ] if use_sensitive else [],
        }