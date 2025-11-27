import re
from typing import List, Dict, Tuple

class RegexDetector:
    def __init__(self):
        # 시/도 패턴
        self.sido_pattern = "서울특별시|서울|서울시|부산광역시|부산|대구광역시|대구|인천광역시|인천|광주광역시|광주|대전광역시|대전|울산광역시|울산|세종특별자치시|세종|경기도|경기|강원도|강원|충청북도|충북|충청남도|충남|전라북도|전북|전라남도|전남|경상북도|경북|경상남도|경남|제주특별자치도|제주|제주도"
        
        # 시군구 패턴
        self.sigungu_pattern = r"[가-힣]+(?:시|군|구)"
        # 읍면동 패턴
        self.eup_myeon_dong_pattern = r"[가-힣]+(?:읍|면|동)"
        # 법정리명 패턴 (읍면 지역에서 사용)
        self.legal_ri_pattern = r"[가-힣]+리"
        # 도로명 패턴
        self.road_name_pattern = r"[가-힣0-9]+(?:로|길|대로)"
        # 건물번호 패턴 (본번-부번 또는 본번만)
        self.building_number_pattern = r"\d+(?:-\d+)?(?:번길)?"
        # 지번 패턴 (산 + 본번-부번 또는 본번만)
        self.jibun_pattern = r"(?:산\s*)?\d+(?:-\d+)?"
        # 상세주소 패턴 (선택적)
        self.detail_pattern = r"(?:\s*[가-힣0-9\s,\-()]+)?"

        # 캐시된 정규식 객체
        self._unified_address_regex = self._compile_unified_address_regex()
        self._tracking_regex = re.compile(
            r'운송장번호|운송장|송장번호|(?:CJ|우체국|한진|롯데|로젠|경동|합동|CJ대한통운|대한통운|호남|천일|대신|건영|일양)(?:\s*택배\b)?',
            re.IGNORECASE
        )
        self._account_keywords_regex = re.compile(
            r'(?:입금|환불)?\s*계좌|계좌번호',
            re.IGNORECASE
        )
        self._bank_name_patterns = [
            re.compile(rf'{re.escape(k)}(?:\s*(?:은행|뱅크))?', re.IGNORECASE)
            for k in ['신한', '국민', '하나', '우리', '기업', '농협', '수협', '씨티', '카카오', '토스', '케이', '제일', 'SC', '산업', '한국', '신협', '우체국', '새마을', '대구', '부산', '경남', '광주', '전북', '제주', '산림', '수출입', 'KDB', 'KB', 'IBK', 'NH']
        ]
        
    def get_road_address_regex(self) -> re.Pattern:
        """도로명주소 정규식"""
        pattern = (
            rf"({self.sido_pattern})\s+"
            rf"({self.sigungu_pattern})\s+"
            rf"(?:({self.eup_myeon_dong_pattern})\s+)?"
            rf"({self.road_name_pattern})\s+"
            rf"({self.building_number_pattern})"
            rf"({self.detail_pattern})"
        )
        return re.compile(pattern)
    
    def get_jibun_address_regex(self) -> re.Pattern:
        """지번주소 정규식"""
        pattern = (
            rf"({self.sido_pattern})\s+"
            rf"({self.sigungu_pattern})\s+"
            rf"({self.eup_myeon_dong_pattern})\s+"
            rf"(?:({self.legal_ri_pattern})\s+)?"
            rf"({self.jibun_pattern})"
            rf"({self.detail_pattern})"
        )
        return re.compile(pattern)

    def _compile_unified_address_regex(self) -> re.Pattern:
        """통합 주소 정규식 컴파일"""
        road_pattern = (
            rf"({self.sido_pattern})\s+"
            rf"({self.sigungu_pattern})\s+"
            rf"(?:({self.eup_myeon_dong_pattern})\s+)?"
            rf"({self.road_name_pattern})\s+"
            rf"({self.building_number_pattern})"
            rf"({self.detail_pattern})"
        )
        
        jibun_pattern = (
            rf"({self.sido_pattern})\s+"
            rf"({self.sigungu_pattern})\s+"
            rf"({self.eup_myeon_dong_pattern})\s+"
            rf"(?:({self.legal_ri_pattern})\s+)?"
            rf"({self.jibun_pattern})"
            rf"({self.detail_pattern})"
        )
        
        unified_pattern = f"(?:{road_pattern})|(?:{jibun_pattern})"
        return re.compile(unified_pattern)

    def get_unified_address_regex(self) -> re.Pattern:
        """캐시된 통합 주소 정규식 반환"""
        return self._unified_address_regex

    def detect_addresses(self, text: str) -> List[str]:
        """주소를 정규식으로 검출"""
        results = []
        unified_regex = self.get_unified_address_regex()
        
        for match in unified_regex.finditer(text):
            address = match.group(0).strip()
            results.append(address)
        
        return results

    def _is_fuzzy_date_candidate(self, s: str) -> bool:
        """날짜 후보인지 확인하는 헬퍼 함수"""
        s = s.strip()
        n = len(s)
        for i in range(0, max(1, n - 7)):
            part8 = s[i:i+8]
            if len(part8) == 8 and part8.isdigit():
                try:
                    year = int(part8[0:4])
                    month = int(part8[4:6])
                    day = int(part8[6:8])
                    if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                        return True
                except Exception:
                    pass
        for i in range(0, max(1, n - 5)):
            part6 = s[i:i+6]
            if len(part6) == 6 and part6.isdigit():
                try:
                    mm = int(part6[2:4])
                    dd = int(part6[4:6])
                    if 1 <= mm <= 12 and 1 <= dd <= 31:
                        return True
                except Exception:
                    pass
        return False

    def _detect_trackings(self, text: str) -> List[Tuple[str, Tuple[int, int]]]:
        """운송장 번호를 탐지하고, 번호와 위치(span)를 반환"""
        tracking_patterns = [
            r'운송장번호|운송장|송장번호',
            r'(?:CJ|우체국|한진|롯데|로젠|경동|합동|CJ대한통운|대한통운|호남|천일|대신|건영|일양)(?:\s*택배\b)?'
        ]
        order_patterns = [
            r'주문\s*번호\s*[:：]?\s*',
            r'미검수\s*번호\s*[:：]?\s*',
        ]
        tracking_regex = re.compile('|'.join(f'(?:{p})' for p in tracking_patterns), re.IGNORECASE)
        order_regex = re.compile('|'.join(f'(?:{p})' for p in order_patterns), re.IGNORECASE)

        tracking_candidates = re.finditer(
            r'(?<!\d)(?:'
            r'\d{4}(?:[-\s]?\d{4}){2}'
            r'|[A-Z]{2}\d{9}[A-Z]{0,2}'
            r'|\d{9,16}'
            r')(?!\d)',
            text, flags=re.IGNORECASE
        )

        trackings: List[Tuple[str, Tuple[int, int]]] = []
        for match in tracking_candidates:
            candidate = match.group(0)
            clean_candidate = re.sub(r'[\s\-\.]', '', candidate)
            if len(clean_candidate) < 8 or len(clean_candidate) > 16:
                continue
            
            start, end = match.span()
            context_before = text[max(0, start - 30):start].lower()
            
            if order_regex.search(context_before):
                continue

            context_after = text[end:end + 30].lower()
            if tracking_regex.search(context_before + context_after) or tracking_regex.search(candidate):
                # 이 부분이 수정되었습니다.
                trackings.append((candidate.strip(), match.span()))

        return trackings

    def detect_trackings(self, text: str) -> List[str]:
        return [t[0] for t in self._detect_trackings(text)]

    def _detect_accounts(self, text: str) -> List[Tuple[str, Tuple[int, int]]]:
        """계좌 번호를 탐지하고, 번호와 위치(span)를 반환"""
        account_keywords = [
            r'(?:입금|환불)?\s*계좌',
            r'계좌번호'
        ]
        bank_keywords = [
            '신한', '국민', '하나', '우리', '기업', '농협', '수협', '씨티',
            '카카오', '토스', '케이', '제일', 'SC', '산업', '한국', '신협',
            '우체국', '새마을', '대구', '부산', '경남', '광주', '전북', '제주',
            '산림', '수출입', 'KDB', 'KB', 'IBK', 'NH'
        ]

        accounts: List[Tuple[str, Tuple[int, int]]] = []
        seen_spans = set()

        account_keyword_pattern = re.compile(
            r'(?:' + '|'.join(account_keywords) + r')[\s:]*[^\d]*(\d[\d\s-]{8,24}\d)(?![\d-])',
            re.IGNORECASE
        )

        bank_pattern = re.compile(
            r'(?:' + '|'.join(
                rf'(?:{re.escape(keyword)})(?:\s*(?:은행|뱅크))?'
                for keyword in bank_keywords
            ) + r')\s*[^\d]*(\d[\d\s-]*\d)(?![\d-])',
            re.IGNORECASE
        )

        bank_name_patterns = [
            re.compile(rf'{re.escape(k)}(?:\s*(?:은행|뱅크))?', re.IGNORECASE)
            for k in bank_keywords
        ]

        account_number_pattern = re.compile(r'(?<!\d)(\d{10,14})(?!\d)', re.UNICODE)

        for match in account_keyword_pattern.finditer(text):
            raw = match.group(1) or ""
            account = re.sub(r'\D', '', raw)
            span = match.span(1)
            key = (account, span[0], span[1])
            if 10 <= len(account) <= 14 and key not in seen_spans:
                accounts.append((account, span))
                seen_spans.add(key)

        for match in bank_pattern.finditer(text):
            raw = match.group(1) or ""
            account = re.sub(r'\D', '', raw)
            span = match.span(1)
            key = (account, span[0], span[1])
            if 10 <= len(account) <= 14 and key not in seen_spans:
                accounts.append((account, span))
                seen_spans.add(key)

        prefix_exclude_patterns = [
            r'주문\s*번호[\s\-:：]*',
            r'운송장\s*번호\s*:?'
        ]

        for match in account_number_pattern.finditer(text):
            account = match.group(1)
            span = match.span(1)
            key = (account, span[0], span[1])

            pre_context = text[max(0, match.start() - 30):match.start()].lower()
            if any(re.search(p, pre_context) for p in prefix_exclude_patterns):
                continue
            if len(account) == 10 and self._is_fuzzy_date_candidate(account):
                continue

            context = text[max(0, match.start() - 20):match.end() + 20]

            has_account_kw = any(re.search(k, context, re.IGNORECASE) for k in account_keywords)
            has_bank_name = any(p.search(context) for p in bank_name_patterns)

            if not (has_account_kw or has_bank_name):
                continue

            if key not in seen_spans:
                accounts.append((account, span))
                seen_spans.add(key)

        return accounts

    def detect_accounts(self, text: str) -> List[str]:
        return [a[0] for a in self._detect_accounts(text)]

    def detect_all(self, text: str) -> Dict[str, List[str]]:
        # 1단계: 각 함수를 실행하여 번호와 위치 정보를 모두 가져옴
        tracking_candidates_with_span = self._detect_trackings(text)
        account_candidates_with_span = self._detect_accounts(text)

        # 2단계: 중복 번호를 식별
        tracking_numbers = {t[0] for t in tracking_candidates_with_span}
        account_numbers = {a[0] for a in account_candidates_with_span}
        
        common_numbers = tracking_numbers.intersection(account_numbers)

        final_trackings = []
        final_accounts = []
        
        # 3단계: 중복을 해결하고 최종 리스트를 구성
        for num, span in tracking_candidates_with_span:
            if num in common_numbers:
                # 중복 번호의 문맥을 분석하여 우선순위를 결정
                context_before = text[max(0, span[0] - 15):span[0]].lower()
                
                # 계좌 관련 키워드가 더 명확하면 계좌로 분류
                if self._account_keywords_regex.search(context_before) or any(p.search(context_before) for p in self._bank_name_patterns):
                    final_accounts.append(num)
                # 그렇지 않으면 운송장으로 분류
                else:
                    final_trackings.append(num)
                
                # 이미 처리된 번호는 common_numbers에서 제거
                if num in common_numbers:
                    common_numbers.remove(num)
            else:
                final_trackings.append(num)

        # 4단계: 계좌 번호 후보 리스트에서 중복을 다시 확인 - 운송장으로 분류되지 않은 계좌 후보만 최종 리스트에 추가
        for num, span in account_candidates_with_span:
            if num not in final_trackings and num not in final_accounts:
                final_accounts.append(num)

        return {
            "trackings": sorted(list(set(final_trackings))),
            "accounts": sorted(list(set(final_accounts)))
        }