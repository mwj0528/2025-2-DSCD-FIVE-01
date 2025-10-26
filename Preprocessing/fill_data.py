# fill_product_fields.py
# -*- coding: utf-8 -*-
"""
엑셀의 HS 메타정보(한글/영문 품목명, HS부호 등)를 바탕으로
LLM에게 '일반 사용자가 입력한 것 같은' 자연스러운 상품명/상품설명을 생성해
새 컬럼으로 저장합니다.

의존성:
    pip install pandas openpyxl python-dotenv openai tenacity

환경변수(.env 권장):
    OPENAI_API_KEY=sk-xxxx
    # 필요 시 모델 변경:
    OPENAI_MODEL=gpt-4.1-mini  # 권장: 비용/품질 균형
"""

import os
import time
import json
import math
from typing import Dict, Any, Optional

import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# OpenAI Responses API (>= 1.2.0 SDK)
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

# ===================== 사용자 설정 =====================
IN_XLSX  = r"dataset/HScode_100개.xlsx"  # 입력 파일
OUT_XLSX = r"output/HScode_100개_filled.xlsx"   # 결과 저장 파일
SHEET_NAME = 0   # 특정 시트명/인덱스 가능
BATCH_SLEEP = 0.0  # 행당 대기(초). 속도/쿼터 여유 없으면 0~0.1 권장
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
# ======================================================


# -------- 유틸: LLM 호출 준비 --------
load_dotenv(override=False)

class LLMClient:
    """
    OpenAI Responses API 우선 사용.
    SDK가 없거나 실패 시 명시적으로 예외 발생시켜 재시도.
    """
    def __init__(self, model: str):
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("openai SDK 미설치: pip install openai")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("환경변수 OPENAI_API_KEY가 없습니다. .env 또는 시스템 환경변수로 설정하세요.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _build_system_prompt(self) -> str:
        return (
            "당신은 HS코드 보조정보를 참고하여, HS코드 자동 추천 모델 성능테스트에 쓰일 정답 데이터를 생성해야함 "
            "‘상품명’과 ‘상품 설명’을 간결하게 작성하는 어시스턴트. "
            "공식 분류 용어(관세/HS/GRI 문구)는 노출하지 말고, 품질/재질/원재료/주요 용도/특징을 자연스럽게 개조식으로 작성. "
            "결과는 JSON만 출력."
        )

    def _build_user_prompt(self, row: Dict[str, Any]) -> str:
        # 행에서 쓸만한 컬럼 키 추출(있으면 전달)
        def get(k):
            return row.get(k)
        # 컬럼 후보(파일에 실제 존재하는 이름을 폭넓게 커버)
        KR = _first_key(row, ["한글품목명", "품목명(한글)", "품목명_한글", "KoreanName"])
        EN = _first_key(row, ["영문품목명", "품목명(영문)", "품목명_영문", "EnglishName"])
        HS = _first_key(row, ["HS부호", "HS", "HS 코드", "HSCode"])
        ATTR = _first_key(row, ["성질통합분류코드명", "성질", "성질코드명", "PropertyClass"])
        GOODS = _first_key(row, ["상품명", "기존상품명"])
        DESC = _first_key(row, ["상품설명", "기존상품설명"])

        # 프롬프트: 모델이 참고할 정보만 간략·구조적으로 제공
        context = {
            "HS부호": get(HS),
            "한글품목명": get(KR),
            "영문품목명": get(EN),
            "성질통합분류코드명": get(ATTR),
            "참고_기존상품명": get(GOODS),
            "참고_기존상품설명": get(DESC),
        }

        # 출력 규격: JSON만, 제약사항 명시
        instruction = (
            "다음 정보를 참고하여 아래 JSON 형식으로만 출력.\n"
            "- 상품명: 40자 이하\n"
            "- 상품설명: 200자 이하, 원재료/재질, 주요 사용 용도, 특징(크기/사용감/호환성 등)을 포함\n"
            "반환 JSON 스키마:\n"
            "{\n"
            '  "product_name": "string",\n'
            '  "product_description": "string"\n'
            "}\n\n"
            f"[참고정보]\n{json.dumps(context, ensure_ascii=False)}"
        )
        return instruction

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(6),
        reraise=True
    )
    def infer(self, row: Dict[str, Any]) -> Dict[str, str]:
        system = self._build_system_prompt()
        prompt = self._build_user_prompt(row)

        # Responses API
        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_output_tokens=300,
        )

        # 텍스트 추출
        text = _extract_text(resp)
        if not text:
            raise RuntimeError("빈 응답")

        # JSON 파싱 시도(관용적으로 가끔 코드블록/잡텍스트 섞일 수 있으니 정리)
        data = _safe_json_loads(text)
        if not isinstance(data, dict):
            raise RuntimeError(f"JSON 파싱 실패: {text[:200]}...")
        name = data.get("product_name") or data.get("상품명")
        desc = data.get("product_description") or data.get("상품설명")
        if not name or not desc:
            raise RuntimeError(f"필드 누락: {data}")

        # 길이 가드(과하면 잘라줌)
        name = name.strip()
        desc = " ".join(str(desc).split())
        if len(name) > 40:
            name = name[:40].rstrip()
        if len(desc) > 230:
            desc = desc[:230].rstrip()

        return {"name": name, "desc": desc}


# -------- 보조 함수들 --------
def _extract_text(resp) -> str:
    """
    OpenAI Responses API의 출력에서 텍스트만 안전하게 뽑는다.
    """
    try:
        for item in resp.output:
            if item.type == "message":
                # message.content -> list of content parts
                parts = item.message.content
                if parts and hasattr(parts[0], "text"):
                    return parts[0].text
        # fallback
        return getattr(resp, "output_text", None) or ""
    except Exception:
        # 일부 SDK 버전 호환 보정
        try:
            return resp.output_text
        except Exception:
            return ""

def _safe_json_loads(s: str) -> Any:
    # 코드블록/앞뒤 잡텍스트 제거 시도
    s2 = s.strip()
    # 백틱 코드블록 제거
    if s2.startswith("```"):
        s2 = s2.strip("`")
        # 언어태그 제거
        lines = s2.splitlines()
        if lines and not lines[0].startswith("{"):
            lines = lines[1:]
        s2 = "\n".join(lines)
    # JSON 부분만 추출 시도
    lpos = s2.find("{")
    rpos = s2.rfind("}")
    if lpos != -1 and rpos != -1 and rpos > lpos:
        s2 = s2[lpos:rpos+1]
    try:
        return json.loads(s2)
    except Exception:
        return {}

def _first_key(row: Dict[str, Any], candidates):
    for k in candidates:
        if k in row:
            return k
    return None

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

# -------- 메인 처리 --------
def main():
    if not os.path.exists(IN_XLSX):
        raise FileNotFoundError(f"입력 파일이 없습니다: {IN_XLSX}")

    df = pd.read_excel(IN_XLSX, sheet_name=SHEET_NAME)
    df = _normalize_columns(df)

    out_cols = ["상품명(사용자입력 가정)", "상품설명(사용자입력 가정)"]
    for c in out_cols:
        if c not in df.columns:
            df[c] = pd.NA

    client = LLMClient(model=MODEL)

    # 이미 작성된 행은 스킵(resume)
    total = len(df)
    filled = 0
    for idx, row in df.iterrows():
        if pd.notna(df.at[idx, out_cols[0]]) and pd.notna(df.at[idx, out_cols[1]]):
            continue

        row_dict = row.to_dict()
        try:
            result = client.infer(row_dict)
            df.at[idx, out_cols[0]] = result["name"]
            df.at[idx, out_cols[1]] = result["desc"]
            filled += 1
        except Exception as e:
            # 실패 시 빈값 유지하고 로그만 남김
            df.at[idx, out_cols[0]] = df.at[idx, out_cols[0]]
            df.at[idx, out_cols[1]] = df.at[idx, out_cols[1]]
            print(f"[경고] {idx}행 처리 실패: {e}")

        if BATCH_SLEEP > 0:
            time.sleep(BATCH_SLEEP)

        if (idx + 1) % 1 == 0:
            print(f"진행 현황: {idx + 1}/{total}")

    # 저장
    os.makedirs(os.path.dirname(OUT_XLSX) or ".", exist_ok=True)
    df.to_excel(OUT_XLSX, index=False)
    print(f"완료: {filled}행 작성 → {OUT_XLSX}")

if __name__ == "__main__":
    main()
