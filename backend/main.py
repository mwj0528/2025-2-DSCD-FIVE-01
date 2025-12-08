# backend/main.py
import os
import sys
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# === 프로젝트 루트 경로를 sys.path에 추가 ===
current_dir = os.path.dirname(os.path.abspath(__file__))   # .../project/backend
root_dir = os.path.dirname(current_dir)                    # .../project

if root_dir not in sys.path:
    sys.path.append(root_dir)

# 이제 LLM 패키지 안의 rag_service를 가져온다
from LLM.rag_service_final import classify_hs


app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ 1) API 스키마 ------------

class ItemInput(BaseModel):
    name: str
    desc: str

class HierarchyLevel(BaseModel):
    code: str
    definition: str

class HierarchyDefinitions(BaseModel):
    chapter_2digit: Optional[HierarchyLevel] = None
    heading_4digit: Optional[HierarchyLevel] = None
    subheading_6digit: Optional[HierarchyLevel] = None
    national_10digit: Optional[HierarchyLevel] = None

class Citation(BaseModel):
    type: str
    code: Optional[str] = None
    doc_id: Optional[str] = None

class Candidate(BaseModel):
    hs_code: str
    title: Optional[str] = None
    reason: Optional[str] = None
    citations: Optional[List[Citation]] = None
    hierarchy_definitions: Optional[HierarchyDefinitions] = None

class HSResponse(BaseModel):
    candidates: List[Candidate]


# ------------ 2) 유니버설 HS Code 포맷 함수 ------------

def format_hs_code(code: str) -> str:
    """
    모든 HS Code를 XXXX.XX-XXXX 형태로 강제 포맷.
    이미 포맷된 값은 그대로 유지.
    숫자만 들어오면 10자리 기준으로 포맷 적용.
    """
    if code is None:
        return ""

    s = str(code).strip()

    # 이미 '.' 또는 '-'가 있으면 포맷된 것으로 간주
    if "." in s or "-" in s:
        return s

    # 숫자만 추출
    digits = "".join(ch for ch in s if ch.isdigit())

    # 10자리인 경우 → XXXX.XX-XXXX 형태
    if len(digits) == 10:
        return f"{digits[:4]}.{digits[4:6]}-{digits[6:]}"

    # 포맷을 특정할 수 없으면 원본 유지
    return s


# ------------ 3) 분류 엔드포인트 ------------

@app.post("/api/classify", response_model=HSResponse)
async def api_classify(item: ItemInput):
    try:
        # rag_service.classify_hs -> classifier.classify_hs_code_hierarchical()
        raw = classify_hs(item.name, item.desc, top_n=5)

        # 기본적으로 "candidates", 과거 형식은 "top_k_results"
        cand_list = raw.get("candidates", raw.get("top_k_results", []))

        candidates: List[Candidate] = []

        for c in cand_list:
            # 기본 필드
            hs_code = c.get("hs_code") or c.get("hs10") or ""
            hs_code = format_hs_code(hs_code)   # ★ 강제 포맷 적용

            title = c.get("title") or c.get("label") or ""
            reason = c.get("reason") or c.get("rationale") or ""

            if not isinstance(hs_code, str):
                hs_code = str(hs_code)

            # citations 파싱
            citations_raw = c.get("citations") or []
            parsed_citations: Optional[List[Citation]] = None
            if isinstance(citations_raw, list) and citations_raw:
                tmp: List[Citation] = []
                for ci in citations_raw:
                    if not isinstance(ci, dict):
                        continue
                    ci_type = ci.get("type")
                    if not ci_type:
                        continue
                    tmp.append(
                        Citation(
                            type=str(ci_type),
                            code=ci.get("code"),
                            doc_id=ci.get("doc_id"),
                        )
                    )
                if tmp:
                    parsed_citations = tmp

            # hierarchy_definitions 파싱
            hier_raw = c.get("hierarchy_definitions")
            parsed_hier: Optional[HierarchyDefinitions] = None
            if isinstance(hier_raw, dict):
                try:
                    parsed_hier = HierarchyDefinitions(**hier_raw)
                except Exception:
                    parsed_hier = None

            candidates.append(
                Candidate(
                    hs_code=hs_code,
                    title=title,
                    reason=reason,
                    citations=parsed_citations,
                    hierarchy_definitions=parsed_hier,
                )
            )

        return HSResponse(candidates=candidates)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------ 4) 정적 프론트엔드 서빙 ------------

frontend_dir = os.path.join(root_dir, "frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

# uvicorn 실행 명령:
# uvicorn backend.main:app --host localhost --port 8000

