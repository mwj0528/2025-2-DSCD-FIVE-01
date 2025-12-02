from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

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
from LLM.rag_service import classify_hs


app = FastAPI()

# ==== 프론트엔드 경로 설정 ====
BASE_DIR = Path(__file__).resolve().parent.parent  # project-root
FRONTEND_DIR = BASE_DIR / "frontend"

# /static 아래로 main.js, style.css 서빙
app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR), html=False),
    name="static",
)

# 루트 URL에서 index.html 반환
@app.get("/")
def read_root():
    return FileResponse(FRONTEND_DIR / "index.html")

# CORS 설정 (필요시 수정)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 발표용이니 전체 허용
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


# ------------ 2) 분류 엔드포인트 ------------

@app.post("/api/classify", response_model=HSResponse)
async def api_classify(item: ItemInput):
    try:
        # rag_service.classify_hs -> HSClassifier.classify_hs_code_hierarchical()
        raw = classify_hs(item.name, item.desc, top_n=5)

        # 기본적으로 "candidates"를 사용, 혹시 과거 형식(top_k_results)이면 그걸 사용
        cand_list = raw.get("candidates", raw.get("top_k_results", []))

        candidates: List[Candidate] = []

        for c in cand_list:
            # 기본 필드
            hs_code = c.get("hs_code") or c.get("hs10") or ""
            title = c.get("title") or c.get("label") or ""
            reason = c.get("reason") or c.get("rationale") or ""

            if not isinstance(hs_code, str):
                hs_code = str(hs_code)

            # citations 파싱 (없어도 에러 안 나게)
            citations_raw = c.get("citations") or []
            parsed_citations: Optional[List[Citation]] = None
            if isinstance(citations_raw, list) and citations_raw:
                tmp: List[Citation] = []
                for ci in citations_raw:
                    if not isinstance(ci, dict):
                        continue
                    # type 필드는 필수, 나머지는 선택
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

            # hierarchy_definitions 파싱 (rag_module._add_hierarchy_definitions 구조와 일치)
            hier_raw = c.get("hierarchy_definitions")
            parsed_hier: Optional[HierarchyDefinitions] = None
            if isinstance(hier_raw, dict):
                try:
                    parsed_hier = HierarchyDefinitions(**hier_raw)
                except Exception:
                    # 구조가 살짝 달라도 전체 API가 터지지는 않도록 무시
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
        # 백엔드에서 어떤 오류가 나든 프론트에서는 500으로 인지
        raise HTTPException(status_code=500, detail=str(e))


# ------------ 3) 정적 프론트엔드 서빙 ------------

# 프로젝트 루트 기준: frontend/ 폴더에 index.html, main.js, style.css
frontend_dir = os.path.join(root_dir, "frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

# uvicorn 실행 명령
# uvicorn backend.main:app --host 0.0.0.0 --port 8000
