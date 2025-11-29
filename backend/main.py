# backend/main.py
import os, sys
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

# (원하면 CORS도 같이)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 필요시 localhost만 허용으로 조이기
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== 1) API 스키마 ====
class ItemInput(BaseModel):
    name: str
    desc: str

class Candidate(BaseModel):
    hs_code: str
    title: str | None = None
    reason: str | None = None

class HSResponse(BaseModel):
    candidates: list[Candidate]

# ==== 2) 분류 엔드포인트 ====
@app.post("/api/classify", response_model=HSResponse)
async def api_classify(item: ItemInput):
    try:
        raw = classify_hs(item.name, item.desc, top_n=5) #5개로 수정

        # chainlit에서 쓰던 result_json 형식 재사용
        cand_list = raw.get("candidates", raw.get("top_k_results", []))

        candidates = []
        for c in cand_list:
            candidates.append(Candidate(
                hs_code=c.get("hs_code") or c.get("hs10") or "",
                title=c.get("title") or c.get("label") or "",
                reason=c.get("reason") or c.get("rationale") or "",
            ))

        return HSResponse(candidates=candidates)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==== 3) 정적 프론트엔드 서빙 ====
# 프로젝트 루트 기준: frontend/ 폴더에 index.html, main.js, style.css 두기
frontend_dir = os.path.join(current_dir, "..", "frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

# 아래 명령어 실행:
# uvicorn backend.main:app --host 0.0.0.0 --port 8000
