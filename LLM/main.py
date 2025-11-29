from fastapi import FastAPI
from pydantic import BaseModel
import os, sys

# ===== 0. 경로 설정 =====
# (rag_module을 찾기 위해 LLM 폴더 경로 추가)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# ===== 1. 최신 엔진 Import =====
try:
    from rag_module import HSClassifier
except ImportError:
    print("오류: rag_module.py를 찾을 수 없습니다.")
    sys.exit(1)

app = FastAPI()

# ===== 2. RAG 엔진 초기화 (서버 시작 시 1번만 실행) =====
# (streamlit_app.py와 동일한 '기본 조건' 설정)
print("RAG 엔진 로딩 중...")
classifier = HSClassifier(
    parser_type="both",                         # --parser both
    embed_model="text-embedding-3-large",       # --embed-model openai_large
    chroma_dir="data/chroma_db_openai_large_kw",# DB 경로
    collection_name="hscode_collection",
    use_keyword_extraction=True
)
print("RAG 엔진 로딩 완료!")

# 3. 입력 데이터 모델
class ItemInput(BaseModel):
    name: str
    desc: str

# 4. API 엔드포인트
@app.post("/classify")
async def run_rag_classification(item: ItemInput):
    """
    [POST] 상품명과 설명을 받아 HS Code를 추천합니다.
    (계층적 3단계 분석 / Hierarchical RAG 사용)
    """
    try:
        # 최신 엔진의 계층적 분류 함수 호출
        result = classifier.classify_hs_code_hierarchical(
            product_name=item.name,
            product_description=item.desc,
            top_n=3
        )
        return result
    except Exception as e:
        return {"error": str(e)}

# 실행 명령 (터미널):
# python -m uvicorn LLM.main:app --reload