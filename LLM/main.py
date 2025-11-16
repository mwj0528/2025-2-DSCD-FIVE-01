from fastapi import FastAPI
from pydantic import BaseModel
from LLM.RAG_db_full import classify_hs_code_rag # 스크립트 import

app = FastAPI()

# 1. 사용자 입력을 받을 모델 정의
class ItemInput(BaseModel):
    name: str
    desc: str

# 2. API 엔드포인트 생성
@app.post("/classify")
async def run_rag_classification(item: ItemInput):
    # 님의 RAG 함수를 그대로 호출
    result = classify_hs_code_rag(item.name, item.desc, top_n=3)
    return result


###**.env 파일이 있는 parent_dir (최상위 폴더)**에서 실행해야 합니다! 아래 명령어를
#### parent_dir/ 경로에서 실행
###-----------------------------------
###    python -m uvicorn LLM.main:app --reload
###     streamlit run LLM/streamlit_app.py