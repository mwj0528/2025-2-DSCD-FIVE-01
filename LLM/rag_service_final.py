# LLM/rag_service.py
# RAG 엔진을 앱 시작할 때 단 한 번만 로딩하기 위해 필요 -> 속도 메모리 효율 높이기 위해서 생성
# 백엔드(API)와 프론트엔드(UI)에서 모두 불러 쓸 수 있는 공통 함수를 제공

import os
import sys

# === rag_module import 가능한 경로 추가 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if current_dir not in sys.path:
    sys.path.append(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from rag_module_final import HSClassifier


# === 1. 앱 시작 시 단 한 번만 RAG 엔진 초기화 ===
print("[rag_service] RAG 엔진 로딩 중... (최종 모델)")

classifier = HSClassifier(
    embed_model="text-embedding-3-large",
    chroma_dir=os.path.join(parent_dir, "data", "chroma_db_openai_large_kw"),
    collection_name="hscode_collection",
    use_keyword_extraction=True,
    use_nomenclature=True,  # 항상 사용
)

print("[rag_service] RAG 엔진 로딩 완료!")


# === 2. FastAPI/Chainlit에서 공동으로 사용할 함수 ===
def classify_hs(product_name: str, product_desc: str, top_n: int = 5):
    """
    Chainlit, FastAPI 어디서든 호출 가능한 공통 함수.
    RAG 기반 HS 코드 추천을 수행한다.
    """
    return classifier.classify_hs_code_hierarchical(
        product_name=product_name,
        product_description=product_desc,
        top_n=top_n,
    )
