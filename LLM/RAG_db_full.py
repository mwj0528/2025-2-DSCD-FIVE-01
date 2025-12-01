# rag_hs_prompt.py
# -*- coding: utf-8 -*-
"""
- 네가 준 classify_hs_code() 형태를 RAG로 확장한 버전
- Chroma Persistent DB(=chromadb 폴더) 연동
- 인덱싱 때 사용한 임베딩 모델과 동일 모델 사용(중요!)
- LLM은 JSON 모드(response_format={"type":"json_object"})로 강제
"""
#########

import sys
import os
# 1. 이 파일(rag_hs_prompt.py)의 현재 경로를 찾음
#    -> /home/oohga/project/LLM
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 그것의 '부모 폴더'(project) 경로를 찾음
#    -> /home/oohga/project
parent_dir = os.path.dirname(current_dir)

# 3. Python이 '부모 폴더'를 검색하도록 경로(sys.path)에 추가
sys.path.append(parent_dir)

# --- (이제서야 Python이 'project/' 폴더를 뒤지기 시작함) ---

# 4. 이제 Python이 'RAG_embedding' 폴더를 찾을 수 있음
from RAG_embedding.graph_rag import GraphRAG


#################
from openai import OpenAI
from dotenv import load_dotenv
import os, re, json
from typing import List, Dict, Any, Tuple
from konlpy.tag import Okt
import sys
import os

# 현재 파일의 디렉토리에서 상위 디렉토리로 이동 후 RAG_embedding 폴더 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
rag_embedding_dir = os.path.join(parent_dir, 'RAG_embedding')

sys.path.append(rag_embedding_dir)
from RAG_embedding.graph_rag import GraphRAG

# ===== 0) 환경설정 =====
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Chroma 설정(폴더 경로 = chroma.sqlite3가 들어있는 디렉터리)
CHROMA_DIR = os.getenv("CHROMA_DIR", r"./chroma_db")  # 예: C:\...\embedding\chroma_db
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "hscode_collection")

# 인덱싱 때 썼던 임베딩 모델과 반드시 동일하게!
EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

okt_analyzer = Okt()
STOPWORDS = [
    '의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다',
    '상품명', '설명', '사유', '이름', '제품', '관련', '내용', '항목', '분류', '기준',
    'hs', 'code', 'item', 'des', 'description', 'name'
]
def extract_keywords_advanced(text: str) -> str:
    """사용자 입력을 DB와 동일한 '키워드' 형식으로 변환"""
    
    # (1단계(인덱싱)에서 쓴 로직과 100% 동일해야 함)
    tagged_words = okt_analyzer.pos(text, norm=True, stem=False)
    keywords = []

    for word, tag in tagged_words:
        if tag in ['Noun', 'Alpha']:
            keywords.append(word)
    regex_keywords = re.findall(r'\b[a-zA-Z0-9]{2,}\b', text)
    keywords.extend(regex_keywords)
    filtered_keywords = set() 

    for k in keywords:
        k_lower = k.lower() 
        if k_lower not in STOPWORDS:
            filtered_keywords.add(k)

    return " ".join(sorted(list(filtered_keywords)))
    
  

# ===== 1) 안전 JSON 파서(네 코드 그대로) =====
def _parse_json_safely(text: str):
    """
    1) 그대로 json.loads 시도
    2) ```json ... ``` 또는 ``` ... ``` 감싸진 경우 벗겨서 재시도
    3) 마지막으로 중괄호/대괄호 범위만 추출해서 재시도
    """
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass

    # 코드펜스 제거
    fenced = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip(), flags=re.DOTALL)
    try:
        return json.loads(fenced), None
    except json.JSONDecodeError:
        pass

    # JSON 스니펫만 추출 (가장 바깥 { ... } 또는 [ ... ])
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1)), None
        except json.JSONDecodeError as e:
            return None, f"JSON decode failed after extraction: {e}"
    return None, "JSON decode failed: unrecognized format"


# ===== 2) 임베딩 & Chroma 유틸 =====
from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb
from chromadb.config import Settings

class QueryEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = True

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=self.normalize
        )
        return np.asarray(vecs, dtype="float32")

def open_chroma_collection(persist_dir: str, collection_name: str):
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
    try:
        return client.get_collection(name=collection_name)
    except Exception:
        names = [c.name for c in client.list_collections()]
        raise RuntimeError(f"컬렉션 '{collection_name}' 없음. 현재 컬렉션들: {names}")


def search_chroma(collection, embedder, query_text: str, top_k: int = 12):
    qvec = embedder.embed([query_text])[0]

    # ❌ include에 "ids" 넣지 마세요
    res = collection.query(
        query_embeddings=[qvec.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances", "embeddings"],  # ← 여기서 수정
    )

    # ids는 include에 없어도 기본으로 내려옵니다.
    ids      = (res.get("ids") or [[]])[0]
    dists    = (res.get("distances") or [[]])[0]
    docs_txt = (res.get("documents") or [[]])[0]
    metas    = (res.get("metadatas") or [[]])[0]
    embeds   = (res.get("embeddings") or [[]])[0]

    docs = []
    for id_, dist, txt, meta, emb in zip(ids, dists, docs_txt, metas, embeds):
        docs.append({
            "id": id_,
            "distance": float(dist),
            "document": txt,
            "metadata": meta,
            # 이후 MMR 등에서 쓰려면 벡터도 보관
            "embedding": np.asarray(emb, dtype="float32")
        })

    # 거리(작을수록 유사) 기준 정렬
    docs.sort(key=lambda d: d["distance"])
    return docs



# ===== 3) RAG 프롬프트 빌더 =====
def build_rag_prompt(product_name: str, product_description: str, retrieved: List[Dict[str, Any]], top_n: int, graph_context: str = "") -> Tuple[str, str]:
    """
    - system: 컨텍스트 내에서만 답하도록 가드
    - user: 질문 + 컨텍스트 + JSON 스키마(엄격)
    """
    system = (
        "당신은 국제무역 HS 코드 분류 전문가입니다.\n\n"
        "규칙:\n"
        "1) 제공된 context 내의 정보만 사용하여 판단합니다.\n"
        "2) HS Code 계층 구조(GraphDB Context)는 전체 HS Code data이며\n"
        "   품목분류사례(VectorDB Context)는 classify Case data입니다.\n"
        "3) 계층 구조와 다른 코드는 절대 제시하지 않습니다.\n"
        "4) 추천하는 HS Code는 반드시 10자리여야 합니다.\n"
        "5) 항상 응답은 strict JSON format으로만 출력합니다.\n"
        "6) 확신이 없을 경우 'candidates': [] 로 응답합니다."
    )

    # ===== 여기부터 컨텍스트 블록 구성 부분만 요청한 방식으로 교체 =====
    def _pick(meta, keys, default=""):
        for k in keys:
            v = meta.get(k)
            if v not in (None, ""):
                return str(v)
        return default

    def _fallback_name_from_body(body: str) -> str:
        # 본문 첫 줄에 "상품명: ..." 패턴이 자주 있으므로 거기서 보정 추출
        m = re.search(r"^상품명:\s*(.+)$", body, flags=re.MULTILINE)
        return m.group(1).strip() if m else ""

    # 컨텍스트 블록(최대 6~10개 권장)
    blocks = []
    for d in retrieved[:10]:
        meta = d.get("metadata", {}) or {}
        body = (d.get("document") or "").strip()

        # 1) 메타 기준(실측 키에 맞춤) + 2) 본문 보정 추출
        hs   = _pick(meta, ["HSCode", "hs_code", "HS", "HS부호"])
        name = _pick(meta, ["상품명", "한글품목명", "title", "품목명"]) or _fallback_name_from_body(body)
        date = _pick(meta, ["시행일자", "발행일"])

        # 선택: body 길이 안전 절단
        max_chars = 1200
        if len(body) > max_chars:
            body = body[:max_chars] + "…"

        # distances는 query() 사용 시에만 존재. get()이면 없음 → 0.0
        dist = d.get("distance", 0.0)
        try:
            dist = float(dist)
        except Exception:
            dist = 0.0

        blocks.append(
            f"[DOC id={d.get('id')} dist={dist:.4f}]\n"
            f"상품명: {name}\nHSCode: {hs}\n시행일자: {date}\n본문:\n{body}\n"
        )

    vector_context = "\n\n".join(blocks) if blocks else "(검색 결과 없음)"
    # ===== 교체 끝 =====

    # GraphDB context 추가
    graph_section = ""
    if graph_context.strip():
        graph_section = f"{graph_context}"
    else:
        graph_section = "(GraphDB 검색 결과 없음)"

    user = f"""
다음 제품의 HS 코드 상위 {top_n} 후보를 추천하세요 
중요: 추천하는 모든 HS Code는 반드시 10자리여야 합니다 (예: 9405.40.10.00)

[입력]
- Product Name: {product_name}
- Product Description: {product_description}

================================================
[context]
[HS Code 계층 구조 Context — GraphDB Retrieved]
(모든 데이터는 HS 공식 nomenclature 기반)
{graph_section}
================================================

[품목분류사례 Context — VectorDB Retrieved]
(정부 품목분류사례 문서 기반 근거 자료)
{vector_context}  
================================================

[응답 형식: strict JSON — 추가 키 금지]
{{
  "candidates": [
    {{
      "hs_code": "string",          // 반드시 10자리 HS Code (예: 9405.40.10.00)
      "title": "string",
      "reason": "string",           // 한국어, 200자 이내
      "citations": [
        {{"type": "graph", "code": "string"}},   // GraphDB 근거
        {{"type": "case", "doc_id": "string"}}   // VectorDB 근거
      ]
    }}
  ]
}}

필수 규칙:
1) 후보는 최대 {top_n}개
2) hs_code는 반드시 10자리여야 합니다 (예: 9405.40.10.00)
3) citations는 최소 1개 이상 포함
4) citations.type은 반드시 "graph" 또는 "case"만 가능
"""
    return system, user


# ===== 4) 최종 함수: RAG + JSON 모드 =====
def classify_hs_code_rag(product_name: str, product_description: str, top_n: int = 3) -> Dict[str, Any]:
    """
    1) Chroma에서 유사 문서 검색
    2) GraphRAG에서 계층 구조 정보 검색
    3) 컨텍스트를 붙여 LLM(JSON 강제) 호출
    4) JSON 파싱해 반환
    """
    # 1) Chroma 검색
    emb = QueryEmbedder(EMBED_MODEL)
    col = open_chroma_collection(CHROMA_DIR, COLLECTION_NAME)
    original_query_text = f"{product_name}\n{product_description}"
    keyword_query_text = extract_keywords_advanced(original_query_text)
    
    hits = search_chroma(col, emb, keyword_query_text, top_k=max(8, top_n*3))

    # 2) GraphRAG에서 계층 구조 정보 검색
    try:
        graph_rag = GraphRAG()
        graph_context = graph_rag.get_final_context(original_query_text, k=5)
    except Exception as e:
        print(f"GraphRAG 오류: {e}")
        graph_context = ""

    # 3) 프롬프트 구성
    sys_prompt, user_prompt = build_rag_prompt(product_name, product_description, hits, top_n, graph_context)

    # 4) JSON 모드 호출
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.2,
        response_format={"type": "json_object"}  # <- 핵심: 순수 JSON만 반환
    )

    output_text = response.choices[0].message.content.strip()

    # 5) 안전 파싱
    result, err = _parse_json_safely(output_text)
    if err:
        result = {"error": err, "raw_output": output_text}

    return result


# ===== 5) GraphRAG 통합 헬퍼 함수 =====
def get_enhanced_context(product_name: str, product_description: str, k: int = 5) -> Dict[str, str]:
    """
    ChromaDB와 GraphDB의 컨텍스트를 모두 가져오는 헬퍼 함수
    
    Args:
        product_name: 상품명
        product_description: 상품설명
        k: GraphDB에서 검색할 후보 개수
        
    Returns:
        Dict[str, str]: chroma_context와 graph_context를 포함한 딕셔너리
    """
    # ChromaDB 컨텍스트
    emb = QueryEmbedder(EMBED_MODEL)
    col = open_chroma_collection(CHROMA_DIR, COLLECTION_NAME)
    original_query_text = f"{product_name}\n{product_description}"
    keyword_query_text = extract_keywords_advanced(original_query_text)
    
    hits = search_chroma(col, emb, keyword_query_text, top_k=8)
    
    # ChromaDB 컨텍스트 구성
    def _pick(meta, keys, default=""):
        for k in keys:
            v = meta.get(k)
            if v not in (None, ""):
                return str(v)
        return default

    def _fallback_name_from_body(body: str) -> str:
        m = re.search(r"^상품명:\s*(.+)$", body, flags=re.MULTILINE)
        return m.group(1).strip() if m else ""

    blocks = []
    for d in hits[:10]:
        meta = d.get("metadata", {}) or {}
        body = (d.get("document") or "").strip()

        hs   = _pick(meta, ["HSCode", "hs_code", "HS", "HS부호"])
        name = _pick(meta, ["상품명", "한글품목명", "title", "품목명"]) or _fallback_name_from_body(body)
        date = _pick(meta, ["시행일자", "발행일"])

        max_chars = 1200
        if len(body) > max_chars:
            body = body[:max_chars] + "…"

        dist = d.get("distance", 0.0)
        try:
            dist = float(dist)
        except Exception:
            dist = 0.0

        blocks.append(
            f"[DOC id={d.get('id')} dist={dist:.4f}]\n"
            f"상품명: {name}\nHSCode: {hs}\n시행일자: {date}\n본문:\n{body}\n"
        )

    chroma_context = "\n\n".join(blocks) if blocks else "(검색 결과 없음)"
    
    # GraphDB 컨텍스트
    try:
        graph_rag = GraphRAG()
        graph_context = graph_rag.get_final_context(original_query_text, k=k)
    except Exception as e:
        print(f"GraphRAG 오류: {e}")
        graph_context = ""
    
    return {
        "vector_context": chroma_context,
        "graph_context": graph_context,
        "query_text": original_query_text
    }


# ===== 6) 예시 실행 =====
if __name__ == "__main__":
    print("=== 통합 RAG 테스트 (ChromaDB + GraphDB) ===")
    name = "LED 조명"
    desc = "플라스틱 하우징에 장착된 LED 조명 모듈로, 실내용 조명 기구"
    
    print(f"상품명: {name}")
    print(f"상품설명: {desc}")
    print("\n처리 중...")
    
    # 1. 개별 컨텍스트 확인
    print("\n=== 1. 컨텍스트 확인 ===")
    contexts = get_enhanced_context(name, desc, k=5)
    print(f"VectorDB 컨텍스트 길이: {len(contexts['vector_context'])}")
    print(f"GraphDB 컨텍스트 길이: {len(contexts['graph_context'])}")
    
    # 2. 통합 RAG 실행
    print("\n=== 2. 통합 RAG 실행 ===")
    out = classify_hs_code_rag(name, desc, top_n=3)
    print("\n=== 최종 결과 ===")
    print(json.dumps(out, ensure_ascii=False, indent=2))

# python LLM/RAG.py
