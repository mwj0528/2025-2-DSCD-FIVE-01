# rag_hs_prompt.py
# -*- coding: utf-8 -*-
"""
- 네가 준 classify_hs_code() 형태를 RAG로 확장한 버전
- Chroma Persistent DB(=chromadb 폴더) 연동
- 인덱싱 때 사용한 임베딩 모델과 동일 모델 사용(중요!)
- LLM은 JSON 모드(response_format={"type":"json_object"})로 강제
"""

from openai import OpenAI
from dotenv import load_dotenv
import os, re, json
from typing import List, Dict, Any, Tuple

# ===== 0) 환경설정 =====
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Chroma 설정(폴더 경로 = chroma.sqlite3가 들어있는 디렉터리)
CHROMA_DIR = os.getenv("CHROMA_DIR", r"../chroma_db")  # 예: C:\...\embedding\chroma_db
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "hscode_collection")

# 인덱싱 때 썼던 임베딩 모델과 반드시 동일하게!
EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

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
def build_rag_prompt(product_name: str, product_description: str, retrieved: List[Dict[str, Any]], top_n: int) -> Tuple[str, str]:
    """
    - system: 컨텍스트 내에서만 답하도록 가드
    - user: 질문 + 컨텍스트 + JSON 스키마(엄격)
    """
    system = (
        "당신은 HS 코드 분류 전문가입니다. 제공된 context의 내용을 참고해서 답하세요. "
        "context는 사용자가 입력한 내용과 유사한 HS코드 품목분류사례 데이터입니다"
        "HS코드 품목분류사례에 모든 HS코드 사례가 있지 않으므로 유사도가 떨어질 수도 있으니 그런 경우 참고만 할 것"
        "항상 JSON 스키마로만 응답하세요."
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

    context = "\n\n".join(blocks) if blocks else "(검색 결과 없음)"
    # ===== 교체 끝 =====

    user = f"""
다음 제품의 HS 코드 상위 {top_n} 후보를 추천.

[입력]
- Product Name: {product_name}
- Product Description: {product_description}

[컨텍스트]
{context}

[응답 형식: strict JSON — 추가 키 금지]
{{
  "candidates": [
    {{
      "hs_code": "string",
      "title": "string",
      "reason": "string",          // 한국어, 200자 이내
      "gri": ["string"],           // 적용 GRI; 불명확하면 빈 배열
      "citations": [{{"doc_id":"string"}}]  // [DOC id=...]의 id 사용
    }}
  ]
}}

규칙:
- 후보는 최대 {top_n}개.
- 최대한 context 근거를 인용, 'citations'에 근거 문서 id를추가.
- 근거가 부족하면 "candidates": [] 로 빈칸 처리.
"""
    return system, user


# ===== 4) 최종 함수: RAG + JSON 모드 =====
def classify_hs_code_rag(product_name: str, product_description: str, top_n: int = 3) -> Dict[str, Any]:
    """
    1) Chroma에서 유사 문서 검색
    2) 컨텍스트를 붙여 LLM(JSON 강제) 호출
    3) JSON 파싱해 반환
    """
    # 1) 검색
    emb = QueryEmbedder(EMBED_MODEL)
    col = open_chroma_collection(CHROMA_DIR, COLLECTION_NAME)
    query_text = f"{product_name}\n{product_description}"
    hits = search_chroma(col, emb, query_text, top_k=max(8, top_n*3))

    # 2) 프롬프트 구성
    sys_prompt, user_prompt = build_rag_prompt(product_name, product_description, hits, top_n)

    # 3) JSON 모드 호출
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}  # <- 핵심: 순수 JSON만 반환
    )

    output_text = response.choices[0].message.content.strip()

    # 4) 안전 파싱
    result, err = _parse_json_safely(output_text)
    if err:
        result = {"error": err, "raw_output": output_text}

    return result


# ===== 5) 예시 실행 =====
if __name__ == "__main__":
    name = "LED 조명"
    desc = "플라스틱 하우징에 장착된 LED 조명 모듈로, 실내용 조명 기구"
    out = classify_hs_code_rag(name, desc, top_n=3)
    print(json.dumps(out, ensure_ascii=False, indent=2))
