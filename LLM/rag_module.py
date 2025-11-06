# -*- coding: utf-8 -*-
"""
HS Code 분류를 위한 RAG 모듈
- ChromaDB와 GraphDB를 선택적으로 사용 가능
- Parser 설정을 통해 사용할 DB 선택
"""

from openai import OpenAI
from dotenv import load_dotenv
import os, re, json
from typing import List, Dict, Any, Tuple, Optional, Literal
from konlpy.tag import Okt
import sys

# 현재 파일의 디렉토리에서 상위 디렉토리로 이동 후 RAG_embedding 폴더 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
rag_embedding_dir = os.path.join(parent_dir, 'RAG_embedding')

sys.path.append(rag_embedding_dir)

# 임베딩 & ChromaDB 관련
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import torch
import numpy as np
import chromadb
from chromadb.config import Settings

# GraphDB는 선택적으로 import
try:
    from graph_rag import GraphRAG
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    print("경고: GraphRAG 모듈을 찾을 수 없습니다. GraphDB 기능은 사용할 수 없습니다.")


# ===== Parser 타입 정의 =====
ParserType = Literal["chroma", "graph", "both"]


# ===== 환경설정 =====
load_dotenv()

# 기본 설정
DEFAULT_CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma_db")
DEFAULT_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "hscode_collection")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# 키워드 추출 관련
okt_analyzer = Okt()
STOPWORDS = [
    '의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다',
    '상품명', '설명', '사유', '이름', '제품', '관련', '내용', '항목', '분류', '기준',
    'hs', 'code', 'item', 'des', 'description', 'name'
]


# ===== 유틸리티 함수 =====
def extract_keywords_advanced(text: str) -> str:
    """사용자 입력을 DB와 동일한 '키워드' 형식으로 변환"""
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
        if len(k) > 1 and k_lower not in STOPWORDS:
            filtered_keywords.add(k)

    return " ".join(sorted(list(filtered_keywords)))


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


# ===== ChromaDB 관련 클래스 및 함수 =====
class QueryEmbedder:
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, device: str = "cpu"):
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

    res = collection.query(
        query_embeddings=[qvec.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

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
            "embedding": np.asarray(emb, dtype="float32")
        })

    docs.sort(key=lambda d: d["distance"])
    return docs


# ===== 메인 HSClassifier 클래스 =====
class HSClassifier:
    """
    HS Code 분류를 위한 RAG 클래스
    Parser 설정을 통해 ChromaDB, GraphDB, 또는 둘 다 선택적으로 사용 가능
    """
    
    def __init__(
        self,
        parser_type: ParserType = "both",
        chroma_dir: str = None,
        collection_name: str = None,
        embed_model: str = None,
        openai_model: str = None,
        openai_api_key: str = None,
        use_keyword_extraction: bool = True,
        use_rerank: bool = False,
        rerank_model: str = None,
        rerank_top_m: int = 5,
        use_graph_rerank: bool = False,
        graph_rerank_model: str = None,
        graph_rerank_top_m: int = 5,
        temperature: float = 0.0,
        seed: Optional[int] = None
    ):
        """
        Args:
            parser_type: 사용할 DB 설정 ("chroma", "graph", "both")
            chroma_dir: ChromaDB 디렉토리 경로
            collection_name: ChromaDB 컬렉션 이름
            embed_model: 임베딩 모델 이름
            openai_model: OpenAI 모델 이름
            openai_api_key: OpenAI API 키
            use_keyword_extraction: ChromaDB 검색 시 키워드 추출 사용 여부 (기본값: True)
        """
        # Parser 타입 설정
        if parser_type not in ["chroma", "graph", "both"]:
            raise ValueError(f"parser_type은 'chroma', 'graph', 'both' 중 하나여야 합니다. 입력값: {parser_type}")
        
        self.parser_type = parser_type
        self.use_keyword_extraction = use_keyword_extraction
        self.use_rerank = bool(use_rerank)
        
        # GraphDB 사용 시 가용성 확인
        if parser_type in ["graph", "both"]:
            if not GRAPH_AVAILABLE:
                raise RuntimeError("GraphDB를 사용하려고 했지만 GraphRAG 모듈을 찾을 수 없습니다.")
        
        # 설정값 초기화
        self.chroma_dir = chroma_dir or DEFAULT_CHROMA_DIR
        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME
        self.embed_model = embed_model or DEFAULT_EMBED_MODEL
        self.openai_model = openai_model or DEFAULT_OPENAI_MODEL
        self.rerank_model = rerank_model or DEFAULT_RERANK_MODEL
        self.rerank_top_m = max(1, int(rerank_top_m)) if isinstance(rerank_top_m, int) else 8
        self.temperature = float(temperature)
        # seed 우선순위: 인자 > 환경변수 > None
        env_seed = os.getenv("LLM_SEED")
        self.seed = seed if seed is not None else (int(env_seed) if env_seed and env_seed.isdigit() else None)
        
        # OpenAI 클라이언트 초기화
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API 키가 필요합니다. 환경변수 OPENAI_API_KEY를 설정하거나 openai_api_key를 제공하세요.")
        self.client = OpenAI(api_key=api_key)
        
        # ChromaDB 관련 초기화 (필요한 경우)
        self._chroma_embedder = None
        self._chroma_collection = None
        self._reranker = None
        if parser_type in ["chroma", "both"]:
            try:
                self._chroma_embedder = QueryEmbedder(self.embed_model)
                self._chroma_collection = open_chroma_collection(self.chroma_dir, self.collection_name)
            except Exception as e:
                print(f"경고: ChromaDB 초기화 실패: {e}")
                if parser_type == "chroma":
                    raise
            # ReRank 초기화 (선택)
            if self.use_rerank:
                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._reranker = CrossEncoder(self.rerank_model, device=device)
                except Exception as e:
                    print(f"경고: ReRank 모델 초기화 실패: {e}")
                    self._reranker = None
        
        # GraphDB 관련 초기화 (필요한 경우)
        self._graph_rag = None
        if parser_type in ["graph", "both"]:
            try:
                self._graph_rag = GraphRAG(
                    use_graph_rerank=use_graph_rerank,
                    graph_rerank_model=graph_rerank_model,
                    graph_rerank_top_m=graph_rerank_top_m
                )
            except Exception as e:
                print(f"경고: GraphDB 초기화 실패: {e}")
                if parser_type == "graph":
                    raise
    
    def _get_chroma_context(self, query_text: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """ChromaDB에서 컨텍스트 검색"""
        if self._chroma_collection is None or self._chroma_embedder is None:
            return []
        
        # 키워드 추출 사용 여부에 따라 쿼리 텍스트 결정
        if self.use_keyword_extraction:
            query_for_search = extract_keywords_advanced(query_text)
        else:
            query_for_search = query_text
        
        hits = search_chroma(self._chroma_collection, self._chroma_embedder, query_for_search, top_k=top_k)

        # 선택적 ReRank 적용
        if self.use_rerank and self._reranker is not None and hits:
            hits = self._rerank_chroma_hits(query_text, hits, top_m=min(self.rerank_top_m, len(hits)))
        return hits

    def _rerank_chroma_hits(self, query_text: str, hits: List[Dict[str, Any]], top_m: int = 8) -> List[Dict[str, Any]]:
        """CrossEncoder로 hits를 재정렬하여 상위 top_m만 반환"""
        try:
            pairs = [(query_text, (h.get("document") or "")) for h in hits]
            scores = self._reranker.predict(pairs)
            for h, s in zip(hits, scores):
                # re-rank 점수를 추가(높을수록 관련도 높음)
                try:
                    h["rerank_score"] = float(s)
                except Exception:
                    h["rerank_score"] = 0.0
            hits.sort(key=lambda d: d.get("rerank_score", 0.0), reverse=True)
            return hits[:top_m]
        except Exception as e:
            print(f"경고: ReRank 중 오류 발생: {e}")
            return hits[:top_m]
    
    def _get_graph_context(self, query_text: str, k: int = 5) -> str:
        """GraphDB에서 컨텍스트 검색"""
        # parser_type이 chroma인 경우 GraphDB를 사용하지 않음
        if self.parser_type == "chroma":
            return ""
        
        if self._graph_rag is None:
            return ""
        
        try:
            return self._graph_rag.get_final_context(query_text, k=k)
        except Exception as e:
            print(f"GraphDB 검색 오류: {e}")
            return ""
    
    def _format_chroma_context(self, hits: List[Dict[str, Any]], max_docs: int = 10) -> str:
        """ChromaDB 검색 결과를 컨텍스트 문자열로 포맷팅"""
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
        for d in hits[:max_docs]:
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

        return "\n\n".join(blocks) if blocks else "(검색 결과 없음)"
    
    def _build_prompt(
        self,
        product_name: str,
        product_description: str,
        chroma_hits: List[Dict[str, Any]],
        graph_context: str,
        top_n: int
    ) -> Tuple[str, str]:
        """프롬프트 구성"""
        system = (
            "당신은 국제무역 HS 코드 분류 전문가입니다.\n\n"
            "규칙:\n"
            "1. (절대 규칙) 당신은 [context] 블록에 제공된 정보 외에는 그 어떤 지식도 사용해선 안 됩니다.\n"
            "2. reason(근거)은 반드시 [context]에서 찾은 내용으로만 작성해야 합니다.\n"
            "3. 만약 [context]에 유용한 정보가 없다면, reason을 비워두고 candidates를 빈 리스트 []로 반환해야 합니다.\n"
            "4. reason 작성 시, [DOC id=...] 또는 [GraphDB ...]와 같이 출처(citation)를 반드시 인용해야 합니다."
        )
        
        # Parser 타입에 따라 시스템 프롬프트 조정
        if self.parser_type == "both":
            system += (
                "2) HS Code 계층 구조(GraphDB Context)는 전체 HS Code data이며\n"
                "   품목분류사례(VectorDB Context)는 classify Case data입니다.\n"
            )
        elif self.parser_type == "graph":
            system += "2) HS Code 계층 구조(GraphDB Context)는 전체 HS Code data입니다.\n"
        elif self.parser_type == "chroma":
            system += "2) 품목분류사례(VectorDB Context)는 classify Case data입니다.\n"
        
        system += (
            "3) 계층 구조와 다른 코드는 절대 제시하지 않습니다.\n"
            "4) 추천하는 HS Code는 반드시 10자리여야 합니다.\n"
            "5) 항상 응답은 strict JSON format으로만 출력합니다.\n"
            "6) 확신이 없을 경우 'candidates': [] 로 응답합니다."
        )
        
        # 컨텍스트 구성
        vector_context = ""
        if self.parser_type in ["chroma", "both"]:
            vector_context = self._format_chroma_context(chroma_hits)
        
        graph_section = ""
        if self.parser_type in ["graph", "both"]:
            graph_section = graph_context if graph_context.strip() else "(GraphDB 검색 결과 없음)"
        
        # User 프롬프트 구성
        user = f"""
다음 제품의 HS 코드 상위 {top_n} 후보를 추천하세요. 
**중요: 추천하는 모든 HS Code는 반드시 10자리여야 합니다 (예: 9405.40.10.00).**

[입력]
- Product Name: {product_name}
- Product Description: {product_description}

================================================
[context]
"""
        
        # GraphDB 컨텍스트 추가 (있는 경우)
        if self.parser_type in ["graph", "both"]:
            user += f"""[HS Code 계층 구조 Context — GraphDB Retrieved]
(모든 데이터는 HS 공식 nomenclature 기반)
{graph_section}
================================================
"""
        
        # ChromaDB 컨텍스트 추가 (있는 경우)
        if self.parser_type in ["chroma", "both"]:
            user += f"""
[품목분류사례 Context — VectorDB Retrieved]
(정부 품목분류사례 문서 기반 근거 자료)
{vector_context}  
================================================
"""
        
        user += f"""
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
1) 후보는 최대 {top_n}개.
2) hs_code는 반드시 10자리여야 합니다 (예: 9405.40.10.00).
3) citations는 최소 1개 이상 포함.
4) citations.type은 반드시 "graph" 또는 "case"만 가능.
"""
        # ===== (추가!) 평가용 RAG 컨텍스트 문자열 생성 =====
        # (LLM이 본 컨텍스트 원본을 '할루시네이션' 평가에 사용)
        full_rag_context = ""
        if self.parser_type in ["graph", "both"]:
            full_rag_context += f"[GraphDB Context]\n{graph_section}\n\n"
        if self.parser_type in ["chroma", "both"]:
            full_rag_context += f"[ChromaDB Context]\n{vector_context}\n"
        # =================================================
        
        return system, user, full_rag_context.strip() # (수정!) 컨텍스트 원본 추가
    
    def classify_hs_code(
        self,
        product_name: str,
        product_description: str,
        top_n: int = 3,
        chroma_top_k: int = None,
        graph_k: int = 5,
        debug_return_context: bool = False # 추가 평가 스크립트용
    ) -> Dict[str, Any]:
        """
        HS Code 분류 실행
        
        Args:
            product_name: 상품명
            product_description: 상품 설명
            top_n: 추천할 후보 개수
            chroma_top_k: ChromaDB에서 검색할 문서 개수 (기본값: top_n * 3)
            graph_k: GraphDB에서 검색할 후보 개수
            
        Returns:
            Dict[str, Any]: 분류 결과 (JSON 형태)
        """
        original_query_text = f"{product_name}\n{product_description}"
        
        # ChromaDB 검색 (필요한 경우)
        chroma_hits = []
        if self.parser_type in ["chroma", "both"]:
            if chroma_top_k is None:
                chroma_top_k = max(8, top_n * 3)
            chroma_hits = self._get_chroma_context(original_query_text, top_k=chroma_top_k)
        
        # GraphDB 검색 (필요한 경우)
        graph_context = ""
        if self.parser_type in ["graph", "both"]:
            graph_context = self._get_graph_context(original_query_text, k=graph_k)
        
        # 프롬프트 구성
        sys_prompt, user_prompt, full_rag_context = self._build_prompt(
            product_name, product_description, chroma_hits, graph_context, top_n
        )
        
        # LLM 호출
        response = self.client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={"type": "json_object"},
            **({"seed": self.seed} if self.seed is not None else {})
        )
        
        output_text = response.choices[0].message.content.strip()
        
        # JSON 파싱
        result, err = _parse_json_safely(output_text)
        if err:
            result = {"error": err, "raw_output": output_text}
        
        if debug_return_context:
            # evaluate.py가 호출한 경우, 답변과 컨텍스트를 모두 반환
            return result, full_rag_context
        else:
            # API 서버(main.py)가 호출한 경우, 답변만 반환
            return result
    
    def get_enhanced_context(
        self,
        product_name: str,
        product_description: str,
        chroma_top_k: int = 8,
        graph_k: int = 5
    ) -> Dict[str, str]:
        """
        ChromaDB와 GraphDB의 컨텍스트를 모두 가져오는 헬퍼 함수
        
        Args:
            product_name: 상품명
            product_description: 상품설명
            chroma_top_k: ChromaDB에서 검색할 문서 개수
            graph_k: GraphDB에서 검색할 후보 개수
            
        Returns:
            Dict[str, str]: vector_context, graph_context, query_text를 포함한 딕셔너리
        """
        original_query_text = f"{product_name}\n{product_description}"
        
        # ChromaDB 컨텍스트
        chroma_context = ""
        if self.parser_type in ["chroma", "both"]:
            hits = self._get_chroma_context(original_query_text, top_k=chroma_top_k)
            chroma_context = self._format_chroma_context(hits)
        else:
            chroma_context = "(ChromaDB 미사용)"
        
        # GraphDB 컨텍스트
        graph_context = ""
        if self.parser_type in ["graph", "both"]:
            graph_context = self._get_graph_context(original_query_text, k=graph_k)
        else:
            graph_context = "(GraphDB 미사용)"
        
        return {
            "vector_context": chroma_context,
            "graph_context": graph_context,
            "query_text": original_query_text,
            "parser_type": self.parser_type
        }