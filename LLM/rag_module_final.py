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
import random

# 현재 파일의 디렉토리에서 상위 디렉토리로 이동 후 RAG_embedding 폴더 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
rag_embedding_dir = os.path.join(parent_dir, 'RAG_embedding')

# RAG_embedding을 패키지로 import할 수 있도록 상위 경로를 우선 추가
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# 혹시 모듈 직접 경로가 필요할 때를 대비해 하위 디렉터리도 추가
if rag_embedding_dir not in sys.path:
    sys.path.append(rag_embedding_dir)

# 임베딩 & ChromaDB 관련
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import chromadb
from chromadb.config import Settings


# ===== 재현성을 위한 랜덤 시드 설정 =====
def set_all_seeds(seed: int = 42):
    """
    모든 랜덤 시드를 고정하여 재현성 확보
    
    Args:
        seed: 랜덤 시드 값 (기본값: 42)
    """
    # Python 내장 random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # PyTorch 재현성 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Python hash seed (딕셔너리 순서 등에 영향)
    os.environ['PYTHONHASHSEED'] = str(seed)

# GraphDB는 선택적으로 import
GRAPH_AVAILABLE = False
GraphRAG = None

def _try_import_graph_rag():
    """graph_rag 모듈을 다양한 경로에서 시도하여 불러온다."""
    global GraphRAG, GRAPH_AVAILABLE

    # 1) 패키지 경로(RAG_embedding.__init__)에서 우선 시도
    try:
        from RAG_embedding import GraphRAG as _GraphRAG  # type: ignore
        GraphRAG = _GraphRAG
        GRAPH_AVAILABLE = True
        return
    except ImportError:
        pass

    # 2) 모듈 경로(RAG_embedding.graph_rag)로 재시도
    try:
        from RAG_embedding.graph_rag import GraphRAG as _GraphRAG  # type: ignore
        GraphRAG = _GraphRAG
        GRAPH_AVAILABLE = True
        return
    except ImportError as err:
        first_err = err

    # 3) 마지막으로 로컬 모듈 직접 import
    try:
        from graph_rag import GraphRAG as _GraphRAG
        GraphRAG = _GraphRAG
        GRAPH_AVAILABLE = True
        return
    except ImportError as err:
        GRAPH_AVAILABLE = False
        print("경고: GraphRAG 모듈을 찾을 수 없습니다. GraphDB 기능은 사용할 수 없습니다.")
        print(f"세부 정보: {first_err or err}")

_try_import_graph_rag()


# ===== 환경설정 =====
load_dotenv()

# 기본 설정
DEFAULT_CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma_db_openai_large_kw")
DEFAULT_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "hscode_collection")
DEFAULT_NOMENCLATURE_DIR = os.getenv("NOMENCLATURE_DIR", "data/nomenclature_chroma_db")
DEFAULT_NOMENCLATURE_COLLECTION = os.getenv("NOMENCLATURE_COLLECTION", "hscode_nomenclature")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_ENV_MAP = {
    "text-embedding-3-small": os.getenv("OPENAI_SMALL_EMBED_API_KEY"),
    "text-embedding-3-large": os.getenv("OPENAI_LARGE_EMBED_API_KEY"),
}

# 키워드 추출 관련
okt_analyzer = Okt()
STOPWORDS = [
    '의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다',
    '상품명', '설명', '사유', '이름', '제품', '관련', '내용', '항목', '분류', '기준',
    'hs', 'code', 'item', 'des', 'description', 'name'
]


# ===== 유틸리티 함수 ====
def translate_to_korean(text: str, client: OpenAI, model: str = DEFAULT_OPENAI_MODEL) -> str:
    """
    영어 텍스트를 한국어로 번역
    """
    if not text or not text.strip():
        return text

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional translator. "
                        "Translate the given English text to Korean. "
                        "Only return the translated text without any additional explanation or formatting."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Translate the following English text to Korean:\n\n{text}",
                },
            ],
            temperature=0.0,
            max_tokens=1000,
        )
        translated = (response.choices[0].message.content or "").strip()
        return translated
    except Exception as e:
        print(f"경고: 계층 정의 번역 실패, 원문 사용: {e}")
        return text
    
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
def _is_openai_embedding_model(model_name: str) -> bool:
    return model_name.startswith("text-embedding-")


def _resolve_openai_embed_api_key(model_name: str) -> Optional[str]:
    if not _is_openai_embedding_model(model_name):
        return None
    return OPENAI_EMBED_ENV_MAP.get(model_name) or os.getenv("OPENAI_API_KEY")


def _chunk_list(items: List[str], chunk_size: int):
    for idx in range(0, len(items), chunk_size):
        yield items[idx: idx + chunk_size]


class QueryEmbedder:
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.normalize = True

        if _is_openai_embedding_model(model_name):
            self.provider = "openai"
            self._init_openai_embedder()
        else:
            self.provider = "sentence_transformer"
            self.model = SentenceTransformer(model_name, device=device)

    def _init_openai_embedder(self) -> None:
        key = _resolve_openai_embed_api_key(self.model_name)
        if not key:
            raise ValueError(
                f"OpenAI 임베딩 모델 '{self.model_name}' 사용을 위해 OPENAI_API_KEY 또는 "
                f"전용 키(OPENAI_SMALL_EMBED_API_KEY 등)를 설정하세요."
            )
        self.client = OpenAI(api_key=key)
        self.openai_batch = int(os.getenv("OPENAI_EMBED_BATCH", "16"))

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype="float32")

        if self.provider == "openai":
            return self._embed_with_openai(texts)

        vecs = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )
        return np.asarray(vecs, dtype="float32")

    def _embed_with_openai(self, texts: List[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for chunk in _chunk_list(texts, self.openai_batch):
            response = self.client.embeddings.create(
                model=self.model_name,
                input=chunk,
            )
            chunk_vecs = [
                np.asarray(item.embedding, dtype="float32")
                for item in response.data
            ]
            vectors.extend(chunk_vecs)
        return np.stack(vectors, axis=0)


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
    ChromaDB와 GraphDB를 모두 사용
    """
    
    def __init__(
        self,
        chroma_dir: str = None,
        collection_name: str = None,
        nomenclature_dir: str = None,
        nomenclature_collection_name: str = None,
        embed_model: str = None,
        openai_model: str = None,
        openai_api_key: str = None,
        use_keyword_extraction: bool = True,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        use_nomenclature: bool = True
    ):
        """
        Args:
            chroma_dir: ChromaDB 디렉토리 경로 (case data용)
            collection_name: ChromaDB 컬렉션 이름 (case data용)
            nomenclature_dir: Nomenclature ChromaDB 디렉토리 경로
            nomenclature_collection_name: Nomenclature ChromaDB 컬렉션 이름
            embed_model: 임베딩 모델 이름
            openai_model: OpenAI 모델 이름
            openai_api_key: OpenAI API 키
            use_keyword_extraction: ChromaDB 검색 시 키워드 추출 사용 여부 (기본값: True)
            use_nomenclature: Nomenclature ChromaDB 사용 여부 (기본값: True)
        """
        self.use_keyword_extraction = use_keyword_extraction
        
        # GraphDB 가용성 확인
        if not GRAPH_AVAILABLE:
            raise RuntimeError("GraphDB를 사용하려고 했지만 GraphRAG 모듈을 찾을 수 없습니다.")
        
        # 설정값 초기화
        self.chroma_dir = chroma_dir or DEFAULT_CHROMA_DIR
        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME
        self.nomenclature_dir = nomenclature_dir or DEFAULT_NOMENCLATURE_DIR
        self.nomenclature_collection_name = nomenclature_collection_name or DEFAULT_NOMENCLATURE_COLLECTION
        self.embed_model = embed_model or DEFAULT_EMBED_MODEL
        self.openai_model = openai_model or DEFAULT_OPENAI_MODEL
        self.temperature = float(temperature)
        # seed 우선순위: 인자 > 환경변수 > 기본값(42)
        env_seed = os.getenv("SEED", None)
        if seed is not None:
            self.seed = seed
        elif env_seed and str(env_seed).isdigit():
            self.seed = int(env_seed)
        else:
            # 재현성을 위해 기본값 42 사용
            self.seed = 42
        
        # 모든 랜덤 시드 고정
        set_all_seeds(self.seed)
        
        # Nomenclature 사용 여부 설정
        self.use_nomenclature = bool(use_nomenclature)
        
        # OpenAI 클라이언트 초기화
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API 키가 필요합니다. 환경변수 OPENAI_API_KEY를 설정하거나 openai_api_key를 제공하세요.")
        self.client = OpenAI(api_key=api_key)
        
        # ChromaDB 관련 초기화
        self._chroma_embedder = None
        self._chroma_collection = None
        self._nomenclature_collection = None
        try:
            self._chroma_embedder = QueryEmbedder(self.embed_model)
            self._chroma_collection = open_chroma_collection(self.chroma_dir, self.collection_name)
        except Exception as e:
            print(f"경고: ChromaDB 초기화 실패: {e}")
            raise
        
        # Nomenclature ChromaDB 초기화 (use_nomenclature 파라미터로 제어)
        self._nomenclature_collection = None
        if self.use_nomenclature:
            try:
                # nomenclature는 별도의 embedder가 필요할 수 있지만, 같은 모델 사용
                if self._chroma_embedder is None:
                    self._chroma_embedder = QueryEmbedder(self.embed_model)
                self._nomenclature_collection = open_chroma_collection(self.nomenclature_dir, self.nomenclature_collection_name)
                print(f"Nomenclature ChromaDB 초기화 완료: {self.nomenclature_dir}/{self.nomenclature_collection_name}")
            except Exception as e:
                print(f"경고: Nomenclature ChromaDB 초기화 실패 (계속 진행): {e}")
                self._nomenclature_collection = None
        
        # GraphDB 관련 초기화
        self._graph_rag = None
        try:
            self._graph_rag = GraphRAG(
                use_graph_rerank=False,
                graph_rerank_model=None,
                graph_rerank_top_m=5,
                use_graph_hybrid_rrf=False,
                graph_bm25_k=5,
                graph_rrf_k=60,
                graph_embed_model=self.embed_model,
                graph_openai_api_key=_resolve_openai_embed_api_key(self.embed_model)
            )
        except Exception as e:
            print(f"경고: GraphDB 초기화 실패: {e}")
            raise
        
        # HS Code 통칙 로드 및 캐싱
        self.hscode_rules = self._load_hscode_rules()
        # self.hscode_rules = ""
    def _get_chroma_context(self, query_text: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """ChromaDB에서 컨텍스트 검색 (case data용)"""
        if self._chroma_collection is None or self._chroma_embedder is None:
            return []
        
        # 키워드 추출 사용 여부에 따라 쿼리 텍스트 결정
        if self.use_keyword_extraction:
            query_for_search = extract_keywords_advanced(query_text)
        else:
            query_for_search = query_text
        
        hits = search_chroma(self._chroma_collection, self._chroma_embedder, query_for_search, top_k=top_k)
        return hits
    
    def _get_nomenclature_context(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Nomenclature ChromaDB에서 컨텍스트 검색"""
        if self._nomenclature_collection is None or self._chroma_embedder is None:
            return []
        
        # 키워드 추출 사용 여부에 따라 쿼리 텍스트 결정
        if self.use_keyword_extraction:
            query_for_search = extract_keywords_advanced(query_text)
        else:
            query_for_search = query_text
        
        hits = search_chroma(self._nomenclature_collection, self._chroma_embedder, query_for_search, top_k=top_k)
        return hits
    
    def _get_graph_context(self, query_text: str, k: int = 5) -> str:
        """GraphDB에서 컨텍스트 검색"""
        if self._graph_rag is None:
            return ""
        
        try:
            return self._graph_rag.get_final_context(query_text, k=k)
        except Exception as e:
            print(f"GraphDB 검색 오류: {e}")
            return ""
    
    def _get_code_definition(self, code: str) -> str:
        """GraphDB에서 특정 코드의 정의(description)를 가져옴"""
        if self._graph_rag is None or not code:
            return ""

        try:
            # 코드 정규화 (점/하이픈 제거)
            code_clean = code.replace('.', '').replace('-', '')

            # 다양한 표현 형태 시도
            code_variants = [code_clean]

            # 길이에 따라 점 포함 버전도 추가
            if len(code_clean) == 4:
                # 4자리: 94xx 같은 heading
                code_variants.append(code_clean)
            elif len(code_clean) == 6:
                # 6자리: 4+2 형태
                code_variants.append(f"{code_clean[:4]}.{code_clean[4:6]}")
            elif len(code_clean) == 10:
                # 10자리: 4.2.2.2 형태
                code_variants.append(
                    f"{code_clean[:4]}.{code_clean[4:6]}.{code_clean[6:8]}.{code_clean[8:10]}"
                )

            # 중복 제거
            code_variants = list(set(code_variants))
            candidates_str = str(code_variants).replace("'", '"')

            # # 라벨 제한 없이 code 속성만으로 매칭
            # cypher_query = f"""
            # UNWIND {candidates_str} AS code_str
            # MATCH (item {{code: code_str}})
            # RETURN 
            #   item.code AS code,
            #   coalesce(
            #     item.description_ko,
            #     item.description,
            #     item.name_ko,
            #     item.name
            #   ) AS description
            # LIMIT 1
            # """
            cypher_query = f"""
            UNWIND {candidates_str} AS code_str
            MATCH (item:HSItem {{code: code_str}})
            RETURN item.code AS code, item.description AS description
            LIMIT 1
            """

            results = self._graph_rag.graph.query(cypher_query)
            if results and len(results) > 0:
                desc = results[0].get("description", "") or ""
                if desc:
                    # 영어라면 한국어로 번역
                    desc = translate_to_korean(desc, self.client, self.openai_model)
                return desc

            return ""

        except Exception as e:
            print(f"경고: 코드 정의 조회 실패 ({code}): {e}")
            return ""


    def _add_hierarchy_definitions(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """각 candidate의 10자리 코드에서 계층별 정의를 추출하여 추가"""
        if not candidates or self._graph_rag is None:
            return candidates

        for candidate in candidates:
            hs_code = str(candidate.get('hs_code', ''))
            if not hs_code:
                continue

            code_clean = hs_code.replace('.', '').replace('-', '')

            if len(code_clean) < 10:
                continue

            code_2digit = code_clean[:2]
            code_4digit = code_clean[:4]
            code_6digit = f"{code_clean[:4]}.{code_clean[4:6]}"
            code_10digit = f"{code_clean[:4]}.{code_clean[4:6]}.{code_clean[6:8]}.{code_clean[8:10]}"

            definition_2digit = self._get_code_definition(code_2digit)
            definition_4digit = self._get_code_definition(code_4digit)
            definition_6digit = self._get_code_definition(code_6digit)
            definition_10digit = self._get_code_definition(code_10digit)

            candidate['hierarchy_definitions'] = {
                'chapter_2digit': {'code': code_2digit, 'definition': definition_2digit},
                'heading_4digit': {'code': code_4digit, 'definition': definition_4digit},
                'subheading_6digit': {'code': code_6digit, 'definition': definition_6digit},
                'national_10digit': {'code': code_10digit, 'definition': definition_10digit}
            }

        return candidates

    
    def _format_chroma_context(self, hits: List[Dict[str, Any]], max_docs: int = 10) -> str:
        """ChromaDB 검색 결과를 컨텍스트 문자열로 포맷팅 (case data용)"""
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
    
    def _format_nomenclature_context(self, hits: List[Dict[str, Any]], max_docs: int = 5) -> str:
        """Nomenclature ChromaDB 검색 결과를 컨텍스트 문자열로 포맷팅"""
        blocks = []
        for d in hits[:max_docs]:
            body = (d.get("document") or "").strip()
            
            max_chars = 1500
            if len(body) > max_chars:
                body = body[:max_chars] + "…"
            
            dist = d.get("distance", 0.0)
            try:
                dist = float(dist)
            except Exception:
                dist = 0.0
            
            blocks.append(
                f"[NOMENCLATURE id={d.get('id')} dist={dist:.4f}]\n{body}\n"
            )
        
        return "\n\n".join(blocks) if blocks else "(Nomenclature 검색 결과 없음)"
    
    def _load_hscode_rules(self) -> str:
        """HS Code 통칙 파일을 읽어서 반환"""
        rules_file_path = os.path.join(current_dir, 'hscode_rule.txt')
        try:
            with open(rules_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"경고: HS Code 통칙 파일을 찾을 수 없습니다: {rules_file_path}")
            return ""
        except Exception as e:
            print(f"경고: HS Code 통칙 파일 읽기 실패: {e}")
            return ""
    
    def _build_prompt_6digit(
        self,
        product_name: str,
        product_description: str,
        chroma_hits: List[Dict[str, Any]],
        graph_context: str,
        nomenclature_context: str,
        top_n: int
    ) -> Tuple[str, str]:
        """6자리 코드 예측용 프롬프트 구성"""
        system = (
            "당신은 국제무역 HS 코드 분류 전문가입니다.\n\n"
            "규칙:\n"
            "1) 제공된 context 내의 정보만 사용하여 판단합니다.\n"
            "2) HS Code 계층 구조(GraphDB Context)는 전체 HS Code data이며\n"
            "   품목분류사례(VectorDB Context)는 classify Case data이고\n"
            "   Nomenclature Context는 HS 공식 명명법 문서입니다.\n"
        )
        
        system += (
            "3) 계층 구조와 다른 코드는 절대 제시하지 않습니다.\n"
            "4) 추천하는 HS Code는 반드시 6자리여야 합니다 (예: 9405.40).\n"
            "5) 항상 응답은 strict JSON format으로만 출력합니다.\n\n"
            f"HS Code 분류 통칙:\n{self.hscode_rules}\n"
        )
        
        # 컨텍스트 구성
        vector_context = self._format_chroma_context(chroma_hits)
        graph_section = graph_context if graph_context.strip() else "(GraphDB 검색 결과 없음)"
        
        # User 프롬프트 구성
        user = f"""
다음 제품의 HS 코드 상위 {top_n} 후보를 추천하세요. 
**중요: 추천하는 모든 HS Code는 반드시 6자리여야 합니다 (예: 9405.40).**

[입력]
- Product Name: {product_name}
- Product Description: {product_description}

================================================
[context]
"""
        
        # GraphDB 컨텍스트 추가
        user += f"""[HS Code 계층 구조 Context — GraphDB Retrieved]
(모든 데이터는 HS 공식 nomenclature 기반)
{graph_section}
================================================
"""
        
        # ChromaDB 컨텍스트 추가
        user += f"""
[품목분류사례 Context — VectorDB Retrieved]
(정부 품목분류사례 문서 기반 근거 자료)
{vector_context}  
================================================
"""
        
        # Nomenclature 컨텍스트 추가 (항상 추가)
        if nomenclature_context and nomenclature_context != "(Nomenclature 검색 결과 없음)":
            user += f"""
[Nomenclature Context — HS 공식 명명법 문서]
(HS Code 공식 Nomenclature 문서 기반)
{nomenclature_context}
================================================
"""
        
        user += f"""
[응답 형식: strict JSON — 추가 키 금지]
{{
  "candidates": [
    {{
      "hs_code": "string",          // 반드시 6자리 HS Code (예: 9405.40)
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
1) 후보는 최소 {top_n}개 이상이어야 합니다. {top_n}개 미만이면 안 됩니다. (더 많이 제시해도 됩니다)
2) hs_code는 반드시 6자리여야 합니다 (예: 9405.40).
3) **중요: 모든 후보의 hs_code는 서로 달라야 합니다. 중복된 6자리 코드는 제시하지 마세요.**
4) citations는 최소 1개 이상 포함.
5) citations.type은 반드시 "graph" 또는 "case"만 가능.
"""
        return system, user
    
    def _build_prompt_10digit_hierarchical(
        self,
        product_name: str,
        product_description: str,
        chroma_hits: List[Dict[str, Any]],
        graph_context: str,
        ten_digit_context: str,
        nomenclature_context: str,
        top_n: int
    ) -> Tuple[str, str]:
        """10자리 코드 예측용 계층적 프롬프트 구성 (2단계)"""
        system = (
            "당신은 국제무역 HS 코드 분류 전문가입니다.\n\n"
            "규칙:\n"
            "1) 제공된 context 내의 정보만 사용하여 판단합니다.\n"
            "2) HS Code 계층 구조(GraphDB Context)는 전체 HS Code data이며\n"
            "   품목분류사례(VectorDB Context)는 classify Case data이고\n"
            "   Nomenclature Context는 HS 공식 명명법 문서입니다.\n"
            "3) 계층 구조와 다른 코드는 절대 제시하지 않습니다.\n"
            "4) 추천하는 HS Code는 반드시 10자리여야 합니다 (예: 9405.40-1000).\n"
            "5) **중요: 우선적으로 '10자리 HS Code 후보' 컨텍스트에 있는 코드를 추천하되, "
            "해당 컨텍스트에 적합한 코드가 없으면 전체 GraphDB Context에서 적절한 코드를 찾아 추천할 수 있습니다.**\n"
            "6) 항상 응답은 strict JSON format으로만 출력합니다.\n\n"
            f"HS Code 분류 통칙:\n{self.hscode_rules}\n"
        )
        
        # 컨텍스트 구성
        vector_context = self._format_chroma_context(chroma_hits)
        graph_section = graph_context if graph_context.strip() else "(GraphDB 검색 결과 없음)"
        
        # User 프롬프트 구성
        user = f"""
다음 제품의 HS 코드 상위 {top_n} 후보를 추천하세요. 
**중요: 추천하는 모든 HS Code는 반드시 10자리여야 합니다.**
**우선적으로 '10자리 HS Code 후보' 컨텍스트에 있는 코드를 추천하되, 적절한 코드가 없으면 전체 GraphDB Context에서 찾아 추천할 수 있습니다.**

[입력]
- Product Name: {product_name}
- Product Description: {product_description}

================================================
[context]
"""
        
        # GraphDB 컨텍스트 추가
        user += f"""[HS Code 계층 구조 Context — GraphDB Retrieved]
(모든 데이터는 HS 공식 nomenclature 기반)
{graph_section}
================================================
"""
        
        # ChromaDB 컨텍스트 추가
        user += f"""
[품목분류사례 Context — VectorDB Retrieved]
(정부 품목분류사례 문서 기반 근거 자료)
{vector_context}  
================================================
"""
        
        # Nomenclature 컨텍스트 추가 (항상 추가)
        if nomenclature_context and nomenclature_context != "(Nomenclature 검색 결과 없음)":
            user += f"""
[Nomenclature Context — HS 공식 명명법 문서]
(HS Code 공식 Nomenclature 문서 기반)
{nomenclature_context}
================================================
"""
        
        # 10자리 코드 후보 컨텍스트 추가
        user += f"""
[10자리 HS Code 후보 — 1단계에서 예측된 6자리 코드의 하위 코드]
{ten_digit_context}
================================================
"""
        
        user += f"""
[응답 형식: strict JSON — 추가 키 금지]
{{
  "candidates": [
    {{
      "hs_code": "string",          // 반드시 10자리 HS Code (예: 9405.40-1000)
      "title": "string",
      "reason": "string",           // 한국어
      "citations": [
        {{"type": "graph", "code": "string"}},   // GraphDB 근거
        {{"type": "case", "doc_id": "string"}}   // VectorDB 근거
      ]
    }}
  ]
}}

필수 규칙:
1) 후보는 최대 {top_n}개.
2) hs_code는 반드시 10자리여야 합니다 (예: 9405.40-1000).
3) 우선적으로 '10자리 HS Code 후보' 컨텍스트에 있는 코드를 선택하세요. 해당 컨텍스트에 적절한 코드가 없으면 전체 GraphDB Context에서 찾아 추천할 수 있습니다.
4) citations는 최소 1개 이상 포함
5) citations.type은 반드시 "graph" 또는 "case"만 가능
6) reason은 추천한 코드에 대한 정의와 사용자의 상품에 대한 비교를 기반으로 해당 코드를 추천한 이유를 길고 자세하게 작성
"""
        return system, user
    
    def classify_hs_code_hierarchical(
        self,
        product_name: str,
        product_description: str,
        top_n: int = 5,
        chroma_top_k: int = None,
        graph_k: int = 5
    ) -> Dict[str, Any]:
        """
        계층적 2단계 RAG를 통한 HS Code 분류
        1단계: 6자리 HS Code 예측
        2단계: 예측된 6자리 코드의 하위 10자리 코드만 컨텍스트로 추가하여 최종 10자리 예측
        
        Args:
            product_name: 상품명
            product_description: 상품 설명
            top_n: 추천할 후보 개수 (기본값: 5)
            chroma_top_k: ChromaDB에서 검색할 문서 개수 (기본값: top_n * 3)
            graph_k: GraphDB에서 검색할 후보 개수 (기본값: 5)
            
        Returns:
            Dict[str, Any]: 분류 결과 (JSON 형태)
        """
        original_query_text = f"{product_name}\n{product_description}"
        
        print("=== 1단계: 6자리 HS Code 예측 ===")
        
        # ChromaDB 검색
        chroma_hits = []
        if chroma_top_k is None:
            chroma_top_k = max(8, top_n * 3)
        chroma_hits = self._get_chroma_context(original_query_text, top_k=chroma_top_k)

        # GraphDB 검색
        graph_context = self._get_graph_context(original_query_text, k=graph_k)
        
        # Nomenclature ChromaDB 검색 (항상 시도)
        nomenclature_hits = self._get_nomenclature_context(original_query_text, top_k=5)
        nomenclature_context = self._format_nomenclature_context(nomenclature_hits, max_docs=5)
        
        # 1단계: 6자리 코드 예측 - LLM이 n개 미만을 반환하는 경우를 대비해 더 많은 후보 요청
        # 중복이 있을 수 있으므로 top_n * 2를 요청하고, 중복 제거 후 정확히 top_n개 선택
        request_count = max(top_n * 2, top_n + 5)  # 최소 top_n + 5개 요청
        
        # 1단계: 6자리 코드 예측 프롬프트 구성
        sys_prompt_6digit, user_prompt_6digit = self._build_prompt_6digit(
            product_name, product_description, chroma_hits, graph_context, nomenclature_context, request_count
        )
        
        # 1단계 LLM 호출
        response_6digit = self.client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": sys_prompt_6digit},
                {"role": "user", "content": user_prompt_6digit}
            ],
            temperature=self.temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={"type": "json_object"},
            **({"seed": self.seed} if self.seed is not None else {})
        )
        
        output_6digit = response_6digit.choices[0].message.content.strip()
        result_6digit, err_6digit = _parse_json_safely(output_6digit)
        
        if err_6digit:
            return {"error": f"1단계 6자리 예측 실패: {err_6digit}", "raw_output": output_6digit}
        
        # 예측된 6자리 코드 추출
        candidates_6digit = result_6digit.get('candidates', [])
        if not candidates_6digit:
            return {"error": "1단계에서 6자리 코드 후보를 찾을 수 없습니다.", "step1_result": result_6digit}
        
        # 6자리 코드 리스트 추출 (점 제거 후 6자리 확인, 중복 제거, 정확히 top_n개)
        six_digit_codes = []
        seen_codes = set()  # 중복 제거를 위한 set
        
        for cand in candidates_6digit:
            # 이미 top_n개를 채웠으면 중단
            if len(six_digit_codes) >= top_n:
                break
                
            hs_code = str(cand.get('hs_code', '')).replace('.', '').replace('-', '')
            if len(hs_code) == 6:
                # 'XXXX.XX' 형식으로 변환
                formatted = f"{hs_code[:4]}.{hs_code[4:6]}"
                # 중복 제거: 이미 본 코드가 아니면 추가
                if formatted not in seen_codes:
                    seen_codes.add(formatted)
                    six_digit_codes.append(formatted)
            elif '.' in str(cand.get('hs_code', '')):
                # 이미 포맷된 경우
                code_str = str(cand.get('hs_code', ''))
                code_clean = code_str.replace('.', '').replace('-', '')
                if len(code_clean) == 6:
                    formatted = f"{code_clean[:4]}.{code_clean[4:6]}"
                    # 중복 제거: 이미 본 코드가 아니면 추가
                    if formatted not in seen_codes:
                        seen_codes.add(formatted)
                        six_digit_codes.append(formatted)
        
        if not six_digit_codes:
            return {"error": "유효한 6자리 코드를 추출할 수 없습니다.", "step1_result": result_6digit}
        
        # 정확히 top_n개가 아니면 재시도 또는 경고
        if len(six_digit_codes) < top_n:
            print(f"경고: 6자리 코드가 {len(six_digit_codes)}개만 추출되었습니다. 요청한 {top_n}개보다 적습니다.")
            print(f"      LLM이 충분한 후보를 반환하지 않았거나, 중복이 많았을 수 있습니다.")
            print(f"      현재 {len(six_digit_codes)}개로 진행합니다.")
            # 부족한 경우라도 진행 (에러는 아님)
        
        # 상위 단계에서 더 많은 후보 유지 (오류 전파 방지: top_n * 2까지 유지)
        # 정답이 top_n에 없어도 top_n * 2에 있으면 복구 가능
        max_codes_to_keep = max(top_n * 2, top_n + 5)
        six_digit_codes = six_digit_codes[:max_codes_to_keep]
        
        print(f"예측된 6자리 코드 (중복 제거 후, 최종 {len(six_digit_codes)}개, 상위 {top_n}개 우선 사용): {six_digit_codes}")
        
        # 2단계: 예측된 6자리 코드의 하위 10자리 코드 조회
        print("=== 2단계: 10자리 HS Code 예측 ===")
        
        if self._graph_rag is None:
            return {"error": "GraphDB가 필요합니다."}
        
        try:
            # 상위 top_n개만 우선 사용, 나머지는 백업으로 유지
            ten_digit_context = self._graph_rag.get_10digit_codes_from_6digit(six_digit_codes[:top_n])
        except Exception as e:
            return {"error": f"10자리 코드 조회 실패: {e}", "step1_result": result_6digit}
        
        # 하위 10자리 코드가 없어도 전체 GraphDB 컨텍스트로 폴백 가능
        if not ten_digit_context or "(하위 10자리 코드를 찾을 수 없습니다)" in ten_digit_context:
            print("경고: 예측된 6자리 코드의 하위 10자리 코드를 찾을 수 없습니다. 전체 GraphDB 컨텍스트로 폴백합니다.")
            ten_digit_context = "(하위 10자리 코드를 찾을 수 없어 전체 GraphDB 컨텍스트를 참고하세요.)"
        
        # 2단계: 10자리 코드 예측 프롬프트 구성
        
        sys_prompt_10digit, user_prompt_10digit = self._build_prompt_10digit_hierarchical(
            product_name, product_description, chroma_hits, graph_context, ten_digit_context, nomenclature_context, top_n
        )
        
        # 2단계 LLM 호출
        response_10digit = self.client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": sys_prompt_10digit},
                {"role": "user", "content": user_prompt_10digit}
            ],
            temperature=self.temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={"type": "json_object"},
            **({"seed": self.seed} if self.seed is not None else {})
        )
        
        output_10digit = response_10digit.choices[0].message.content.strip()
        result_10digit, err_10digit = _parse_json_safely(output_10digit)
        
        if err_10digit:
            return {
                "error": f"2단계 10자리 예측 실패: {err_10digit}",
                "raw_output": output_10digit,
                "step1_result": result_6digit
            }
        
        # 각 candidate에 계층별 정의 추가
        if "candidates" in result_10digit and isinstance(result_10digit["candidates"], list):
            result_10digit["candidates"] = self._add_hierarchy_definitions(result_10digit["candidates"])

        
        # context 정보 추가 (1단계: 6자리 예측용 context)
        vector_context_step1 = self._format_chroma_context(chroma_hits)
        result_10digit["chromaDB_context_step1"] = vector_context_step1
        result_10digit["graphDB_context_step1"] = graph_context
        
        # context 정보 추가 (2단계: 10자리 예측용 context)
        vector_context_step2 = self._format_chroma_context(chroma_hits)
        result_10digit["chromaDB_context_step2"] = vector_context_step2
        # graph_context는 이미 문자열이고, ten_digit_context가 추가로 포함됨
        final_graph_context = graph_context
        if ten_digit_context and ten_digit_context != "(하위 10자리 코드를 찾을 수 없어 전체 GraphDB 컨텍스트를 참고하세요.)":
            final_graph_context = f"{graph_context}\n\n=== 예측된 6자리 코드의 하위 10자리 코드 ===\n{ten_digit_context}"
        result_10digit["graphDB_context_step2"] = final_graph_context
        
        # 하위 호환성을 위해 기존 필드명도 유지 (2단계 context)
        result_10digit["chromaDB_context"] = vector_context_step2
        result_10digit["graphDB_context"] = final_graph_context

        # Nomenclature 컨텍스트 공유 (Stage1/Stage2)
        # nom_ctx = nomenclature_context if self.use_nomenclature else ""
        # result_10digit["nomenclature_context_step1"] = nom_ctx
        # result_10digit["nomenclature_context_step2"] = nom_ctx
        # result_10digit["nomenclature_context"] = nom_ctx        
        return result_10digit
    
    def get_enhanced_context(
        self,
        product_name: str,
        product_description: str,
        chroma_top_k: int = 8,
        graph_k: int = 5
    ) -> Dict[str, str]:
        """
        ChromaDB, GraphDB, Nomenclature ChromaDB의 컨텍스트를 모두 가져오는 헬퍼 함수
        
        Args:
            product_name: 상품명
            product_description: 상품설명
            chroma_top_k: ChromaDB에서 검색할 문서 개수
            graph_k: GraphDB에서 검색할 후보 개수
            
        Returns:
            Dict[str, str]: vector_context, graph_context, nomenclature_context, query_text를 포함한 딕셔너리
        """
        original_query_text = f"{product_name}\n{product_description}"
        
        # ChromaDB 컨텍스트
        hits = self._get_chroma_context(original_query_text, top_k=chroma_top_k)
        chroma_context = self._format_chroma_context(hits)
        
        # GraphDB 컨텍스트
        graph_context = self._get_graph_context(original_query_text, k=graph_k)
        
        # Nomenclature ChromaDB 컨텍스트 (항상 시도)
        nomenclature_context = ""
        if self._nomenclature_collection is not None:
            nomenclature_hits = self._get_nomenclature_context(original_query_text, top_k=5)
            nomenclature_context = self._format_nomenclature_context(nomenclature_hits, max_docs=5)
        else:
            nomenclature_context = "(Nomenclature ChromaDB 미사용)"
        
        return {
            "vector_context": chroma_context,
            "graph_context": graph_context,
            "nomenclature_context": nomenclature_context,
            "query_text": original_query_text
        }
