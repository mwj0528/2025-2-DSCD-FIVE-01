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
from rank_bm25 import BM25Okapi
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
from sentence_transformers import CrossEncoder
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


# ===== Parser 타입 정의 =====
ParserType = Literal["chroma", "graph", "both"]


# ===== 환경설정 =====
load_dotenv()

# 기본 설정
DEFAULT_CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma_db_openai_large_kw")
DEFAULT_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "hscode_collection")
DEFAULT_NOMENCLATURE_DIR = os.getenv("NOMENCLATURE_DIR", "data/nomenclature_chroma_db")
DEFAULT_NOMENCLATURE_COLLECTION = os.getenv("NOMENCLATURE_COLLECTION", "hscode_nomenclature")
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
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


# ===== 유틸리티 함수 =====
def translate_to_english(text: str, client: OpenAI, model: str = "gpt-4o-mini") -> str:
    """
    한국어 텍스트를 영어로 번역
    
    Args:
        text: 번역할 텍스트
        client: OpenAI 클라이언트
        model: 사용할 OpenAI 모델
        
    Returns:
        str: 번역된 영어 텍스트
    """
    if not text or not text.strip():
        return text
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional translator. Translate the given Korean text to English. Only return the translated text without any additional explanation or formatting."
                },
                {
                    "role": "user",
                    "content": f"Translate the following Korean text to English:\n\n{text}"
                }
            ],
            temperature=0.0,
            max_tokens=1000
        )
        translated = response.choices[0].message.content.strip()
        return translated
    except Exception as e:
        print(f"경고: 번역 실패, 원본 텍스트 사용: {e}")
        return text

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
    Parser 설정을 통해 ChromaDB, GraphDB, 또는 둘 다 선택적으로 사용 가능
    """
    
    def __init__(
        self,
        parser_type: ParserType = "both",
        chroma_dir: str = None,
        collection_name: str = None,
        nomenclature_dir: str = None,
        nomenclature_collection_name: str = None,
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
        # Listwise LLM-as-Reranker (Sliding Window)
        use_llm_rerank_listwise: bool = False,
        llm_rerank_window: int = 10,
        llm_rerank_step: int = 5,
        llm_rerank_max_candidates: int = 16,
        llm_rerank_top_m: int = 5,
        # Hybrid Search (Semantic K + BM25 K → RRF) 옵션
        use_rrf_hybrid: bool = False,
        bm25_k: int = 8,
        rrf_k: int = 60,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        translate_to_english: bool = False,
        use_nomenclature: bool = True
    ):
        """
        Args:
            parser_type: 사용할 DB 설정 ("chroma", "graph", "both")
            chroma_dir: ChromaDB 디렉토리 경로 (case data용)
            collection_name: ChromaDB 컬렉션 이름 (case data용)
            nomenclature_dir: Nomenclature ChromaDB 디렉토리 경로
            nomenclature_collection_name: Nomenclature ChromaDB 컬렉션 이름
            embed_model: 임베딩 모델 이름
            openai_model: OpenAI 모델 이름
            openai_api_key: OpenAI API 키
            use_keyword_extraction: ChromaDB 검색 시 키워드 추출 사용 여부 (기본값: True)
            translate_to_english: 사용자 입력을 영어로 번역하여 RAG 검색 수행 (기본값: False)
            use_nomenclature: Nomenclature ChromaDB 사용 여부 (기본값: True)
        """
        # Parser 타입 설정
        if parser_type not in ["chroma", "graph", "both"]:
            raise ValueError(f"parser_type은 'chroma', 'graph', 'both' 중 하나여야 합니다. 입력값: {parser_type}")
        
        self.parser_type = parser_type
        self.use_keyword_extraction = use_keyword_extraction
        self.use_rerank = bool(use_rerank)
        self.use_rrf_hybrid = bool(use_rrf_hybrid)
        
        # GraphDB 사용 시 가용성 확인
        if parser_type in ["graph", "both"]:
            if not GRAPH_AVAILABLE:
                raise RuntimeError("GraphDB를 사용하려고 했지만 GraphRAG 모듈을 찾을 수 없습니다.")
        
        # 설정값 초기화
        self.chroma_dir = chroma_dir or DEFAULT_CHROMA_DIR
        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME
        self.nomenclature_dir = nomenclature_dir or DEFAULT_NOMENCLATURE_DIR
        self.nomenclature_collection_name = nomenclature_collection_name or DEFAULT_NOMENCLATURE_COLLECTION
        self.embed_model = embed_model or DEFAULT_EMBED_MODEL
        self.openai_model = openai_model or DEFAULT_OPENAI_MODEL
        self.rerank_model = rerank_model or DEFAULT_RERANK_MODEL
        self.rerank_top_m = max(1, int(rerank_top_m)) if isinstance(rerank_top_m, int) else 8
        # LLM Listwise Rerank 설정
        self.use_llm_rerank_listwise = bool(use_llm_rerank_listwise)
        self.llm_rerank_window = max(2, int(llm_rerank_window))
        self.llm_rerank_step = max(1, int(llm_rerank_step))
        self.llm_rerank_max_candidates = max(2, int(llm_rerank_max_candidates))
        self.llm_rerank_top_m = max(1, int(llm_rerank_top_m))
        # BM25 / RRF 파라미터
        try:
            self.bm25_k = max(1, int(bm25_k))
        except Exception:
            self.bm25_k = 8
        try:
            self.rrf_k = max(1, int(rrf_k))
        except Exception:
            self.rrf_k = 60
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
        
        # 번역 옵션 설정
        self.translate_to_english = bool(translate_to_english)
        
        # Nomenclature 사용 여부 설정
        self.use_nomenclature = bool(use_nomenclature)
        
        # OpenAI 클라이언트 초기화
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API 키가 필요합니다. 환경변수 OPENAI_API_KEY를 설정하거나 openai_api_key를 제공하세요.")
        self.client = OpenAI(api_key=api_key)
        
        # ChromaDB 관련 초기화 (필요한 경우)
        self._chroma_embedder = None
        self._chroma_collection = None
        self._nomenclature_collection = None
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
        
        # GraphDB 관련 초기화 (필요한 경우)
        self._graph_rag = None
        if parser_type in ["graph", "both"]:
            try:
                self._graph_rag = GraphRAG(
                    use_graph_rerank=use_graph_rerank,
                    graph_rerank_model=graph_rerank_model,
                    graph_rerank_top_m=graph_rerank_top_m,
                    use_graph_hybrid_rrf=getattr(self, 'use_rrf_hybrid', False),
                    graph_bm25_k=getattr(self, 'bm25_k', 5),
                    graph_rrf_k=getattr(self, 'rrf_k', 60),
                    graph_embed_model=self.embed_model,
                    graph_openai_api_key=_resolve_openai_embed_api_key(self.embed_model)
                )
            except Exception as e:
                print(f"경고: GraphDB 초기화 실패: {e}")
                if parser_type == "graph":
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

        # Hybrid Search: Semantic K + BM25 K → RRF 융합
        if self.use_rrf_hybrid:
            try:
                bm25_hits = self._bm25_search(query_for_search, top_k=self.bm25_k)
                hits = self._rrf_fuse(hits, bm25_hits, k=self.rrf_k, top_k=top_k)
            except Exception as e:
                print(f"경고: RRF 하이브리드 실패, semantic만 사용: {e}")

        # 선택적 ReRank 적용
        if self.use_rerank and self._reranker is not None and hits:
            hits = self._rerank_chroma_hits(query_text, hits, top_m=min(self.rerank_top_m, len(hits)))

        # 선택적 LLM Listwise 재랭킹 (슬라이딩 윈도우)
        if self.use_llm_rerank_listwise and hits:
            hits = self._llm_listwise_rerank(
                product_query=query_text,
                hits=hits,
                max_candidates=min(self.llm_rerank_max_candidates, len(hits)),
                window=self.llm_rerank_window,
                step=self.llm_rerank_step,
                top_m=min(self.llm_rerank_top_m, len(hits))
            )
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

    # ===== BM25 / RRF Hybrid 구현 =====
    def _ensure_bm25_index(self):
        if getattr(self, '_bm25_ready', False):
            return
        if self._chroma_collection is None:
            raise RuntimeError("BM25 인덱스를 만들기 위해 Chroma 컬렉션이 필요합니다.")
        # 전체 문서 로딩 (배치 반복)
        total = self._chroma_collection.count()
        corpus_texts = []
        corpus_ids = []
        corpus_metas = []
        batch = 500
        for offset in range(0, total, batch):
            res = self._chroma_collection.get(include=["documents", "metadatas"], limit=batch, offset=offset)
            docs = res.get("documents", [])
            metas = res.get("metadatas", [])
            ids = res.get("ids", [])
            for i, txt in enumerate(docs):
                corpus_texts.append(txt or "")
                corpus_ids.append(ids[i] if i < len(ids) else None)
                corpus_metas.append(metas[i] if i < len(metas) else {})
        # 토큰화: JVM 비의존 정규식 기반
        tokenized_corpus = [self._bm25_tokenize(t) for t in corpus_texts]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._bm25_corpus = {
            i: {"id": corpus_ids[i], "text": corpus_texts[i], "meta": corpus_metas[i]}
            for i in range(len(corpus_texts))
        }
        self._bm25_ready = True

    def _bm25_search(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        self._ensure_bm25_index()
        q_tokens = self._bm25_tokenize(query_text)
        scores = self._bm25.get_scores(q_tokens)
        # 상위 top_k 인덱스 추출
        idx_sorted = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, i in enumerate(idx_sorted):
            item = self._bm25_corpus.get(int(i))
            if not item:
                continue
            results.append({
                "id": item["id"],
                "document": item["text"],
                "metadata": item["meta"],
                "bm25_score": float(scores[int(i)]),
                "rank": rank + 1
            })
        return results

    def _bm25_tokenize(self, text: str) -> List[str]:
        """BM25 전용 경량 토크나이저: 소문자화 + 한글/영문/숫자 토큰, 길이>=2"""
        if not text:
            return []
        s = str(text).lower()
        # 단위/기호 간단 정규화(선택적으로 확장 가능)
        s = re.sub(r"[%㎜]+", " ", s)
        tokens = re.findall(r"[a-z0-9가-힣]+", s)
        return [t for t in tokens if len(t) >= 2]

    def _rrf_fuse(self, semantic_hits: List[Dict[str, Any]], bm25_hits: List[Dict[str, Any]], k: int = 60, top_k: int = 8) -> List[Dict[str, Any]]:
        # 랭크 부여
        sem_rank = { (h.get('id')): (r+1) for r, h in enumerate(semantic_hits) }
        bm25_rank = { (h.get('id')): (r+1) for r, h in enumerate(bm25_hits) }
        all_ids = set([h.get('id') for h in semantic_hits if h.get('id') is not None]) | set([h.get('id') for h in bm25_hits if h.get('id') is not None])
        fused = []
        # id → 대표 문서 병합
        id_to_doc = {}
        for h in semantic_hits:
            id_to_doc[h.get('id')] = h
        for h in bm25_hits:
            if h.get('id') not in id_to_doc:
                id_to_doc[h.get('id')] = h
        for doc_id in all_ids:
            r1 = sem_rank.get(doc_id)
            r2 = bm25_rank.get(doc_id)
            score = 0.0
            if r1 is not None:
                score += 1.0 / (k + r1)
            if r2 is not None:
                score += 1.0 / (k + r2)
            d = dict(id_to_doc.get(doc_id) or {})
            d['rrf_score'] = float(score)
            fused.append(d)
        fused.sort(key=lambda x: x.get('rrf_score', 0.0), reverse=True)
        return fused[:top_k]

    # ===== Listwise LLM-as-Reranker (Sliding Window) =====
    def _format_listwise_prompt(self, product_query: str, items: List[Dict[str, Any]]) -> Tuple[str, str]:
        system = (
            "당신은 국제무역 HS 코드 분류 전문가입니다. 주어진 후보를 쿼리와의 관련성에 따라 내림차순으로 재정렬하세요. "
            "계층 일치, 10자리 코드 일관성, 근거-입력 정합성을 중시합니다. 반드시 JSON만 출력합니다."
        )
        # 후보를 [id]로 표기
        lines = ["주어진 쿼리와의 관련성에 따라 다음 문서들의 순위를 매기세요."]
        for it in items:
            doc = (it.get("document") or "").strip()
            if len(doc) > 500:
                doc = doc[:500] + "…"
            name = (it.get("metadata") or {}).get("상품명") or (it.get("metadata") or {}).get("title") or ""
            hs = (it.get("metadata") or {}).get("HSCode") or (it.get("metadata") or {}).get("hs_code") or ""
            lines.append(f"[{it['id']}] HS: {hs} | {name}\n{doc}")
        lines.append("\n쿼리:\n" + product_query)
        lines.append("\n순위는 식별자를 사용하여 내림차순으로 나열하세요. JSON만 출력:\n{\"ranking\": [\"[id]\", ...]}")
        user = "\n".join(lines)
        return system, user

    def _llm_listwise_rerank(
        self,
        product_query: str,
        hits: List[Dict[str, Any]],
        max_candidates: int,
        window: int,
        step: int,
        top_m: int
    ) -> List[Dict[str, Any]]:
        # 상위 max_candidates만 사용
        pool = hits[:max_candidates]
        if len(pool) <= 1:
            return pool

        # 윈도우는 뒤에서 앞으로 이동 (끝쪽부터 더 정밀 평가)
        M = len(pool)
        start_indices = list(range(max(0, M - window), -1, -step))
        # Borda 점수 집계
        id_to_borda = { (d.get('id')): 0.0 for d in pool }
        id_to_doc = { d.get('id'): d for d in pool }

        for start in start_indices:
            end = min(M, start + window)
            window_items = pool[start:end]
            if len(window_items) <= 1:
                continue

            sys_prompt, user_prompt = self._format_listwise_prompt(product_query, window_items)
            resp = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                top_p=1,
                response_format={"type": "json_object"},
            )
            out = (resp.choices[0].message.content or "").strip()
            parsed, err = _parse_json_safely(out)
            if err or not isinstance(parsed, dict) or "ranking" not in parsed:
                # 실패 시 현재 윈도우는 스킵 (전역 폴백은 원순위 유지)
                continue
            ranking_ids = parsed.get("ranking") or []
            # 기대 형식: ["[id]", "[id2]"] 혹은 raw id
            ranked_clean = []
            for rid in ranking_ids:
                if not isinstance(rid, str):
                    continue
                m = re.findall(r"\[(.+?)\]", rid)
                ranked_clean.append(m[0] if m else rid)

            # Borda: 윈도우 길이 기준 점수 부여
            L = len(window_items)
            local_rank = { (d.get('id')): (L - idx) for idx, d in enumerate(window_items) }
            for idx, doc_id in enumerate(ranked_clean):
                if doc_id in id_to_borda:
                    # 높은 순위에 더 많은 점수 (L, L-1, ...)
                    id_to_borda[doc_id] += (L - idx)
            # 랭크에 없는 항목은 로컬 기본 점수 부여(가벼운 패널티)
            for did in local_rank:
                if did not in ranked_clean:
                    id_to_borda[did] += 1.0

        # 최종 정렬
        fused = sorted(pool, key=lambda d: id_to_borda.get(d.get('id'), 0.0), reverse=True)
        return fused[:top_m]

    def _rerank_chroma_hits(self, query_text: str, hits: List[Dict[str, Any]], top_m: int = 8) -> List[Dict[str, Any]]:
        """CrossEncoder로 hits를 재정렬하여 상위 top_m만 반환"""
        if self._reranker is None:
            return hits[:top_m]
        
        try:
            pairs = [(query_text, (h.get("document") or "")) for h in hits]
            # 메모리 절약: 작은 배치 크기 사용 (기본값 32에서 4로 감소)
            # 환경변수 RERANK_BATCH_SIZE로 조절 가능 (기본: 4)
            batch_size = int(os.getenv("RERANK_BATCH_SIZE", "4"))
            scores = self._reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
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

            # 라벨 제한 없이 code 속성만으로 매칭
            cypher_query = f"""
            UNWIND {candidates_str} AS code_str
            MATCH (item {{code: code_str}})
            RETURN 
              item.code AS code,
              coalesce(
                item.description_ko,
                item.description,
                item.name_ko,
                item.name
              ) AS description
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
    
    def _build_prompt(
        self,
        product_name: str,
        product_description: str,
        chroma_hits: List[Dict[str, Any]],
        graph_context: str,
        nomenclature_context: str,
        top_n: int
    ) -> Tuple[str, str]:
        """프롬프트 구성"""
        system = (
            "당신은 국제무역 HS 코드 분류 전문가입니다.\n\n"
            "규칙:\n"
            "1) 제공된 context 내의 정보만 사용하여 판단합니다.\n"
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
            "5) 항상 응답은 strict JSON format으로만 출력합니다.\n\n"
            f"HS Code 분류 통칙:\n{self.hscode_rules}\n"
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
중요: 추천하는 모든 HS Code는 반드시 10자리여야 합니다 (예: 9405.40-1000).**

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
3) citations는 최소 1개 이상 포함.
4) citations.type은 반드시 "graph" 또는 "case"만 가능.
"""
        return system, user
    
    def _build_prompt_4digit(
        self,
        product_name: str,
        product_description: str,
        chroma_hits: List[Dict[str, Any]],
        graph_context: str,
        nomenclature_context: str,
        top_n: int
    ) -> Tuple[str, str]:
        """4자리 코드 예측용 프롬프트 구성"""
        system = (
            "당신은 국제무역 HS 코드 분류 전문가입니다.\n\n"
            "HS Code 통칙: \n"
            
            "규칙:\n"
            "1) 제공된 context 내의 정보만 사용하여 판단합니다.\n"
        )
        
        # Parser 타입에 따라 시스템 프롬프트 조정
        if self.parser_type == "both":
            system += (
                "2) HS Code 계층 구조(GraphDB Context)는 전체 HS Code data이며\n"
                "   품목분류사례(VectorDB Context)는 classify Case data이고\n"
                "   Nomenclature Context는 HS 공식 명명법 문서입니다.\n"
            )
        elif self.parser_type == "graph":
            system += (
                "2) HS Code 계층 구조(GraphDB Context)는 전체 HS Code data이며\n"
                "   Nomenclature Context는 HS 공식 명명법 문서입니다.\n"
            )
        elif self.parser_type == "chroma":
            system += (
                "2) 품목분류사례(VectorDB Context)는 classify Case data이며\n"
                "   Nomenclature Context는 HS 공식 명명법 문서입니다.\n"
            )
        
        system += (
            "3) 계층 구조와 다른 코드는 절대 제시하지 않습니다.\n"
            "4) 추천하는 HS Code는 반드시 4자리여야 합니다 (예: 9405).\n"
            "5) 항상 응답은 strict JSON format으로만 출력합니다.\n\n"
            f"HS Code 분류 통칙:\n{self.hscode_rules}\n"
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
**중요: 추천하는 모든 HS Code는 반드시 4자리여야 합니다 (예: 9405).**

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
      "hs_code": "string",          // 반드시 4자리 HS Code (예: 9405)
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
2) hs_code는 반드시 4자리여야 합니다 (예: 9405).
3) **중요: 모든 후보의 hs_code는 서로 달라야 합니다. 중복된 4자리 코드는 제시하지 마세요.**
4) citations는 최소 1개 이상 포함.
5) citations.type은 반드시 "graph" 또는 "case"만 가능.
"""
        return system, user
    
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
        )
        
        # Parser 타입에 따라 시스템 프롬프트 조정
        if self.parser_type == "both":
            system += (
                "2) HS Code 계층 구조(GraphDB Context)는 전체 HS Code data이며\n"
                "   품목분류사례(VectorDB Context)는 classify Case data이고\n"
                "   Nomenclature Context는 HS 공식 명명법 문서입니다.\n"
            )
        elif self.parser_type == "graph":
            system += (
                "2) HS Code 계층 구조(GraphDB Context)는 전체 HS Code data이며\n"
                "   Nomenclature Context는 HS 공식 명명법 문서입니다.\n"
            )
        elif self.parser_type == "chroma":
            system += (
                "2) 품목분류사례(VectorDB Context)는 classify Case data이며\n"
                "   Nomenclature Context는 HS 공식 명명법 문서입니다.\n"
            )
        
        system += (
            "3) 계층 구조와 다른 코드는 절대 제시하지 않습니다.\n"
            "4) 추천하는 HS Code는 반드시 6자리여야 합니다 (예: 9405.40).\n"
            "5) 항상 응답은 strict JSON format으로만 출력합니다.\n\n"
            f"HS Code 분류 통칙:\n{self.hscode_rules}\n"
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
**중요: 추천하는 모든 HS Code는 반드시 6자리여야 합니다 (예: 9405.40).**

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
    
    def _build_prompt_6digit_from_4digit(
        self,
        product_name: str,
        product_description: str,
        chroma_hits: List[Dict[str, Any]],
        graph_context: str,
        six_digit_context: str,
        nomenclature_context: str,
        top_n: int
    ) -> Tuple[str, str]:
        """4자리에서 예측된 6자리 코드 예측용 프롬프트 구성 (2단계)"""
        system = (
            "당신은 국제무역 HS 코드 분류 전문가입니다.\n\n"
            "규칙:\n"
            "1) 제공된 context 내의 정보만 사용하여 판단합니다.\n"
        )
        
        # Parser 타입에 따라 시스템 프롬프트 조정
        if self.parser_type == "both":
            system += (
                "2) HS Code 계층 구조(GraphDB Context)는 전체 HS Code data이며\n"
                "   품목분류사례(VectorDB Context)는 classify Case data이고\n"
                "   Nomenclature Context는 HS 공식 명명법 문서입니다.\n"
            )
        elif self.parser_type == "graph":
            system += (
                "2) HS Code 계층 구조(GraphDB Context)는 전체 HS Code data이며\n"
                "   Nomenclature Context는 HS 공식 명명법 문서입니다.\n"
            )
        elif self.parser_type == "chroma":
            system += (
                "2) 품목분류사례(VectorDB Context)는 classify Case data이며\n"
                "   Nomenclature Context는 HS 공식 명명법 문서입니다.\n"
            )
        
        system += (
            "3) 계층 구조와 다른 코드는 절대 제시하지 않습니다.\n"
            "4) 추천하는 HS Code는 반드시 6자리여야 합니다 (예: 9405.40).\n"
            "5) **중요: 우선적으로 '6자리 HS Code 후보' 컨텍스트에 있는 코드를 추천하되, "
            "해당 컨텍스트에 적합한 코드가 없으면 전체 GraphDB Context에서 적절한 코드를 찾아 추천할 수 있습니다.**\n"
            "6) 항상 응답은 strict JSON format으로만 출력합니다.\n\n"
            f"HS Code 분류 통칙:\n{self.hscode_rules}\n"
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
**중요: 추천하는 모든 HS Code는 반드시 6자리여야 합니다.**
**우선적으로 '6자리 HS Code 후보' 컨텍스트에 있는 코드를 추천하되, 적절한 코드가 없으면 전체 GraphDB Context에서 찾아 추천할 수 있습니다.**

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
        
        # Nomenclature 컨텍스트 추가 (항상 추가)
        if nomenclature_context and nomenclature_context != "(Nomenclature 검색 결과 없음)":
            user += f"""
[Nomenclature Context — HS 공식 명명법 문서]
(HS Code 공식 Nomenclature 문서 기반)
{nomenclature_context}
================================================
"""
        
        # 6자리 코드 후보 컨텍스트 추가
        user += f"""
[6자리 HS Code 후보 — 1단계에서 예측된 4자리 코드의 하위 코드]
{six_digit_context}
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
3) **우선적으로 '6자리 HS Code 후보' 컨텍스트에 있는 코드를 선택하세요. 해당 컨텍스트에 적절한 코드가 없으면 전체 GraphDB Context에서 찾아 추천할 수 있습니다.**
4) **중요: 모든 후보의 hs_code는 서로 달라야 합니다. 중복된 6자리 코드는 제시하지 마세요.**
5) citations는 최소 1개 이상 포함.
6) citations.type은 반드시 "graph" 또는 "case"만 가능.
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
        )
        
        # Parser 타입에 따라 시스템 프롬프트 조정
        if self.parser_type == "both":
            system += (
                "2) HS Code 계층 구조(GraphDB Context)는 전체 HS Code data이며\n"
                "   품목분류사례(VectorDB Context)는 classify Case data이고\n"
                "   Nomenclature Context는 HS 공식 명명법 문서입니다.\n"
            )
        elif self.parser_type == "graph":
            system += (
                "2) HS Code 계층 구조(GraphDB Context)는 전체 HS Code data이며\n"
                "   Nomenclature Context는 HS 공식 명명법 문서입니다.\n"
            )
        elif self.parser_type == "chroma":
            system += (
                "2) 품목분류사례(VectorDB Context)는 classify Case data이며\n"
                "   Nomenclature Context는 HS 공식 명명법 문서입니다.\n"
            )
        
        system += (
            "3) 계층 구조와 다른 코드는 절대 제시하지 않습니다.\n"
            "4) 추천하는 HS Code는 반드시 10자리여야 합니다 (예: 9405.40-1000).\n"
            "5) **중요: 우선적으로 '10자리 HS Code 후보' 컨텍스트에 있는 코드를 추천하되, "
            "해당 컨텍스트에 적합한 코드가 없으면 전체 GraphDB Context에서 적절한 코드를 찾아 추천할 수 있습니다.**\n"
            "6) 항상 응답은 strict JSON format으로만 출력합니다.\n\n"
            f"HS Code 분류 통칙:\n{self.hscode_rules}\n"
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
**중요: 추천하는 모든 HS Code는 반드시 10자리여야 합니다.**
**우선적으로 '10자리 HS Code 후보' 컨텍스트에 있는 코드를 추천하되, 적절한 코드가 없으면 전체 GraphDB Context에서 찾아 추천할 수 있습니다.**

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
4) citations는 최소 1개 이상 포함.
5) citations.type은 반드시 "graph" 또는 "case"만 가능
6) reason은 추천한 코드에 대한 정의와 사용자의 상품에 대한 비교를 기반으로 해당 코드를 추천한 이유를 길고 자세하게 작성. 
"""
        return system, user
    
    def classify_hs_code(
        self,
        product_name: str,
        product_description: str,
        top_n: int = 3,
        chroma_top_k: int = None,
        graph_k: int = 5
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
        
        # 영어 번역 옵션이 켜져 있으면 번역 수행
        search_query_text = original_query_text
        if self.translate_to_english:
            try:
                translated_name = translate_to_english(product_name, self.client, self.openai_model)
                translated_desc = translate_to_english(product_description, self.client, self.openai_model)
                search_query_text = f"{translated_name}\n{translated_desc}"
                print(f"[번역] 원본: {product_name} / {product_description[:50]}...")
                print(f"[번역] 영어: {translated_name} / {translated_desc[:50]}...")
            except Exception as e:
                print(f"경고: 번역 실패, 원본 텍스트 사용: {e}")
                search_query_text = original_query_text
        
        # ChromaDB 검색 (필요한 경우)
        chroma_hits = []
        if self.parser_type in ["chroma", "both"]:
            if chroma_top_k is None:
                chroma_top_k = max(8, top_n * 3)
            chroma_hits = self._get_chroma_context(original_query_text, top_k=chroma_top_k)

        # Nomenclature ChromaDB 검색 (항상 시도)
        nomenclature_hits = self._get_nomenclature_context(search_query_text, top_k=5)
        nomenclature_context = self._format_nomenclature_context(nomenclature_hits, max_docs=5)

        # GraphDB: 기본은 context 문자열, late-fusion 재랭크 시에는 후보 텍스트 포함 리스트 사용
        graph_context = ""
        graph_candidates_with_text = []
        if self.parser_type in ["graph", "both"]:
            if self.use_rerank and self.parser_type == "both":
                try:
                    # 텍스트 동반 후보 반환 (최대 graph_k)
                    graph_candidates_with_text = self._graph_rag.get_vector_candidates_with_text(search_query_text, k=graph_k)
                except Exception as e:
                    print(f"경고: Graph 후보 텍스트 조회 실패: {e}")
                    graph_candidates_with_text = []
            else:
                graph_context = self._get_graph_context(search_query_text, k=graph_k)

        # Late-fusion: --rerank 사용 시 Chroma+Graph 통합 재랭킹으로 최종 Top-N만 컨텍스트에 사용 (parser_type=both일 때)
        if self.use_rerank and self.parser_type == "both":
            combined = []
            # Chroma → 공통 스키마로 정규화
            for h in chroma_hits[:max(0, graph_k)]:
                combined.append({
                    "id": f"chroma::{h.get('id')}",
                    "document": h.get("document") or "",
                    "metadata": dict(h.get("metadata") or {}) | {"source": "chroma"},
                    "_raw": h,
                })
            # Graph → 공통 스키마로 정규화
            for item in graph_candidates_with_text[:graph_k]:
                code = item.get("code")
                text = item.get("text") or ""
                combined.append({
                    "id": f"graph::{code}",
                    "document": text,
                    "metadata": {"hs_code": code, "source": "graph"},
                })

            selected_chroma_hits: List[Dict[str, Any]] = []
            selected_graph_codes: List[str] = []

            if combined:
                # CrossEncoder가 있으면 통합 재랭크, 없으면 간단 폴백: Chroma 우선 후 Graph 보충
                if self._reranker is not None:
                    try:
                        pairs = [(search_query_text, c.get("document") or "") for c in combined]
                        batch_size = int(os.getenv("RERANK_BATCH_SIZE", "4"))
                        scores = self._reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
                        for c, s in zip(combined, scores):
                            c["rerank_score"] = float(s) if s is not None else 0.0
                        combined.sort(key=lambda d: d.get("rerank_score", 0.0), reverse=True)
                    except Exception as e:
                        print(f"경고: 통합 ReRank 오류, 기본 순서 사용: {e}")
                # 최종 Top-N 선택
                fused_top = combined[:max(1, self.rerank_top_m or top_n)]
                for c in fused_top:
                    src = (c.get("metadata") or {}).get("source")
                    if src == "chroma":
                        # 원본 hit를 유지해야 포맷팅 가능
                        raw = c.get("_raw") or {}
                        selected_chroma_hits.append(raw)
                    elif src == "graph":
                        code = (c.get("metadata") or {}).get("hs_code")
                        if code:
                            selected_graph_codes.append(code)

            # 컨텍스트 구성: 선택된 항목만 사용
            chroma_ctx_hits = selected_chroma_hits if selected_chroma_hits else chroma_hits[:top_n]
            vector_context = self._format_chroma_context(chroma_ctx_hits)
            graph_context = ""
            if selected_graph_codes:
                try:
                    graph_context = self._graph_rag.get_graph_context(selected_graph_codes)
                except Exception as e:
                    print(f"경고: Graph context 생성 실패: {e}")
                    graph_context = ""

            # 프롬프트 구성
            sys_prompt, user_prompt = self._build_prompt(
                product_name, product_description, chroma_ctx_hits, graph_context, nomenclature_context, top_n
            )
        else:
            # 기존 동작: 개별 컨텍스트 사용
            # vector_context 포맷팅 (ChromaDB 사용 시)
            vector_context = ""
            if self.parser_type in ["chroma", "both"]:
                vector_context = self._format_chroma_context(chroma_hits)
            
            sys_prompt, user_prompt = self._build_prompt(
                product_name, product_description, chroma_hits, graph_context, nomenclature_context, top_n
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
 
        # context 정보 추가
        # result["chromaDB_context"] = vector_context if self.parser_type in ["chroma", "both"] else ""
        # result["graphDB_context"] = graph_context if self.parser_type in ["graph", "both"] else ""
        
        return result
    
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
        
        # 영어 번역 옵션이 켜져 있으면 번역 수행
        search_query_text = original_query_text
        if self.translate_to_english:
            try:
                translated_name = translate_to_english(product_name, self.client, self.openai_model)
                translated_desc = translate_to_english(product_description, self.client, self.openai_model)
                search_query_text = f"{translated_name}\n{translated_desc}"
                print(f"[번역] 원본: {product_name} / {product_description[:50]}...")
                print(f"[번역] 영어: {translated_name} / {translated_desc[:50]}...")
            except Exception as e:
                print(f"경고: 번역 실패, 원본 텍스트 사용: {e}")
                search_query_text = original_query_text
        
        print("=== 1단계: 6자리 HS Code 예측 ===")
        
        # ChromaDB 검색 (필요한 경우)
        chroma_hits = []
        if self.parser_type in ["chroma", "both"]:
            if chroma_top_k is None:
                chroma_top_k = max(8, top_n * 3)
            chroma_hits = self._get_chroma_context(search_query_text, top_k=chroma_top_k)

        # GraphDB 검색
        graph_context = ""
        if self.parser_type in ["graph", "both"]:
            graph_context = self._get_graph_context(search_query_text, k=graph_k)
        
        # Nomenclature ChromaDB 검색 (항상 시도)
        nomenclature_hits = self._get_nomenclature_context(search_query_text, top_k=5)
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
        
        if self.parser_type not in ["graph", "both"] or self._graph_rag is None:
            return {"error": "GraphDB가 필요합니다. parser_type을 'graph' 또는 'both'로 설정하세요."}
        
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
        vector_context_step1 = ""
        if self.parser_type in ["chroma", "both"]:
            vector_context_step1 = self._format_chroma_context(chroma_hits)
        result_10digit["chromaDB_context_step1"] = vector_context_step1 if self.parser_type in ["chroma", "both"] else ""
        result_10digit["graphDB_context_step1"] = graph_context if self.parser_type in ["graph", "both"] else ""
        
        # context 정보 추가 (2단계: 10자리 예측용 context)
        vector_context_step2 = ""
        if self.parser_type in ["chroma", "both"]:
            vector_context_step2 = self._format_chroma_context(chroma_hits)
        result_10digit["chromaDB_context_step2"] = vector_context_step2 if self.parser_type in ["chroma", "both"] else ""
        # graph_context는 이미 문자열이고, ten_digit_context가 추가로 포함됨
        final_graph_context = graph_context
        if ten_digit_context and ten_digit_context != "(하위 10자리 코드를 찾을 수 없어 전체 GraphDB 컨텍스트를 참고하세요.)":
            final_graph_context = f"{graph_context}\n\n=== 예측된 6자리 코드의 하위 10자리 코드 ===\n{ten_digit_context}"
        result_10digit["graphDB_context_step2"] = final_graph_context if self.parser_type in ["graph", "both"] else ""
        
        # 하위 호환성을 위해 기존 필드명도 유지 (2단계 context)
        result_10digit["chromaDB_context"] = vector_context_step2 if self.parser_type in ["chroma", "both"] else ""
        result_10digit["graphDB_context"] = final_graph_context if self.parser_type in ["graph", "both"] else ""

        # Nomenclature 컨텍스트 공유 (Stage1/Stage2)
        # nom_ctx = nomenclature_context if self.use_nomenclature else ""
        # result_10digit["nomenclature_context_step1"] = nom_ctx
        # result_10digit["nomenclature_context_step2"] = nom_ctx
        # result_10digit["nomenclature_context"] = nom_ctx        
        return result_10digit
    
    def classify_hs_code_hierarchical_3stage(
        self,
        product_name: str,
        product_description: str,
        top_n: int = 5,
        chroma_top_k: int = None,
        graph_k: int = 5
    ) -> Dict[str, Any]:
        """
        계층적 3단계 RAG를 통한 HS Code 분류
        1단계: 4자리 HS Code 예측
        2단계: 예측된 4자리 코드의 하위 6자리 코드만 컨텍스트로 추가하여 6자리 예측
        3단계: 예측된 6자리 코드의 하위 10자리 코드만 컨텍스트로 추가하여 최종 10자리 예측
        
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
        
        # 영어 번역 옵션이 켜져 있으면 번역 수행
        search_query_text = original_query_text
        if self.translate_to_english:
            try:
                translated_name = translate_to_english(product_name, self.client, self.openai_model)
                translated_desc = translate_to_english(product_description, self.client, self.openai_model)
                search_query_text = f"{translated_name}\n{translated_desc}"
                print(f"[번역] 원본: {product_name} / {product_description[:50]}...")
                print(f"[번역] 영어: {translated_name} / {translated_desc[:50]}...")
            except Exception as e:
                print(f"경고: 번역 실패, 원본 텍스트 사용: {e}")
                search_query_text = original_query_text
        
        print("=== 1단계: 4자리 HS Code 예측 ===")
        
        # ChromaDB 검색 (필요한 경우)
        chroma_hits = []
        if self.parser_type in ["chroma", "both"]:
            if chroma_top_k is None:
                chroma_top_k = max(8, top_n * 3)
            chroma_hits = self._get_chroma_context(search_query_text, top_k=chroma_top_k)

        # GraphDB 검색
        graph_context = ""
        if self.parser_type in ["graph", "both"]:
            graph_context = self._get_graph_context(search_query_text, k=graph_k)
        
        # Nomenclature ChromaDB 검색 (항상 시도)
        nomenclature_hits = self._get_nomenclature_context(search_query_text, top_k=5)
        nomenclature_context = self._format_nomenclature_context(nomenclature_hits, max_docs=5)
        
        # 1단계: 4자리 코드 예측 - LLM이 n개 미만을 반환하는 경우를 대비해 더 많은 후보 요청
        request_count = max(top_n * 2, top_n + 5)  # 최소 top_n + 5개 요청
        
        # 1단계: 4자리 코드 예측 프롬프트 구성
        sys_prompt_4digit, user_prompt_4digit = self._build_prompt_4digit(
            product_name, product_description, chroma_hits, graph_context, nomenclature_context, request_count
        )
        
        # 1단계 LLM 호출
        response_4digit = self.client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": sys_prompt_4digit},
                {"role": "user", "content": user_prompt_4digit}
            ],
            temperature=self.temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={"type": "json_object"},
            **({"seed": self.seed} if self.seed is not None else {})
        )
        
        output_4digit = response_4digit.choices[0].message.content.strip()
        result_4digit, err_4digit = _parse_json_safely(output_4digit)
        
        if err_4digit:
            return {"error": f"1단계 4자리 예측 실패: {err_4digit}", "raw_output": output_4digit}
        
        # 예측된 4자리 코드 추출
        candidates_4digit = result_4digit.get('candidates', [])
        if not candidates_4digit:
            return {"error": "1단계에서 4자리 코드 후보를 찾을 수 없습니다.", "step1_result": result_4digit}
        
        # 4자리 코드 리스트 추출 (점 제거 후 4자리 확인, 중복 제거, 정확히 top_n개)
        four_digit_codes = []
        seen_codes = set()  # 중복 제거를 위한 set
        
        for cand in candidates_4digit:
            # 이미 top_n개를 채웠으면 중단
            if len(four_digit_codes) >= top_n:
                break
                
            hs_code = str(cand.get('hs_code', '')).replace('.', '').replace('-', '')
            if len(hs_code) == 4:
                formatted = hs_code
                # 중복 제거: 이미 본 코드가 아니면 추가
                if formatted not in seen_codes:
                    seen_codes.add(formatted)
                    four_digit_codes.append(formatted)
            elif '.' in str(cand.get('hs_code', '')):
                # 'XXXX' 형식으로 변환
                code_str = str(cand.get('hs_code', ''))
                code_clean = code_str.split('.')[0].replace('-', '').replace('.', '')
                if len(code_clean) == 4:
                    formatted = code_clean
                    # 중복 제거: 이미 본 코드가 아니면 추가
                    if formatted not in seen_codes:
                        seen_codes.add(formatted)
                        four_digit_codes.append(formatted)
        
        if not four_digit_codes:
            return {"error": "유효한 4자리 코드를 추출할 수 없습니다.", "step1_result": result_4digit}
        
        # 정확히 top_n개가 아니면 경고
        if len(four_digit_codes) < top_n:
            print(f"경고: 4자리 코드가 {len(four_digit_codes)}개만 추출되었습니다. 요청한 {top_n}개보다 적습니다.")
            print(f"      LLM이 충분한 후보를 반환하지 않았거나, 중복이 많았을 수 있습니다.")
            print(f"      현재 {len(four_digit_codes)}개로 진행합니다.")
        
        # 상위 단계에서 더 많은 후보 유지 (오류 전파 방지: top_n * 2까지 유지)
        max_codes_to_keep = max(top_n * 2, top_n + 5)
        four_digit_codes = four_digit_codes[:max_codes_to_keep]
        
        print(f"예측된 4자리 코드 (중복 제거 후, 최종 {len(four_digit_codes)}개, 상위 {top_n}개 우선 사용): {four_digit_codes}")
        
        # 2단계: 예측된 4자리 코드의 하위 6자리 코드 조회
        print("=== 2단계: 6자리 HS Code 예측 ===")
        
        if self.parser_type not in ["graph", "both"] or self._graph_rag is None:
            return {"error": "GraphDB가 필요합니다. parser_type을 'graph' 또는 'both'로 설정하세요."}
        
        try:
            # 상위 top_n개만 우선 사용, 나머지는 백업으로 유지
            six_digit_context = self._graph_rag.get_6digit_codes_from_4digit(four_digit_codes[:top_n])
        except Exception as e:
            return {"error": f"6자리 코드 조회 실패: {e}", "step1_result": result_4digit}
        
        # 하위 6자리 코드가 없어도 전체 GraphDB 컨텍스트로 폴백 가능
        if not six_digit_context or "(하위 6자리 코드를 찾을 수 없습니다)" in six_digit_context:
            print("경고: 예측된 4자리 코드의 하위 6자리 코드를 찾을 수 없습니다. 전체 GraphDB 컨텍스트로 폴백합니다.")
            six_digit_context = "(하위 6자리 코드를 찾을 수 없어 전체 GraphDB 컨텍스트를 참고하세요.)"
        
        # 2단계: 6자리 코드 예측 프롬프트 구성
        sys_prompt_6digit, user_prompt_6digit = self._build_prompt_6digit_from_4digit(
            product_name, product_description, chroma_hits, graph_context, six_digit_context, nomenclature_context, request_count
        )
        
        # 2단계 LLM 호출
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
            return {
                "error": f"2단계 6자리 예측 실패: {err_6digit}",
                "raw_output": output_6digit,
                "step1_result": result_4digit
            }
        
        # 예측된 6자리 코드 추출
        candidates_6digit = result_6digit.get('candidates', [])
        if not candidates_6digit:
            return {
                "error": "2단계에서 6자리 코드 후보를 찾을 수 없습니다.",
                "step1_result": result_4digit,
                "step2_result": result_6digit
            }
        
        # 6자리 코드 리스트 추출 (점 제거 후 6자리 확인, 중복 제거, 정확히 top_n개)
        six_digit_codes = []
        seen_codes_6 = set()
        
        for cand in candidates_6digit:
            if len(six_digit_codes) >= top_n:
                break
                
            hs_code = str(cand.get('hs_code', '')).replace('.', '').replace('-', '')
            if len(hs_code) == 6:
                formatted = f"{hs_code[:4]}.{hs_code[4:6]}"
                if formatted not in seen_codes_6:
                    seen_codes_6.add(formatted)
                    six_digit_codes.append(formatted)
            elif '.' in str(cand.get('hs_code', '')):
                code_str = str(cand.get('hs_code', ''))
                code_clean = code_str.replace('.', '').replace('-', '')
                if len(code_clean) == 6:
                    formatted = f"{code_clean[:4]}.{code_clean[4:6]}"
                    if formatted not in seen_codes_6:
                        seen_codes_6.add(formatted)
                        six_digit_codes.append(formatted)
        
        if not six_digit_codes:
            return {
                "error": "유효한 6자리 코드를 추출할 수 없습니다.",
                "step1_result": result_4digit,
                "step2_result": result_6digit
            }
        
        if len(six_digit_codes) < top_n:
            print(f"경고: 6자리 코드가 {len(six_digit_codes)}개만 추출되었습니다. 요청한 {top_n}개보다 적습니다.")
            print(f"      현재 {len(six_digit_codes)}개로 진행합니다.")
        
        # 상위 단계에서 더 많은 후보 유지 (오류 전파 방지)
        max_codes_to_keep = max(top_n * 2, top_n + 5)
        six_digit_codes = six_digit_codes[:max_codes_to_keep]
        
        print(f"예측된 6자리 코드 (중복 제거 후, 최종 {len(six_digit_codes)}개, 상위 {top_n}개 우선 사용): {six_digit_codes}")
        
        # 3단계: 예측된 6자리 코드의 하위 10자리 코드 조회
        print("=== 3단계: 10자리 HS Code 예측 ===")
        
        try:
            # 상위 top_n개만 우선 사용
            ten_digit_context = self._graph_rag.get_10digit_codes_from_6digit(six_digit_codes[:top_n])
        except Exception as e:
            return {
                "error": f"10자리 코드 조회 실패: {e}",
                "step1_result": result_4digit,
                "step2_result": result_6digit
            }
        
        # 하위 10자리 코드가 없어도 전체 GraphDB 컨텍스트로 폴백 가능
        if not ten_digit_context or "(하위 10자리 코드를 찾을 수 없습니다)" in ten_digit_context:
            print("경고: 예측된 6자리 코드의 하위 10자리 코드를 찾을 수 없습니다. 전체 GraphDB 컨텍스트로 폴백합니다.")
            ten_digit_context = "(하위 10자리 코드를 찾을 수 없어 전체 GraphDB 컨텍스트를 참고하세요.)"
        
        # 3단계: 10자리 코드 예측 프롬프트 구성
        sys_prompt_10digit, user_prompt_10digit = self._build_prompt_10digit_hierarchical(
            product_name, product_description, chroma_hits, graph_context, ten_digit_context, nomenclature_context, top_n
        )
        
        # 3단계 LLM 호출
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
                "error": f"3단계 10자리 예측 실패: {err_10digit}",
                "raw_output": output_10digit,
                "step1_result": result_4digit,
                "step2_result": result_6digit
            }
        
        # 최종 결과에 1, 2단계 정보도 포함
        result_10digit["step1_4digit_codes"] = four_digit_codes
        result_10digit["step1_result"] = result_4digit
        result_10digit["step2_6digit_codes"] = six_digit_codes
        result_10digit["step2_result"] = result_6digit
        
        # 각 candidate에 계층별 정의 추가
        if "candidates" in result_10digit and isinstance(result_10digit["candidates"], list):
            result_10digit["candidates"] = self._add_hierarchy_definitions(result_10digit["candidates"])
        
        # context 정보 추가 (1단계: 4자리 예측용 context)
        vector_context_step1 = ""
        if self.parser_type in ["chroma", "both"]:
            vector_context_step1 = self._format_chroma_context(chroma_hits)
        result_10digit["chromaDB_context_step1"] = vector_context_step1 if self.parser_type in ["chroma", "both"] else ""
        result_10digit["graphDB_context_step1"] = graph_context if self.parser_type in ["graph", "both"] else ""
        
        # context 정보 추가 (2단계: 6자리 예측용 context)
        vector_context_step2 = ""
        if self.parser_type in ["chroma", "both"]:
            vector_context_step2 = self._format_chroma_context(chroma_hits)
        result_10digit["chromaDB_context_step2"] = vector_context_step2 if self.parser_type in ["chroma", "both"] else ""
        graph_context_step2 = graph_context
        if six_digit_context and six_digit_context != "(하위 6자리 코드를 찾을 수 없어 전체 GraphDB 컨텍스트를 참고하세요.)":
            graph_context_step2 = f"{graph_context}\n\n=== 예측된 4자리 코드의 하위 6자리 코드 ===\n{six_digit_context}"
        result_10digit["graphDB_context_step2"] = graph_context_step2 if self.parser_type in ["graph", "both"] else ""
        
        # context 정보 추가 (3단계: 10자리 예측용 context)
        vector_context_step3 = ""
        if self.parser_type in ["chroma", "both"]:
            vector_context_step3 = self._format_chroma_context(chroma_hits)
        result_10digit["chromaDB_context_step3"] = vector_context_step3 if self.parser_type in ["chroma", "both"] else ""
        # graph_context는 이미 문자열이고, ten_digit_context가 추가로 포함됨
        final_graph_context = graph_context
        if ten_digit_context and ten_digit_context != "(하위 10자리 코드를 찾을 수 없어 전체 GraphDB 컨텍스트를 참고하세요.)":
            final_graph_context = f"{graph_context}\n\n=== 예측된 6자리 코드의 하위 10자리 코드 ===\n{ten_digit_context}"
        result_10digit["graphDB_context_step3"] = final_graph_context if self.parser_type in ["graph", "both"] else ""
        
        # 하위 호환성을 위해 기존 필드명도 유지 (3단계 context)
        result_10digit["chromaDB_context"] = vector_context_step3 if self.parser_type in ["chroma", "both"] else ""
        result_10digit["graphDB_context"] = final_graph_context if self.parser_type in ["graph", "both"] else ""

        # Nomenclature 컨텍스트 공유 (Stage1/Stage2/Stage3)
        # nom_ctx = nomenclature_context if self.use_nomenclature else ""
        # result_10digit["nomenclature_context_step1"] = nom_ctx
        # result_10digit["nomenclature_context_step2"] = nom_ctx
        # result_10digit["nomenclature_context_step3"] = nom_ctx
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
            "query_text": original_query_text,
            "parser_type": self.parser_type
        }
