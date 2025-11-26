from dotenv import load_dotenv
import os
import random
import numpy as np
from langchain_neo4j import Neo4jGraph
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sentence_transformers import CrossEncoder
import torch
from rank_bm25 import BM25Okapi
import re
from typing import List, Dict, Optional


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


def _is_openai_embedding_model(model_name: Optional[str]) -> bool:
    return bool(model_name) and model_name.startswith("text-embedding-")


class GraphRAG:
    """HS Code 추천을 위한 Graph RAG 클래스"""
    
    def __init__(
        self,
        use_graph_rerank: bool = False,
        graph_rerank_model: str = None,
        graph_rerank_top_m: int = 5,
        use_graph_hybrid_rrf: bool = False,
        graph_bm25_k: int = 5,
        graph_rrf_k: int = 60,
        graph_embed_model: Optional[str] = None,
        graph_openai_api_key: Optional[str] = None,
    ):
        """GraphRAG 인스턴스 초기화"""
        # .env 파일 로드
        load_dotenv()
        
        # 재현성을 위한 랜덤 시드 설정
        seed = int(os.getenv("SEED", "42"))
        set_all_seeds(seed)
        
        # 환경 변수 로드
        self.NEO4J_URI = os.getenv("NEO4J_URI")
        self.NEO4J_USER = os.getenv("NEO4J_USER")
        self.NEO4J_PASS = os.getenv("NEO4J_PASS")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.INDEX_NAME = os.getenv("INDEX_NAME")
        self.DEFAULT_GRAPH_RERANK_MODEL = os.getenv("GRAPH_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        self.use_graph_rerank = bool(use_graph_rerank)
        self.graph_rerank_model = graph_rerank_model or self.DEFAULT_GRAPH_RERANK_MODEL
        try:
            self.graph_rerank_top_m = max(1, int(graph_rerank_top_m))
        except Exception:
            self.graph_rerank_top_m = 5
        # Hybrid (RRF) 설정
        self.use_graph_hybrid_rrf = bool(use_graph_hybrid_rrf)
        try:
            self.graph_bm25_k = max(1, int(graph_bm25_k))
        except Exception:
            self.graph_bm25_k = 5
        try:
            self.graph_rrf_k = max(1, int(graph_rrf_k))
        except Exception:
            self.graph_rrf_k = 60
        
        # Neo4j Graph 연결
        self.graph = Neo4jGraph(
            url=self.NEO4J_URI, 
            username=self.NEO4J_USER, 
            password=self.NEO4J_PASS
        )
        
        # Vector DB 설정
        default_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.MODEL_NAME = graph_embed_model or default_model
        if _is_openai_embedding_model(self.MODEL_NAME):
            api_key = graph_openai_api_key or self.OPENAI_API_KEY
            if not api_key:
                raise ValueError(
                    f"OpenAI 임베딩 모델 '{self.MODEL_NAME}' 사용을 위해 OPENAI_API_KEY 또는 전용 키를 설정하세요."
                )
            self.embedding_model = OpenAIEmbeddings(model=self.MODEL_NAME, api_key=api_key)
        else:
            self.embedding_model = SentenceTransformerEmbeddings(
                model_name=self.MODEL_NAME,
                encode_kwargs={"normalize_embeddings": True}
            )
        
        # Neo4j Vector DB 인스턴스 생성
        self.neo4j_vector_db = Neo4jVector.from_existing_graph(
            embedding=self.embedding_model,
            url=self.NEO4J_URI,
            username=self.NEO4J_USER,
            password=self.NEO4J_PASS,
            index_name=self.INDEX_NAME,          
            node_label="HSItem",            
            text_node_properties=["description"],
            embedding_node_property="embedding",
        )

        # 선택적 ReRank 초기화
        self._reranker = None
        if self.use_graph_rerank:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._reranker = CrossEncoder(self.graph_rerank_model, device=device)
            except Exception as e:
                print(f"경고: Graph ReRank 모델 초기화 실패: {e}")
                self._reranker = None

    def get_vector_candidates(self, user_query: str, k: int = 5) -> List[str]:
        """유사도 순서를 유지하면서 4/6자리 코드만 필터링 후 최대 k개 반환.
        부족하면 점증적으로 검색 범위를 늘려 k개를 최대한 채운다.
        """
        current_fetch = max(20, k * 5)
        max_fetch = max(100, k * 50)
        last_count = -1

        while True:
            results = self.neo4j_vector_db.similarity_search(user_query, k=current_fetch)

            # 기본: 유사도 순서 유지하며 필터링 + 중복 제거
            base_ordered: List[str] = []
            seen: set = set()
            filtered_docs = []  # (code, text)
            for doc in results:
                code = doc.metadata.get("code")
                if not code:
                    continue
                # 점과 하이픈 제거 후 길이 확인 (4자리 또는 6자리만)
                code_clean = str(code).replace('.', '').replace('-', '')
                if len(code_clean) not in (4, 6):
                    continue
                text = getattr(doc, "page_content", "")
                if code not in seen:
                    seen.add(code)
                    base_ordered.append(code)
                filtered_docs.append((code, text))

            # ReRank 활성화 시 CrossEncoder로 재정렬
            if self.use_graph_rerank and self._reranker is not None and filtered_docs:
                try:
                    # 같은 코드가 여러 문서에 나타나면 최고 점수 채택
                    code_to_best_score = {}
                    pairs = [(user_query, text) for _, text in filtered_docs]
                    # 메모리 절약: 작은 배치 크기 사용
                    batch_size = int(os.getenv("GRAPH_RERANK_BATCH_SIZE", "4"))
                    scores = self._reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
                    for (code, _), s in zip(filtered_docs, scores):
                        score = float(s) if s is not None else 0.0
                        if code not in code_to_best_score or score > code_to_best_score[code]:
                            code_to_best_score[code] = score
                    reranked = sorted(code_to_best_score.items(), key=lambda x: x[1], reverse=True)
                    # 상위 graph_rerank_top_m를 우선 사용하되 최종 반환은 최대 k개까지
                    reranked_codes = [c for c, _ in reranked][:max(self.graph_rerank_top_m, k)]
                    # 부족 시 기본 순서로 보충
                    for c in base_ordered:
                        if len(reranked_codes) >= k:
                            break
                        if c not in reranked_codes:
                            reranked_codes.append(c)
                    if len(reranked_codes) >= k:
                        return reranked_codes[:k]
                except Exception as e:
                    print(f"경고: Graph ReRank 중 오류: {e}")

            # Hybrid RRF (Semantic + BM25) 적용: filtered_docs 텍스트에 대해 BM25 구축 후 RRF 결합
            if self.use_graph_hybrid_rrf and filtered_docs:
                try:
                    texts = [text or "" for _, text in filtered_docs]
                    tokens = [self._bm25_tokenize(t) for t in texts]
                    bm25 = BM25Okapi(tokens)
                    q_tokens = self._bm25_tokenize(user_query or "")
                    scores = bm25.get_scores(q_tokens)
                    # 코드별 최고 점수
                    code_to_best_bm25 = {}
                    for (code, _), s in zip(filtered_docs, scores):
                        score = float(s) if s is not None else 0.0
                        if code not in code_to_best_bm25 or score > code_to_best_bm25[code]:
                            code_to_best_bm25[code] = score
                    # semantic 순위
                    sem_rank = {c: r+1 for r, c in enumerate(base_ordered)}
                    # bm25 순위
                    bm25_sorted = sorted(code_to_best_bm25.items(), key=lambda x: x[1], reverse=True)
                    bm25_rank = {c: r+1 for r, (c, _) in enumerate(bm25_sorted[:max(self.graph_bm25_k, k)])}
                    # RRF 융합
                    all_codes = set(sem_rank.keys()) | set(bm25_rank.keys())
                    fused = []
                    for code in all_codes:
                        r1 = sem_rank.get(code)
                        r2 = bm25_rank.get(code)
                        score = 0.0
                        if r1 is not None:
                            score += 1.0 / (self.graph_rrf_k + r1)
                        if r2 is not None:
                            score += 1.0 / (self.graph_rrf_k + r2)
                        fused.append((code, score))
                    fused.sort(key=lambda x: x[1], reverse=True)
                    rrf_codes = [c for c, _ in fused][:k]
                    if rrf_codes:
                        return rrf_codes
                except Exception as e:
                    print(f"경고: Graph RRF 하이브리드 실패: {e}")

            # ReRank/Hybrid 미사용 또는 불충분 시 기본 순서 반환 시도
            if len(base_ordered) >= k:
                return base_ordered[:k]

            # 더 가져와도 증가가 없거나 상한 도달 시 종료
            if len(results) == last_count or current_fetch >= max_fetch:
                # 마지막으로 가능한 만큼 반환
                if self.use_graph_rerank and self._reranker is not None:
                    # 위에서 이미 시도했으므로 base_ordered 반환
                    return base_ordered
                return base_ordered

            last_count = len(results)
            current_fetch = min(current_fetch * 2, max_fetch)

    def get_vector_candidates_with_text(self, user_query: str, k: int = 5) -> List[dict]:
        """유사도 순서를 유지하면서 4/6자리 코드만 필터링 후 최대 k개에 대해 대표 텍스트를 함께 반환.
        반환 형식: [{"code": str, "text": str}]
        """
        current_fetch = max(20, k * 5)
        max_fetch = max(100, k * 50)
        last_count = -1

        while True:
            results = self.neo4j_vector_db.similarity_search(user_query, k=current_fetch)

            # 기본: 유사도 순서 유지하며 필터링 + 코드별 대표 텍스트 저장
            base_ordered: List[str] = []
            seen: set = set()
            filtered_docs = []  # (code, text)
            code_to_text = {}
            for doc in results:
                code = doc.metadata.get("code")
                if not code:
                    continue
                if len(code) not in (4, 6):
                    continue
                text = getattr(doc, "page_content", "")
                if code not in seen:
                    seen.add(code)
                    base_ordered.append(code)
                    code_to_text[code] = text
                filtered_docs.append((code, text))

            # ReRank 활성화 시 CrossEncoder로 재정렬하고 코드별 베스트 텍스트 선택
            if self.use_graph_rerank and self._reranker is not None and filtered_docs:
                try:
                    code_to_best_score = {}
                    code_to_best_text = {}
                    pairs = [(user_query, text) for _, text in filtered_docs]
                    batch_size = int(os.getenv("GRAPH_RERANK_BATCH_SIZE", "4"))
                    scores = self._reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
                    for (code, text), s in zip(filtered_docs, scores):
                        score = float(s) if s is not None else 0.0
                        if code not in code_to_best_score or score > code_to_best_score[code]:
                            code_to_best_score[code] = score
                            code_to_best_text[code] = text
                    reranked = sorted(code_to_best_score.items(), key=lambda x: x[1], reverse=True)
                    reranked_codes = [c for c, _ in reranked][:max(self.graph_rerank_top_m, k)]
                    # 부족 시 기본 순서로 보충
                    for c in base_ordered:
                        if len(reranked_codes) >= k:
                            break
                        if c not in reranked_codes:
                            reranked_codes.append(c)
                    out = []
                    for c in reranked_codes[:k]:
                        out.append({"code": c, "text": code_to_best_text.get(c, code_to_text.get(c, ""))})
                    if out:
                        return out
                except Exception as e:
                    print(f"경고: Graph ReRank 중 오류: {e}")

            # Hybrid RRF 미사용 또는 실패 시 기본 순서에서 반환
            if len(base_ordered) >= k:
                return [{"code": c, "text": code_to_text.get(c, "")} for c in base_ordered[:k]]

            if len(results) == last_count or current_fetch >= max_fetch:
                # 가능한 만큼 반환
                return [{"code": c, "text": code_to_text.get(c, "")} for c in base_ordered]

            last_count = len(results)
            current_fetch = min(current_fetch * 2, max_fetch)

    def _bm25_tokenize(self, text: str):
        if not text:
            return []
        s = str(text).lower()
        s = re.sub(r"[%㎜]+", " ", s)
        return [t for t in re.findall(r"[a-z0-9가-힣]+", s) if len(t) >= 2]

    def get_graph_context(self, candidate_codes: List[str]) -> str:
        """후보 코드를 기반으로 계층 경로를 탐색하고 LLM Context를 생성"""
        
        # 🚨 동적 Cypher 쿼리 생성
        # candidates_str = "['8541', '9405']" 형태의 Cypher 리스트로 변환
        candidates_str = str(candidate_codes).replace("'", '"')

        # LLM이 직접 쿼리를 생성하는 대신, 코드를 삽입하여 실행
        # cypher_query = f"""
        # UNWIND {candidates_str} AS root_code_str
        # MATCH p = (root:HSItem {{code: root_code_str}})-[:HAS_CHILD*1..]->(n)
        # WHERE NOT (n)-[:HAS_CHILD]->()
        # RETURN nodes(p) AS Path_Nodes, relationships(p) AS Path_Relationships
        # """

        cypher_query = f"""
        UNWIND {candidates_str} AS root_code_str
        MATCH p = (root:HSItem {{code: root_code_str}})-[:HAS_CHILD*1..]->(n)
        WHERE NOT (n)-[:HAS_CHILD]->()
        RETURN nodes(p) AS Path_Nodes, relationships(p) AS Path_Relationships
        """
        
        results = self.graph.query(cypher_query)
        
        final_context = "# [검색된 HS Code 계층 구조 데이터]\n\n"
        
        # --- LLM Context 문자열 변환 로직 ---
        for result in results:
            nodes = result['Path_Nodes']
            
            # 1. 시각적 계층 경로 구성 (고객님의 예시 형태)
            if not nodes: continue

            path_text = ""
            table_rows = []
            
            for i, node in enumerate(nodes):
                code = node['code']
                desc = node['description']
                
                # 경로 텍스트 생성
                if i == 0:
                    path_text += f"[시작 노드: {code} ({desc})]\n"
                    level_desc = "상위 레벨"
                elif i == len(nodes) - 1:
                    path_text += f"    |--[:HAS_CHILD]-> [최종 노드: {code} ({desc})]\n"
                    level_desc = "최종 레벨"
                else:
                    path_text += f"    |--[:HAS_CHILD]-> [중간 노드: {code} ({desc})]\n"
                    level_desc = "중간 레벨"
                
                # 테이블 행 데이터 수집
                table_rows.append(f"| {code} | {desc} | {level_desc} |")
            
            # Context에 경로 추가
            final_context += path_text + "\n"
            
            # 2. 추론 요약 테이블 구성
            final_context += "---"
            final_context += "\n[추론 요약 테이블]\n"
            final_context += "| 코드 | 영문 품목명 | 계층 |\n"
            final_context += "|:---|:---|:---|\n"
            final_context += "\n".join(table_rows) + "\n\n"
        
        return final_context

    def get_10digit_codes_from_6digit(self, six_digit_codes: List[str]) -> str:
        """6자리 코드들의 하위 10자리 코드만 반환하는 메서드
        
        Args:
            six_digit_codes: 6자리 HS 코드 리스트 (예: ['9405.40', '9405.50'])
            
        Returns:
            str: 10자리 코드들의 컨텍스트 문자열
        """
        # 6자리 코드를 정규화 (점 있는 형식과 점 없는 형식 모두 시도)
        # Neo4j에 저장된 형식이 일관되지 않을 수 있으므로 두 형식 모두 시도
        normalized_codes = []
        for code in six_digit_codes:
            code_clean = code.replace('.', '').replace('-', '')
            if len(code_clean) == 6:
                # 점 없는 형식 추가 (예: '842139')
                normalized_codes.append(code_clean)
                # 점 있는 형식도 추가 (예: '8421.39')
                formatted = f"{code_clean[:4]}.{code_clean[4:6]}"
                if formatted not in normalized_codes:
                    normalized_codes.append(formatted)
            elif '.' in code:
                # 이미 점이 있으면 그대로 추가
                normalized_codes.append(code)
                # 점 없는 형식도 추가
                code_no_dot = code.replace('.', '').replace('-', '')
                if code_no_dot not in normalized_codes:
                    normalized_codes.append(code_no_dot)
        
        if not normalized_codes:
            return "# [10자리 HS Code 후보]\n\n(6자리 코드가 없습니다.)\n"
        
        # 중복 제거
        normalized_codes = list(set(normalized_codes))
        candidates_str = str(normalized_codes).replace("'", '"')
        
        # 6자리 코드의 직접 자식 중 10자리 코드만 가져오는 쿼리
        # 리프 노드(더 이상 자식이 없는 노드)만 가져온 후 Python에서 10자리 필터링
        cypher_query = f"""
        UNWIND {candidates_str} AS parent_code_str
        MATCH (parent:HSItem {{code: parent_code_str}})-[:HAS_CHILD*1..]->(child:HSItem)
        WHERE NOT (child)-[:HAS_CHILD]->()
        RETURN DISTINCT parent.code AS parent_code, child.code AS child_code, child.description AS child_description
        ORDER BY parent.code, child.code
        """
        
        try:
            results = self.graph.query(cypher_query)
        except Exception as e:
            print(f"경고: 10자리 코드 조회 실패: {e}")
            return "# [10자리 HS Code 후보]\n\n(조회 중 오류가 발생했습니다.)\n"
        
        if not results:
            return "# [10자리 HS Code 후보]\n\n(하위 10자리 코드를 찾을 수 없습니다.)\n"
        
        # Python에서 10자리 코드만 필터링 (점 제거 후 10자리)
        parent_to_children = {}
        for result in results:
            parent = result.get('parent_code', '')
            child_code = str(result.get('child_code', ''))
            child_desc = result.get('child_description', '')
            
            # 점 제거 후 10자리인지 확인
            code_clean = child_code.replace('.', '').replace('-', '')
            if len(code_clean) == 10:
                if parent not in parent_to_children:
                    parent_to_children[parent] = []
                parent_to_children[parent].append({
                    'code': child_code,
                    'description': child_desc
                })
        
        if not parent_to_children:
            return "# [10자리 HS Code 후보]\n\n(하위 10자리 코드를 찾을 수 없습니다.)\n"
        
        # 컨텍스트 문자열 생성
        final_context = "# [10자리 HS Code 후보]\n\n"
        final_context += "다음은 예측된 6자리 코드들의 하위 10자리 코드 목록입니다.\n\n"
        
        for parent_code, children in parent_to_children.items():
            final_context += f"## 부모 코드: {parent_code}\n\n"
            final_context += "| 10자리 코드 | 영문 품목명 |\n"
            final_context += "|:---|:---|\n"
            
            for child in children:
                code = child['code']
                desc = child['description']
                final_context += f"| {code} | {desc} |\n"
            
            final_context += "\n"
        
        return final_context

    def get_6digit_codes_from_4digit(self, four_digit_codes: List[str]) -> str:
        """4자리 코드들의 하위 6자리 코드만 반환하는 메서드
        
        Args:
            four_digit_codes: 4자리 HS 코드 리스트 (예: ['9405', '9406'])
            
        Returns:
            str: 6자리 코드들의 컨텍스트 문자열
        """
        # 4자리 코드를 정규화
        normalized_codes = []
        for code in four_digit_codes:
            # 점 제거 후 4자리 확인
            code_clean = code.replace('.', '').replace('-', '')
            if len(code_clean) == 4:
                normalized_codes.append(code_clean)
            elif '.' in code:
                # 'XXXX' 형식으로 변환
                code_clean = code.split('.')[0].replace('-', '')
                if len(code_clean) == 4:
                    normalized_codes.append(code_clean)
        
        if not normalized_codes:
            return "# [6자리 HS Code 후보]\n\n(4자리 코드가 없습니다.)\n"
        
        candidates_str = str(normalized_codes).replace("'", '"')
        
        # 4자리 코드의 직접 자식 중 6자리 코드만 가져오는 쿼리
        # 리프 노드가 아닌 6자리 코드만 가져옴
        cypher_query = f"""
        UNWIND {candidates_str} AS parent_code_str
        MATCH (parent:HSItem {{code: parent_code_str}})-[:HAS_CHILD*1..]->(child:HSItem)
        WHERE (child)-[:HAS_CHILD]->()
        RETURN DISTINCT parent.code AS parent_code, child.code AS child_code, child.description AS child_description
        ORDER BY parent.code, child.code
        """
        
        try:
            results = self.graph.query(cypher_query)
        except Exception as e:
            print(f"경고: 6자리 코드 조회 실패: {e}")
            return "# [6자리 HS Code 후보]\n\n(조회 중 오류가 발생했습니다.)\n"
        
        if not results:
            return "# [6자리 HS Code 후보]\n\n(하위 6자리 코드를 찾을 수 없습니다.)\n"
        
        # Python에서 6자리 코드만 필터링 (점 제거 후 6자리, 자식이 있는 노드)
        parent_to_children = {}
        for result in results:
            parent = result.get('parent_code', '')
            child_code = result.get('child_code', '')
            child_desc = result.get('child_description', '')
            
            # 점 제거 후 6자리인지 확인
            code_clean = child_code.replace('.', '').replace('-', '')
            if len(code_clean) == 6:
                if parent not in parent_to_children:
                    parent_to_children[parent] = []
                parent_to_children[parent].append({
                    'code': child_code,
                    'description': child_desc
                })
        
        if not parent_to_children:
            return "# [6자리 HS Code 후보]\n\n(하위 6자리 코드를 찾을 수 없습니다.)\n"
        
        # 컨텍스트 문자열 생성
        final_context = "# [6자리 HS Code 후보]\n\n"
        final_context += "다음은 예측된 4자리 코드들의 하위 6자리 코드 목록입니다.\n\n"
        
        for parent_code, children in parent_to_children.items():
            final_context += f"## 부모 코드: {parent_code}\n\n"
            final_context += "| 6자리 코드 | 영문 품목명 |\n"
            final_context += "|:---|:---|\n"
            
            for child in children:
                code = child['code']
                desc = child['description']
                final_context += f"| {code} | {desc} |\n"
            
            final_context += "\n"
        
        return final_context

    def generate_recommendation(self, user_input: str):
        """LLM을 사용하여 HS Code 추천 생성"""
        # 1. Context 검색
        context = self.get_graph_context(self.get_vector_candidates(user_input))
        # 2. LLM Prompt 구성
        template = """
        당신은 HS Code 추천 전문가입니다. 
        제공된 [검색된 HS Code 계층 구조 데이터] 정보만 사용하여 사용자의 상품에 가장 적합한 10자리 HS Code를 추천하고, 
        왜 그 코드를 선택했는지 계층 경로를 설명하십시오. 
        만약 여러 경로가 검색되었다면, 모든 경로를 제시하고 최종 선택을 사용자에게 맡기십시오.

        사용자 상품: {user_input}

        [검색된 HS Code 계층 구조 데이터]:
        {context}
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # 3. LLM Chain 실행
        chain = prompt | ChatOpenAI(model="gpt-4-turbo") 
        
        response = chain.invoke({"user_input": user_input, "context": context})
        
        return response.content

    def get_final_context(self, user_input: str, k: int = 5) -> str:
        """
        다른 파일에서 사용할 수 있는 메인 메서드
        Input과 k를 받아서 가장 가까운 Top-k 후보의 final_context를 반환
        
        Args:
            user_input (str): 사용자 입력 (상품명 등)
            k (int): 검색할 후보 개수 (기본값: 5)
            
        Returns:
            str: 검색된 HS Code 계층 구조 데이터의 final_context
        """
        # 1. Vector Search로 후보 코드 검색
        candidate_codes = self.get_vector_candidates(user_input, k)
        
        # 2. 후보 코드를 기반으로 계층 구조 Context 생성
        final_context = self.get_graph_context(candidate_codes)
        
        return final_context


# 사용 예시 (다른 파일에서 import할 때는 이 부분이 실행되지 않음)
if __name__ == "__main__":
    # GraphRAG 인스턴스 생성
    graph_rag = GraphRAG()
    
    # 테스트 실행
    print("=== GraphRAG 테스트 ===")
    result = graph_rag.get_final_context("방부처리한 적송 나무", k=5)
    print(result)


# final_context = graph_rag.get_final_context(user_input, k) -> context 반환
