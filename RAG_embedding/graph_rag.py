from dotenv import load_dotenv
import os
from langchain_neo4j import Neo4jGraph
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
import torch

from typing import List, Dict


class GraphRAG:
    """HS Code 추천을 위한 Graph RAG 클래스"""
    
    def __init__(self, use_graph_rerank: bool = False, graph_rerank_model: str = None, graph_rerank_top_m: int = 5):
        """GraphRAG 인스턴스 초기화"""
        # .env 파일 로드
        load_dotenv()
        
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
        
        # Neo4j Graph 연결
        self.graph = Neo4jGraph(
            url=self.NEO4J_URI, 
            username=self.NEO4J_USER, 
            password=self.NEO4J_PASS
        )
        
        # Vector DB 설정
        self.MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.embedding_model = SentenceTransformerEmbeddings(model_name=self.MODEL_NAME,
                                encode_kwargs={"normalize_embeddings": True})
        
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
                if len(code) not in (4, 6):
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
                    scores = self._reranker.predict(pairs)
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
                    # 그래도 부족하면 계속 fetch 확대
                except Exception as e:
                    print(f"경고: Graph ReRank 중 오류: {e}")

            # ReRank 미사용 또는 불충분 시 기본 순서 반환 시도
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

    def get_graph_context(self, candidate_codes: List[str]) -> str:
        """후보 코드를 기반으로 계층 경로를 탐색하고 LLM Context를 생성"""
        
        # 🚨 동적 Cypher 쿼리 생성
        # candidates_str = "['8541', '9405']" 형태의 Cypher 리스트로 변환
        candidates_str = str(candidate_codes).replace("'", '"')

        # LLM이 직접 쿼리를 생성하는 대신, 코드를 삽입하여 실행
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
