from dotenv import load_dotenv
import os
from langchain_neo4j import Neo4jGraph
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from typing import List, Dict


# .env 파일 로드
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

# Neo4j Graph 연결
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASS)

# Vector DB 설정 (graph_embedding.py와 동일한 설정)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedding_model = SentenceTransformerEmbeddings(model_name=MODEL_NAME)

# Neo4j Vector DB 인스턴스 생성
neo4j_vector_db = Neo4jVector.from_existing_graph(
    embedding=embedding_model,
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASS,
    index_name=INDEX_NAME,          
    node_label="HSItem",            
    text_node_properties=["description"],
    embedding_node_property="embedding",
)

def get_vector_candidates(user_query: str, k: int = 5) -> List[str]:
    """Vector Search를 실행하여 상위 k개의 후보 코드(4~6자리)를 반환"""
    # neo4j_vector_db 인스턴스를 사용하여 유사도 검색 실행
    search_results = neo4j_vector_db.similarity_search(user_query, k=k)
    
    # 4자리 또는 6자리 코드만 추출하여 상위 레벨로 사용 (전략적 필터링)
    candidate_codes = set()
    for doc in search_results:
        code = doc.metadata.get('code')
        if code and len(code) in [4, 6]:
             candidate_codes.add(code)
    
    return list(candidate_codes)

# # 예시: 'LED 램프'에 대한 후보 코드 검색
# user_input = "Mules and hinnies; live"
# candidate_codes = get_vector_candidates(user_input)
# print(f"Vector Search 후보 코드: {candidate_codes}") 



def get_graph_context(candidate_codes: List[str]) -> str:
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
    
    results = graph.query(cypher_query)
    
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

# 3단계: LLM 답변 생성 (최종 RAG)
# LangChain의 PromptTemplate을 사용하여 최종 Context와 사용자 질문을 LLM에 전달합니다.




def generate_recommendation(user_input: str):
    # 1. Context 검색
    context = get_graph_context(get_vector_candidates(user_input))
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

# 🚀 최종 실행 
print(generate_recommendation("방부처리한 적송 나무"))