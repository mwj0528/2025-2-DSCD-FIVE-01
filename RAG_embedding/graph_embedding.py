from dotenv import load_dotenv
import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings

load_dotenv()
# --- 1. AuraDB 연결 정보 ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")
INDEX_NAME = os.getenv("INDEX_NAME")

# MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_NAME = "Salesforce/SFR-Embedding-Mistral"
EMBEDDING_DIM = 4092

# --- 2. 임베딩 모델 정의 ---
embedding_model = SentenceTransformerEmbeddings(
    model_name=MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True,
                   "batch_size": 4,
                    }
)
# embedding_model = OpenAIEmbeddings(
#     model="text-embedding-3-large"  # ✅ openai_embedding_small에 해당
# )

# --- 3. Vector Index 생성 및 데이터 쓰기 (통합) ---
# 기존 오류 (text_node_property)는 이미 수정되었고, 
# 새로운 오류 (index_options)를 제거합니다.

neo4j_vector_db = Neo4jVector.from_existing_graph(
    embedding=embedding_model,
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASS,
    index_name=INDEX_NAME,          
    node_label="HSItem",            
    text_node_properties=["description"], # 복수형 인자 사용
    embedding_node_property="embedding", # 생성된 벡터가 저장될 속성 이름
    # 🚨 오류 발생 인자 제거: index_options 인자는 이제 사용하지 않습니다.
)

print("Python 환경에서 Neo4j Vector Index 생성 및 임베딩 업데이트 완료.")
# 이 과정이 성공하면 Vector Search를 위한 준비가 완료됩니다.


# python RAG_embedding/graph_embedding.py