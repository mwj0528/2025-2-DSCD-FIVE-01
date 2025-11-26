# LLM 기반 HS 코드 추천 시스템

> **목표**: 상품명/설명 같은 **텍스트 입력**만으로 HS 코드 **Top-K 후보**와 **설명 가능한 근거**를 구조화 JSON으로 제공합니다.  
> **핵심**: RAG 기반 검색 근거 + LLM 생성 + 사후 검증(코드 유효성·근거 일치·스키마 룰)로 **환각(할루시네이션) 최소화**.  

---

## 1) 개요

- **프로젝트명**: LLM 기반 HS Code Recommendation
- **과목**: 데이터사이언스 캡스톤디자인  
- **목표**: 자연어로 입력된 상품 정보(상품명·설명)를 바탕으로 **HS 코드 Top-K**과 **근거 텍스트**를 반환

### 주요 제공 기능
- 🔎 **Top-N HS 코드 추천** (LLM + RAG)
- 📚 **근거 텍스트 제공**: 검색된 규정/해설/사례의 관련 문단을 함께 제시
- 🗄️ **다중 데이터베이스 지원**: ChromaDB (Vector DB) + Neo4j (Graph DB)
- 🏗️ **계층적 RAG**: 2단계(6자리→10자리) 분류
- 🧪 **평가 시스템**: 자리수별 정확도 제공

---

## 2) 시스템 아키텍처

```text
[입력(상품명·설명)]
         │
         ▼
  [전처리·정규화] ──▶ [임베딩] ──▶ [VectorDB/GraphDB 검색]
         │                              │
         │                              ├─ ChromaDB (Vector Search)
         │                              └─ Neo4j (Graph Search)
         │
         ▼
      [LLM 생성(JSON 스키마 강제)]
         │
         ├─ 계층적 모드: 6자리 → 10자리
         │
         ├─ 검증① 코드 유효성(존재/형식/계층 규칙)
         ├─ 검증② 근거-응답 일치도(semantic entailment)
         └─ 검증③ 스키마/룰(JSON 키/타입/금지표현)
         ▼
      [최종 JSON 응답]
```

### 지원 데이터베이스
- **Case ChromaDB**: 품목분류사례 검색
- **Neo4j GraphDB**: HS 코드 계층 구조를 그래프로 표현하여 검색
- **Nomenclature ChromaDB**: HS 해설서 검색

### 지원 임베딩 모델
- `text-embedding-3-large` (OpenAI)

---

## 3) 프로젝트 구조

```
2025-2-DSCD-FIVE-01/
├── LLM/                          # LLM 관련 코드
│   ├── run_rag.py               # 메인 실행 스크립트
│   ├── rag_module.py            # 핵심 RAG 모듈 (HSClassifier)
│   ├── evaluate.py              # 평가 스크립트
│   ├── evaluate_and_reason.py   # 추론 과정 포함 평가
│   ├── Stage1_prompt.txt        # 계층적 1단계 프롬프트
│   ├── Stage2_prompt.txt        # 계층적 2단계 프롬프트
│   └── hscode_rule.txt          # HS 코드 규칙
│
├── RAG_embedding/                # 임베딩 및 RAG 관련 코드
│   ├── graph_rag.py             # GraphDB RAG 클래스
│   ├── graph_embedding.py       # GraphDB 임베딩 생성
│   ├── embedding_ver2.ipynb     # ChromaDB 임베딩 생성 노트북
│   └── pdf_to_markdown.py       # PDF → Markdown 변환
│
├── Preprocessing/                # 데이터 전처리
│   ├── all_hscode_preprocessing.ipynb
│   └── fill_data.py
│
├── Crawling/                     # 크롤링 관련
│   └── ...
│
├── data/                         # 데이터 파일
│   ├── all_hscode.csv           # HS 코드 전체 목록
│   ├── eval_dataset_*.csv       # 평가 데이터셋
│   ├── chroma_db_*/             # ChromaDB 인덱스
│   └── nomenclature_chroma_db/  # 명명법 ChromaDB
│
├── output/                       # 결과 파일
│   └── results/                  # 평가 결과
│
└── check.py                      # 유틸리티 스크립트
```

---

## 4) 설치 및 실행

### 4.1 요구사항
- Python 3.10+
- Neo4j 데이터베이스 (GraphDB 사용 시)
- 인터넷 연결 (LLM·임베딩 모델 사용 시)

### 4.2 의존성 설치
```bash

# 의존성 설치 (requirements.txt가 있다면)
pip install -r requirements.txt

```

### 4.3 환경 변수 설정
프로젝트 루트에 `.env` 파일을 생성합니다.

```env
# OpenAI API
OPENAI_API_KEY=your_openai_key

# Neo4j GraphDB (GraphDB 사용 시)
NEO4J_URI=your_neo4j_url
NEO4J_USER=your_username
NEO4J_PASS=your_password
INDEX_NAME=hs_code_index

# ChromaDB
CHROMA_DIR=data/chroma_db_openai_large_kw
CHROMA_COLLECTION=default

# 재현성
SEED=42
```

### 4.4 데이터베이스 준비

#### ChromaDB 인덱스 구축
```bash
# Jupyter 노트북 실행
jupyter notebook RAG_embedding/embedding_ver2.ipynb
```

#### Neo4j GraphDB 설정
1. Neo4j 데이터베이스 설치 및 실행
2. HS 코드 데이터를 그래프로 로드
3. 벡터 인덱스 생성:
```bash
python RAG_embedding/graph_embedding.py
```

---

## 5) 사용 방법

### 5.1 실행 (CLI)

#### 2단계 (6자리 → 10자리)
```bash
python LLM/run_rag.py \
  --parser both+nomenclature \
  --hierarchical \
  --name "LED 조명" \
  --desc "플라스틱 하우징에 장착된 LED 조명 모듈" \
  --top-n 5
```

---

## 6) 평가


---

## 7) 입력/출력 형식

### 7.1 입력
- **상품명** (`--name`): 상품의 이름
- **상품 설명** (`--desc`): 상품에 대한 상세 설명

### 7.2 출력 예시시(JSON)
```json
{
  "candidates": [
    {
      "hs_code": "8539.50.0000",
      "title": "LED 조명",
      "hierarchy": {
        "chapter": "85",
        "heading": "8539",
        "subheading": "8539.50",
        "national": "8539.50.0000"
      },
      "evidence": [
        {
          "source_id": "doc_001",
          "source_title": "품목분류 해설",
          "evidence_text": "LED 조명은 전기 조명 기구로 분류됩니다...",
          "loc": "Chapter 85, Section 3"
        }
      ],
      "confidence": {
        "retrieval_score": 0.85,
        "entailment_score": 0.92
      }
    }
  ],
  "meta": {
    "top_n_requested": 5,
    "top_k_retrieval": 15,
    "inference_time_seconds": 2.345
  }
}
```

---

## 8) 주요 기능 상세

### 8.1 계층적 RAG
- **2단계**: 6자리 코드 예측 → 해당 코드 하위에서 10자리 예측
- 각 단계에서 GraphDB를 활용하여 계층 구조를 고려한 검색 수행

### 8.2 키워드 추출
- 한국어 입력에 대해 KoNLPy (Okt)를 사용한 키워드 추출
- 추출된 키워드를 검색 쿼리에 추가하여 검색 품질 향상

---

## 9) 참고 자료

- HS 코드 공식 명명법 문서: `data/HS_code_Nomenclature.md`
- 프롬프트 템플릿: `LLM/Stage1_prompt.txt`, `LLM/Stage2_prompt.txt`