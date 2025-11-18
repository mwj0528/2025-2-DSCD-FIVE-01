# -*- coding: utf-8 -*-
"""
HS Code 분류 RAG 실행 스크립트
- Parser 설정을 통해 ChromaDB, GraphDB, 또는 둘 다 선택적으로 사용 가능
"""

import json
import argparse
import os
from rag_module import HSClassifier, ParserType, set_all_seeds

EMBED_MODEL_CHOICES = {
    "openai_small": "text-embedding-3-small",
    "openai_large": "text-embedding-3-large",
    "paraphrase-multilingual-minilm-l12-v2": "paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-small": "intfloat/multilingual-e5-small",
}

EMBED_CHROMA_DIR = {
    "openai_small": "data/chroma_db_openai_small_kw",
    "openai_large": "data/chroma_db_openai_large_kw",
    "intfloat/multilingual-e5-small": "data/chroma_db_e5_small_kw",
}


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="HS Code 분류 RAG 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # ChromaDB만 사용
  python run_rag.py --parser chroma --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈"
  
  # GraphDB만 사용
  python run_rag.py --parser graph --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈"
  
  # 둘 다 사용
  python run_rag.py --parser both --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈"
        """
    )
    
    # 필수 인자
    parser.add_argument(
        "--parser",
        type=str,
        choices=["chroma", "graph", "both"],
        default="both",
        help="사용할 DB 설정: 'chroma'(ChromaDB만), 'graph'(GraphDB만), 'both'(둘 다)"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="상품명"
    )
    parser.add_argument(
        "--desc",
        type=str,
        required=True,
        help="상품 설명"
    )
    
    # 선택 인자
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="추천할 후보 개수 (기본값: 3)"
    )
    parser.add_argument(
        "--chroma-top-k",
        type=int,
        default=None,
        help="ChromaDB에서 검색할 문서 개수 (기본값: top_n * 3)"
    )
    parser.add_argument(
        "--graph-k",
        type=int,
        default=5,
        help="GraphDB에서 검색할 후보 개수 (기본값: 5)"
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=None,
        help="ChromaDB 디렉토리 경로 (기본값: 환경변수 CHROMA_DIR)"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="ChromaDB 컬렉션 이름 (기본값: 환경변수 CHROMA_COLLECTION)"
    )
    parser.add_argument(
        "--no-keyword-extraction",
        action="store_false",
        dest="use_keyword_extraction",
        help="ChromaDB 검색 시 키워드 추출 사용 안 함 (기본값: 키워드 추출 사용)"
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="계층적 2단계 RAG 사용: 1단계에서 6자리 예측, 2단계에서 10자리 예측 (기본: 미사용)"
    )
    parser.add_argument(
        "--hierarchical-3stage",
        action="store_true",
        help="계층적 3단계 RAG 사용: 1단계에서 4자리 예측, 2단계에서 6자리 예측, 3단계에서 10자리 예측 (기본: 미사용)"
    )
    parser.add_argument(
        "--translate-to-english",
        action="store_true",
        help="사용자 입력을 영어로 번역하여 RAG 검색 수행 (기본: 미사용)"
    )
    # RRF Hybrid 옵션 (Semantic K + BM25 K → RRF)
    parser.add_argument(
        "--hybrid-rrf",
        action="store_true",
        help="Chroma에서 Semantic K + BM25 K를 RRF로 융합"
    )
    parser.add_argument(
        "--bm25-k",
        type=int,
        default=5,
        help="BM25에서 검색할 문서 수 (기본: 5)"
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF 상수 k (기본: 60)"
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Chroma 검색 결과에 CrossEncoder ReRank 적용 (기본: 미적용)"
    )
    parser.add_argument(
        "--rerank-model",
        type=str,
        default=None,
        help="ReRank에 사용할 CrossEncoder 모델명 (기본값: 환경변수 RERANK_MODEL 또는 ms-marco MiniLM)"
    )
    parser.add_argument(
        "--rerank-top-m",
        type=int,
        default=5,
        help="ReRank 후 상위 몇 개를 컨텍스트에 사용할지 (기본: 5)"
    )
    # Graph ReRank 옵션
    parser.add_argument(
        "--graph-rerank",
        action="store_true",
        help="Graph 후보 코드(4/6자리)에 CrossEncoder ReRank 적용 (기본: 미적용)"
    )
    parser.add_argument(
        "--graph-rerank-model",
        type=str,
        default=None,
        help="Graph ReRank에 사용할 CrossEncoder 모델명 (기본: 환경변수 GRAPH_RERANK_MODEL 또는 ms-marco MiniLM)"
    )
    parser.add_argument(
        "--graph-rerank-top-m",
        type=int,
        default=5,
        help="Graph ReRank 후 상위 몇 개 후보 코드를 사용할지 (기본: 5)"
    )
    # Listwise LLM-as-Reranker 옵션
    parser.add_argument(
        "--llm-listwise",
        action="store_true",
        help="Listwise LLM 재랭킹(슬라이딩 윈도우) 활성화"
    )
    parser.add_argument(
        "--llm-listwise-window",
        type=int,
        default=10,
        help="Listwise 윈도우 크기 w (기본: 10)"
    )
    parser.add_argument(
        "--llm-listwise-step",
        type=int,
        default=5,
        help="Listwise 스텝 s (기본: 5)"
    )
    parser.add_argument(
        "--llm-listwise-max-cand",
        type=int,
        default=16,
        help="Listwise 평가에 사용할 상위 후보 수 M (기본: 16)"
    )
    parser.add_argument(
        "--llm-listwise-top-m",
        type=int,
        default=5,
        help="Listwise 최종 상위 문서 수 (기본: 5)"
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        choices=list(EMBED_MODEL_CHOICES.keys()),
        default="paraphrase-multilingual-minilm-l12-v2",
        help="쿼리 임베딩에 사용할 모델 선택",
    )
    # Graph 하이브리드는 공통 스위치(--hybrid-rrf)와 공통 K(--bm25-k, --rrf-k)를 그대로 사용합니다
    
    args = parser.parse_args()
    
    # ===== 재현성을 위한 랜덤 시드 설정 =====
    # 환경변수에서 seed 가져오기 (기본값: 42)
    seed = int(os.getenv("SEED", "42"))
    set_all_seeds(seed)
    print(f"랜덤 시드 고정: {seed}")
    
    # HSClassifier 인스턴스 생성
    print(f"=== HS Code 분류 시스템 초기화 ===")
    resolved_chroma_dir = args.chroma_dir or EMBED_CHROMA_DIR.get(args.embed_model)
    print(f"Parser 설정: {args.parser}")
    print(f"영어 번역: {'사용' if args.translate_to_english else '미사용'}")
    print(f"임베딩 모델: {args.embed_model}")
    if resolved_chroma_dir:
        print(f"ChromaDB 디렉터리: {resolved_chroma_dir}")
    print(f"상품명: {args.name}")
    print(f"상품설명: {args.desc}")
    print()
    
    # Hybrid 설정: 하나의 스위치/파라미터로 두 DB 모두에 적용
    unified_hybrid = args.hybrid_rrf
    unified_bm25_k = args.bm25_k
    unified_rrf_k = args.rrf_k

    try:
        classifier = HSClassifier(
            parser_type=args.parser,
            chroma_dir=resolved_chroma_dir or args.chroma_dir,
            collection_name=args.collection_name,
            embed_model=EMBED_MODEL_CHOICES[args.embed_model],
            use_keyword_extraction=args.use_keyword_extraction,
            use_rrf_hybrid=unified_hybrid,
            bm25_k=unified_bm25_k,
            rrf_k=unified_rrf_k,
            use_rerank=args.rerank,
            rerank_model=args.rerank_model,
            rerank_top_m=args.rerank_top_m,
            use_graph_rerank=args.graph_rerank,
            graph_rerank_model=args.graph_rerank_model,
            graph_rerank_top_m=args.graph_rerank_top_m,
            use_llm_rerank_listwise=args.llm_listwise,
            llm_rerank_window=args.llm_listwise_window,
            llm_rerank_step=args.llm_listwise_step,
            llm_rerank_max_candidates=args.llm_listwise_max_cand,
            llm_rerank_top_m=args.llm_listwise_top_m,
            seed=seed,
            translate_to_english=args.translate_to_english
        )
    except Exception as e:
        print(f"오류: HSClassifier 초기화 실패: {e}")
        return 1
    
    # 컨텍스트 먼저 출력
    print("=== DB에서 검색된 컨텍스트 ===")
    contexts = classifier.get_enhanced_context(
        args.name,
        args.desc,
        chroma_top_k=args.chroma_top_k or max(8, args.top_n * 3),
        graph_k=args.graph_k
    )
    
    print(f"Parser 타입: {contexts['parser_type']}")
    print(f"VectorDB 컨텍스트 길이: {len(contexts['vector_context'])}")
    print(f"GraphDB 컨텍스트 길이: {len(contexts['graph_context'])}")
    
    if contexts['parser_type'] in ["chroma", "both"]:
        print("\n--- VectorDB 컨텍스트 (ChromaDB) ---")
        print(contexts['vector_context'])
    
    if contexts['parser_type'] in ["graph", "both"]:
        print("\n--- GraphDB 컨텍스트 ---")
        print(contexts['graph_context'])
    
    # HS Code 분류 실행
    print("\n=== HS Code 분류 실행 ===")
    if args.hierarchical_3stage:
        print("계층적 3단계 RAG 모드 사용")
    elif args.hierarchical:
        print("계층적 2단계 RAG 모드 사용")
    print("처리 중...")
    
    try:
        if args.hierarchical_3stage:
            # 계층적 3단계 RAG 사용
            if args.parser not in ["graph", "both"]:
                print("오류: 계층적 모드는 GraphDB가 필요합니다. --parser를 'graph' 또는 'both'로 설정하세요.")
                return 1
            
            result = classifier.classify_hs_code_hierarchical_3stage(
                product_name=args.name,
                product_description=args.desc,
                top_n=args.top_n,
                chroma_top_k=args.chroma_top_k,
                graph_k=args.graph_k
            )
        elif args.hierarchical:
            # 계층적 2단계 RAG 사용
            if args.parser not in ["graph", "both"]:
                print("오류: 계층적 모드는 GraphDB가 필요합니다. --parser를 'graph' 또는 'both'로 설정하세요.")
                return 1
            
            result = classifier.classify_hs_code_hierarchical(
                product_name=args.name,
                product_description=args.desc,
                top_n=args.top_n,
                chroma_top_k=args.chroma_top_k,
                graph_k=args.graph_k
            )
        else:
            # 기존 1단계 RAG 사용
            result = classifier.classify_hs_code(
                product_name=args.name,
                product_description=args.desc,
                top_n=args.top_n,
                chroma_top_k=args.chroma_top_k,
                graph_k=args.graph_k
            )
        
        print("\n=== 분류 결과 ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
        
    except Exception as e:
        print(f"오류: 분류 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

"""

 ### 키워드 추출하는 버전 ###
 
  # ChromaDB만 사용
  python LLM/run_rag.py --parser chroma --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈"
  
  # GraphDB만 사용
  python LLM/run_rag.py --parser graph --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈"
  
  # 둘 다 사용
  python LLM/run_rag.py --parser both --embed-model intfloat/multilingual-e5-small --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈"


 ### 영어 번역 버전 ###

  # ChromaDB만 사용
  python LLM/run_rag.py --parser chroma --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈" --translate-to-english
  
  # GraphDB만 사용
  python LLM/run_rag.py --parser graph --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈" --translate-to-english
  
  # 둘 다 사용
  python LLM/run_rag.py --parser both --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈" --translate-to-english
  
 ### 키워드 추출 사용 안 하는 버전 ###

 # ChromaDB만 사용
  python LLM/run_rag.py --parser chroma --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈"--no-keyword-extraction
  
  # GraphDB만 사용
  python LLM/run_rag.py --parser graph --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈"--no-keyword-extraction
  
  # 둘 다 사용
  python LLM/run_rag.py --parser both --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈" --no-keyword-extraction


  ### 계층적 2단계 RAG 버전 ###

  # 계층적 2단계 모드 (GraphDB 또는 both 필요)
  python LLM/run_rag.py --parser both --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈" --hierarchical

  ### 계층적 3단계 RAG 버전 ###

  # 계층적 3단계 모드 (GraphDB 또는 both 필요)
  python LLM/run_rag.py --parser both --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈" --hierarchical-3stage


  ### ReRank 적용 버전 ###

  # ChromaDB ReRank 적용
  python LLM/run_rag.py --parser chroma --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈" --rerank
  
  # GraphDB ReRank 적용
  python LLM/run_rag.py --parser graph --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈" --graph-rerank
  
  # 둘 다 ReRank 적용
  python LLM/run_rag.py --parser both --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈" --rerank --graph-rerank

  python LLM/run_rag.py --parser both --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈" --hierarchical --top-n 5
"""