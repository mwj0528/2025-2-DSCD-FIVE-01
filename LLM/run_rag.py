# -*- coding: utf-8 -*-
"""
HS Code 분류 RAG 실행 스크립트
- Parser 설정을 통해 ChromaDB, GraphDB, 또는 둘 다 선택적으로 사용 가능
"""

import json
import argparse
from rag_module import HSClassifier, ParserType


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
    
    args = parser.parse_args()
    
    # HSClassifier 인스턴스 생성
    print(f"=== HS Code 분류 시스템 초기화 ===")
    print(f"Parser 설정: {args.parser}")
    print(f"상품명: {args.name}")
    print(f"상품설명: {args.desc}")
    print()
    
    try:
        classifier = HSClassifier(
            parser_type=args.parser,
            chroma_dir=args.chroma_dir,
            collection_name=args.collection_name,
            use_keyword_extraction=args.use_keyword_extraction,
            use_rerank=args.rerank,
            rerank_model=args.rerank_model,
            rerank_top_m=args.rerank_top_m,
            use_graph_rerank=args.graph_rerank,
            graph_rerank_model=args.graph_rerank_model,
            graph_rerank_top_m=args.graph_rerank_top_m
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
    print("처리 중...")
    
    try:
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
  python LLM/run_rag.py --parser both --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈"


 ### 키워드 추출 사용 안 하는 버전 ###

 # ChromaDB만 사용
  python LLM/run_rag.py --parser chroma --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈"--no-keyword-extraction
  
  # GraphDB만 사용
  python LLM/run_rag.py --parser graph --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈"--no-keyword-extraction
  
  # 둘 다 사용
  python LLM/run_rag.py --parser both --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈" --no-keyword-extraction


  ### ReRank 적용 버전 ###

  # ChromaDB ReRank 적용
  python LLM/run_rag.py --parser chroma --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈" --rerank
  
  # GraphDB ReRank 적용
  python LLM/run_rag.py --parser graph --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈" --graph-rerank
  
  # 둘 다 ReRank 적용
  python LLM/run_rag.py --parser both --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈" --rerank --graph-rerank

"""