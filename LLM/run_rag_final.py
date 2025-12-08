# -*- coding: utf-8 -*-
"""
HS Code 분류 RAG 실행 스크립트 (최종 모델)
- ChromaDB와 GraphDB를 모두 사용하는 계층적 2단계 RAG
- 항상 both 모드로 동작 (ChromaDB + GraphDB)
"""

import json
import argparse
import os
import time
# from rag_module import HSClassifier, ParserType, set_all_seeds
from rag_module_final import HSClassifier, set_all_seeds

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
        description="HS Code 분류 RAG 시스템 (최종 모델)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 사용 (계층적 2단계 RAG, OpenAI Large 임베딩, Nomenclature 포함)
  python run_rag_final.py --name "LED 조명" --desc "플라스틱 하우징에 장착된 LED 조명 모듈"
        """
    )
    
    # 필수 인자
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
        default=5,
        help="추천할 후보 개수 (기본값: 5)"
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
    
    args = parser.parse_args()
    
    # 임베딩 모델은 항상 openai_large로 고정
    embed_model_key = "openai_large"
    
    # ===== 재현성을 위한 랜덤 시드 설정 =====
    # 환경변수에서 seed 가져오기 (기본값: 42)
    seed = int(os.getenv("SEED", "42"))
    set_all_seeds(seed)
    print(f"랜덤 시드 고정: {seed}")
    
    # HSClassifier 인스턴스 생성
    print(f"=== HS Code 분류 시스템 초기화 (최종 모델) ===")
    resolved_chroma_dir = args.chroma_dir or EMBED_CHROMA_DIR.get(embed_model_key)
    print(f"모드: ChromaDB + GraphDB (both)")
    print(f"Nomenclature ChromaDB: 사용 (고정)")
    print(f"임베딩 모델: {embed_model_key} ({EMBED_MODEL_CHOICES[embed_model_key]}) (고정)")
    if resolved_chroma_dir:
        print(f"ChromaDB 디렉터리: {resolved_chroma_dir}")
    print(f"상품명: {args.name}")
    print(f"상품설명: {args.desc}")
    print()

    try:
        classifier = HSClassifier(
            embed_model=EMBED_MODEL_CHOICES[embed_model_key],
            use_keyword_extraction=args.use_keyword_extraction,
            seed=seed,
            use_nomenclature=True  # 항상 사용
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
    
    print(f"VectorDB 컨텍스트 길이: {len(contexts['vector_context'])}")
    print(f"GraphDB 컨텍스트 길이: {len(contexts['graph_context'])}")
    print(f"Nomenclature 컨텍스트 길이: {len(contexts.get('nomenclature_context', ''))}")
    
    print("\n--- VectorDB 컨텍스트 (ChromaDB) ---")
    print(contexts['vector_context'])
    
    print("\n--- GraphDB 컨텍스트 ---")
    print(contexts['graph_context'])
    
    # Nomenclature 컨텍스트 출력
    if 'nomenclature_context' in contexts and contexts['nomenclature_context']:
        print("\n--- Nomenclature 컨텍스트 (HS 공식 명명법 문서) ---")
        print(contexts['nomenclature_context'])
    
    # HS Code 분류 실행 (항상 계층적 2단계 RAG 사용)
    print("\n=== HS Code 분류 실행 (계층적 2단계 RAG) ===")
    print("처리 중...")
    
    try:
        # 추론 시작 시간 측정
        start_time = time.perf_counter()
        
        # 계층적 2단계 RAG 사용
        result = classifier.classify_hs_code_hierarchical(
            product_name=args.name,
            product_description=args.desc,
            top_n=args.top_n,
            chroma_top_k=args.chroma_top_k,
            graph_k=args.graph_k
        )
        
        # 추론 종료 시간 측정
        end_time = time.perf_counter()
        inference_time = end_time - start_time

        # 계층형 Stage 컨텍스트 출력
        if isinstance(result, dict):
            print("\n=== 계층형 Stage 컨텍스트 ===")

            step1_codes = result.get("step1_6digit_codes")
            if step1_codes:
                print(f"- Stage1 (6자리) 예측 코드: {', '.join(step1_codes)}")

            stage2_graph = result.get("graphDB_context_step2") or ""
            if stage2_graph.strip():
                print("\n--- Stage2 입력용 GraphDB 컨텍스트 (Stage1 결과 반영) ---")
                print(stage2_graph)

            stage2_chroma = result.get("chromaDB_context_step2") or ""
            if stage2_chroma.strip():
                print("\n--- Stage2 입력용 VectorDB 컨텍스트 (Stage1 결과 반영) ---")
                print(stage2_chroma)
        
        # 최종 JSON에서는 컨텍스트 문자열 제거
        if isinstance(result, dict):
            context_keys = [k for k in list(result.keys()) if "context" in k.lower()]
            for key in context_keys:
                result.pop(key, None)
        
        print("\n=== 분류 결과 ===")
        print(f"추론 시간: {inference_time:.2f}초 ({inference_time:.3f}초)")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 결과에 시간 정보 추가
        if isinstance(result, dict):
            result["inference_time_seconds"] = round(inference_time, 3)
        
        return 0
        
    except Exception as e:
        print(f"오류: 분류 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

"""
사용 예시:


  # 실제 사용 예시
  python LLM/run_rag_final.py --name "오실로스코프(oscilloscope)와 오실로그래프(oscillograph)" --desc "(전자계측기). 대분류: Instruments, apparatus for measuring, checking electrical quantities not meters of heading no. 9028; instruments, apparatus for measuring or detecting alpha, beta, gamma, x-ray, cosmic and other radiations. 중분류: Oscilloscopes and oscillographs."
  
  python LLM/run_rag_final.py --name "LED 조명" --desc "알루미늄 하우징의 실내용 LED 조명기구, 220V 전원 사용"
  
  python LLM/run_rag_final.py --name "DISK NUT ASSY(TES62-300)" --desc "- 나선 가공한 홀(볼트 삽입용)을 가진 원형의 구리 합금 재질 물품 (신청 물품) - 재질: C3771(황동), 크기: M12(나사지름) × ∅106(전체 외경) × 15(두께) - 용도: 변압기의 누설 전류를 줄이는 탱크 실드(차폐판)를 탱크 내벽에 지지 및 고정하기 위해 사용"

  """