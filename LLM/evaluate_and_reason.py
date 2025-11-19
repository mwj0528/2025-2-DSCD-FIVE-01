import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
import torch
import json
from rag_module import HSClassifier, set_all_seeds

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
    # ===== 커맨드라인 인자 파싱 =====
    parser = argparse.ArgumentParser(description="HS Code 분류 RAG 평가 스크립트")
    parser.add_argument(
        "--parser",
        type=str,
        choices=["chroma", "graph", "both"],
        default="both",
        help="사용할 DB 설정: 'chroma'(ChromaDB만), 'graph'(GraphDB만), 'both'(둘 다, 기본값)"
    )
    parser.add_argument(
        "--no-keyword-extraction",
        action="store_false",
        dest="use_keyword_extraction",
        help="ChromaDB 검색 시 키워드 추출 사용 안 함 (기본값: 키워드 추출 사용)"
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
        "--data-path",
        type=str,
        default=None,
        help="평가 데이터셋 경로 (기본값: ../data/eval_dataset.csv)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="결과 저장 경로 (기본값: ../output/results/eval_result.csv)"
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
    # Graph 하이브리드는 공통 스위치(--hybrid-rrf)와 공통 K(--bm25-k, --rrf-k)를 그대로 사용합니다
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
    parser.add_argument(
        "--embed-model",
        type=str,
        choices=list(EMBED_MODEL_CHOICES.keys()),
        default="paraphrase-multilingual-minilm-l12-v2",
        help="쿼리 임베딩에 사용할 SentenceTransformer/OpenAI 모델",
    )
    parser.add_argument(
        "--reason",
        action="store_true",
        help="CSV에 RAG context 정보 저장 (기본: 미저장)"
    )
    
    args = parser.parse_args()
    
    # ===== 재현성을 위한 랜덤 시드 설정 =====
    # 환경변수에서 seed 가져오기 (기본값: 42)
    seed = int(os.getenv("SEED", "42"))
    set_all_seeds(seed)
    print(f"랜덤 시드 고정: {seed}")
    
    # ===== 데이터 로딩 =====
    if args.data_path:
        DATA_PATH = args.data_path
    else:
        DATA_PATH = "data/eval_dataset_1031.csv"
    
    df = pd.read_csv(DATA_PATH)
    # 재현성을 위해 데이터 순서 고정 (인덱스 기준 정렬)
    df = df.sort_index().reset_index(drop=True)
    
    # 평가를 위한 주요 컬럼
    PRODUCT_NAME_COL = '사용자_상품명'
    PRODUCT_DESC_COL = '사용자_상품설명'
    GT_HSCODE_COL = 'HS부호'
    
    # ===== 결과 저장 경로 선언 =====
    if args.output_path:
        out_path = args.output_path
    else:
        out_path = "output/results/eval_result.csv"
    # 출력 디렉터리 생성(없으면 생성)
    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    except Exception:
        pass
    
    if os.path.exists(out_path):
        os.remove(out_path)
    
    # JSON 파일 경로 (--reason 옵션이 있을 때만)
    json_path = None
    if args.reason:
        json_path = os.path.splitext(out_path)[0] + "_context.json"
        if os.path.exists(json_path):
            os.remove(json_path)
    
    # ===== HSClassifier 초기화 =====
    print(f"=== HS Code 분류 평가 시작 ===")
    print(f"Parser 설정: {args.parser}")
    resolved_chroma_dir = EMBED_CHROMA_DIR.get(args.embed_model)
    print(f"키워드 추출: {'사용' if args.use_keyword_extraction else '미사용'}")
    print(f"영어 번역: {'사용' if args.translate_to_english else '미사용'}")
    print(f"임베딩 모델: {args.embed_model}")
    if resolved_chroma_dir:
        print(f"ChromaDB 디렉터리: {resolved_chroma_dir}")
    if args.hierarchical_3stage:
        print(f"계층적 모드: 3단계 사용")
    elif args.hierarchical:
        print(f"계층적 모드: 2단계 사용")
    else:
        print(f"계층적 모드: 미사용")
    print(f"데이터셋: {DATA_PATH}")
    print(f"결과 저장: {out_path}")
    print()
    
    # Hybrid 설정: 하나의 스위치/파라미터로 두 DB 모두에 적용
    unified_hybrid = args.hybrid_rrf
    unified_bm25_k = args.bm25_k
    unified_rrf_k = args.rrf_k

    # 계층적 모드 검증
    if (args.hierarchical or args.hierarchical_3stage) and args.parser not in ["graph", "both"]:
        print("오류: 계층적 모드는 GraphDB가 필요합니다. --parser를 'graph' 또는 'both'로 설정하세요.")
        return 1
    
    try:
        classifier = HSClassifier(
            parser_type=args.parser,
            chroma_dir=resolved_chroma_dir,
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
    
    # ===== 평가 루프 =====
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    # Prefix 정확도 집계를 위한 카운터 (CSV는 변경하지 않음)
    prefix_lengths = [2, 4, 6, 10]
    prefix_correct = {
        'Top1': {k: 0 for k in prefix_lengths},
        'Top3': {k: 0 for k in prefix_lengths},
        'Top5': {k: 0 for k in prefix_lengths},
    }
    results = []
    # context 확인을 위한 샘플 출력 개수 (--reason 옵션이 있을 때만)
    context_sample_count = 3 if args.reason else 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="평가 진행"):
        prod_name = str(row[PRODUCT_NAME_COL])
        prod_desc = str(row[PRODUCT_DESC_COL])
        gt_hs = str(row[GT_HSCODE_COL])
        try:
            if args.hierarchical_3stage:
                # 계층적 3단계 RAG 사용
                pred = classifier.classify_hs_code_hierarchical_3stage(
                    product_name=prod_name,
                    product_description=prod_desc,
                    top_n=5,
                    chroma_top_k=args.chroma_top_k,
                    graph_k=args.graph_k
                )
            elif args.hierarchical:
                # 계층적 2단계 RAG 사용
                pred = classifier.classify_hs_code_hierarchical(
                    product_name=prod_name,
                    product_description=prod_desc,
                    top_n=5,
                    chroma_top_k=args.chroma_top_k,
                    graph_k=args.graph_k
                )
            else:
                # 기존 1단계 RAG 사용
                pred = classifier.classify_hs_code(
                    product_name=prod_name,
                    product_description=prod_desc,
                    top_n=5,
                    chroma_top_k=args.chroma_top_k,
                    graph_k=args.graph_k
                )
            candidates = pred.get('candidates', [])

            # 비교 단순화: gt/예측 모두 점/하이픈 제거 후 비교
            pred_hs_list = [str(c['hs_code']).replace('.', '').replace('-', '') for c in candidates]
            gt_clean = gt_hs.replace('.', '').replace('-', '')
            gt_hs_10 = gt_clean[:10]
            match_top1 = (len(pred_hs_list) > 0 and pred_hs_list[0][:10] == gt_hs_10)
            match_top3 = any(hs[:10] == gt_hs_10 for hs in pred_hs_list[:3])
            match_top5 = any(hs[:10] == gt_hs_10 for hs in pred_hs_list[:5])
            correct_top1 += int(match_top1)
            correct_top3 += int(match_top3)
            correct_top5 += int(match_top5)

            # Prefix(2/4/6/10) 매칭 집계 (CSV는 변경하지 않음)
            def any_prefix_match(gt: str, preds: list, k: int) -> bool:
                if not gt or not preds:
                    return False
                gtp = gt[:k]
                return any(str(p)[:k] == gtp for p in preds)

            top1_list = pred_hs_list[:1]
            top3_list = pred_hs_list[:3]
            top5_list = pred_hs_list[:5]

            for k in prefix_lengths:
                prefix_correct['Top1'][k] += int(any_prefix_match(gt_clean, top1_list, k))
                prefix_correct['Top3'][k] += int(any_prefix_match(gt_clean, top3_list, k))
                prefix_correct['Top5'][k] += int(any_prefix_match(gt_clean, top5_list, k))

            # context 정보 추출 (--reason 옵션이 있을 때만)
            # 리스트를 문자열로 변환 (세미콜론으로 구분)
            top3_pred_str = ';'.join(pred_hs_list[:3]) if len(pred_hs_list) >= 3 else ';'.join(pred_hs_list)
            top5_pred_str = ';'.join(pred_hs_list[:5]) if len(pred_hs_list) >= 5 else ';'.join(pred_hs_list)
            
            result_row = {
                'GT': gt_hs,
                'Top1_pred': pred_hs_list[0] if len(pred_hs_list) > 0 else '',
                'Top3_pred': top3_pred_str,
                'Top5_pred': top5_pred_str,
                'Top1_match': match_top1,
                'Top3_match': match_top3,
                'Top5_match': match_top5
            }
            
            # context 정보는 JSON에 저장 (--reason 옵션이 있을 때만)
            if args.reason:
                context_entry = {
                    'index': int(idx),
                    'product_name': prod_name,
                    'product_description': prod_desc,
                    'GT': gt_hs,
                    'Top1_pred': pred_hs_list[0] if len(pred_hs_list) > 0 else '',
                    'Top3_pred': pred_hs_list[:3],
                    'Top5_pred': pred_hs_list[:5]
                }
                
                # 계층적 모드인 경우 각 단계의 context 저장
                if args.hierarchical_3stage:
                    # 3단계: step1, step2, step3 모두 저장
                    chroma_ctx_step1 = pred.get('chromaDB_context_step1', '')
                    graph_ctx_step1 = pred.get('graphDB_context_step1', '')
                    chroma_ctx_step2 = pred.get('chromaDB_context_step2', '')
                    graph_ctx_step2 = pred.get('graphDB_context_step2', '')
                    chroma_ctx_step3 = pred.get('chromaDB_context_step3', '')
                    graph_ctx_step3 = pred.get('graphDB_context_step3', '')
                    
                    context_entry['step1'] = {
                        'chromaDB_context': chroma_ctx_step1,
                        'graphDB_context': graph_ctx_step1
                    }
                    context_entry['step2'] = {
                        'chromaDB_context': chroma_ctx_step2,
                        'graphDB_context': graph_ctx_step2
                    }
                    context_entry['step3'] = {
                        'chromaDB_context': chroma_ctx_step3,
                        'graphDB_context': graph_ctx_step3
                    }
                    
                    # 처음 몇 개 샘플의 context 출력
                    if idx < context_sample_count:
                        print(f"\n=== 샘플 {idx+1}: Context 확인 ===")
                        print(f"상품명: {prod_name[:50]}...")
                        print(f"GT: {gt_hs}")
                        print(f"\n[1단계 - 4자리 예측]")
                        print(f"  ChromaDB context 길이: {len(chroma_ctx_step1)} 문자")
                        print(f"  GraphDB context 길이: {len(graph_ctx_step1)} 문자")
                        if chroma_ctx_step1:
                            print(f"  ChromaDB context 미리보기: {chroma_ctx_step1[:200]}...")
                        if graph_ctx_step1:
                            print(f"  GraphDB context 미리보기: {graph_ctx_step1[:200]}...")
                        print(f"\n[2단계 - 6자리 예측]")
                        print(f"  ChromaDB context 길이: {len(chroma_ctx_step2)} 문자")
                        print(f"  GraphDB context 길이: {len(graph_ctx_step2)} 문자")
                        if chroma_ctx_step2:
                            print(f"  ChromaDB context 미리보기: {chroma_ctx_step2[:200]}...")
                        if graph_ctx_step2:
                            print(f"  GraphDB context 미리보기: {graph_ctx_step2[:200]}...")
                        print(f"\n[3단계 - 10자리 예측]")
                        print(f"  ChromaDB context 길이: {len(chroma_ctx_step3)} 문자")
                        print(f"  GraphDB context 길이: {len(graph_ctx_step3)} 문자")
                        if chroma_ctx_step3:
                            print(f"  ChromaDB context 미리보기: {chroma_ctx_step3[:200]}...")
                        if graph_ctx_step3:
                            print(f"  GraphDB context 미리보기: {graph_ctx_step3[:200]}...")
                        print()
                elif args.hierarchical:
                    # 2단계: step1, step2 모두 저장
                    chroma_ctx_step1 = pred.get('chromaDB_context_step1', '')
                    graph_ctx_step1 = pred.get('graphDB_context_step1', '')
                    chroma_ctx_step2 = pred.get('chromaDB_context_step2', '')
                    graph_ctx_step2 = pred.get('graphDB_context_step2', '')
                    
                    context_entry['step1'] = {
                        'chromaDB_context': chroma_ctx_step1,
                        'graphDB_context': graph_ctx_step1
                    }
                    context_entry['step2'] = {
                        'chromaDB_context': chroma_ctx_step2,
                        'graphDB_context': graph_ctx_step2
                    }
                    
                    # 처음 몇 개 샘플의 context 출력
                    if idx < context_sample_count:
                        print(f"\n=== 샘플 {idx+1}: Context 확인 ===")
                        print(f"상품명: {prod_name[:50]}...")
                        print(f"GT: {gt_hs}")
                        print(f"\n[1단계 - 6자리 예측]")
                        print(f"  ChromaDB context 길이: {len(chroma_ctx_step1)} 문자")
                        print(f"  GraphDB context 길이: {len(graph_ctx_step1)} 문자")
                        if chroma_ctx_step1:
                            print(f"  ChromaDB context 미리보기: {chroma_ctx_step1[:200]}...")
                        if graph_ctx_step1:
                            print(f"  GraphDB context 미리보기: {graph_ctx_step1[:200]}...")
                        print(f"\n[2단계 - 10자리 예측]")
                        print(f"  ChromaDB context 길이: {len(chroma_ctx_step2)} 문자")
                        print(f"  GraphDB context 길이: {len(graph_ctx_step2)} 문자")
                        if chroma_ctx_step2:
                            print(f"  ChromaDB context 미리보기: {chroma_ctx_step2[:200]}...")
                        if graph_ctx_step2:
                            print(f"  GraphDB context 미리보기: {graph_ctx_step2[:200]}...")
                        print()
                else:
                    # 일반 모드
                    chroma_context = pred.get('chromaDB_context', '')
                    graph_context = pred.get('graphDB_context', '')
                    
                    context_entry['chromaDB_context'] = chroma_context
                    context_entry['graphDB_context'] = graph_context
                    
                    # 처음 몇 개 샘플의 context 출력
                    if idx < context_sample_count:
                        print(f"\n=== 샘플 {idx+1}: Context 확인 ===")
                        print(f"상품명: {prod_name[:50]}...")
                        print(f"GT: {gt_hs}")
                        print(f"  ChromaDB context 길이: {len(chroma_context)} 문자")
                        print(f"  GraphDB context 길이: {len(graph_context)} 문자")
                        if chroma_context:
                            print(f"  ChromaDB context 미리보기: {chroma_context[:200]}...")
                        if graph_context:
                            print(f"  GraphDB context 미리보기: {graph_context[:200]}...")
                        print()
                
                # JSON 파일에 한 개씩 저장 (pretty print 형식, 각 속성 줄넘김)
                # 문자열 내부의 \n을 실제 줄넘김으로 변환 (JSON 표준은 아니지만 보기 좋게)
                if json_path:
                    def convert_newlines_in_strings(obj):
                        """문자열 내부의 \n을 실제 줄넘김으로 변환"""
                        if isinstance(obj, dict):
                            return {k: convert_newlines_in_strings(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_newlines_in_strings(item) for item in obj]
                        elif isinstance(obj, str):
                            # \\n (이스케이프된 줄넘김)을 실제 줄넘김으로 변환
                            return obj.replace('\\n', '\n')
                        return obj
                    
                    # JSON 문자열로 변환
                    json_str = json.dumps(context_entry, ensure_ascii=False, indent=2)
                    # JSON 문자열 내부의 이스케이프된 \n을 실제 줄넘김으로 변환
                    # 하지만 JSON 문자열 값 내부의 \n만 변환해야 함
                    import re
                    # JSON 문자열 값 내부의 \\n을 실제 줄넘김으로 변환
                    # "key": "value\\nmore" -> "key": "value\nmore"
                    json_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrt])', '', json_str)  # 잘못된 이스케이프 제거
                    json_str = json_str.replace('\\n', '\n')  # 모든 \n을 실제 줄넘김으로
                    
                    with open(json_path, 'a', encoding='utf-8') as f:
                        f.write(json_str)
                        f.write('\n\n')  # 객체 사이에 빈 줄 추가
            
            results.append(result_row)

            # 즉시 한 줄씩 결과 저장(append), 헤더는 최초 1회만
            # 파일이 존재하지 않을 때만 utf-8-sig (BOM 포함) 사용하여 Excel 호환성 확보
            encoding = 'utf-8-sig' if not os.path.exists(out_path) else 'utf-8'
            # CSV 저장 시 줄바꿈과 특수문자 처리 (quoting=1은 QUOTE_ALL)
            pd.DataFrame([result_row]).to_csv(
                out_path, 
                mode='a', 
                header=not os.path.exists(out_path) or idx == 0, 
                index=False, 
                encoding=encoding,
                quoting=1,  # QUOTE_ALL: 모든 필드를 따옴표로 감싸기
                escapechar=None  # 기본 이스케이프 사용
            )
            
        except Exception as e:
            result_row = {
                'GT': gt_hs,
                'Top1_pred': '',
                'Top3_pred': '',
                'Top5_pred': '',
                'Top1_match': False,
                'Top3_match': False,
                'Top5_match': False,
                'error': str(e)
            }
            # 에러 발생 시에도 context 데이터에 추가 (--reason 옵션이 있을 때만)
            if args.reason:
                context_entry = {
                    'index': int(idx),
                    'product_name': prod_name,
                    'product_description': prod_desc,
                    'GT': gt_hs,
                    'error': str(e)
                }
                # JSON 파일에 한 개씩 저장 (pretty print 형식, 각 속성 줄넘김)
                # 문자열 내부의 \n을 실제 줄넘김으로 변환 (JSON 표준은 아니지만 보기 좋게)
                if json_path:
                    def convert_newlines_in_strings(obj):
                        """문자열 내부의 \n을 실제 줄넘김으로 변환"""
                        if isinstance(obj, dict):
                            return {k: convert_newlines_in_strings(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_newlines_in_strings(item) for item in obj]
                        elif isinstance(obj, str):
                            # \\n (이스케이프된 줄넘김)을 실제 줄넘김으로 변환
                            return obj.replace('\\n', '\n')
                        return obj
                    
                    # JSON 문자열로 변환
                    json_str = json.dumps(context_entry, ensure_ascii=False, indent=2)
                    # JSON 문자열 내부의 이스케이프된 \n을 실제 줄넘김으로 변환
                    # 하지만 JSON 문자열 값 내부의 \n만 변환해야 함
                    import re
                    # JSON 문자열 값 내부의 \\n을 실제 줄넘김으로 변환
                    # "key": "value\\nmore" -> "key": "value\nmore"
                    json_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrt])', '', json_str)  # 잘못된 이스케이프 제거
                    json_str = json_str.replace('\\n', '\n')  # 모든 \n을 실제 줄넘김으로
                    
                    with open(json_path, 'a', encoding='utf-8') as f:
                        f.write(json_str)
                        f.write('\n\n')  # 객체 사이에 빈 줄 추가
            results.append(result_row)
            # 파일이 존재하지 않을 때만 utf-8-sig (BOM 포함) 사용하여 Excel 호환성 확보
            encoding = 'utf-8-sig' if not os.path.exists(out_path) else 'utf-8'
            # CSV 저장 시 줄바꿈과 특수문자 처리 (quoting=1은 QUOTE_ALL)
            pd.DataFrame([result_row]).to_csv(
                out_path, 
                mode='a', 
                header=not os.path.exists(out_path) or idx == 0, 
                index=False, 
                encoding=encoding,
                quoting=1,  # QUOTE_ALL: 모든 필드를 따옴표로 감싸기
                escapechar=None  # 기본 이스케이프 사용
            )
    
    # ===== 결과 집계 및 출력 =====
    total = len(df)
    top1_acc = correct_top1 / total if total > 0 else 0
    top3_acc = correct_top3 / total if total > 0 else 0
    top5_acc = correct_top5 / total if total > 0 else 0
    
    print("\n=== 평가 결과 ===")
    print(f"총 샘플 수: {total}")
    print(f"Top-1 정확도: {top1_acc:.3f} ({correct_top1}/{total})")
    print(f"Top-3 정확도: {top3_acc:.3f} ({correct_top3}/{total})")
    print(f"Top-5 정확도: {top5_acc:.3f} ({correct_top5}/{total})")
    print(f"상세 결과 저장: {out_path}")
    
    # JSON 파일 저장 완료 메시지 (--reason 옵션이 있을 때만)
    if args.reason and json_path:
        print(f"Context 정보 저장: {json_path} (Pretty print 형식, 각 속성 줄넘김, 한 개씩 저장됨)")
    
    # ===== 요약 결과 TXT 저장 =====
    summary_path = os.path.splitext(out_path)[0] + "_summary.txt"
    summary_lines = [
        "=== 평가 결과 요약 ===\n",
        f"총 샘플 수: {total}\n",
        f"Top-1 정확도: {top1_acc:.3f} ({correct_top1}/{total})\n",
        f"Top-3 정확도: {top3_acc:.3f} ({correct_top3}/{total})\n",
        f"Top-5 정확도: {top5_acc:.3f} ({correct_top5}/{total})\n",
        "Top-1 정확도\n",
        f"2자리 정확도: { (prefix_correct['Top1'][2] / total if total else 0):.3f} ({prefix_correct['Top1'][2]}/{total})\n",
        f"4자리 정확도: { (prefix_correct['Top1'][4] / total if total else 0):.3f} ({prefix_correct['Top1'][4]}/{total})\n",
        f"6자리 정확도: { (prefix_correct['Top1'][6] / total if total else 0):.3f} ({prefix_correct['Top1'][6]}/{total})\n",
        f"10자리 정확도: { (prefix_correct['Top1'][10] / total if total else 0):.3f} ({prefix_correct['Top1'][10]}/{total})\n",
        "Top-3 정확도\n",
        f"2자리 정확도: { (prefix_correct['Top3'][2] / total if total else 0):.3f} ({prefix_correct['Top3'][2]}/{total})\n",
        f"4자리 정확도: { (prefix_correct['Top3'][4] / total if total else 0):.3f} ({prefix_correct['Top3'][4]}/{total})\n",
        f"6자리 정확도: { (prefix_correct['Top3'][6] / total if total else 0):.3f} ({prefix_correct['Top3'][6]}/{total})\n",
        f"10자리 정확도: { (prefix_correct['Top3'][10] / total if total else 0):.3f} ({prefix_correct['Top3'][10]}/{total})\n",
        "Top-5 정확도\n",
        f"2자리 정확도: { (prefix_correct['Top5'][2] / total if total else 0):.3f} ({prefix_correct['Top5'][2]}/{total})\n",
        f"4자리 정확도: { (prefix_correct['Top5'][4] / total if total else 0):.3f} ({prefix_correct['Top5'][4]}/{total})\n",
        f"6자리 정확도: { (prefix_correct['Top5'][6] / total if total else 0):.3f} ({prefix_correct['Top5'][6]}/{total})\n",
        f"10자리 정확도: { (prefix_correct['Top5'][10] / total if total else 0):.3f} ({prefix_correct['Top5'][10]}/{total})\n",
        f"세부 CSV: {out_path}\n",
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.writelines(summary_lines)
    print(f"요약 결과 저장: {summary_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())

"""


### openai_large 임베딩 모델 사용 버전 ###

# 2stage
python LLM/evaluate_and_reason.py --parser both --hierarchical --embed-model openai_large --reason --output-path "output/results/hierarchical_2stage_openai_large_reason_1119/eval_result.csv"


"""