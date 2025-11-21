import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
import torch
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
        choices=["chroma", "graph", "both", "both+nomenclature"],
        default="both",
        help="사용할 DB 설정: 'chroma'(ChromaDB만), 'graph'(GraphDB만), 'both'(ChromaDB+GraphDB), 'both+nomenclature'(ChromaDB+GraphDB+Nomenclature)"
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
    
    # ===== HSClassifier 초기화 =====
    # parser 옵션 파싱: nomenclature 사용 여부 확인 (출력용)
    use_nomenclature = "+nomenclature" in args.parser
    parser_type = args.parser.replace("+nomenclature", "")
    
    print(f"=== HS Code 분류 평가 시작 ===")
    print(f"Parser 설정: {parser_type}")
    print(f"Nomenclature ChromaDB: {'사용' if use_nomenclature else '미사용'}")
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
    if (args.hierarchical or args.hierarchical_3stage) and parser_type not in ["graph", "both"]:
        print("오류: 계층적 모드는 GraphDB가 필요합니다. --parser를 'graph' 또는 'both'로 설정하세요.")
        return 1
    
    try:
        classifier = HSClassifier(
            parser_type=parser_type,
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
            translate_to_english=args.translate_to_english,
            use_nomenclature=use_nomenclature
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

            result_row = {
                'GT': gt_hs,
                'Top1_pred': pred_hs_list[0] if len(pred_hs_list) > 0 else None,
                'Top3_pred': pred_hs_list[:3],
                'Top5_pred': pred_hs_list,
                'Top1_match': match_top1,
                'Top3_match': match_top3,
                'Top5_match': match_top5
            }
            results.append(result_row)

            # 즉시 한 줄씩 결과 저장(append), 헤더는 최초 1회만
            pd.DataFrame([result_row]).to_csv(out_path, mode='a', header=not os.path.exists(out_path) or idx == 0, index=False)
            
        except Exception as e:
            result_row = {
                'GT': gt_hs,
                'Top1_pred': None,
                'Top3_pred': None,
                'Top5_pred': None,
                'Top1_match': False,
                'Top3_match': False,
                'Top5_match': False,
                'error': str(e)
            }
            results.append(result_row)
            pd.DataFrame([result_row]).to_csv(out_path, mode='a', header=not os.path.exists(out_path) or idx == 0, index=False)
    
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

### Input 키워드 추출하는 버전 ###

# ChromaDB만 사용
python LLM/evaluate.py --parser chroma --output-path "output/results/ChromaDB_keyword+input_keyword_1031/eval_result.csv"

# GraphDB만 사용
python LLM/evaluate.py --parser graph --output-path "output/results/graphDB+input_keyword_1031/eval_result.csv"

# 둘 다 사용
python LLM/evaluate.py --parser both --output-path "output/results/base_1117/eval_result.csv"


### openai_small 임베딩 모델 사용 버전 ###
# 둘 다 사용
python LLM/evaluate.py --parser both --embed-model openai_small --output-path "output/results/base_openai_small_1117/eval_result.csv"




### e5_small 임베딩 모델 사용 버전 ###
# 둘 다 사용
python LLM/evaluate.py --parser both --embed-model intfloat/multilingual-e5-small --output-path "output/results/base_e5_small_1117/eval_result.csv"


### 영어 번역 버전 ###

# ChromaDB만 사용
python LLM/evaluate.py --parser chroma --translate-to-english --output-path "output/results/ChromaDB_translate_1117/eval_result.csv"

# GraphDB만 사용
python LLM/evaluate.py --parser graph --translate-to-english --output-path "output/results/graphDB_translate_1117/eval_result.csv"

# 둘 다 사용
python LLM/evaluate.py --parser both --translate-to-english --output-path "output/results/base_translate_1117/eval_result.csv"


### Input 키워드 추출 사용 안 하는 버전 ###

# ChromaDB만 사용
python LLM/evaluate.py --parser chroma --no-keyword-extraction --output-path "output/results/no_keyword_input_chroma_1031/eval_result.csv"

# GraphDB만 사용
python LLM/evaluate.py --parser graph --no-keyword-extraction --output-path "output/results/no_keyword_input_graph_1031/eval_result.csv"

# 둘 다 사용
python LLM/evaluate.py --parser both --no-keyword-extraction --output-path "output/results/no_keyword_input_both_1031/eval_result.csv"

### ReRank 적용 버전 ###

# 둘 다 ReRank 적용
python LLM/evaluate.py --parser both --rerank --chroma-top-k 5 --rerank-top-m 5 --graph-k 5 --graph-rerank-top-m 5 --output-path "output/results/base+rerank_1104/eval_result_rerank.csv"

### hybrid search 적용 버전 ###

# ChromaDB만 사용
python LLM/evaluate.py --parser chroma --hybrid-rrf --output-path "output/results/base+chroma_hybrid_1031/eval_result_hybrid.csv"

# GraphDB만 사용
python LLM/evaluate.py --parser graph --hybrid-rrf --output-path "output/results/base+graph_hybrid_1031/eval_result_hybrid.csv"

# 둘 다 사용
python LLM/evaluate.py --parser both --hybrid-rrf --output-path "output/results/base+hybrid_1117/eval_result_hybrid.csv"


### Rerank + Hybrid Search 적용 버전 ###
python LLM/evaluate.py --parser both --rerank --graph-rerank \
    --chroma-top-k 10 --rerank-top-m 5 --graph-k 10 --graph-rerank-top-m 5 \
    --hybrid-rrf --output-path "output/results/base+rerank_hybrid_1031/eval_result_rerank_hybrid.csv"

# LLM ReRank + Hybrid Search 적용 버전
python LLM/evaluate.py --parser both --hybrid-rrf --bm25-k 5 --rrf-k 60 --llm-listwise --llm-listwise-max-cand 16 --llm-listwise-window 10 --llm-listwise-step 5 --llm-listwise-top-m 5 --chroma-top-k 10 --graph-k 8 --output-path "output/results/llm_rerank_hybrid_1031/eval_result_llm_rerank_hybrid.csv"

### 계층적 2단계 RAG 버전 ###

# 계층적 모드 (GraphDB 또는 both 필요)
python LLM/evaluate.py --parser both --hierarchical --output-path "output/results/hierarchical_2stage/eval_result_hierarchical.csv"

# 계층적 모드 + ReRank
python LLM/evaluate.py --parser both --hierarchical --rerank --graph-rerank --chroma-top-k 10 --rerank-top-m 5 --graph-k 10 --graph-rerank-top-m 5 --output-path "output/results/hierarchical_2stage+rerank/eval_result_hierarchical_rerank.csv"

# 계층적 모드 + 영어
python LLM/evaluate.py --parser both --hierarchical --translate-to-english --output-path "output/results/hierarchical_2stage+translate/eval_result_hierarchical_translate.csv"

# 계층적 모드 + Hybrid Search
python LLM/evaluate.py --parser both --hierarchical --hybrid-rrf --output-path "output/results/hierarchical_2stage+hybrid/eval_result_hierarchical_hybrid.csv"

# 계층적 모드 + Hybrid Search + 영어 번역
python LLM/evaluate.py --parser both --hierarchical --hybrid-rrf --translate-to-english --output-path "output/results/hierarchical_2stage+hybrid+translate+[]/eval_result_hierarchical_hybrid_translate.csv"


### 계층적 3단계 RAG 버전 ###

# 계층적 3단계 모드 (GraphDB 또는 both 필요)
python LLM/evaluate.py --parser both --hierarchical-3stage --output-path "output/results/hierarchical_3stage/eval_result_hierarchical_3stage.csv"

# 계층적 3단계 모드 + ReRank
python LLM/evaluate.py --parser both --hierarchical-3stage --rerank --graph-rerank --chroma-top-k 10 --rerank-top-m 5 --graph-k 10 --graph-rerank-top-m 5 --output-path "output/results/hierarchical_3stage+rerank/eval_result_hierarchical_3stage_rerank.csv"

# 계층적 3단계 모드 + Hybrid Search
python LLM/evaluate.py --parser both --hierarchical-3stage --hybrid-rrf --output-path "output/results/hierarchical_3stage+hybrid/eval_result_hierarchical_3stage_hybrid.csv"

python LLM/evaluate.py --parser both --hierarchical-3stage --embed-model openai_large --output-path "output/results/hierarchical_3stage_openai_large/eval_result_hierarchical_3stage.csv"

# 2stage openai_large case eval dataset 버전
python LLM/evaluate.py --parser both --hierarchical --embed-model openai_large --data-path "data/only_case_100sample_1118.csv" --output-path "output/results/hierarchical_2stage_openai_large_casedataset_1118/eval_result_hierarchical_2stage.csv"




### openai_large 임베딩 모델 사용 버전 ###
# 둘 다 사용
python LLM/evaluate.py --parser both --embed-model openai_large --output-path "output/results/base_openai_large_1121/eval_result.csv"

# 2stage
python LLM/evaluate.py --parser both --hierarchical --embed-model openai_large --output-path "output/results/hierarchical_2stage_openai_large_rule/eval_result.csv"

# 2stage + nomenclature
python LLM/evaluate.py --parser both+nomenclature --hierarchical --embed-model openai_large --output-path "output/results/hierarchical_2stage_openai_large_rule_nomenclature/eval_result.csv"

# 3stage
python LLM/evaluate.py --parser both --hierarchical-3stage --embed-model openai_large --output-path "output/results/hierarchical_3stage_openai_large/eval_result.csv"

# 3stage + nomenclature
python LLM/evaluate.py --parser both+nomenclature --hierarchical-3stage --embed-model openai_large --output-path "output/results/hierarchical_3stage_openai_large_rule_nomenclature/eval_result.csv"
"""