import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from rag_module import HSClassifier

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
    
    args = parser.parse_args()
    
    # ===== 데이터 로딩 =====
    if args.data_path:
        DATA_PATH = args.data_path
    else:
        DATA_PATH = "data/eval_dataset_1031.csv"
    
    df = pd.read_csv(DATA_PATH)
    
    # 평가를 위한 주요 컬럼
    PRODUCT_NAME_COL = '사용자_상품명'
    PRODUCT_DESC_COL = '사용자_상품설명'
    GT_HSCODE_COL = 'HS부호'
    
    # ===== 결과 저장 경로 선언 =====
    if args.output_path:
        out_path = args.output_path
    else:
        out_path = "output/results/eval_result.csv"
    
    if os.path.exists(out_path):
        os.remove(out_path)
    
    # ===== HSClassifier 초기화 =====
    print(f"=== HS Code 분류 평가 시작 ===")
    print(f"Parser 설정: {args.parser}")
    print(f"키워드 추출: {'사용' if args.use_keyword_extraction else '미사용'}")
    print(f"데이터셋: {DATA_PATH}")
    print(f"결과 저장: {out_path}")
    print()
    
    try:
        classifier = HSClassifier(
            parser_type=args.parser,
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
    
    # ===== 평가 루프 =====
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="평가 진행"):
        prod_name = str(row[PRODUCT_NAME_COL])
        prod_desc = str(row[PRODUCT_DESC_COL])
        gt_hs = str(row[GT_HSCODE_COL])
        try:
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
            gt_hs_10 = gt_hs.replace('.', '').replace('-', '')[:10]
            match_top1 = (len(pred_hs_list) > 0 and pred_hs_list[0][:10] == gt_hs_10)
            match_top3 = any(hs[:10] == gt_hs_10 for hs in pred_hs_list[:3])
            match_top5 = any(hs[:10] == gt_hs_10 for hs in pred_hs_list[:5])
            correct_top1 += int(match_top1)
            correct_top3 += int(match_top3)
            correct_top5 += int(match_top5)

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
        f"세부 CSV: {out_path}\n",
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.writelines(summary_lines)
    print(f"요약 결과 저장: {summary_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())

"""

### 키워드 추출하는 버전 ###

# ChromaDB만 사용
python LLM/evaluate.py --parser chroma

# GraphDB만 사용
python LLM/evaluate.py --parser graph

# 둘 다 사용
python LLM/evaluate.py --parser both

### 키워드 추출 사용 안 하는 버전 ###

# ChromaDB만 사용
python LLM/evaluate.py --parser chroma --no-keyword-extraction

# GraphDB만 사용
python LLM/evaluate.py --parser graph --no-keyword-extraction

# 둘 다 사용
python LLM/evaluate.py --parser both --no-keyword-extraction

### ReRank 적용 버전 ###

# ChromaDB ReRank 적용
python LLM/evaluate.py --parser chroma --no-keyword-extraction --rerank

# GraphDB ReRank 적용
python LLM/evaluate.py --parser graph --no-keyword-extraction --graph-rerank

# 둘 다 ReRank 적용
python LLM/evaluate.py --parser both --no-keyword-extraction --rerank --graph-rerank

"""