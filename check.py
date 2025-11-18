import argparse
import os
import pandas as pd
import numpy as np
from ast import literal_eval


def _normalize_code(value: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value)
    return s.replace(".", "").replace("-", "").strip()


def _parse_pred_list(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    s = str(value).strip()
    # 이미 리스트면 그대로
    if isinstance(value, list):
        return [
            _normalize_code(v) for v in value if _normalize_code(v)
        ]
    # 문자열로 직렬화된 리스트 처리
    try:
        parsed = literal_eval(s)
        if isinstance(parsed, list):
            return [
                _normalize_code(v) for v in parsed if _normalize_code(v)
            ]
    except Exception:
        pass
    # 콤마 구분 문자열 등 방어
    if "," in s:
        parts = [p.strip().strip("'") for p in s.split(",")]
        return [
            _normalize_code(p) for p in parts if _normalize_code(p)
        ]
    norm = _normalize_code(s)
    return [norm] if norm else []


def _any_prefix_match(gt: str, preds: list[str], prefix_len: int) -> bool:
    if not gt or not preds:
        return False
    gt_prefix = gt[:prefix_len]
    for p in preds:
        if p[:prefix_len] == gt_prefix and gt_prefix != "":
            return True
    return False


def add_prefix_match_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 컬럼 존재 가정: GT, Top1_pred, Top3_pred, Top5_pred
    gt_col = "GT"
    t1_col = "Top1_pred"
    t3_col = "Top3_pred"
    t5_col = "Top5_pred"

    def compute_row(row):
        gt = _normalize_code(row.get(gt_col, ""))
        top1_list = []
        top3_list = []
        top5_list = []

        # Top1: 단일 값 또는 리스트 첫 요소로 판단
        t1 = row.get(t1_col, None)
        if isinstance(t1, list):
            top1_list = [
                _normalize_code(t1[0])
            ] if len(t1) > 0 else []
        else:
            t1n = _normalize_code(t1)
            top1_list = [t1n] if t1n else []

        # Top3/Top5: 문자열로 직렬화된 리스트 가능
        top3_list = _parse_pred_list(row.get(t3_col, None))[:3]
        top5_list = _parse_pred_list(row.get(t5_col, None))[:5]

        res = {}
        for k in (2, 4, 6):
            res[f"Top1_prefix{k}"] = _any_prefix_match(gt, top1_list, k)
            res[f"Top3_prefix{k}"] = _any_prefix_match(gt, top3_list, k)
            res[f"Top5_prefix{k}"] = _any_prefix_match(gt, top5_list, k)
        return pd.Series(res)

    prefix_df = df.apply(compute_row, axis=1)
    return pd.concat([df, prefix_df], axis=1)


def summarize_prefix_accuracy(df: pd.DataFrame):
    summary = {}
    for scope in ("Top1", "Top3", "Top5"):
        for k in (2, 4, 6):
            col = f"{scope}_prefix{k}"
            if col in df:
                total = len(df)
                correct = int(df[col].sum())
                acc = correct / total if total else 0.0
                summary[col] = {
                    "correct": correct,
                    "total": total,
                    "accuracy": acc,
                }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prefix(2/4/6) 매칭률 계산")
    parser.add_argument(
        "--input",
        type=str,
        default="output/results/base+hybrid_1031/eval_result_hybrid.csv",
        help="입력 CSV 경로 (GT/Top1_pred/Top3_pred/Top5_pred 포함)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 CSV 저장 경로 (미지정 시 저장하지 않음)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {args.input}")

    df = pd.read_csv(args.input)
    df_aug = add_prefix_match_columns(df)

    # 요약 출력
    summary = summarize_prefix_accuracy(df_aug)
    print("=== Prefix 매칭 요약 ===")
    for key, val in summary.items():
        print(f"{key}: {val['accuracy']:.3f} ({val['correct']}/{val['total']})")

    # 저장 옵션
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        df_aug.to_csv(args.output, index=False)
        print(f"결과 저장: {args.output}")


if __name__ == "__main__":
    main()


