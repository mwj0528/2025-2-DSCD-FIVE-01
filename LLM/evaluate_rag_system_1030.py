# evaluate_rag_system.py
"""
RAG 기반 HS Code 추천 시스템 성능 평가 (간결/안정 버전)
- Top-1, Top-3, Top-5 정확도만 계산/출력
- 행별 타임아웃/하트비트 로그/샘플 제한 지원
- 필요 시 요약 JSON만 저장

실행:
    # 첫 실행(스모크 테스트 권장)
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 \
    EVAL_MAX_SAMPLES=5 EVAL_TIMEOUT_SEC=45 \
    python evaluate_rag_system.py

필수:
    1) RAG.py 또는 rag_hs_prompt.py 내 classify_hs_code_rag 함수가 import 가능
    2) HScode_100개_filled.xlsx 존재(혹은 상위 폴더)
    3) .env 에 OPENAI_API_KEY 설정
"""

import os
# 교착/과점유 방지 권장 설정(없으면 기본값 사용)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import re
import sys
import json
import time
import signal
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

# =========================
# RAG 모듈 import (RAG.py → rag_hs_prompt.py 순서로 시도)
# 이 함수가 VectorDB + GraphDB + LLM의 전체 과정을 실행
# =========================
try:
    from RAG import classify_hs_code_rag  # 사용자 환경 우선
except ImportError:
    try:
        from rag_hs_prompt import classify_hs_code_rag  # 대체 경로
    except ImportError:
        print("❌ classify_hs_code_rag 임포트 실패. RAG.py 또는 rag_hs_prompt.py 경로를 확인하세요.")
        sys.exit(1)


class HSCodeEvaluator:
    def __init__(self, excel_path: str = None):
        # 디버그/성능 제어 환경변수
        self.max_samples = int(os.getenv("EVAL_MAX_SAMPLES", "0"))     # 0이면 전체
        self.per_item_timeout = int(os.getenv("EVAL_TIMEOUT_SEC", "45"))  # 행별 타임아웃(초)
        self.top_n = int(os.getenv("EVAL_TOP_N", "5"))                 # 예측 상한(정확도는 1/3/5만 계산)

        # 엑셀 경로 자동 탐색
        if excel_path is None:
            for p in [
                "HScode_100개_with_user_input.xlsx",
                "../HScode_100개_with_user_input.xlsx",
                "../../HScode_100개_with_user_input.xlsx",
            ]:
                if os.path.exists(p):
                    excel_path = p
                    break
        if excel_path is None or not os.path.exists(excel_path):
            raise FileNotFoundError("HScode_100개_with_user_input.xlsx 파일을 찾을 수 없습니다.")
        self.excel_path = excel_path

    @staticmethod
    def normalize_hs(code: Optional[str], keep_digits: int = 10) -> Optional[str]:
        if code is None or (isinstance(code, float) and np.isnan(code)):
            return None 
        digits = re.sub(r"[^0-9]", "", str(code)) # 점 제거, 10자리 맞춤
        if not digits:
            return None
        return (digits[:keep_digits]).ljust(keep_digits, "0")

    def load_test_data(self) -> pd.DataFrame:
        df = pd.read_excel(self.excel_path)

        required_cols = {
            'id': ['번호', 'id', 'ID'],
            'product_name': ['사용자_상품명', '상품명'],
            'product_desc': ['사용자_상품설명', '상품설명'],
            'gold_hs': ['HS부호', 'HSCode', 'HS코드']
        }
        # 실제 엑셀에 있는 컬럼명 찾기
        col_map = {}
        for target, cands in required_cols.items():
            for c in cands:
                if c in df.columns:
                    col_map[c] = target
                    break
            else:
                raise ValueError(f"필수 컬럼 누락: {cands}")

        test_df = df[list(col_map.keys())].rename(columns=col_map)
        test_df['gold_hs'] = test_df['gold_hs'].apply(lambda x: self.normalize_hs(x, 10))
        test_df = test_df.dropna(subset=['product_name', 'product_desc', 'gold_hs'])

        if self.max_samples > 0:
            test_df = test_df.head(self.max_samples)
            print(f"🔎 디버그 모드: 상위 {len(test_df)}개만 평가(EVAL_MAX_SAMPLES).")

        return test_df

    def generate_predictions(self, test_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """조용히 예측만 수행 + 행별 타임아웃 + 하트비트 로그"""
        preds: List[Dict] = []

        class _Timeout(Exception):
            ...

        def _handler(signum, frame):
            raise _Timeout()

        signal.signal(signal.SIGALRM, _handler)

        total = len(test_df)
        for i, (_, row) in enumerate(test_df.iterrows(), start=1):
            product_name = str(row['product_name']).strip()
            product_desc = str(row['product_desc']).strip()
            pred_list: List[str] = []

            try:
                signal.alarm(self.per_item_timeout)  # ⏱️ 행별 타임아웃
                t0 = time.time()
                result = classify_hs_code_rag(       # RAG 시스템 호출 (VectorDB + GraphDB + LLM 전체 과정)
                    product_name=product_name,
                    product_description=product_desc,
                    top_n=top_n
                )
                signal.alarm(0)

                if isinstance(result, dict):
                    for cand in result.get("candidates", []):
                        norm = self.normalize_hs(cand.get("hs_code", ""), 10)
                        if norm:
                            pred_list.append(norm)

                # 5개마다 진행 상황 출력 (하트비트)
                if i % 5 == 0 or i == total:
                    dt = time.time() - t0
                    print(f"   · 진행 {i}/{total} (last {dt:.1f}s, preds={len(pred_list)})")

            except _Timeout:
                signal.alarm(0)
                print(f"   · 진행 {i}/{total} (timeout {self.per_item_timeout}s, 건너뜀)")
                pred_list = []
            except Exception as e:
                signal.alarm(0)
                # 조용하지만 원인 힌트는 한 줄
                print(f"   · 진행 {i}/{total} (error: {str(e)[:80]})")
                pred_list = []

            preds.append({'id': row['id'], 'pred_list': pred_list})

        return pd.DataFrame(preds)

    @staticmethod
    def compute_metrics(test_df: pd.DataFrame, pred_df: pd.DataFrame) -> Dict:
        """Top-1/3/5 정확도만 계산"""
        df = test_df.merge(pred_df, on='id', how='left')
        df['pred_list'] = df['pred_list'].apply(lambda x: x if isinstance(x, list) else [])

        def calc_hits(row):
            gold = row['gold_hs']
            preds = row['pred_list']
            return pd.Series({
                'hit_top1': 1 if (len(preds) >= 1 and gold == preds[0]) else 0,
                'hit_top3': 1 if gold in preds[:3] else 0,
                'hit_top5': 1 if gold in preds[:5] else 0,
            })

        hits = df.apply(calc_hits, axis=1)
        detailed = pd.concat([df, hits], axis=1)

        total = len(detailed)
        top1_correct = int(detailed['hit_top1'].sum())
        top3_correct = int(detailed['hit_top3'].sum())
        top5_correct = int(detailed['hit_top5'].sum())

        report = {
            'total_samples': int(total),
            'valid_predictions': int(detailed['pred_list'].apply(len).gt(0).sum()),
            'top1_accuracy': float(top1_correct / total) if total else 0.0,
            'top3_accuracy': float(top3_correct / total) if total else 0.0,
            'top5_accuracy': float(top5_correct / total) if total else 0.0,
            'top1_correct': top1_correct,
            'top3_correct': top3_correct,
            'top5_correct': top5_correct
        }
        return report

    @staticmethod
    def save_report(report: Dict, output_dir: str = "."):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"evaluation_report_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return path

    def run(self, save_output: bool = True, output_dir: str = "."):
        # 1) 데이터 로드
        test_df = self.load_test_data()

        # 2) 예측 (Top-N 생성 → Top-5 정확도까지 계산 가능)
        pred_df = self.generate_predictions(test_df, top_n=max(5, self.top_n))

        # 3) 지표 계산(Top-1/3/5만)
        report = self.compute_metrics(test_df, pred_df)

        # 4) 최소 출력(정확도만)
        print("🚀 HS Code RAG 성능 평가 (Top-1/3/5)")
        print(f"전체 샘플 수: {report['total_samples']} | 유효 예측 수: {report['valid_predictions']}")
        print(f"Top-1 정확도: {report['top1_accuracy']:.2%} ({report['top1_correct']}/{report['total_samples']})")
        print(f"Top-3 정확도: {report['top3_accuracy']:.2%} ({report['top3_correct']}/{report['total_samples']})")
        print(f"Top-5 정확도: {report['top5_accuracy']:.2%} ({report['top5_correct']}/{report['total_samples']})")

        saved = None
        if save_output:
            saved = self.save_report(report, output_dir=output_dir)
            print(f"(요약 리포트 저장: {saved})")
        return report, saved


if __name__ == "__main__":
    try:
        evaluator = HSCodeEvaluator()
        evaluator.run(save_output=True, output_dir=".")
    except Exception as e:
        # 한 줄만 간단히 표기 (자세한 스택 출력 없음)
        print(f"실행 오류: {str(e)}")
