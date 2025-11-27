"""
RAG 기반 HS Code 추천 시스템 성능 평가 (간결/안정 버전)
- Top-1, Top-3, Top-5 정확도만 계산/출력
- 행별 타임아웃/하트비트 로그/샘플 제한 지원
- 필요 시 요약 JSON 및 상세 Excel 저장

실행:
    # 첫 실행(스모크 테스트 권장)
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 \
    $env:EVAL_MAX_SAMPLES=5 $EVAL_TIMEOUT_SEC=45 \
    python evaluate_rag_system.py

필수:
    1) RAG.py 또는 rag_hs_prompt.py 내 classify_hs_code_rag 함수가 import 가능
    2) 테스트 데이터 파일 존재 (예: sample_data.csv)
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
from typing import Dict, List, Optional, Tuple # 👈 Tuple 추가
from datetime import datetime

# =========================
# RAG 모듈 import (RAG.py → rag_hs_prompt.py 순서로 시도)
# 이 함수가 VectorDB + GraphDB + LLM의 전체 과정을 실행
# =========================
try:
    from RAG_openai_small_kw import classify_hs_code_rag  # 사용자 환경 우선 (RAG_e5_small_kw로 단일화)
except ImportError:
    # 대체 경로를 찾을 수 없으므로, RAG_e5_small_kw 파일의 경로를 확인하도록 안내
    print("❌ classify_hs_code_rag 임포트 실패. RAG_e5_small_kw.py 파일이 없거나 경로가 잘못되었습니다.")
    sys.exit(1)


class HSCodeEvaluator:
    def __init__(self, excel_path: str = None):
        # 디버그/성능 제어 환경변수
        self.max_samples = int(os.getenv("EVAL_MAX_SAMPLES", "0"))      # 0이면 전체
        self.per_item_timeout = int(os.getenv("EVAL_TIMEOUT_SEC", "45"))  # 행별 타임아웃(초)
        self.top_n = int(os.getenv("EVAL_TOP_N", "5"))                  # 예측 상한(정확도는 1/3/5만 계산)

        # 엑셀 경로 자동 탐색 수정
        if excel_path is None:
            # 💡 경로 재구성을 통해 숨겨진 문자열 오류 방지
            BASE_DIR = r"C:\Users\user\Desktop\수업\4-2\캡스톤디자인\share\DSCD_NEW"
            file_name = "sample_data.csv"
            absolute_path = os.path.join(BASE_DIR, "output", file_name)
            
            if os.path.exists(absolute_path):
                excel_path = absolute_path
                
            if excel_path is None or not os.path.exists(excel_path):
                # 에러 메시지도 새 파일명에 맞게 수정
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {absolute_path}")
            self.excel_path = excel_path

    @staticmethod
    def normalize_hs(code: Optional[str], keep_digits: int = 10) -> Optional[str]:
        if code is None or (isinstance(code, float) and np.isnan(code)):
            return None 
        digits = re.sub(r"[^0-9]", "", str(code))
        if not digits:
            return None
        return (digits[:keep_digits]).rjust(keep_digits, "0")

    def load_test_data(self) -> pd.DataFrame:
        file_ext = os.path.splitext(self.excel_path)[1].lower()
        hs_dtype = {'HS부호': str, 'HSCode': str, 'HS코드': str}
        
        # 인코딩 문제 회피를 위한 try-except 블록
        try:
            if file_ext == '.csv':
                df = pd.read_csv(self.excel_path, dtype=hs_dtype, encoding='utf-8')
            elif file_ext in ['.xlsx', '.xls']:
                # CSV 로직이 실패할 경우를 대비해 Excel 엔진 필요
                df = pd.read_excel(self.excel_path, converters=hs_dtype)
            else:
                df = pd.read_excel(self.excel_path) # Default to excel if extension is unknown
        except UnicodeDecodeError:
             # UTF-8 실패 시 CP949로 재시도
            df = pd.read_csv(self.excel_path, dtype=hs_dtype, encoding='cp949') 

        required_cols = {
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

        # Pandas 내부 인덱스를 ID로 사용 (별도 ID 컬럼 불필요)
        test_df = test_df.reset_index().rename(columns={'index': 'idx'})
        test_df['idx'] = test_df['idx'] + 1 # 1부터 시작하도록 조정

        if self.max_samples > 0:
            test_df = test_df.head(self.max_samples)
            print(f"🔎 디버그 모드: 상위 {len(test_df)}개만 평가(EVAL_MAX_SAMPLES).")

        return test_df

    def generate_predictions(self, test_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """조용히 예측만 수행 + 행별 타임아웃 + 하트비트 로그"""
        preds: List[Dict] = []

        class _Timeout(Exception):
            pass

        def _handler(signum, frame):
            raise _Timeout()

        # Windows 환경은 SIGALRM을 지원하지 않으므로 조건부 실행
        use_alarm = os.name == 'posix'

        if use_alarm:
            signal.signal(signal.SIGALRM, _handler)

        total = len(test_df)
        for i, row in enumerate(test_df.itertuples(), start=1):
            product_name = str(row.product_name).strip()
            product_desc = str(row.product_desc).strip()
            pred_list: List[str] = []
            
            # 예측된 후보 전체를 저장할 필드 추가
            raw_candidates = []
            error_msg = ""
            
            try:
                if use_alarm:
                    signal.alarm(self.per_item_timeout) # ⏱️ 행별 타임아웃
                t0 = time.time()
                
                # classify_hs_code_rag 호출
                result = classify_hs_code_rag(
                    product_name=product_name,
                    product_description=product_desc,
                    top_n=top_n
                )
                
                if use_alarm:
                    signal.alarm(0) # 알람 해제

                if isinstance(result, dict):
                    raw_candidates = result.get("candidates", [])
                    for cand in raw_candidates:
                        norm = self.normalize_hs(cand.get("hs_code", ""), 10)
                        if norm:
                            pred_list.append(norm)

                # 5개마다 진행 상황 출력 (하트비트)
                if i % 5 == 0 or i == total:
                    dt = time.time() - t0
                    print(f"   · 진행 {i}/{total} (last {dt:.1f}s, preds={len(pred_list)})")

            except _Timeout:
                if use_alarm:
                    signal.alarm(0)
                error_msg = f"TIMEOUT ({self.per_item_timeout}s)"
                print(f"   · 진행 {i}/{total} (timeout {self.per_item_timeout}s, 건너뜜)")
                pred_list = []
            except Exception as e:
                if use_alarm:
                    signal.alarm(0)
                error_msg = str(e)[:80]
                print(f"   · 진행 {i}/{total} (error: {error_msg})")
                pred_list = []

            preds.append({
                'idx': row.idx, 
                'pred_list': pred_list,
                'raw_candidates': raw_candidates, # LLM의 원본 후보 목록 저장
                'error_msg': error_msg
            })

        pred_df = pd.DataFrame(preds).set_index('idx')
        return pred_df

    @staticmethod
    def compute_metrics(test_df: pd.DataFrame, pred_df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
        """Top-1/3/5 정확도 계산 및 상세 데이터 반환"""
        
        # test_df와 pred_df를 인덱스(idx) 기준으로 조인
        df = test_df.set_index('idx').join(pred_df, how='left')
        df['pred_list'] = df['pred_list'].apply(lambda x: x if isinstance(x, list) else [])

        def calc_hits(row):
            gold = row['gold_hs']
            preds = row['pred_list']
            return pd.Series({
                'Hit_Top1': 1 if (len(preds) >= 1 and gold == preds[0]) else 0,
                'Hit_Top3': 1 if gold in preds[:3] else 0,
                'Hit_Top5': 1 if gold in preds[:5] else 0,
                'Prediction_1st': preds[0] if len(preds) >= 1 else None, # 1순위 예측 코드
                'Is_Correct': 1 if gold in preds[:5] else 0 # Top-5 내 정답 여부
            })

        hits = df.apply(calc_hits, axis=1)
        detailed = pd.concat([df, hits], axis=1).reset_index(drop=True)
        
        # 최종 리포트 계산
        total = len(detailed)
        top1_correct = int(detailed['Hit_Top1'].sum())
        top3_correct = int(detailed['Hit_Top3'].sum())
        top5_correct = int(detailed['Hit_Top5'].sum())

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
        return report, detailed

    @staticmethod
    def save_report(report: Dict, output_dir: str = "."):
        """요약 JSON 리포트 저장"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"evaluation_report_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return path

    @staticmethod
    def save_detailed_excel(detailed_df: pd.DataFrame, output_dir: str = "."):
        """상세 평가 결과를 Excel 파일로 저장"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"evaluation_detailed_{ts}.xlsx")
        
        # 엑셀 저장 시 컬럼 순서 지정 및 정리
        cols_to_keep = ['idx', 'product_name', 'product_desc', 'gold_hs', 
                        'Prediction_1st', 'Hit_Top1', 'Hit_Top3', 'Hit_Top5', 
                        'pred_list', 'raw_candidates', 'error_msg']
        
        detailed_df = detailed_df.reindex(columns=cols_to_keep)
        
        # Excel 저장 (index=False: Pandas 인덱스 미포함)
        detailed_df.to_excel(path, index=False)
        return path

    def run(self, save_output: bool = True, output_dir: str = "."):
        # 1) 데이터 로드
        test_df = self.load_test_data()

        # 2) 예측 (Top-N 생성 → Top-5 정확도까지 계산 가능)
        pred_df = self.generate_predictions(test_df, top_n=max(5, self.top_n))

        # 3) 지표 계산(Top-1/3/5만) 및 상세 데이터프레임 반환
        report, detailed_df = self.compute_metrics(test_df, pred_df)

        # 4) 최소 출력(정확도만)
        print("🚀 HS Code RAG 성능 평가 (Top-1/3/5)")
        print(f"전체 샘플 수: {report['total_samples']} | 유효 예측 수: {report['valid_predictions']}")
        print(f"Top-1 정확도: {report['top1_accuracy']:.2%} ({report['top1_correct']}/{report['total_samples']})")
        print(f"Top-3 정확도: {report['top3_accuracy']:.2%} ({report['top3_correct']}/{report['total_samples']})")
        print(f"Top-5 정확도: {report['top5_accuracy']:.2%} ({report['top5_correct']}/{report['total_samples']})")

        saved_json = None
        saved_excel = None
        if save_output:
            saved_json = self.save_report(report, output_dir=output_dir)
            print(f"(요약 리포트 저장: {saved_json})")
            
            # 5) 상세 Excel 저장
            saved_excel = self.save_detailed_excel(detailed_df, output_dir=output_dir)
            print(f"(상세 Excel 저장: {saved_excel})")

        return report, saved_json, saved_excel


if __name__ == "__main__":
    try:
        evaluator = HSCodeEvaluator()
        evaluator.run(save_output=True, output_dir=".")
    except Exception as e:
        # 한 줄만 간단히 표기 (자세한 스택 출력 없음)
        print(f"실행 오류: {str(e)}")
