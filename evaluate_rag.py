# evaluate_rag.py
"""
RAG 기반 HS Code 추천 시스템 성능 평가

플로우:
1. 정답 데이터 로드 (상품명, 상품설명, 정답 HS Code)
2. 각 샘플마다 RAG로 Top-K 예측
3. 정답과 비교해서 Top-1, Top-3, Top-5 정확도 계산

출력예시:

최종 성능 평가 결과
============================================================
전체 샘플 수:       100
유효 샘플 수:       98

Top-1 정확도:       45.00%  (45/100)
Top-3 정확도:       72.00%  (72/100)
Top-5 정확도:       85.00%  (85/100)

MRR (평균):         0.5892
============================================================

💾 상세 결과 저장: evaluation_detailed.csv
💾 요약 리포트 저장: evaluation_report.json

============================================================
Top-1에서 틀린 케이스 (상위 10개)
============================================================

ID: 5
상품명: USB 케이블
정답: 8544420000
예측: ['8544300000', '8544420000', '8544700000']

ID: 12
상품명: 면 티셔츠
정답: 6109100000
예측: ['6109900000', '6109100000', '6110300000']

...
"""

import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Optional
from pathlib import Path

# RAG 엔진 import (팀원 코드)
from rag_hs_prompt import classify_hs_code_rag


class HSCodeEvaluator:
    def __init__(self, gold_path: str):
        """
        Args:
            gold_path: 정답 데이터 엑셀 파일 경로
        """
        self.gold_path = gold_path
        
    def normalize_hs(self, code: Optional[str], keep_digits: int = 10) -> Optional[str]:
        """
        HS Code를 숫자만 추출해서 정규화
        
        예시:
            "9405.42-0000" -> "9405420000"
            "8531.20" -> "8531200000" (10자리로 패딩)
        """
        if code is None or (isinstance(code, float) and np.isnan(code)):
            return None
        
        # 숫자만 추출
        digits = re.sub(r"[^0-9]", "", str(code))
        if not digits:
            return None
        
        # 지정된 자릿수로 자르기
        return digits[:min(len(digits), keep_digits)]
    
    def load_gold_data(self) -> pd.DataFrame:
        """
        정답 데이터 로드
        
        필수 컬럼:
            - id (또는 번호): 샘플 식별자
            - 상품명 (또는 품목명): 상품 이름
            - 상품설명 (또는 설명): 상품 설명
            - HS코드 (또는 HSCode, HS부호): 정답 HS Code
        
        Returns:
            DataFrame with columns: [id, product_name, product_desc, gold_hs_10, gold_hs_6, gold_hs_4]
        """
        df = pd.read_excel(self.gold_path)
        print(f"📂 엑셀 로드 완료: {len(df)}개 행")
        print(f"   컬럼: {list(df.columns)}\n")
        
        # 1) ID 컬럼 찾기
        id_col = None
        for col in ["id", "ID", "번호", "row_id", "index"]:
            if col in df.columns:
                id_col = col
                break
        
        if id_col is None:
            print("⚠️ ID 컬럼이 없어서 자동 생성합니다")
            df["id"] = np.arange(1, len(df) + 1)
            id_col = "id"
        
        # 2) 상품명 컬럼 찾기
        name_col = None
        for col in ["상품명", "품목명", "name", "product_name", "제품명"]:
            if col in df.columns:
                name_col = col
                break
        
        if name_col is None:
            raise ValueError(f"상품명 컬럼을 찾을 수 없습니다. 현재 컬럼: {list(df.columns)}")
        
        # 3) 상품설명 컬럼 찾기
        desc_col = None
        for col in ["상품설명", "설명", "description", "product_description", "물품설명"]:
            if col in df.columns:
                desc_col = col
                break
        
        if desc_col is None:
            raise ValueError(f"상품설명 컬럼을 찾을 수 없습니다. 현재 컬럼: {list(df.columns)}")
        
        # 4) HS Code 컬럼 찾기
        hs_col = None
        for col in df.columns:
            if "hs" in col.lower() or "코드" in col:
                hs_col = col
                break
        
        if hs_col is None:
            raise ValueError(f"HS Code 컬럼을 찾을 수 없습니다. 현재 컬럼: {list(df.columns)}")
        
        # 5) 필요한 컬럼만 추출
        gold = df[[id_col, name_col, desc_col, hs_col]].copy()
        gold.columns = ["id", "product_name", "product_desc", "gold_hs_raw"]
        
        # 6) HS Code 정규화 (10자리, 6자리, 4자리)
        gold["gold_hs_10"] = gold["gold_hs_raw"].apply(lambda x: self.normalize_hs(x, 10))
        gold["gold_hs_6"] = gold["gold_hs_raw"].apply(lambda x: self.normalize_hs(x, 6))
        gold["gold_hs_4"] = gold["gold_hs_raw"].apply(lambda x: self.normalize_hs(x, 4))
        
        print(f"✅ 정답 데이터 준비 완료")
        print(f"   - ID: {id_col}")
        print(f"   - 상품명: {name_col}")
        print(f"   - 상품설명: {desc_col}")
        print(f"   - HS Code: {hs_col}\n")
        
        return gold
    
    def generate_predictions(self, gold_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """
        RAG로 각 샘플에 대해 예측 생성
        
        Returns:
            DataFrame with columns: [id, pred_list]
            pred_list: Top-K개의 예측 HS Code 리스트
        """
        predictions = []
        total = len(gold_df)
        
        print(f"🤖 RAG로 예측 생성 시작 (총 {total}개)\n")
        
        for idx, row in gold_df.iterrows():
            row_id = row["id"]
            product_name = str(row["product_name"]) if pd.notna(row["product_name"]) else ""
            product_desc = str(row["product_desc"]) if pd.notna(row["product_desc"]) else ""
            
            # 입력 검증
            if not product_name.strip() and not product_desc.strip():
                print(f"⚠️ [{idx+1}/{total}] ID={row_id}: 상품명과 설명이 모두 비어있음 -> 스킵")
                predictions.append({
                    "id": row_id,
                    "pred_list": []
                })
                continue
            
            # RAG로 예측
            try:
                result = classify_hs_code_rag(
                    product_name=product_name,
                    product_description=product_desc,
                    top_n=top_n
                )
                
                # 결과에서 HS Code 추출
                if "error" in result:
                    print(f"❌ [{idx+1}/{total}] ID={row_id}: 예측 실패 - {result['error']}")
                    pred_list = []
                else:
                    candidates = result.get("candidates", [])
                    pred_list = []
                    for cand in candidates:
                        hs_code = cand.get("hs_code", "")
                        # 정규화
                        normalized = self.normalize_hs(hs_code, 10)
                        if normalized:
                            pred_list.append(normalized)
                    
                    print(f"✅ [{idx+1}/{total}] ID={row_id}: {len(pred_list)}개 예측 완료")
            
            except Exception as e:
                print(f"❌ [{idx+1}/{total}] ID={row_id}: 예외 발생 - {str(e)}")
                pred_list = []
            
            predictions.append({
                "id": row_id,
                "pred_list": pred_list
            })
        
        return pd.DataFrame(predictions)
    
    def compute_metrics(self, gold_df: pd.DataFrame, pred_df: pd.DataFrame) -> Dict:
        """
        Top-1, Top-3, Top-5 정확도 계산
        
        정확도 정의:
            Top-1 정확도 = (1순위에 정답이 있는 샘플 수) / (전체 샘플 수)
            Top-3 정확도 = (3순위 안에 정답이 있는 샘플 수) / (전체 샘플 수)
            Top-5 정확도 = (5순위 안에 정답이 있는 샘플 수) / (전체 샘플 수)
        """
        # 병합
        df = gold_df.merge(pred_df, on="id", how="left")
        df["pred_list"] = df["pred_list"].apply(lambda x: x if isinstance(x, list) else [])
        
        # 각 행별로 Top-1, Top-3, Top-5 hit 계산
        def calc_hits(row):
            gold = row["gold_hs_10"]  # 10자리 기준으로 비교
            preds = [p for p in row["pred_list"] if isinstance(p, str) and p]
            
            # 정답이 없으면 모두 0
            if not isinstance(gold, str) or not gold:
                return pd.Series({"hit_top1": 0, "hit_top3": 0, "hit_top5": 0, "mrr": 0.0})
            
            # 예측이 없으면 모두 0
            if not preds:
                return pd.Series({"hit_top1": 0, "hit_top3": 0, "hit_top5": 0, "mrr": 0.0})
            
            # Top-K hit 계산
            hit_top1 = 1 if gold in preds[:1] else 0
            hit_top3 = 1 if gold in preds[:3] else 0
            hit_top5 = 1 if gold in preds[:5] else 0
            
            # MRR 계산 (보너스)
            mrr = 0.0
            for rank, pred in enumerate(preds, start=1):
                if pred == gold:
                    mrr = 1.0 / rank
                    break
            
            return pd.Series({
                "hit_top1": hit_top1,
                "hit_top3": hit_top3,
                "hit_top5": hit_top5,
                "mrr": mrr
            })
        
        # 계산
        hit_cols = df.apply(calc_hits, axis=1)
        detailed = pd.concat([df, hit_cols], axis=1)
        
        # 전체 정확도 계산
        total_samples = len(detailed)
        valid_samples = detailed["gold_hs_10"].notna().sum()
        
        accuracy_top1 = detailed["hit_top1"].sum() / total_samples if total_samples > 0 else 0.0
        accuracy_top3 = detailed["hit_top3"].sum() / total_samples if total_samples > 0 else 0.0
        accuracy_top5 = detailed["hit_top5"].sum() / total_samples if total_samples > 0 else 0.0
        avg_mrr = detailed["mrr"].mean() if total_samples > 0 else 0.0
        
        # 요약 리포트
        report = {
            "total_samples": total_samples,
            "valid_samples": int(valid_samples),
            "top1_accuracy": float(accuracy_top1),
            "top3_accuracy": float(accuracy_top3),
            "top5_accuracy": float(accuracy_top5),
            "mrr": float(avg_mrr)
        }
        
        return report, detailed
    
    def run(self, top_n: int = 5, save_results: bool = True):
        """
        전체 평가 실행
        
        Args:
            top_n: 예측할 HS Code 개수 (보통 5개)
            save_results: 결과를 CSV/JSON으로 저장할지 여부
        """
        print("="*60)
        print("RAG 기반 HS Code 추천 시스템 성능 평가")
        print("="*60 + "\n")
        
        # 1) 정답 데이터 로드
        gold_df = self.load_gold_data()
        
        # 2) RAG로 예측 생성
        pred_df = self.generate_predictions(gold_df, top_n=top_n)
        
        # 3) 정확도 계산
        print("\n" + "="*60)
        print("📊 성능 지표 계산 중...")
        print("="*60 + "\n")
        
        report, detailed = self.compute_metrics(gold_df, pred_df)
        
        # 4) 결과 출력
        print("\n" + "="*60)
        print("최종 성능 평가 결과")
        print("="*60)
        print(f"전체 샘플 수:       {report['total_samples']}")
        print(f"유효 샘플 수:       {report['valid_samples']}")
        print(f"")
        print(f"Top-1 정확도:       {report['top1_accuracy']:.2%}  ({int(report['top1_accuracy'] * report['total_samples'])}/{report['total_samples']})")
        print(f"Top-3 정확도:       {report['top3_accuracy']:.2%}  ({int(report['top3_accuracy'] * report['total_samples'])}/{report['total_samples']})")
        print(f"Top-5 정확도:       {report['top5_accuracy']:.2%}  ({int(report['top5_accuracy'] * report['total_samples'])}/{report['total_samples']})")
        print(f"")
        print(f"MRR (평균):         {report['mrr']:.4f}")
        print("="*60 + "\n")
        
        # 5) 저장
        if save_results:
            # 상세 결과 저장
            detailed_path = "evaluation_detailed.csv"
            detailed.to_csv(detailed_path, index=False, encoding="utf-8-sig")
            print(f"💾 상세 결과 저장: {detailed_path}")
            
            # 요약 리포트 저장
            report_path = "evaluation_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"💾 요약 리포트 저장: {report_path}\n")
        
        return report, detailed


# ============================================================================
# 실행
# ============================================================================
if __name__ == "__main__":
    # 정답 데이터 경로 (여러분의 파일 경로로 ㄷ수정하세요)
    GOLD_FILE = "/mnt/data/HScode_랜덤100개_기타제외.xlsx"
    
    # 평가 실행
    evaluator = HSCodeEvaluator(gold_path=GOLD_FILE)
    report, detailed_df = evaluator.run(top_n=5, save_results=True)
    
    # 추가 분석 예시: 틀린 케이스만 보기
    print("\n" + "="*60)
    print("Top-1에서 틀린 케이스 (상위 10개)")
    print("="*60)
    
    wrong_cases = detailed_df[detailed_df["hit_top1"] == 0].head(10)
    for idx, row in wrong_cases.iterrows():
        print(f"\nID: {row['id']}")
        print(f"상품명: {row['product_name']}")
        print(f"정답: {row['gold_hs_10']}")
        print(f"예측: {row['pred_list'][:3]}")  # 상위 3개만
