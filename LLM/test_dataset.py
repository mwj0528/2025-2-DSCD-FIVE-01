import pandas as pd
import os
import json
import re
import numpy as np 
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Any, List

# --- 전역 설정 및 상수 ---
load_dotenv()
# OpenAI 클라이언트 초기화. API 키는 .env 파일에서 로드됨
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 파일 경로 설정
MAIN_DATA_PATH = '../output/sample_data.csv' 
HS_DESC_PATH = '../output/all_hscode_ver2.csv' # 2, 4, 6, 10자리 설명이 모두 포함된 통합 파일
OUTPUT_FILE_PATH = 'test100_natural.csv' 

# ----------------------------------------------------
# 📌 GPT 프롬프트 정의 (HS 설명 컨텍스트 추가 및 자연스러운 번역 강조)
# ----------------------------------------------------

SYSTEM_PROMPT = """
당신은 국제무역 품목 분류 전문가입니다. 제공된 HS 코드의 **2, 4, 6자리 상위 분류 설명**과 **최종 10자리 코드의 설명**을 모두 참고하여, 
마치 실제 사람이 작성한 것처럼 자연스럽고 매끄러운 한국어 '사용자_상품명'과 '사용자_상품설명'을 생성해야 합니다.

**필수 임무:** '사용자_상품설명'은 '사용자_상품명'과 절대 동일해서는 안 되며, **HS10 코드가 가진 10자리의 세부 분류 기준(용도, 재질, 형태, 수치 등)**을 반드시 반영해야 합니다.

### 💡 품목 설명 생성의 기준 예시
- **HS코드:** 3910.00.9010 (참고용)
- **품목명:** Silicon oil, in primary forms; VINYL SILICON OIL; RH-VI305B
- **생성된 설명의 품질:** 무색 투명한 점조 액상의 Polydimethylsiloxane vinyl terminated와 Polydimethylsiloxane이 혼합된 실리콘 오일, 용도: 플라스틱, 고무 제조용 (재질, 형태, 용도를 반드시 포함)

출력은 반드시 순수 JSON 객체여야 하며, HS 코드의 분류 맥락을 활용하여 품목의 특성을 살리는 번역을 수행하십시오.
"""

# --- GPT 호출 함수 ---
def process_data_with_gpt(input_text: str) -> Dict[str, Any]:
    """GPT 모델을 사용하여 영문 품목명/설명을 한국어로 번역하고 JSON으로 포맷팅"""

    # USER_PROMPT에 HS 분류 설명과 영문 품목명을 함께 전달
    USER_PROMPT = f"""
    ### 입력 데이터
    {input_text}

    ### 추출 규칙 (Rules)
    1.  품목명(title_en): 입력 데이터의 '영문 품목명' 필드 내용을 'title_en' 필드에 그대로 복사합니다.
    2.  품목설명(description_en): 품목명과 동일한 내용을 'description_en' 필드에 복사합니다.
    3.  한국어 번역(title_kr, description_kr): 'title_en'과 'description_en'의 내용을 **최대한 실제 사람이 쓴 것처럼 자연스럽고 매끄러운 한국어**로 번역하여 해당 필드에 넣습니다. 이때 제공된 HS 분류 설명(HS2/HS4/HS6/HS10)의 맥락을 참고하여 번역합니다.

    ### 출력 형식 (Output Format: Strict JSON)
    {{
      "title_kr": "string",
      "title_en": "string",
      "description_kr": "string",
      "description_en": "string"
    }}

    **처리 결과 (JSON 객체만):**
    """
    
    global client, SYSTEM_PROMPT 
    
    try:
        # API 호출 (지정된 모델 사용)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT}, 
                {"role": "user", "content": USER_PROMPT} 
            ],
            temperature=0.2, # 창의적인 번역을 위해 temperature를 약간 높임
            response_format={"type": "json_object"},
            timeout=60 # 처리 시간을 넉넉하게 확보
        )
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        # 오류 발생 시 디버깅을 위해 오류 내용과 입력을 반환
        print(f"GPT 처리 오류 발생: {e}")
        return {"error": str(e), "input": input_text}

# HS 설명 데이터를 병합하는 헬퍼 함수
def merge_hs_desc(df: pd.DataFrame, desc_df: pd.DataFrame, length: int, col_name: str) -> pd.DataFrame:
    """지정된 길이의 HS 코드를 기준으로 설명 데이터를 병합합니다."""
    
    # 해당 길이의 코드만 필터링하여 병합 준비
    temp_df = desc_df[desc_df['code'].str.len() == length].copy()
    temp_df = temp_df.rename(columns={'code': f'HS{length}', 'desc': col_name})
    
    # 메인 데이터프레임에 해당 길이의 코드 생성
    df[f'HS{length}'] = df['HS10'].str[:length]
    
    # 병합 수행 (left merge: 메인 데이터 기준)
    df = pd.merge(df, temp_df[[f'HS{length}', col_name]], on=f'HS{length}', how='left')
    return df

# --- 메인 실행 ---
if __name__ == "__main__":
    try:
        # 1. 메인 데이터 로드 (100개 데이터라고 가정)
        main_df = pd.read_csv(MAIN_DATA_PATH, encoding='utf-8')
        
        # 'HS10'이 존재함을 가정하며, 없으면 KeyError 발생
        # '영문품목명' 컬럼이 존재함을 가정하며, 없으면 KeyError 발생
        
        # 2. HS 코드 표준화 및 분리
        main_df['HS10'] = main_df['HS10'].astype(str).str.replace(r'[^0-9]', '', regex=True).str.zfill(10)
        
        # 3. HS 코드 설명 데이터 로드 및 병합 (2, 4, 6, 10자리 설명 추가)
        desc_df = pd.read_csv(HS_DESC_PATH, encoding='utf-8').rename(columns={'code': 'code', 'description': 'desc'})
        # desc_df의 코드 정제 (숫자만 남기기)
        desc_df['code'] = desc_df['code'].astype(str).str.replace(r'[^0-9]', '', regex=True)
        
        # 2, 4, 6, 10자리 설명 순차적으로 병합
        main_df = merge_hs_desc(main_df, desc_df, 2, 'HS2_설명')
        main_df = merge_hs_desc(main_df, desc_df, 4, 'HS4_설명')
        main_df = merge_hs_desc(main_df, desc_df, 6, 'HS6_설명')
        main_df = merge_hs_desc(main_df, desc_df, 10, 'HS10_설명') # 최종 10자리 설명 추가

        print("✅ HS 코드 2, 4, 6, 10자리 설명 병합 완료.")

        # 4. 샘플링 로직 제거 (전체 100개 데이터 사용)
        data_df_sample = main_df
        print(f"전체 {len(data_df_sample)}개 데이터에 대해 GPT 처리를 시작합니다.")
        
        # 5. GPT 입력 텍스트 조합 (HS 설명 포함)
        def combine_for_gpt(row):
            title_en = str(row.get('영문품목명', '')).strip() 
            
            # GPT에 HS 분류 설명과 영문 품목명을 함께 전달
            return f"""
            [HS 분류 맥락]
            HS2 설명: {row.get('HS2_설명', '정보 없음')}
            HS4 설명: {row.get('HS4_설명', '정보 없음')}
            HS6 설명: {row.get('HS6_설명', '정보 없음')}
            HS10 설명: {row.get('HS10_설명', '정보 없음')}

            [번역 대상]
            영문 품목명: {title_en}
            """

        data_df_sample['raw_text'] = data_df_sample.apply(combine_for_gpt, axis=1)

        # 6. GPT 호출 및 JSON 파싱
        print("🚀 GPT 번역 처리 시작...")
        # 주의: API 호출 속도 제한에 걸릴 수 있으므로, 실제 환경에서는 sleep이나 재시도 로직이 필요할 수 있습니다.
        data_df_sample['processed_json'] = data_df_sample['raw_text'].apply(process_data_with_gpt)
        result_df = pd.json_normalize(data_df_sample['processed_json'])
        final_df = pd.concat([data_df_sample.reset_index(drop=True), result_df], axis=1)
        
        # 7. 번역 오류 보완 (description_kr이 영문과 동일하거나 부족할 경우 title_kr로 대체)
        # GPT가 번역을 거부하고 영문 품목명을 그대로 반환했을 때의 오류 처리
        final_df['description_kr_final'] = np.where(
            final_df['description_kr'].astype(str).str.lower().str.strip() == final_df['description_en'].astype(str).str.lower().str.strip(),
            final_df['title_kr'],
            final_df['description_kr'] 
        )

        # 8. 최종 컬럼 선택 및 저장 (요청된 컬럼 순서대로)
        final_output_df = final_df[[
            'HS10',
            'title_kr',
            'description_kr_final', 
            'HS2_설명',
            'HS4_설명',
            'HS6_설명',
            'HS10_설명', # 최종 10자리 설명 포함
            ]].rename(columns={
            'title_kr': '사용자_상품명',
            'description_kr_final': '사용자_상품설명',
        })

        final_output_df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8')
        
        print(f"\n✅ 최종 결과 {len(final_output_df)}개 샘플 CSV 저장 완료: {OUTPUT_FILE_PATH}")
        print("\n=== 최종 출력 컬럼 순서 ===")
        print(final_output_df.head())

    except FileNotFoundError as e:
        print(f"❌ 오류: 파일을 찾을 수 없습니다: {e.args[0]}. 경로를 확인해 주세요.")
    except KeyError as e:
        print(f"❌ 오류: CSV 컬럼명 누락 또는 불일치: {e}. 실제 CSV 파일의 헤더 ('HS10', '영문품목명' 등)를 확인하고 코드의 컬럼명을 수정하세요.")
    except Exception as e:
        print(f"❌ 최종 처리 중 오류 발생: {e}")