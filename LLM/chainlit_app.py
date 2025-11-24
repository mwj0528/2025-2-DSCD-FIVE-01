import chainlit as cl
import os
import sys
import json
from dotenv import load_dotenv

# ===== 0. 경로 설정 (rag_module 찾기 위해) =====
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# rag_module import
try:
    from rag_module import HSClassifier
except ImportError:
    print("오류: rag_module.py를 찾을 수 없습니다.")
    sys.exit(1)

# .env 로드
load_dotenv()

# ===== 1. RAG 엔진 초기화 (전역 변수 활용) =====
# Chainlit은 세션이 시작될 때 이 함수를 호출합니다.
@cl.on_chat_start
async def start():
    # 1) 로딩 메시지 전송
    msg = cl.Message(content="HS Code 분류 엔진을 가동 중입니다... 잠시만 기다려주세요 ⚙️")
    await msg.send()

    # 2) 엔진 초기화 (비동기로 처리하여 UI 멈춤 방지)
    classifier = await cl.make_async(HSClassifier)(
        parser_type="both",
        embed_model="text-embedding-3-large",
        chroma_dir="data/chroma_db_openai_large_kw",
        collection_name="hscode_collection",
        use_keyword_extraction=True,
        # 필요한 경우 run_rag.py의 기본값들 추가
        translate_to_english=False
    )
    
    # 3) 사용자 세션에 엔진과 상태 저장
    cl.user_session.set("classifier", classifier)
    cl.user_session.set("step", "awaiting_name") # 상태 관리: 이름 입력 대기
    
    # 4) 로딩 완료 메시지 업데이트
    msg.content = """
### 👋 안녕하세요! HS Code 추천 시스템입니다.

먼저 분류하고 싶은 '상품명'을 입력해주세요.
(예: LED 조명, 냉동 삼겹살)
"""
    await msg.update()


# ===== 2. 결과 포맷팅 함수 =====
def format_result_to_markdown(result_json):
    if not result_json or "candidates" not in result_json:
        return "❌ 분석 결과가 없거나 형식이 올바르지 않습니다."

    candidates = result_json["candidates"]
    text = ""

    for i, cand in enumerate(candidates, 1):
        hs = cand.get("hs_code", "N/A")
        title = cand.get("title", "품목명 없음")
        reason = cand.get("reason", "사유 없음")
        
        text += f"### 🥇 추천 {i}: **{hs}**\n"
        text += f"**📦 품목:** {title}\n\n"
        text += f"**💡 사유:** {reason}\n\n"
        
        # 근거 (아코디언 효과 대신 텍스트로 깔끔하게)
        citations = cand.get("citations", [])
        if citations:
            text += "> **📚 근거 자료:**\n"
            for cit in citations:
                ctype = cit.get("type")
                code_info = cit.get('code') or cit.get('doc_id') or "정보 없음"
                icon = "🕸️" if ctype == "graph" else "📄"
                text += f"> - {icon} ({ctype}) {code_info}\n"
        
        text += "\n---\n"
    
    return text


# ===== 3. 메인 채팅 로직 =====
@cl.on_message
async def main(message: cl.Message):
    # 세션에서 현재 상태와 엔진 가져오기
    classifier = cl.user_session.get("classifier")
    step = cl.user_session.get("step")
    user_input = message.content

    # --- Step 1: 상품명 입력 ---
    if step == "awaiting_name":
        # 상품명 저장
        cl.user_session.set("product_name", user_input)
        # 다음 단계로 변경
        cl.user_session.set("step", "awaiting_desc")
        
        await cl.Message(
            content=f"✅ 상품명 '{user_input}'을(를) 입력받았습니다.\n\n이어서 상세한 '상품 설명'을 입력해주세요.\n(재질, 용도, 기능 등을 자세히 적을수록 정확도가 올라갑니다.)"
        ).send()

    # --- Step 2: 상품 설명 입력 & RAG 실행 ---
    elif step == "awaiting_desc":
        product_name = cl.user_session.get("product_name")
        product_desc = user_input
        
        # (간지 포인트!) "생각하는 과정"을 UI에 보여줌
        async with cl.Step(name="HS Code 분석 중...", type="run") as root_step:
            root_step.input = f"상품: {product_name} / 설명: {product_desc}"
            
            # 1. (시각화) 검색 단계
            async with cl.Step(name="🔍 DB 검색 (Vector + Graph)", type="tool") as search_step:
                # 실제로는 RAG 함수 안에서 다 돌지만, UI상 보여주기용 딜레이 혹은 로그
                search_step.output = "ChromaDB 및 Neo4j에서 관련 데이터 추출 완료"
            
            # 2. (시각화) 계층적 추론 단계
            async with cl.Step(name="🧠 계층적 추론 (Hierarchical Reasoning)", type="llm") as logic_step:
                # 실제 RAG 엔진 호출 (비동기로 감싸서 실행)
                # --hierarchical 옵션과 동일한 메서드 호출
                result_json = await cl.make_async(classifier.classify_hs_code_hierarchical)(
                    product_name=product_name,
                    product_description=product_desc,
                    top_n=3
                )
                logic_step.output = "추론 완료"
            
            root_step.output = "최종 결과 생성 완료"

        # 결과 출력
        if "error" in result_json:
            await cl.Message(content=f"🚫 오류 발생: {result_json['error']}").send()
        else:
            formatted_msg = format_result_to_markdown(result_json)
            await cl.Message(content=formatted_msg).send()
            
            # 마무리 멘트 및 초기화
            await cl.Message(content="✅ 분석이 끝났습니다. 새로운 상품 추천을 원하시면 '상품명'을 다시 입력해주세요.").send()
            
            # 상태 리셋
            cl.user_session.set("step", "awaiting_name")
            cl.user_session.set("product_name", "")

    # --- 예외 처리 ---
    else:
        cl.user_session.set("step", "awaiting_name")
        await cl.Message(content="🔄 상태가 초기화되었습니다. 상품명을 입력해주세요.").send()

# local에서 확인
# chainlit run LLM/chainlit_app.py -w 
