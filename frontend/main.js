const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const resetBtn = document.getElementById("reset-btn");
const historyEl = document.getElementById("history-list");

let step = "awaiting_name";   // 'awaiting_name' → 'awaiting_desc' → 'loading'
let productName = "";
let lastUserText = "";
let loadingTimers = [];

// 현재 화면에서 진행 중인 "한 번의 분류 대화" 메시지들(스냅샷용 버퍼)
let currentMessages = [];

// 사이드바에 저장되는 대화 스냅샷들
// { sessionId: { id, title, messages: [{who,text}, ...] } }
let historySessions = {};
let historyCounter = 0;

// ===================== 메시지 출력 관련 =====================

function renderMessage(text, who) {
  const div = document.createElement("div");
  div.className = `msg ${who}`;
  div.innerText = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addMessage(text, who) {
  renderMessage(text, who);
  // 현재 분류 대화의 버퍼에 저장(스냅샷용)
  currentMessages.push({ who, text });
}

function bot(text) { addMessage(text, "bot"); }
function user(text) { addMessage(text, "user"); }

// ===================== placeholder 관리 =====================

function updatePlaceholder() {
  if (step === "awaiting_name") {
    inputEl.placeholder = "상품명을 입력하세요 (예: LED 조명, 냉동 삼겹살)";
  } else if (step === "awaiting_desc") {
    inputEl.placeholder = "상품 설명을 자세히 입력하세요 (재질·용도 등)";
  } else {
    inputEl.placeholder = "메시지를 입력하세요...";
  }
}

// ===================== 초기/리셋 메시지 =====================

function showWelcome() {
  bot(
    "👋 안녕하세요! HS Code 추천 시스템입니다.\n\n" +
    "먼저 분류하고 싶은 '상품명'을 입력해주세요.\n" +
    "예) LED 조명, 냉동 삼겹살, 전기자동차용 리튬이온 배터리"
  );
  updatePlaceholder();
}

// 화면만 깨끗하게 리셋하고 새 분류를 시작하는 용도
function resetConversation() {
  // 로딩 타이머 정리
  loadingTimers.forEach(clearTimeout);
  loadingTimers = [];

  step = "awaiting_name";
  productName = "";
  lastUserText = "";
  currentMessages = [];

  chatEl.innerHTML = "";
  showWelcome();
}

// ===================== 사이드바: 스냅샷 저장 =====================

function addHistoryEntry(name, topCandidate) {
  if (!historyEl) return;

  const empty = historyEl.querySelector(".history-empty");
  if (empty) empty.remove();

  const hs = topCandidate.hs_code || "-";
  const title = topCandidate.title || topCandidate.label || "";

  // 새로운 스냅샷 ID 생성
  historyCounter += 1;
  const sessionId = "h" + historyCounter;

  // 현재 분류 대화의 메시지를 스냅샷으로 저장 (깊은 복사)
  historySessions[sessionId] = {
    id: sessionId,
    title: name,
    messages: currentMessages.map(m => ({ ...m })),
  };

  const item = document.createElement("div");
  item.className = "history-item";
  item.innerHTML = `
    <div class="history-title">${name}</div>
    <div class="history-sub">${hs} · ${title}</div>
  `;
  item.dataset.sessionId = sessionId;

  // 클릭 시 해당 스냅샷 대화 재생
  item.addEventListener("click", () => {
    loadHistorySession(sessionId);
  });

  historyEl.prepend(item);
}

// 사이드바 카드 클릭 시: 저장된 스냅샷 대화 로드
function loadHistorySession(sessionId) {
  const session = historySessions[sessionId];
  if (!session) return;

  // 화면 비우고 해당 스냅샷 메시지 재생
  chatEl.innerHTML = "";
  session.messages.forEach(m => {
    renderMessage(m.text, m.who);
  });
  chatEl.scrollTop = chatEl.scrollHeight;

  // 이 상태에서 다시 입력하면 "새 분류" 시작으로 간주
  step = "awaiting_name";
  productName = "";
  lastUserText = "";
  // 현재 버퍼는 선택한 스냅샷으로 초기화하되,
  // 다음 분류를 위해 handleSend에서 다시 비우게 됨.
  currentMessages = session.messages.map(m => ({ ...m }));
  updatePlaceholder();
}

// ===================== 메인 전송 로직 =====================

// ===== 🆕 로딩 표시 함수 추가 =====
let currentLoader = null; // 로딩 메시지 요소를 저장할 변수

function showLoading() {
  const div = document.createElement("div");
  div.className = "msg loading"; 
  
  // 초기 멘트 + 점 3개
  // span에 id를 줘서 나중에 글씨를 바꿀 수 있게 함
  div.innerHTML = `
    <span id="loading-text">추천 시스템이 분석을 시작합니다...</span>
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
  `;
  
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  currentLoader = div;

  // 🔄 멘트가 3단계로 바뀌는 타이머 설정
  let timePassed = 0;
  const loadingTextEl = div.querySelector("#loading-text");

  loaderInterval = setInterval(() => {
    timePassed += 1;

    if (timePassed === 6) {
      loadingTextEl.innerText = "1단계: 유사 품목 사례와 HS 계층 구조를 검색하고 있습니다...";
    } else if (timePassed === 11) {
      loadingTextEl.innerText = "2단계: 6자리 및 10자리 HS Code 후보를 점수화하고 있습니다...";
    } else if (timePassed === 16) {
      loadingTextEl.innerText = "3단계: 각 후보의 분류 근거를 생성하고 있습니다...";
    } else if (timePassed === 26) {
        loadingTextEl.innerText = "✍️ 결과를 정리하고 있습니다...";
    }
  }, 1000); // 1초마다 체크
}

function hideLoading() {
  // 타이머 멈춤
  if (loaderInterval) {
    clearInterval(loaderInterval);
    loaderInterval = null;
  }
  // 로딩바 제거
  if (currentLoader) {
    currentLoader.remove();
    currentLoader = null;
  }
}

// 로딩 적용//

async function handleSend() {
  if (step === "awaiting_name") {
    // 1. 상품명 입력 단계
    const text = inputEl.value.trim();
    if (!text) return;

    user(text);
    inputEl.value = "";
    productName = text; // 상품명 저장

    step = "awaiting_desc"; // 다음 단계로
    updatePlaceholder();
    
    // 봇 응답 (약간의 딜레이를 주어 자연스럽게)
    setTimeout(() => {
      bot(`✅ 상품명 '${productName}'(을)를 확인했습니다.\n\n` +
      "이제 상품 설명을 입력해주세요.\n" +
      "예) '알루미늄 하우징을 사용한 실내용 LED 조명기구로, 220V 전원에 연결해 사용합니다.'");
    }, 500);

  } else if (step === "awaiting_desc") {
    // 2. 상품 설명 입력 & 분석 요청 단계
    const description = inputEl.value.trim();
    if (!description) return;

    user(description);
    inputEl.value = "";

    // ⏳ [핵심] 분석 시작 전 로딩 표시 띄우기!
    showLoading(); 

    try {
      // API 요청 (시간이 오래 걸림)
      const response = await fetch("/api/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: productName, desc: description })
      });

      const data = await response.json();

      // 🏁 [핵심] 응답 오면 로딩 제거!
      hideLoading();

      // 결과 처리
      if (data.error || data.detail) {
        bot("🚫 오류가 발생했습니다: " + (data.error || data.detail));
      } else {
        const list = data.candidates || [];

        if (!list.length) {
          bot("추천 결과가 없습니다. 설명을 보강하여 다시 시도해주세요.");
        } else {
          
          // 🔄 [수정] 결과를 하나씩 시간차를 두고 출력하는 함수
          const showResultSequentially = async () => {
            for (let i = 0; i < list.length; i++) {
              const c = list[i];
              const hs = c.hs_code || "-";
              const title = c.title || "-";
              const reason = c.reason || "-";

              // 1. 메시지 생성 및 출력
              bot(
              `⭐ 추천 ${i + 1}\n` +
              `HS Code: ${hs}\n` +
              `품목명: ${title}\n\n` +
              `💡 사유:\n${reason}`
            );

              // 2. 다음 메시지 출력 전까지 잠깐 대기 (예: 0.8초)
              // (마지막 메시지 후에는 대기할 필요 없음)
              if (i < list.length - 1) {
                await new Promise(resolve => setTimeout(resolve, 800)); 
              }
            }

            // 3. 모든 결과 출력 후 히스토리 저장 및 마무리 멘트
            if (typeof addHistoryEntry === "function") {
              addHistoryEntry(productName, list[0]);
            }

            step = "awaiting_name";
            updatePlaceholder();
            
            // 마무리 멘트도 약간 딜레이 후 출력
            setTimeout(() => {
              bot("✅ 분석이 끝났습니다. 새로운 상품을 분류하려면 '상품명'을 다시 입력해주세요.");
            }, 600);
          };

          // 함수 실행!
          showResultSequentially();
        }
      }
    } catch (err) {
      hideLoading(); // 에러 나도 로딩은 꺼야 함
      bot("요청 중 통신 오류가 발생했습니다: " + err.message);
      step = "awaiting_name";
      updatePlaceholder();
    }
  }
}

// ===================== 이벤트 바인딩 =====================

sendBtn.addEventListener("click", handleSend);

inputEl.addEventListener("keydown", (e) => {
  // 한글 IME 조합 중 Enter는 무시
  if (e.isComposing || e.keyCode === 229) return;

  if (e.key === "Enter") {
    e.preventDefault();
    handleSend();
  }
});

resetBtn.addEventListener("click", resetConversation);

// ===================== 최초 진입 시 =====================

showWelcome();
