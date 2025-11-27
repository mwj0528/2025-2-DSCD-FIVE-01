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

async function handleSend() {
  const text = inputEl.value.trim();
  if (!text) return;

  // 로딩 중 같은 내용 반복 전송 방지
  if (text === lastUserText && step === "loading") return;
  lastUserText = text;

  // 새 상품명 입력이면 "새 분류 대화" 시작 → 버퍼 초기화
  if (step === "awaiting_name") {
    currentMessages = [];
  }

  user(text);
  inputEl.value = "";

  // --- Step 1: 상품명 입력 ---
  if (step === "awaiting_name") {
    productName = text;

    bot(
      `✅ 상품명 '${productName}'(을)를 확인했습니다.\n\n` +
      "이제 상품 설명을 입력해주세요.\n" +
      "예) '알루미늄 하우징을 사용한 실내용 LED 조명기구로, 220V 전원에 연결해 사용합니다.'"
    );

    step = "awaiting_desc";
    updatePlaceholder();
    return;
  }

  // --- Step 2: 상품 설명 입력 & 길이 검증 ---
  if (step === "awaiting_desc") {
    const desc = text;

    if (desc.length < 10) {
      bot(
        "상품 설명이 너무 짧습니다.\n" +
        "재질, 용도, 구조 등을 조금 더 자세히 적어주세요.\n" +
        "예) '플라스틱 하우징과 LED 모듈로 구성된 실내용 벽부착 조명기구입니다.'"
      );
      return;
    }

    step = "loading";
    updatePlaceholder();

    // ===== 로딩 단계 메시지 =====
    bot("HS Code를 분석 중입니다...");

    loadingTimers.forEach(clearTimeout);
    loadingTimers = [];

    loadingTimers.push(
      setTimeout(() => {
        bot("1단계: 유사 품목 사례와 HS 계층 구조를 검색하고 있습니다.");
      }, 700)
    );

    loadingTimers.push(
      setTimeout(() => {
        bot("2단계: 6자리 및 10자리 HS Code 후보를 점수화하고 있습니다.");
      }, 1500)
    );

    loadingTimers.push(
      setTimeout(() => {
        bot("3단계: 각 후보의 분류 근거를 생성하고 있습니다.");
      }, 2300)
    );

    // ===== 백엔드 요청 =====
    let data;
    try {
      const res = await fetch("/api/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: productName, desc }),
      });
      data = await res.json();
    } catch (err) {
      loadingTimers.forEach(clearTimeout);
      loadingTimers = [];
      bot("요청 중 오류가 발생했습니다: " + err.message);
      step = "awaiting_name";
      updatePlaceholder();
      return;
    }

    loadingTimers.forEach(clearTimeout);
    loadingTimers = [];

    // ===== 결과 처리 =====
    if (data.error || data.detail) {
      bot("🚫 오류가 발생했습니다: " + (data.error || data.detail));
    } else {
      const list = data.candidates || [];

      if (!list.length) {
        bot("추천 결과가 없습니다. 설명을 조금 더 구체적으로 수정해 다시 시도해주세요.");
      } else {
        list.forEach((c, i) => {
          const hs = c.hs_code || "-";
          const title = c.title || "-";
          const reason = c.reason || "-";

          bot(
            `⭐ 추천 ${i + 1}\n` +
            `HS Code: ${hs}\n` +
            `품목명: ${title}\n\n` +
            `사유: ${reason}`
          );
        });

        // 이 분류 대화 전체를 스냅샷으로 저장 → 사이드바 카드에 연결
        addHistoryEntry(productName, list[0]);
      }
    }

    // 다음 분류를 위해 상태만 초기화 (화면은 그대로 두고)
    step = "awaiting_name";
    updatePlaceholder();
    bot("새로운 상품을 분류하려면 다시 상품명을 입력해주세요. (예: LED 조명, 냉동 삼겹살)");
    return;
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
