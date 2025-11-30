const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const resetBtn = document.getElementById("reset-btn");
const historyEl = document.getElementById("history-list");

let step = "awaiting_name";   // 'awaiting_name' → 'awaiting_desc'
let productName = "";
let lastUserText = "";
let loadingTimers = [];
let loaderInterval = null;    // 로딩 문구 변경용 타이머
let isProcessing = false;     // 한 번의 입력이 두 번 처리되는 것 방지

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
  // 말풍선 전용: HTML 대신 텍스트 + 줄바꿈만 사용
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addMessage(text, who) {
  renderMessage(text, who);
  currentMessages.push({ who, text });
}

function bot(text) { addMessage(text, "bot"); }
function user(text) { addMessage(text, "user"); }

// ===================== 추천 결과 말풍선 포맷터 =====================

function formatRecommendationText(rec, index) {
  const rank = index + 1;
  const hs = rec.hs_code || rec.code || "-";
  const title = rec.title || rec.label || "";
  const reason = rec.reason || rec.explanation || "-";
  const h = rec.hierarchy_definitions || {};
  const citations = Array.isArray(rec.citations) ? rec.citations : [];

  let text = `⭐ 추천 ${rank}\n`;
  text += `HS Code: ${hs}`;
  if (title) {
    text += `\n품명: ${title}`;
  }

  text += `\n\n💡 사유\n${reason}`;

  if (
    h &&
    (h.chapter_2digit || h.heading_4digit || h.subheading_6digit || h.national_10digit)
  ) {
    text += `\n\n📚 계층 구조 정의`;
    if (h.chapter_2digit) {
      text += `\n- 2단위(Chapter)  ${h.chapter_2digit.code} — ${h.chapter_2digit.definition || ""}`;
    }
    if (h.heading_4digit) {
      text += `\n- 4단위(Heading)  ${h.heading_4digit.code} — ${h.heading_4digit.definition || ""}`;
    }
    if (h.subheading_6digit) {
      text += `\n- 6단위(Subheading)  ${h.subheading_6digit.code} — ${h.subheading_6digit.definition || ""}`;
    }
    if (h.national_10digit) {
      text += `\n- 10단위(National)  ${h.national_10digit.code} — ${h.national_10digit.definition || ""}`;
    }
  }

  if (citations.length) {
    text += `\n\n📎 근거 출처`;
    citations.forEach((ct) => {
      if (ct.type === "graph") {
        text += `\n- GraphDB 코드: ${ct.code || "-"}`;
      } else if (ct.type === "case") {
        text += `\n- 품목분류사례 문서 ID: ${ct.doc_id || "-"}`;
      } else {
        text += `\n- ${ct.type || "-"}`;
      }
    });
  }

  return text;
}

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

function resetConversation() {
  loadingTimers.forEach(clearTimeout);
  loadingTimers = [];

  if (loaderInterval) {
    clearInterval(loaderInterval);
    loaderInterval = null;
  }

  step = "awaiting_name";
  productName = "";
  lastUserText = "";
  currentMessages = [];
  isProcessing = false;

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

  historyCounter += 1;
  const sessionId = "h" + historyCounter;

  historySessions[sessionId] = {
    id: sessionId,
    title: name,
    messages: currentMessages.map((m) => ({ ...m })),
  };

  const item = document.createElement("div");
  item.className = "history-item";
  item.innerHTML = `
    <div class="history-title">${name}</div>
    <div class="history-sub">${hs} · ${title}</div>
  `;
  item.dataset.sessionId = sessionId;

  item.addEventListener("click", () => {
    loadHistorySession(sessionId);
  });

  historyEl.prepend(item);
}

function loadHistorySession(sessionId) {
  const session = historySessions[sessionId];
  if (!session) return;

  chatEl.innerHTML = "";
  session.messages.forEach((m) => {
    renderMessage(m.text, m.who);
  });
  chatEl.scrollTop = chatEl.scrollHeight;

  step = "awaiting_name";
  productName = "";
  lastUserText = "";
  currentMessages = session.messages.map((m) => ({ ...m }));
  updatePlaceholder();
}

// ===================== 로딩 표시 =====================

let currentLoader = null;

function showLoading() {
  const div = document.createElement("div");
  div.className = "msg loading";
  div.innerHTML = `
    <span id="loading-text">추천 시스템이 분석을 시작합니다...</span>
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
  `;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  currentLoader = div;

  let timePassed = 0;
  const loadingTextEl = div.querySelector("#loading-text");

  loaderInterval = setInterval(() => {
    timePassed += 1;

    if (timePassed === 6) {
      loadingTextEl.innerText =
        "1단계: 유사 품목 사례와 HS 계층 구조를 검색하고 있습니다...";
    } else if (timePassed === 11) {
      loadingTextEl.innerText =
        "2단계: 6자리 및 10자리 HS Code 후보를 점수화하고 있습니다...";
    } else if (timePassed === 16) {
      loadingTextEl.innerText =
        "3단계: 각 후보의 분류 근거를 생성하고 있습니다...";
    } else if (timePassed === 26) {
      loadingTextEl.innerText = "✍️ 결과를 정리하고 있습니다...";
    }
  }, 1000);
}

function hideLoading() {
  if (loaderInterval) {
    clearInterval(loaderInterval);
    loaderInterval = null;
  }
  if (currentLoader) {
    currentLoader.remove();
    currentLoader = null;
  }
}

// ===================== 메인 전송 로직 =====================

async function handleSend() {
  const text = inputEl.value.trim();
  if (!text) return;

  // 동시에 두 번 눌리는 것 방지
  if (isProcessing) return;

  if (step === "awaiting_name") {
    isProcessing = true;

    user(text);
    inputEl.value = "";
    productName = text;

    step = "awaiting_desc";
    updatePlaceholder();

    setTimeout(() => {
      bot(
        `✅ 상품명 '${productName}'(을)를 확인했습니다.\n\n` +
          "정확한 분류를 위해 상품의 특징을 간단히 알려주세요.\n" +
          "예) 재질/성분, 용도·사용 환경, 규격·구성, 제조 방식 등\n\n" +
          "• 예시(공산품): '알루미늄 하우징의 실내용 LED 조명기구, 220V 전원 사용'\n" +
          "• 예시(식품): '냉동 보관된 삼겹살 500g, 가열·조리용'\n"
      );
      isProcessing = false;
    }, 500);

  } else if (step === "awaiting_desc") {
    isProcessing = true;

    const description = text;
    user(description);
    inputEl.value = "";

    showLoading();

    try {
      const response = await fetch("/api/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: productName, desc: description }),
      });

      const data = await response.json();
      hideLoading();

      if (data.error || data.detail) {
        bot("🚫 오류가 발생했습니다: " + (data.error || data.detail));
        step = "awaiting_name";
        updatePlaceholder();
      } else {
        const list = data.candidates || [];

        if (!list.length) {
          bot("추천 결과가 없습니다. 설명을 보강하여 다시 시도해주세요.");
          step = "awaiting_name";
          updatePlaceholder();
        } else {
          const showResultSequentially = async () => {
            for (let i = 0; i < list.length; i++) {
              const c = list[i];
              const recText = formatRecommendationText(c, i);

              // 각 추천 = 하나의 봇 말풍선
              bot(recText);

              if (i < list.length - 1) {
                await new Promise((resolve) => setTimeout(resolve, 800));
              }
            }

            if (typeof addHistoryEntry === "function") {
              addHistoryEntry(productName, list[0]);
            }

            step = "awaiting_name";
            updatePlaceholder();

            setTimeout(() => {
              bot(
                "✅ 분석이 끝났습니다. 새로운 상품을 분류하려면 '상품명'을 다시 입력해주세요."
              );
            }, 600);
          };

          await showResultSequentially();
        }
      }
    } catch (err) {
      hideLoading();
      bot("요청 중 통신 오류가 발생했습니다: " + err.message);
      step = "awaiting_name";
      updatePlaceholder();
    } finally {
      isProcessing = false;
    }
  }
}

// ===================== 이벤트 바인딩 =====================

sendBtn.addEventListener("click", handleSend);

inputEl.addEventListener("keydown", (e) => {
  if (e.isComposing || e.keyCode === 229) return;

  if (e.key === "Enter") {
    e.preventDefault();
    handleSend();
  }
});

resetBtn.addEventListener("click", resetConversation);

// ===================== 최초 진입 시 =====================

showWelcome();
