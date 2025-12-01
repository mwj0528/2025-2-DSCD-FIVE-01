const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const resetBtn = document.getElementById("reset-btn");
const historyEl = document.getElementById("history-list");

let step = "awaiting_name"; // 'awaiting_name' → 'awaiting_desc'
let productName = "";
let lastUserText = "";
let loadingTimers = [];
let loaderInterval = null; // 로딩 문구 변경용 타이머
let isProcessing = false; // 한 번의 입력이 두 번 처리되는 것 방지

// 현재 화면에서 진행 중인 "한 번의 분류 대화" 메시지들(스냅샷용 버퍼)
let currentMessages = [];

// 사이드바에 저장되는 대화 스냅샷들
// { sessionId: { id, title, messages: [{who,text}, ...] } }
let historySessions = {};
let historyCounter = 0;

// ===================== 메시지 출력 관련 =====================

function renderMessage(text, who) {
  // 아무 내용도 없으면 말풍선을 만들지 않음
  if (text == null || String(text).trim().length === 0) return;

  const safeText = String(text);

  const div = document.createElement("div");
  div.className = `msg ${who}`;

  // 봇 메시지는 HTML 허용(HS Code 볼드 등), 사용자 메시지는 순수 텍스트
  if (who === "bot") {
    // 이미 HTML 태그가 있으면 그대로, 없으면 줄바꿈만 <br>로 치환
    if (safeText.includes("<")) {
      div.innerHTML = safeText;
    } else {
      div.innerHTML = safeText.replace(/\n/g, "<br>");
    }
  } else {
    div.textContent = safeText;
  }

  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addMessage(text, who) {
  renderMessage(text, who);
  currentMessages.push({ who, text });
}

function bot(text) {
  addMessage(text, "bot");
}
function user(text) {
  addMessage(text, "user");
}

// ===================== 추천 결과 말풍선 포맷터 =====================

function formatRecommendationText(rec, index) {
  const rank = index + 1;
  const hs = rec.hs_code || rec.code || "";
  const title = rec.title || rec.label || "";

  const rawReason = rec.reason ?? rec.explanation ?? "";
  const reason = String(rawReason).trim();

  // 계층 구조: 백엔드가 hierarchy_definitions 또는 hierarchy 중 무엇이든 보내도 대응
  const hRaw = rec.hierarchy_definitions || rec.hierarchy || {};
  const h2 = hRaw.chapter_2digit ?? hRaw.chapter;
  const h4 = hRaw.heading_4digit ?? hRaw.heading;
  const h6 = hRaw.subheading_6digit ?? hRaw.subheading;
  const h10 = hRaw.national_10digit ?? hRaw.national;

  let text = "";

  // 추천 타이틀
  text += `<div style="font-weight:700; font-size:16px; margin-bottom:4px;">⭐ 추천 ${rank}</div>`;

  // HS Code 라인(볼드 + 폰트 조금 더 크게, CSS .hs-code-line과도 연동)
  if (hs) {
    text += `<div class="hs-code-line">HS Code: ${hs}</div>`;
  }

  if (title) {
    text += `<div>품명: ${title}</div>`;
  }

  // 사유
  if (reason) {
    text += `<br><strong>💡 사유</strong><br>${reason}`;
  }

  // 계층 구조 정의
  if (h2 || h4 || h6 || h10) {
    text += `<br><br><strong>📚 계층 구조 정의</strong>`;

    if (h2) {
      const code = h2.code ?? "";
      // null / undefined만 빈칸 처리, ""(빈문자열)이나 영어 원문은 그대로 둠
      const def =
        h2.definition === undefined || h2.definition === null
          ? ""
          : h2.definition;
      if (code || String(def).trim().length > 0) {
        text += `<br>- 2단위(Chapter)  ${code}${
          code && def ? " — " : ""
        }${def}`;
      }
    }

    if (h4) {
      const code = h4.code ?? "";
      const def =
        h4.definition === undefined || h4.definition === null
          ? ""
          : h4.definition;
      if (code || String(def).trim().length > 0) {
        text += `<br>- 4단위(Heading)  ${code}${
          code && def ? " — " : ""
        }${def}`;
      }
    }

    if (h6) {
      const code = h6.code ?? "";
      const def =
        h6.definition === undefined || h6.definition === null
          ? ""
          : h6.definition;
      if (code || String(def).trim().length > 0) {
        text += `<br>- 6단위(Subheading)  ${code}${
          code && def ? " — " : ""
        }${def}`;
      }
    }

    if (h10) {
      const code = h10.code ?? "";
      const def =
        h10.definition === undefined || h10.definition === null
          ? ""
          : h10.definition;
      if (code || String(def).trim().length > 0) {
        text += `<br>- 10단위(National)  ${code}${
          code && def ? " — " : ""
        }${def}`;
      }
    }
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

  const hs = topCandidate.hs_code || "";
  const title = topCandidate.title || topCandidate.label || "";

  historyCounter += 1;
  const sessionId = "h" + historyCounter;

  historySessions[sessionId] = {
    id: sessionId,
    title: name,
    messages: currentMessages.map((m) => ({ ...m })),
  };

  const subParts = [];
  if (hs) subParts.push(hs);
  if (title) subParts.push(title);
  const sub = subParts.join(" · ");

  const item = document.createElement("div");
  item.className = "history-item";
  item.innerHTML = `
    <div class="history-title">${name}</div>
    <div class="history-sub">${sub}</div>
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
