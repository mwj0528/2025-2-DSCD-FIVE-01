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
  // 카드 UI를 쓰기 위해 HTML 허용
  div.innerHTML = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addMessage(text, who) {
  renderMessage(text, who);
  currentMessages.push({ who, text });
}

function bot(text) { addMessage(text, "bot"); }
function user(text) { addMessage(text, "user"); }

// ===================== 추천 카드 렌더러 =====================

function renderRecommendationCard(rec, index) {
  const hs = rec.hs_code || "-";
  const title = rec.title || "-";
  const reason = rec.reason || "-";
  const h = rec.hierarchy_definitions || {};
  const citations = Array.isArray(rec.citations) ? rec.citations : [];

  return `
    <div class="rec-card">
      <div class="rec-card-header">
        <div class="rec-rank">⭐ 추천 ${index + 1}</div>
        <div class="rec-hscode">
          HS Code:
          <span>${hs}</span>
        </div>
      </div>

      <div class="rec-title">${title}</div>

      <div class="rec-section">
        <div class="rec-section-title">💡 사유</div>
        <div class="rec-section-body">${reason}</div>
      </div>

      ${
        h && (h.chapter_2digit || h.heading_4digit || h.subheading_6digit || h.national_10digit)
          ? `
      <div class="rec-section">
        <div class="rec-section-title">📚 계층 구조 정의</div>
        <ul class="rec-hierarchy-list">
          ${
            h.chapter_2digit
              ? `<li><b>2단위(Chapter)</b> ${h.chapter_2digit.code} — ${h.chapter_2digit.definition || ""}</li>`
              : ""
          }
          ${
            h.heading_4digit
              ? `<li><b>4단위(Heading)</b> ${h.heading_4digit.code} — ${h.heading_4digit.definition || ""}</li>`
              : ""
          }
          ${
            h.subheading_6digit
              ? `<li><b>6단위(Subheading)</b> ${h.subheading_6digit.code} — ${h.subheading_6digit.definition || ""}</li>`
              : ""
          }
          ${
            h.national_10digit
              ? `<li><b>10단위(National)</b> ${h.national_10digit.code} — ${h.national_10digit.definition || ""}</li>`
              : ""
          }
        </ul>
      </div>`
          : ""
      }

      ${
        citations.length
          ? `
      <div class="rec-section">
        <div class="rec-section-title">📎 근거 출처</div>
        <ul class="rec-citations">
          ${citations
            .map((ct) => {
              if (ct.type === "graph") {
                return `<li>GraphDB 코드: ${ct.code || "-"}</li>`;
              } else if (ct.type === "case") {
                return `<li>품목분류사례 문서 ID: ${ct.doc_id || "-"}</li>`;
              }
              return `<li>${ct.type || "-"}</li>`;
            })
            .join("")}
        </ul>
      </div>`
          : ""
      }
    </div>
  `;
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
  if (step === "awaiting_name") {
    const text = inputEl.value.trim();
    if (!text) return;

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
    }, 500);
  } else if (step === "awaiting_desc") {
    const description = inputEl.value.trim();
    if (!description) return;

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
      } else {
        const list = data.candidates || [];

        if (!list.length) {
          bot("추천 결과가 없습니다. 설명을 보강하여 다시 시도해주세요.");
        } else {
          const showResultSequentially = async () => {
            for (let i = 0; i < list.length; i++) {
              const c = list[i];

              // 카드 UI로 출력
              bot(renderRecommendationCard(c, i));

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

          showResultSequentially();
        }
      }
    } catch (err) {
      hideLoading();
      bot("요청 중 통신 오류가 발생했습니다: " + err.message);
      step = "awaiting_name";
      updatePlaceholder();
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

