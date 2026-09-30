/** Chat with your data — used on /chat only. */
let chatHistory = [];

function appendChat(role, text, { thinking } = {}) {
  const empty = el("chatEmpty");
  if (empty) empty.remove();
  const log = el("chatLog");
  const row = document.createElement("div");
  row.className = `chat-row ${role}${thinking ? " thinking" : ""}`;
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble";
  if (thinking) {
    bubble.innerHTML = `<span class="chat-typing"><span></span><span></span><span></span></span>`;
  } else {
    bubble.textContent = text;
  }
  row.appendChild(bubble);
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
  return row;
}

function removeThinking() {
  const log = el("chatLog");
  if (!log) return;
  log.querySelectorAll(".chat-row.thinking").forEach(n => n.remove());
}

function setChatBusy(busy) {
  const input = el("chatInput");
  const btn = el("btnChat");
  if (input) input.disabled = !!busy;
  if (btn) btn.disabled = !!busy;
}

function wireChatSuggestions(root) {
  (root || document).querySelectorAll("[data-suggest]").forEach(btn => {
    btn.onclick = () => {
      el("chatInput").value = btn.getAttribute("data-suggest");
      el("chatInput").focus();
      sendChat();
    };
  });
}

function resetChatLog() {
  chatHistory = [];
  const log = el("chatLog");
  if (!log) return;
  log.innerHTML = `
    <div class="chat-empty" id="chatEmpty">
      <p>Start a conversation about your billing data.</p>
      <div class="chat-suggestions">
        <button type="button" class="chip" data-suggest="Who was billed in 2025-09 without an email?">Who needs a mailed bill?</button>
        <button type="button" class="chip" data-suggest="List clients and their mow prices">List mow prices</button>
        <button type="button" class="chip" data-suggest="Show work items for this month">Show recent work</button>
      </div>
    </div>`;
  wireChatSuggestions(log);
}

function sendChat() {
  const input = el("chatInput");
  if (!input) return;
  const message = input.value.trim();
  if (!message || input.disabled) return;
  appendChat("user", message);
  chatHistory.push({ role: "user", content: message });
  input.value = "";
  input.style.height = "auto";
  el("chatMsg").textContent = "";
  el("chatMsg").className = "msg chat-status";
  appendChat("assistant", "", { thinking: true });
  setChatBusy(true);
  const waiting = typeof modelIsReady === "function" && !modelIsReady();
  if (waiting) {
    el("chatMsg").textContent = "Loading the model…";
    el("chatMsg").className = "msg chat-status";
  }
  const start = typeof ensureModel === "function" ? ensureModel() : Promise.resolve();
  start.then(() => api("/api/chat", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ message, history: chatHistory.slice(0, -1) }),
  })).catch(e => {
    removeThinking();
    appendChat("assistant", e.message || "Request failed");
    el("chatMsg").textContent = e.message;
    el("chatMsg").className = "msg chat-status err";
    setChatBusy(false);
  });
}

window.onJobDone = (p) => {
  if (p.kind !== "chat" || !p.detail || !p.detail.answer) return;
  removeThinking();
  const answer = p.detail.answer;
  chatHistory.push({ role: "assistant", content: answer });
  appendChat("assistant", answer);
  const n = (p.detail.mutations_applied || []).length;
  el("chatMsg").textContent = n ? `Updated datastore (${n} change${n === 1 ? "" : "s"}).` : "";
  el("chatMsg").className = n ? "msg chat-status ok" : "msg chat-status";
  setChatBusy(false);
  el("chatInput").focus();
};

window.onJobError = (p) => {
  if (p.kind !== "chat") return;
  removeThinking();
  appendChat("assistant", p.message || "Chat failed");
  el("chatMsg").textContent = p.message || "Chat failed";
  el("chatMsg").className = "msg chat-status err";
  setChatBusy(false);
};

document.addEventListener("DOMContentLoaded", () => {
  const form = el("chatForm");
  const input = el("chatInput");
  const clear = el("btnClearChat");
  if (!form || !input) return;
  wireChatSuggestions(document);
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    sendChat();
  });
  input.addEventListener("keydown", (e) => {
    const isEnter = e.key === "Enter" || e.keyCode === 13;
    if (isEnter && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      e.stopPropagation();
      sendChat();
    }
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 140) + "px";
  });
  if (clear) {
    clear.onclick = () => {
      resetChatLog();
      el("chatMsg").textContent = "";
      el("chatMsg").className = "msg chat-status";
      input.value = "";
      input.style.height = "auto";
      setChatBusy(false);
      input.focus();
    };
  }
});
