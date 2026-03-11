(function () {
  "use strict";

  const chatMessages = document.getElementById("chatMessages");
  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");

  // ── Helpers ──

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function setInputEnabled(enabled) {
    chatInput.disabled = !enabled;
    sendBtn.disabled = !enabled;
  }

  function addMessage(role, text, sources) {
    const wrapper = document.createElement("div");
    wrapper.className = "message " + role;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    wrapper.appendChild(bubble);

    if (sources && sources.length > 0) {
      const srcEl = document.createElement("div");
      srcEl.className = "sources";
      srcEl.innerHTML =
        "<strong>Sources:</strong> " +
        sources
          .map(function (s) {
            return "<span>" + escapeHtml(s) + "</span>";
          })
          .join(" ");
      wrapper.appendChild(srcEl);
    }

    chatMessages.appendChild(wrapper);
    scrollToBottom();
    return wrapper;
  }

  function showLoading() {
    const wrapper = document.createElement("div");
    wrapper.className = "message bot";
    wrapper.id = "loadingMsg";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML =
      '<div class="loading-dots"><span></span><span></span><span></span></div>';
    wrapper.appendChild(bubble);

    chatMessages.appendChild(wrapper);
    scrollToBottom();
  }

  function hideLoading() {
    var el = document.getElementById("loadingMsg");
    if (el) el.remove();
  }

  function escapeHtml(str) {
    var div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // ── Send question ──

  async function sendQuestion(question) {
    if (!question.trim()) return;

    // Hide the welcome section after the first question
    var welcome = document.querySelector(".welcome");
    if (welcome) welcome.remove();

    addMessage("user", question);
    chatInput.value = "";
    setInputEnabled(false);
    showLoading();

    try {
      var response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question }),
      });

      if (!response.ok) {
        throw new Error("Server responded with " + response.status);
      }

      var data = await response.json();
      hideLoading();

      var answer = data.answer || data.response || "Sorry, I didn't get a response.";
      var sources = data.sources || [];
      addMessage("bot", answer, sources);
    } catch (err) {
      hideLoading();
      addMessage("bot", "Something went wrong. Please try again in a moment.");
      console.error("Chat error:", err);
    } finally {
      setInputEnabled(true);
      chatInput.focus();
    }
  }

  // ── Event listeners ──

  chatForm.addEventListener("submit", function (e) {
    e.preventDefault();
    sendQuestion(chatInput.value);
  });

  document.querySelectorAll(".chip").forEach(function (chip) {
    chip.addEventListener("click", function () {
      var question = chip.getAttribute("data-question");
      chatInput.value = question;
      sendQuestion(question);
    });
  });

  // Auto-focus input
  chatInput.focus();
})();
