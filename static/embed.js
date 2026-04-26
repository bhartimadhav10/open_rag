/**
 * OpenRAG embed widget — drop-in floating chat for any website.
 *
 * Usage:
 *   <script src="https://your-openrag-host.com/embed.js"
 *           data-openrag-url="https://your-openrag-host.com"
 *           data-openrag-title="Ask our docs"
 *           data-openrag-color="#22d3ee"
 *           data-openrag-position="bottom-right"
 *           async></script>
 *
 * The host server must allow your site's origin via CORS_ORIGINS in .env.
 */
(function () {
  // -------- read config from the script tag's data-* attributes --------
  const me =
    document.currentScript ||
    Array.from(document.scripts).reverse().find((s) => /embed\.js(\?|$)/.test(s.src));
  if (!me) return;

  const cfg = {
    url:      (me.getAttribute("data-openrag-url") || "").replace(/\/$/, ""),
    title:    me.getAttribute("data-openrag-title") || "Ask anything",
    color:    me.getAttribute("data-openrag-color") || "#22d3ee",
    position: me.getAttribute("data-openrag-position") || "bottom-right",
    greeting: me.getAttribute("data-openrag-greeting") || "Hi! Ask me anything about this site.",
  };
  if (!cfg.url) {
    console.error("[openrag] missing data-openrag-url on <script> tag");
    return;
  }

  // -------- avoid double-loading --------
  if (window.__openragEmbedLoaded) return;
  window.__openragEmbedLoaded = true;

  // -------- one-time scoped CSS --------
  const POS = {
    "bottom-right": "right:20px;bottom:20px;",
    "bottom-left":  "left:20px;bottom:20px;",
    "top-right":    "right:20px;top:20px;",
    "top-left":     "left:20px;top:20px;",
  }[cfg.position] || "right:20px;bottom:20px;";

  const css = `
.openrag-root, .openrag-root * { box-sizing: border-box; font-family: system-ui,-apple-system,Segoe UI,sans-serif; }
.openrag-bubble {
  position: fixed; ${POS}
  width: 56px; height: 56px; border-radius: 50%;
  background: ${cfg.color}; color: #002;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; box-shadow: 0 8px 24px rgba(0,0,0,0.18);
  z-index: 999998; border: none; transition: transform 150ms;
}
.openrag-bubble:hover { transform: scale(1.05); }
.openrag-bubble svg { width: 26px; height: 26px; }
.openrag-panel {
  position: fixed; ${POS.replace(/(top|bottom):\s*\d+px;/, (m) => m.replace(/\d+px/, "90px"))}
  width: 380px; max-width: calc(100vw - 40px);
  height: 600px; max-height: calc(100vh - 120px);
  background: #fff; color: #111; border-radius: 14px;
  box-shadow: 0 18px 50px rgba(0,0,0,0.22);
  display: none; flex-direction: column; overflow: hidden;
  z-index: 999999; border: 1px solid #e5e5e5;
}
.openrag-panel.open { display: flex; }
.openrag-header {
  background: ${cfg.color}; color: #002;
  padding: 14px 16px; display: flex; align-items: center; justify-content: space-between;
}
.openrag-header h3 { margin: 0; font-size: 15px; font-weight: 600; }
.openrag-header button {
  background: transparent; border: none; color: #002; cursor: pointer;
  font-size: 22px; line-height: 1; padding: 2px 6px;
}
.openrag-msgs {
  flex: 1; overflow-y: auto; padding: 14px; background: #fafafa;
  display: flex; flex-direction: column; gap: 10px;
  font-size: 14px; line-height: 1.5;
}
.openrag-msg { max-width: 85%; padding: 9px 12px; border-radius: 12px; word-wrap: break-word; }
.openrag-msg.user { align-self: flex-end; background: ${cfg.color}; color: #002; border-bottom-right-radius: 4px; }
.openrag-msg.bot  { align-self: flex-start; background: #fff; border: 1px solid #e5e5e5; border-bottom-left-radius: 4px; white-space: pre-wrap; }
.openrag-msg.bot.thinking { color: #888; font-style: italic; }
.openrag-sources { margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee; font-size: 11.5px; color: #666; }
.openrag-sources b { color: #444; font-weight: 600; }
.openrag-src { margin-top: 3px; }
.openrag-src .score { color: ${cfg.color}; filter: brightness(0.8); font-weight: 600; }
.openrag-input {
  display: flex; gap: 8px; padding: 12px; border-top: 1px solid #e5e5e5; background: #fff;
}
.openrag-input input {
  flex: 1; padding: 9px 12px; border: 1px solid #ddd; border-radius: 8px;
  font-size: 14px; outline: none; color: #111; background: #fff;
}
.openrag-input input:focus { border-color: ${cfg.color}; }
.openrag-input button {
  padding: 9px 16px; background: ${cfg.color}; color: #002; border: none;
  border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 14px;
}
.openrag-input button:disabled { opacity: 0.5; cursor: not-allowed; }
.openrag-footer { font-size: 10.5px; color: #999; text-align: center; padding: 6px; background: #fff; }
`;

  const styleEl = document.createElement("style");
  styleEl.textContent = css;
  document.head.appendChild(styleEl);

  // -------- DOM --------
  const root = document.createElement("div");
  root.className = "openrag-root";
  root.innerHTML = `
    <button class="openrag-bubble" aria-label="Open chat">
      <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.5 2 2 6 2 11c0 2.5 1.2 4.7 3 6.3V22l4.3-2.4c.9.2 1.8.4 2.7.4 5.5 0 10-4 10-9s-4.5-9-10-9zm-1 11H7v-2h4v2zm6 0h-4v-2h4v2zm0-4H7V7h10v2z"/></svg>
    </button>
    <div class="openrag-panel" role="dialog" aria-label="${cfg.title}">
      <div class="openrag-header">
        <h3>${cfg.title}</h3>
        <button class="openrag-close" aria-label="Close">×</button>
      </div>
      <div class="openrag-msgs"></div>
      <form class="openrag-input">
        <input type="text" placeholder="Ask a question…" autocomplete="off" />
        <button type="submit">Send</button>
      </form>
      <div class="openrag-footer">Powered by OpenRAG</div>
    </div>
  `;
  document.body.appendChild(root);

  const bubble = root.querySelector(".openrag-bubble");
  const panel  = root.querySelector(".openrag-panel");
  const closer = root.querySelector(".openrag-close");
  const msgs   = root.querySelector(".openrag-msgs");
  const form   = root.querySelector(".openrag-input");
  const input  = form.querySelector("input");
  const sendBtn= form.querySelector("button");

  // -------- session persistence --------
  const SESSION_KEY = `openrag_session_${cfg.url}`;
  let sessionId = localStorage.getItem(SESSION_KEY);

  // -------- helpers --------
  function addMsg(text, who, opts = {}) {
    const el = document.createElement("div");
    el.className = `openrag-msg ${who}` + (opts.thinking ? " thinking" : "");
    el.textContent = text;
    msgs.appendChild(el);
    msgs.scrollTop = msgs.scrollHeight;
    return el;
  }

  function appendSources(botEl, chunks) {
    if (!chunks || !chunks.length) return;
    const wrap = document.createElement("div");
    wrap.className = "openrag-sources";
    wrap.innerHTML = "<b>Sources</b>";
    chunks.forEach((c) => {
      const row = document.createElement("div");
      row.className = "openrag-src";
      const preview = (c.text || "").replace(/\s+/g, " ").slice(0, 90);
      row.innerHTML = `<span class="score">[${(c.rerank_score || 0).toFixed(2)}]</span> ${escapeHTML(c.source)} — "${escapeHTML(preview)}${c.text && c.text.length > 90 ? "…" : ""}"`;
      wrap.appendChild(row);
    });
    botEl.appendChild(wrap);
    msgs.scrollTop = msgs.scrollHeight;
  }

  function escapeHTML(s) {
    return String(s || "").replace(/[&<>"']/g, (c) => (
      {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]
    ));
  }

  function setOpen(open) {
    panel.classList.toggle("open", open);
    if (open) setTimeout(() => input.focus(), 50);
  }

  bubble.addEventListener("click", () => setOpen(!panel.classList.contains("open")));
  closer.addEventListener("click", () => setOpen(false));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && panel.classList.contains("open")) setOpen(false);
  });

  // -------- greeting --------
  if (cfg.greeting) addMsg(cfg.greeting, "bot");

  // -------- send/stream --------
  let activeES = null;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q || activeES) return;
    addMsg(q, "user");
    input.value = "";
    sendBtn.disabled = true;

    const botEl = addMsg("", "bot", { thinking: true });
    botEl.textContent = "Searching…";

    const url = new URL(cfg.url + "/stream");
    url.searchParams.set("q", q);
    if (sessionId) url.searchParams.set("session_id", sessionId);

    let firstToken = true;
    let sources = null;

    activeES = new EventSource(url.toString(), { withCredentials: false });

    activeES.addEventListener("start", (e) => {
      try {
        const d = JSON.parse(e.data);
        if (d.session_id) {
          sessionId = d.session_id;
          localStorage.setItem(SESSION_KEY, sessionId);
        }
      } catch {}
    });

    activeES.addEventListener("retrieval_done", (e) => {
      try { sources = JSON.parse(e.data).chunks; } catch {}
    });

    const onToken = (e) => {
      try {
        const d = JSON.parse(e.data);
        if (firstToken) {
          botEl.textContent = "";
          botEl.classList.remove("thinking");
          firstToken = false;
        }
        botEl.textContent += d.token || "";
        msgs.scrollTop = msgs.scrollHeight;
      } catch {}
    };
    activeES.addEventListener("first_token", onToken);
    activeES.addEventListener("token", onToken);

    activeES.addEventListener("done", () => {
      if (sources) appendSources(botEl, sources);
      cleanup();
    });

    activeES.onerror = () => {
      if (firstToken) {
        botEl.textContent = "Sorry, I couldn't reach the server. Please try again.";
        botEl.classList.remove("thinking");
      }
      cleanup();
    };

    function cleanup() {
      if (activeES) { try { activeES.close(); } catch {} activeES = null; }
      sendBtn.disabled = false;
      input.focus();
    }
  });

  // -------- public API --------
  window.OpenRAG = {
    open:  () => setOpen(true),
    close: () => setOpen(false),
    toggle: () => setOpen(!panel.classList.contains("open")),
    reset: () => {
      sessionId = null;
      localStorage.removeItem(SESSION_KEY);
      msgs.innerHTML = "";
      if (cfg.greeting) addMsg(cfg.greeting, "bot");
    },
  };
})();
