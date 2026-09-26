/* Shared header: model status + global progress */
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (_) {}
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  const ct = r.headers.get("content-type") || "";
  if (ct.includes("application/json")) return r.json();
  return r;
}

function el(id) { return document.getElementById(id); }

async function refreshModel() {
  try {
    const s = await api("/api/model/status");
    const node = el("modelStatus");
    if (node) node.textContent = s.up ? `Model ${s.id}: ready` : `Model ${s.id}: offline`;
    const btn = el("btnLoadModel");
    if (btn) btn.disabled = !!s.up;
  } catch (e) {
    const node = el("modelStatus");
    if (node) node.textContent = "Model: unreachable";
  }
}

async function loadModel() {
  const btn = el("btnLoadModel");
  if (btn) btn.disabled = true;
  try {
    await api("/api/model/load", { method: "POST" });
  } catch (e) {
    alert(e.message);
    if (btn) btn.disabled = false;
  }
}

function setProgressUI(p) {
  const msg = el("progressMsg");
  const fill = el("progressFill");
  const st = el("progressStatus");
  if (msg) {
    const q = p.queue_depth ? ` · queue ${p.queue_depth}` : "";
    msg.textContent = (p.message || "") + q;
  }
  if (st) st.textContent = p.status || "idle";
  if (fill) fill.style.width = `${p.percent || 0}%`;
  if (typeof window.updateQueueMsg === "function") window.updateQueueMsg(p);
  if ((p.ocr_busy || p.status === "running") && p.kind === "ocr") {
    const c = el("btnCancelOcr");
    const a = el("btnCancelAllOcr");
    if (c) c.hidden = false;
    if (a) a.hidden = false;
  }
}

let _lastDetail = null;
async function pollProgress() {
  try {
    const p = await api("/api/progress");
    setProgressUI(p);
    if (p.status === "done" && p.detail && JSON.stringify(p.detail) !== JSON.stringify(_lastDetail)) {
      _lastDetail = p.detail;
      if (typeof window.onJobDone === "function") window.onJobDone(p);
    }
    if ((p.status === "error" || p.status === "cancelled") && typeof window.onJobError === "function") {
      const key = `${p.status}:${p.job_id}:${p.message}`;
      if (key !== _lastDetail) {
        _lastDetail = key;
        window.onJobError(p);
      }
    }
    window._ocrRunning = !!(p.ocr_busy || (p.status === "running" && p.kind === "ocr"));
    refreshModel();
    const ocrLive = p.ocr_busy || (p.status === "running" && p.kind === "ocr");
    if (ocrLive) window._sawOcr = true;
    if (window._sawOcr && !ocrLive && p.status !== "done") {
      const um = el("uploadMsg");
      if (um) {
        um.textContent = "OCR stopped before it finished. Submit the photos again.";
        um.className = "msg err";
      }
      window._sawOcr = false;
    }
  } catch (_) {}
}

document.addEventListener("DOMContentLoaded", () => {
  refreshModel();
  setInterval(pollProgress, 1500);
  const btn = el("btnLoadModel");
  if (btn) btn.addEventListener("click", loadModel);
});
