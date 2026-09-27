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

const JOB_LABELS = {
  model: "Model",
  generate: "Bills",
  email: "Email",
  ocr: "OCR",
  chat: "Chat",
  upload: "Upload",
};

function progressJobs(p) {
  if (Array.isArray(p.jobs)) return p.jobs.filter(job => job && job.job_id && job.status && job.status !== "idle");
  if (p.job_id && p.status && p.status !== "idle") return [p];
  return [];
}

function withQueue(job, p) {
  return Object.assign({}, job, {
    queue_depth: p.queue_depth,
    queued: p.queued,
    ocr_busy: p.ocr_busy,
  });
}

function setProgressUI(p) {
  const host = el("progressBars");
  const jobs = progressJobs(p);
  if (host) {
    host.replaceChildren();
    jobs.forEach(job => {
      const bar = document.createElement("div");
      bar.className = "progress-bar" + (job.status === "error" || job.status === "cancelled" ? " is-error" : "");
      const kind = document.createElement("span");
      kind.className = "progress-kind";
      kind.textContent = JOB_LABELS[job.kind] || "Job";
      const st = document.createElement("span");
      st.className = "progress-status";
      st.textContent = job.status || "";
      const track = document.createElement("div");
      track.className = "track";
      const fill = document.createElement("div");
      fill.className = "fill";
      fill.style.width = `${job.percent || 0}%`;
      track.appendChild(fill);
      const msg = document.createElement("span");
      msg.className = "progress-msg";
      const queue = job.kind === "ocr" && p.queue_depth ? ` · queue ${p.queue_depth}` : "";
      msg.textContent = (job.message || "") + queue;
      bar.append(kind, st, track, msg);
      host.appendChild(bar);
    });
  }
  if (typeof window.updateQueueMsg === "function") window.updateQueueMsg(p);
  const ocrLive = !!(p.ocr_busy || jobs.some(job => job.kind === "ocr" && job.status === "running"));
  const c = el("btnCancelOcr");
  const a = el("btnCancelAllOcr");
  if (c) c.hidden = !ocrLive;
  if (a) a.hidden = !ocrLive;
  return jobs;
}

const _seenDone = new Set();
const _seenProblem = new Set();
window.claimJobDone = (job) => {
  if (!job || !job.job_id || _seenDone.has(job.job_id)) return false;
  _seenDone.add(job.job_id);
  return true;
};
async function pollProgress() {
  try {
    const p = await api("/api/progress");
    const jobs = setProgressUI(p);
    jobs.forEach(job => {
      const tagged = withQueue(job, p);
      if (job.status === "done" && window.claimJobDone(job) && typeof window.onJobDone === "function") {
        window.onJobDone(tagged);
      }
      if (job.status === "error" || job.status === "cancelled") {
        const key = `${job.status}:${job.job_id}:${job.message}`;
        if (!_seenProblem.has(key) && typeof window.onJobError === "function") {
          _seenProblem.add(key);
          window.onJobError(tagged);
        }
      }
    });
    const ocrLive = !!(p.ocr_busy || jobs.some(job => job.kind === "ocr" && job.status === "running"));
    window._ocrRunning = ocrLive;
    refreshModel();
    if (ocrLive) window._sawOcr = true;
    if (jobs.some(job => job.kind === "ocr" && job.status !== "running")) window._sawOcr = false;
    else if (window._sawOcr && !ocrLive) {
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
