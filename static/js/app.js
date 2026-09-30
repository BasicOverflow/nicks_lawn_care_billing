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

let modelCheck = null;
let modelDeploying = false;
let modelReady = false;
let modelLoadStarted = false;
let modelLoadJob = "";
let modelWaiters = [];

function settleModelWaiters(err) {
  const waiting = modelWaiters;
  modelWaiters = [];
  waiting.forEach(waiter => err ? waiter.reject(err) : waiter.resolve());
}

function paintModel(up, label) {
  modelReady = !!up;
  const node = el("modelStatus");
  if (node) {
    node.textContent = label;
    node.classList.toggle("model-ready", !!up);
    node.classList.toggle("model-offline", !up);
  }
  const btn = el("btnLoadModel");
  if (btn) {
    if (up) {
      btn.disabled = true;
      btn.textContent = "Model Loaded";
      btn.classList.add("is-loaded");
    } else if (modelDeploying) {
      btn.disabled = true;
      btn.textContent = "Load model";
      btn.classList.remove("is-loaded");
    } else {
      btn.disabled = false;
      btn.textContent = "Load model";
      btn.classList.remove("is-loaded");
    }
  }
  if (modelReady) {
    modelLoadStarted = false;
    modelLoadJob = "";
    modelDeploying = false;
    settleModelWaiters();
  }
}

async function refreshModel() {
  if (modelCheck) return;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 4000);
  modelCheck = fetch("/api/model/status", { signal: ctrl.signal });
  try {
    const r = await modelCheck;
    if (!r.ok) throw new Error("status");
    const s = await r.json();
    paintModel(!!s.up, s.up ? `Model ${s.id}: ready` : `Model ${s.id}: offline`);
  } catch (_) {
    paintModel(false, "Model: unreachable");
  } finally {
    clearTimeout(timer);
    modelCheck = null;
  }
}

async function loadModel() {
  if (modelReady || modelLoadStarted) return;
  modelLoadStarted = true;
  modelDeploying = true;
  paintModel(false, el("modelStatus") ? el("modelStatus").textContent : "Model: …");
  try {
    const started = await api("/api/model/load", { method: "POST" });
    modelLoadJob = started.job_id || "";
  } catch (e) {
    modelLoadStarted = false;
    modelLoadJob = "";
    modelDeploying = false;
    settleModelWaiters(e);
    alert(e.message);
    refreshModel();
  }
}

function modelIsReady() {
  return modelReady;
}

function ensureModel() {
  if (modelReady) return Promise.resolve();
  if (!modelLoadStarted) loadModel();
  return new Promise((resolve, reject) => {
    if (modelReady) resolve();
    else modelWaiters.push({ resolve, reject });
  });
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
    const modelJobRunning = jobs.some(job => job.kind === "model" && job.status === "running");
    const modelJobFailed = jobs.find(job => job.kind === "model" && (job.status === "error" || job.status === "cancelled"));
    if (modelJobFailed && modelLoadJob && modelJobFailed.job_id === modelLoadJob) {
      modelLoadStarted = false;
      modelLoadJob = "";
      modelDeploying = false;
      settleModelWaiters(new Error(modelJobFailed.message || "Model failed to load"));
    } else if (!modelReady) {
      modelDeploying = modelLoadStarted || modelJobRunning;
    }
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
  refreshModel();
}

document.addEventListener("DOMContentLoaded", () => {
  refreshModel();
  setInterval(pollProgress, 1500);
  const btn = el("btnLoadModel");
  if (btn) btn.addEventListener("click", loadModel);
  const chat = document.querySelector(".chat-card");
  if (chat) {
    chat.addEventListener("pointerdown", (event) => {
      if (event.target.closest("#btnClearChat")) return;
      ensureModel().catch(() => {});
    });
  }
});
