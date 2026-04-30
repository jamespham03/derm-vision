/**
 * DermVision — Analysis Page JS
 * Handles image upload, camera capture, paste, model prediction, results display.
 */

const BAR_COLORS = {
  MEL:   "#e24b4a",
  NV:    "#1d9e75",
  BCC:   "#e24b4a",
  AKIEC: "#ef9f27",
  BKL:   "#1d9e75",
  DF:    "#1d9e75",
  VASC:  "#378add",
  SCC:   "#e24b4a",
};

let currentFile = null;
let stream = null;

// ── DOM refs ──────────────────────────────────────────────────────────────
const uploadZone     = document.getElementById("upload-zone");
const fileInput      = document.getElementById("file-input");
const uploadDefault  = document.getElementById("upload-default");
const uploadPreview  = document.getElementById("upload-preview");
const previewImg     = document.getElementById("preview-img");
const changeImageBtn = document.getElementById("change-image-btn");
const uploadPanel    = document.getElementById("upload-panel");
const resultsPanel   = document.getElementById("results-panel");
const analyzeBtn     = document.getElementById("analyze-btn");
const analyzeBtnText = document.getElementById("analyze-btn-text");
const backBtn        = document.getElementById("back-btn");

// step elements
const step1 = document.getElementById("step-1");
const step2 = document.getElementById("step-2");
const stepConn = document.getElementById("step-connector");

// Source buttons
const btnUpload  = document.getElementById("btn-source-upload");
const btnCamera  = document.getElementById("btn-source-camera");
const btnPaste   = document.getElementById("btn-source-paste");

// Camera modal
const cameraModal = document.getElementById("camera-modal");
const cameraVideo = document.getElementById("camera-video");
const captureBtn  = document.getElementById("capture-btn");
const closeCamBtn = document.getElementById("close-camera");

// Results
const demoNotice     = document.getElementById("demo-notice");
const diagnosisCard  = document.getElementById("diagnosis-card");
const diagName       = document.getElementById("diag-name");
const diagMeta       = document.getElementById("diag-meta");
const riskBadge      = document.getElementById("risk-badge");
const diagDesc       = document.getElementById("diag-desc");
const diagAdvice     = document.getElementById("diag-advice");
const adviceIcon     = document.getElementById("advice-icon");
const breakdownRows  = document.getElementById("breakdown-rows");

// ── Image handling ─────────────────────────────────────────────────────────
function setImage(file) {
  if (!file || !file.type.startsWith("image/")) {
    showToast("Please select a valid image file.");
    return;
  }
  currentFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.onload = () => URL.revokeObjectURL(url);
  uploadDefault.style.display = "none";
  uploadPreview.style.display = "block";
  uploadZone.classList.add("has-image");
  analyzeBtn.disabled = false;
  // Remove file input so drag-over still works
  fileInput.value = "";
}

function clearImage() {
  currentFile = null;
  previewImg.src = "";
  uploadDefault.style.display = "";
  uploadPreview.style.display = "none";
  uploadZone.classList.remove("has-image");
  analyzeBtn.disabled = true;
}

// File input change
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setImage(fileInput.files[0]);
});

// Change image link inside preview
changeImageBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  clearImage();
});

// Source: upload button
btnUpload.addEventListener("click", () => fileInput.click());

// Source: paste button
btnPaste.addEventListener("click", async () => {
  try {
    const items = await navigator.clipboard.read();
    for (const item of items) {
      const type = item.types.find(t => t.startsWith("image/"));
      if (type) {
        const blob = await item.getType(type);
        setImage(new File([blob], "pasted.png", { type }));
        return;
      }
    }
    showToast("No image found in clipboard.");
  } catch {
    showToast("Paste from clipboard requires clipboard permission.");
  }
});

// Global paste (ctrl/cmd+v)
document.addEventListener("paste", (e) => {
  const item = Array.from(e.clipboardData.items).find(i => i.type.startsWith("image/"));
  if (item) setImage(item.getAsFile());
});

// ── Drag & drop ───────────────────────────────────────────────────────────
uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault(); uploadZone.classList.add("dragover");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault(); uploadZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) setImage(file);
});

// Click on upload zone (not on buttons)
uploadZone.addEventListener("click", (e) => {
  if (!uploadZone.classList.contains("has-image") &&
      e.target === uploadZone || e.target.closest(".upload-default-content")) {
    fileInput.click();
  }
});

// ── Camera ────────────────────────────────────────────────────────────────
btnCamera.addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 } }
    });
    cameraVideo.srcObject = stream;
    cameraVideo.play();
    cameraModal.classList.add("open");
  } catch {
    showToast("Camera access denied or not available.");
  }
});

function stopCamera() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  cameraVideo.srcObject = null;
  cameraModal.classList.remove("open");
}

closeCamBtn.addEventListener("click", stopCamera);
cameraModal.addEventListener("click", (e) => { if (e.target === cameraModal) stopCamera(); });

captureBtn.addEventListener("click", () => {
  const canvas = document.createElement("canvas");
  canvas.width = cameraVideo.videoWidth;
  canvas.height = cameraVideo.videoHeight;
  canvas.getContext("2d").drawImage(cameraVideo, 0, 0);
  canvas.toBlob((blob) => {
    setImage(new File([blob], "capture.jpg", { type: "image/jpeg" }));
    stopCamera();
  }, "image/jpeg", 0.92);
});

// ── Prediction ────────────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", async () => {
  if (!currentFile) return;
  setAnalyzing(true);
  try {
    const form = new FormData();
    form.append("file", currentFile);
    const res = await fetch("/api/predict", { method: "POST", body: form });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    showResults(data);
  } catch (err) {
    showToast("Analysis failed: " + (err.message || "Unknown error"));
    setAnalyzing(false);
  }
});

function setAnalyzing(loading) {
  analyzeBtn.disabled = loading;
  if (loading) {
    analyzeBtnText.innerHTML = `<div class="spinner"></div> Analyzing…`;
  } else {
    analyzeBtnText.innerHTML = `Analyze with AI &nbsp;→`;
  }
}

// ── Results ───────────────────────────────────────────────────────────────
function showResults(data) {
  setAnalyzing(false);

  // Step indicator → done
  step1.className = "step done";
  step1.querySelector(".step-circle").innerHTML = `<svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M2 5l2.5 2.5L8 3" stroke="#fff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  step2.className = "step active";
  stepConn.classList.add("done");

  // Demo notice
  demoNotice.style.display = data.demo_mode ? "flex" : "none";

  // Diagnosis header
  diagName.textContent = data.top_name;
  diagMeta.innerHTML = `Code <strong>${data.top_class}</strong> &nbsp;·&nbsp; Confidence <strong>${(data.top_probability * 100).toFixed(1)}%</strong>`;

  riskBadge.textContent = data.risk_label;
  riskBadge.style.background = data.risk_bg + "22";
  riskBadge.style.color = data.risk_text;
  riskBadge.style.borderColor = data.risk_color + "55";

  // Description + advice
  diagDesc.textContent = data.description;
  diagAdvice.style.background = data.risk_bg + "18";
  adviceIcon.style.background = data.risk_color;
  adviceIcon.textContent = data.risk_label.includes("High") ? "✕" :
                            data.risk_label.includes("Moderate") ? "!" : "✓";
  document.getElementById("advice-text").textContent = data.advice;

  // Breakdown bars
  breakdownRows.innerHTML = "";
  data.sorted_predictions.forEach((p, i) => {
    const pct = (p.probability * 100).toFixed(1);
    const isTop = i === 0;
    const color = BAR_COLORS[p.class] || "#888";

    const row = document.createElement("div");
    row.className = "bar-row" + (isTop ? " top-row" : "");
    if (isTop) row.style.background = data.risk_bg + "18";

    row.innerHTML = `
      <div class="bar-names">
        <div class="bar-class-name${isTop ? " top" : ""}">${p.name}</div>
        <div class="bar-code">${p.class}</div>
      </div>
      <div class="bar-track">
        <div class="bar-fill" data-pct="${p.probability * 100}" style="background:${color}"></div>
      </div>
      <span class="bar-pct" style="color:${isTop ? color : (p.probability < 0.001 ? "var(--text-muted)" : color)}">${pct}%</span>`;
    breakdownRows.appendChild(row);
  });

  // Slide upload out, results in
  uploadPanel.style.opacity = "0";
  uploadPanel.style.transform = "translateY(-16px)";
  uploadPanel.style.pointerEvents = "none";
  setTimeout(() => {
    uploadPanel.style.display = "none";
    resultsPanel.classList.add("visible");
    // Animate bars after short delay
    setTimeout(() => {
      document.querySelectorAll(".bar-fill").forEach((bar, i) => {
        setTimeout(() => {
          bar.style.width = bar.dataset.pct + "%";
        }, i * 60);
      });
    }, 100);
  }, 200);
}

backBtn.addEventListener("click", () => {
  // Reset everything
  resultsPanel.classList.remove("visible");
  uploadPanel.style.display = "";
  setTimeout(() => {
    uploadPanel.style.opacity = "1";
    uploadPanel.style.transform = "translateY(0)";
    uploadPanel.style.pointerEvents = "";
  }, 20);
  step1.className = "step active";
  step1.querySelector(".step-circle").textContent = "1";
  step2.className = "step inactive";
  stepConn.classList.remove("done");
  clearImage();
});

// ── Toast notification ────────────────────────────────────────────────────
function showToast(msg) {
  const t = document.createElement("div");
  t.style.cssText = `
    position:fixed;bottom:24px;left:50%;transform:translateX(-50%) translateY(20px);
    background:#1f1f1f;border:0.5px solid rgba(255,255,255,0.1);
    color:#f0eeeb;padding:12px 20px;border-radius:10px;
    font-size:13px;z-index:999;opacity:0;
    transition:opacity 200ms ease,transform 200ms ease;
    box-shadow:0 8px 32px rgba(0,0,0,0.5);font-family:'DM Sans',sans-serif;
  `;
  t.textContent = msg;
  document.body.appendChild(t);
  requestAnimationFrame(() => {
    t.style.opacity = "1"; t.style.transform = "translateX(-50%) translateY(0)";
  });
  setTimeout(() => {
    t.style.opacity = "0"; t.style.transform = "translateX(-50%) translateY(10px)";
    setTimeout(() => t.remove(), 200);
  }, 3500);
}

// Init
document.addEventListener("DOMContentLoaded", () => {
  analyzeBtn.disabled = true;
});
