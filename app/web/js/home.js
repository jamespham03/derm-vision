/**
 * DermVision — Homepage JS
 * Full-screen warp/fisheye grid with mouse parallax.
 */

const CLASS_INFO = {
  MEL:   { name: "Melanoma",                   risk: "high",     riskColor: "#e24b4a" },
  NV:    { name: "Melanocytic Nevus",           risk: "low",      riskColor: "#1d9e75" },
  BCC:   { name: "Basal Cell Carcinoma",        risk: "high",     riskColor: "#e24b4a" },
  AKIEC: { name: "Actinic Keratosis / Bowen's", risk: "moderate", riskColor: "#ef9f27" },
  BKL:   { name: "Benign Keratosis",            risk: "low",      riskColor: "#1d9e75" },
  DF:    { name: "Dermatofibroma",              risk: "low",      riskColor: "#1d9e75" },
  VASC:  { name: "Vascular Lesion",             risk: "low",      riskColor: "#1d9e75" },
  SCC:   { name: "Squamous Cell Carcinoma",     risk: "high",     riskColor: "#e24b4a" },
};

const RISK_LABELS = { high: "High Risk", moderate: "Moderate Risk", low: "Low Risk" };

// 20 cards — 5 cols × 4 rows, all 8 classes represented
const GRID_CARDS = [
  { cls: "MEL"   }, { cls: "NV"    }, { cls: "BCC"   }, { cls: "AKIEC" }, { cls: "BKL"   },
  { cls: "DF"    }, { cls: "VASC"  }, { cls: "SCC"   }, { cls: "MEL"   }, { cls: "NV"    },
  { cls: "BCC"   }, { cls: "AKIEC" }, { cls: "BKL"   }, { cls: "DF"    }, { cls: "VASC"  },
  { cls: "SCC"   }, { cls: "MEL"   }, { cls: "NV"    }, { cls: "BCC"   }, { cls: "AKIEC" },
];

const COLS = 5;

// ── Build grid ────────────────────────────────────────────────────────────────
function buildGrid(realImages) {
  const grid = document.getElementById("image-grid");
  if (!grid) return;

  GRID_CARDS.forEach((card, i) => {
    const info = CLASS_INFO[card.cls];
    const el = document.createElement("div");
    el.className = "grid-card";
    el.setAttribute("data-index", i);

    const thumb = document.createElement("div");
    thumb.className = "card-thumb";
    if (realImages && realImages[i]) {
      const img = document.createElement("img");
      img.src = realImages[i];
      img.alt = info.name;
      img.loading = "lazy";
      thumb.appendChild(img);
    } else {
      thumb.classList.add(`card-bg-${card.cls}`);
    }

    const dot = document.createElement("div");
    dot.className = "card-risk-dot";
    dot.style.color = info.riskColor;
    dot.style.background = info.riskColor;

    const overlay = document.createElement("div");
    overlay.className = "card-overlay";
    overlay.innerHTML = `
      <div class="card-info">
        <span class="card-tag">
          <span style="width:6px;height:6px;border-radius:50%;background:${info.riskColor};display:inline-block;"></span>
          ${card.cls} &nbsp;·&nbsp; ${RISK_LABELS[info.risk]}
        </span>
        <div class="card-name">${info.name}</div>
      </div>`;

    el.appendChild(thumb);
    el.appendChild(dot);
    el.appendChild(overlay);

    el.addEventListener("click", () => { window.location.href = "/analyze"; });

    grid.appendChild(el);
  });
}

// ── Warp / fisheye effect ─────────────────────────────────────────────────────
function applyWarp() {
  const grid = document.getElementById("image-grid");
  if (!grid) return;
  const cards = [...grid.querySelectorAll(".grid-card")];
  const ROWS = Math.ceil(cards.length / COLS);

  cards.forEach((card, i) => {
    const col = i % COLS;
    const row = Math.floor(i / COLS);

    // Normalized position: -1 (left/top edge) to +1 (right/bottom edge)
    const nx = (col / (COLS - 1)) * 2 - 1;   // -1 to 1
    const ny = (row / (ROWS - 1)) * 2 - 1;   // -1 to 1

    // Barrel distortion: outer cards rotate away from viewer
    const rotY = nx * -22;          // columns: outer tilt ±22deg
    const rotX = ny * 12;           // rows: top/bottom tilt ±12deg
    const scale = 1 - Math.abs(nx) * 0.10 - Math.abs(ny) * 0.06; // edge scale-down

    card.dataset.baseRY = rotY;
    card.dataset.baseRX = rotX;
    card.dataset.baseS  = scale;
    applyCardTransform(card, rotY, rotX, scale, false);

    card.addEventListener("mouseenter", () => {
      // Flatten warp on hover + pop forward
      applyCardTransform(card, rotY * 0.25, rotX * 0.25, scale * 1.10, true);
      card.style.zIndex = "20";
    });
    card.addEventListener("mouseleave", () => {
      applyCardTransform(card, rotY, rotX, scale, false);
      card.style.zIndex = "";
    });
  });
}

function applyCardTransform(card, ry, rx, s, isHover) {
  const tz = isHover ? 28 : 0;
  card.style.transform =
    `perspective(900px) rotateY(${ry}deg) rotateX(${rx}deg) scale(${s}) translateZ(${tz}px)`;
  card.style.transition = isHover
    ? "transform 180ms cubic-bezier(0.34,1.56,0.64,1), box-shadow 180ms ease, opacity 150ms ease"
    : "transform 350ms cubic-bezier(0.34,1.56,0.64,1), box-shadow 350ms ease, opacity 200ms ease";
}

// ── Mouse parallax ────────────────────────────────────────────────────────────
function initParallax() {
  const stage = document.getElementById("warp-stage");
  if (!stage) return;

  let targetRX = 8, targetRY = 0;
  let currentRX = 8, currentRY = 0;
  let rafId = null;

  function lerp(a, b, t) { return a + (b - a) * t; }

  function tick() {
    currentRX = lerp(currentRX, targetRX, 0.08);
    currentRY = lerp(currentRY, targetRY, 0.08);
    stage.style.transform = `perspective(1100px) rotateX(${currentRX}deg) rotateY(${currentRY}deg)`;
    rafId = requestAnimationFrame(tick);
  }
  tick();

  document.addEventListener("mousemove", (e) => {
    const mx = (e.clientX / window.innerWidth  - 0.5) * 2; // -1 to 1
    const my = (e.clientY / window.innerHeight - 0.5) * 2;
    targetRX = 8 + my * -3.5;   // base tilt + mouse-Y influence
    targetRY = mx *  4;          // rotate with mouse-X
  });

  document.addEventListener("mouseleave", () => {
    targetRX = 8; targetRY = 0;
  });
}

// ── Sibling dim effect ────────────────────────────────────────────────────────
function initSiblingDim() {
  const grid = document.getElementById("image-grid");
  if (!grid) return;
  grid.addEventListener("mouseenter", () => {}, { passive: true });

  grid.querySelectorAll(".grid-card").forEach(card => {
    card.addEventListener("mouseenter", () => {
      grid.querySelectorAll(".grid-card").forEach(c => {
        if (c !== card) c.style.opacity = "0.55";
      });
    });
    card.addEventListener("mouseleave", () => {
      grid.querySelectorAll(".grid-card").forEach(c => { c.style.opacity = ""; });
    });
  });
}

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  let realImages = null;
  try {
    const res = await fetch("/api/samples");
    const data = await res.json();
    if (data.available) realImages = data.images;
  } catch { /* fallback to gradients */ }

  buildGrid(realImages);

  // Wait one frame for layout, then apply warp
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      applyWarp();
      initParallax();
      initSiblingDim();
    });
  });
}

document.addEventListener("DOMContentLoaded", init);
