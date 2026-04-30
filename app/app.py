"""
DermVision: AI-Powered Skin Lesion Analysis
Redesigned multi-page Gradio UI — image upload + classification only.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import gradio as gr
import numpy as np
import torch
from PIL import Image

from src.dataset import CLASS_NAMES
from src.models.efficientnet import EfficientNetB3Classifier
from src.transforms import get_val_transforms

# ── Globals ───────────────────────────────────────────────────────────────────
MODEL = None
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
IMAGE_SIZE = 300

# ── Class metadata ────────────────────────────────────────────────────────────
CLASS_INFO = {
    "MEL":   {
        "name": "Melanoma", "risk": "high", "color": "#a32d2d",
        "bar_color": "#e24b4a",
        "desc": "A serious form of skin cancer that develops in melanocytes (pigment-producing cells).",
    },
    "NV":    {
        "name": "Melanocytic Nevus", "risk": "low", "color": "#0f6e56",
        "bar_color": "#1d9e75",
        "desc": "A common benign mole formed by a cluster of melanocytes.",
    },
    "BCC":   {
        "name": "Basal Cell Carcinoma", "risk": "high", "color": "#a32d2d",
        "bar_color": "#e24b4a",
        "desc": "The most common skin cancer. Slow-growing and rarely spreads if treated early.",
    },
    "AKIEC": {
        "name": "Actinic Keratosis / Bowen's", "risk": "moderate", "color": "#854f0b",
        "bar_color": "#ef9f27",
        "desc": "A pre-cancerous rough skin patch caused by years of sun exposure.",
    },
    "BKL":   {
        "name": "Benign Keratosis", "risk": "low", "color": "#0f6e56",
        "bar_color": "#1d9e75",
        "desc": "Non-cancerous skin growth including seborrheic keratosis and solar lentigo.",
    },
    "DF":    {
        "name": "Dermatofibroma", "risk": "low", "color": "#0f6e56",
        "bar_color": "#1d9e75",
        "desc": "A common benign fibrous nodule typically found on the lower legs.",
    },
    "VASC":  {
        "name": "Vascular Lesion", "risk": "low", "color": "#185fa5",
        "bar_color": "#378add",
        "desc": "Abnormality of blood vessels including cherry angioma and pyogenic granuloma.",
    },
    "SCC":   {
        "name": "Squamous Cell Carcinoma", "risk": "high", "color": "#a32d2d",
        "bar_color": "#e24b4a",
        "desc": "The second most common skin cancer. Can spread to other parts if left untreated.",
    },
}

RISK_STYLES = {
    "high": {
        "badge_bg": "#fcebeb", "badge_color": "#501313", "badge_border": "#e24b4a",
        "label": "High Risk",
        "advice_bg": "#fcebeb", "advice_border": "#e24b4a", "advice_color": "#791f1f",
        "icon": "✕",
        "advice": "Please consult a dermatologist as soon as possible.",
    },
    "moderate": {
        "badge_bg": "#faeeda", "badge_color": "#633806", "badge_border": "#ef9f27",
        "label": "Moderate Risk",
        "advice_bg": "#faeeda", "advice_border": "#ef9f27", "advice_color": "#633806",
        "icon": "!",
        "advice": "Schedule a dermatologist appointment for evaluation.",
    },
    "low": {
        "badge_bg": "#e1f5ee", "badge_color": "#085041", "badge_border": "#5dcaa5",
        "label": "Low Risk",
        "advice_bg": "#e1f5ee", "advice_border": "#5dcaa5", "advice_color": "#085041",
        "icon": "✓",
        "advice": "Appears likely benign. Continue monitoring for any changes over time.",
    },
}

# ── Model ─────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str) -> None:
    global MODEL
    MODEL = EfficientNetB3Classifier(num_classes=8, pretrained=False)
    MODEL.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    MODEL.to(DEVICE)
    MODEL.eval()
    print(f"Model loaded from {checkpoint_path}")


def run_predict(image: Image.Image) -> dict:
    if MODEL is None:
        return {name: 1.0 / 8 for name in CLASS_NAMES}
    image_np = np.array(image.convert("RGB"))
    transform = get_val_transforms(IMAGE_SIZE)
    tensor = transform(image=image_np)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return {name: float(p) for name, p in zip(CLASS_NAMES, probs)}


# ── Shared HTML blocks ────────────────────────────────────────────────────────
NAV_HTML = """
<nav style="display:flex;align-items:center;justify-content:space-between;
    padding:0 32px;height:60px;background:#ffffff;
    border-bottom:0.5px solid rgba(0,0,0,0.1);font-family:'DM Sans',sans-serif;">
    <div style="display:flex;align-items:center;gap:10px;">
        <div style="width:32px;height:32px;border-radius:8px;background:#0f6e56;
            display:flex;align-items:center;justify-content:center;
            color:#fff;font-size:17px;font-family:'DM Serif Display',serif;font-style:italic;">D</div>
        <span style="font-family:'DM Serif Display',serif;font-size:18px;color:#1a1a18;">DermVision</span>
        <span style="font-size:10px;color:#0f6e56;font-weight:500;letter-spacing:.5px;text-transform:uppercase;margin-left:2px;opacity:.8;">Beta</span>
    </div>
    <div style="display:flex;align-items:center;gap:28px;">
        <span style="font-size:13px;color:#5f5e5a;cursor:pointer;">How it works</span>
        <span style="font-size:13px;color:#5f5e5a;cursor:pointer;">Research</span>
        <span style="font-size:13px;color:#5f5e5a;cursor:pointer;">Disclaimer</span>
    </div>
</nav>
"""

FOOTER_HTML = """
<div style="border-top:0.5px solid rgba(0,0,0,0.1);padding:20px 32px;
    display:flex;align-items:center;justify-content:space-between;
    background:#f9f8f5;font-family:'DM Sans',sans-serif;margin-top:8px;">
    <span style="font-size:12px;color:#888780;">© 2025 DermVision. For educational use only.</span>
    <div style="display:flex;gap:20px;">
        <span style="font-size:12px;color:#888780;cursor:pointer;">Privacy</span>
        <span style="font-size:12px;color:#888780;cursor:pointer;">Terms</span>
        <span style="font-size:12px;color:#888780;cursor:pointer;">Contact</span>
    </div>
    <div style="display:flex;align-items:center;gap:5px;">
        <div style="width:5px;height:5px;border-radius:50%;background:#0f6e56;"></div>
        <span style="font-size:11px;color:#888780;">EfficientNet-B3 · ISIC 2019</span>
    </div>
</div>
"""

FONT_IMPORT = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&family=DM+Serif+Display:ital@0;1&display=swap" rel="stylesheet">
"""

INPUT_HERO_HTML = """
<div style="background:#0f6e56;color:#fff;padding:40px 32px 36px;
    position:relative;overflow:hidden;font-family:'DM Sans',sans-serif;">
    <div style="position:absolute;top:-60px;right:-60px;width:220px;height:220px;
        border-radius:50%;background:rgba(255,255,255,0.06);"></div>
    <div style="display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:500;
        letter-spacing:.8px;text-transform:uppercase;background:rgba(255,255,255,0.15);
        border-radius:20px;padding:4px 12px;margin-bottom:14px;color:rgba(255,255,255,0.9);">
        <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
            <circle cx="4" cy="4" r="3" stroke="currentColor" stroke-width="1.2"/>
            <circle cx="4" cy="4" r="1.2" fill="currentColor"/>
        </svg>
        AI-powered analysis
    </div>
    <div style="color:#fff;font-family:'DM Serif Display',serif;font-size:28px;font-weight:400;
        line-height:1.2;margin-bottom:8px;">Upload a Skin Image</div>
    <p style="font-size:13px;opacity:.8;max-width:480px;margin:0;">
        Upload a clear photo of your skin lesion for instant AI classification across 8 lesion types.
    </p>
</div>
"""

INPUT_PROGRESS_HTML = """
<div style="background:#f9f8f5;border-bottom:0.5px solid rgba(0,0,0,0.1);
    padding:13px 32px;display:flex;align-items:center;font-family:'DM Sans',sans-serif;">
    <div style="display:flex;align-items:center;gap:8px;">
        <div style="width:22px;height:22px;border-radius:50%;background:#1a1a18;color:#fff;
            display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:500;">1</div>
        <span style="font-size:13px;font-weight:500;color:#1a1a18;">Upload image</span>
    </div>
    <div style="width:60px;height:1px;background:rgba(0,0,0,0.15);margin:0 12px;"></div>
    <div style="display:flex;align-items:center;gap:8px;">
        <div style="width:22px;height:22px;border-radius:50%;background:#d3d1c7;color:#888780;
            display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:500;">2</div>
        <span style="font-size:13px;color:#888780;">AI analysis results</span>
    </div>
</div>
"""

UPLOAD_TIPS_HTML = """
<div style="background:#f9f8f5;border:0.5px solid rgba(0,0,0,0.1);border-radius:10px;
    padding:12px 16px;margin-bottom:14px;font-family:'DM Sans',sans-serif;
    font-size:13px;color:#5f5e5a;line-height:1.6;">
    <div style="font-weight:500;margin-bottom:5px;display:flex;align-items:center;gap:6px;color:#1a1a18;">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <circle cx="7" cy="7" r="6" stroke="#5f5e5a" stroke-width="1.2"/>
            <path d="M7 4v3.5" stroke="#5f5e5a" stroke-width="1.2" stroke-linecap="round"/>
            <circle cx="7" cy="10" r=".6" fill="#5f5e5a"/>
        </svg>
        Photo tips for best results
    </div>
    <ul style="margin:0;padding-left:16px;display:flex;flex-direction:column;gap:2px;">
        <li>Use a clear, well-lit, close-up photo of the affected area</li>
        <li>Avoid shadows, blur, and extreme angles</li>
        <li>Dermoscopy images yield the most accurate classification results</li>
    </ul>
</div>
"""

INPUT_DISCLAIMER_HTML = """
<div style="background:#f9f8f5;border:0.5px solid rgba(0,0,0,0.1);border-radius:10px;
    padding:11px 14px;margin-top:12px;font-family:'DM Sans',sans-serif;
    font-size:12px;color:#888780;line-height:1.6;display:flex;gap:8px;align-items:flex-start;">
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style="flex-shrink:0;margin-top:1px;">
        <path d="M7 2L12.5 11.5H1.5L7 2Z" stroke="#b4b2a9" stroke-width="1.2" stroke-linejoin="round"/>
        <path d="M7 6v2.5" stroke="#b4b2a9" stroke-width="1.2" stroke-linecap="round"/>
        <circle cx="7" cy="10" r=".5" fill="#b4b2a9"/>
    </svg>
    <span><strong style="color:#5f5e5a;">Medical disclaimer:</strong> This tool is for educational
    and screening purposes only. It does not provide medical advice, diagnosis, or treatment.
    Always consult a licensed dermatologist for any skin concerns.</span>
</div>
"""

# ── Results HTML ──────────────────────────────────────────────────────────────
def build_results_html(probs):
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_cls, top_prob = sorted_probs[0]
    info = CLASS_INFO[top_cls]
    rs = RISK_STYLES[info["risk"]]

    bars = ""
    for i, (cls, prob) in enumerate(sorted_probs):
        ci = CLASS_INFO[cls]
        pct = prob * 100
        is_top = cls == top_cls
        if is_top:
            row_bg = f"background:{rs['advice_bg']};margin:-4px -12px;padding:10px 12px;border-radius:8px;"
            name_weight = "500"
            pct_color = ci["color"]
        else:
            row_bg = ""
            name_weight = "400"
            pct_color = "#888780" if pct < 0.1 else ci["bar_color"]

        separator = "border-top:0.5px solid rgba(0,0,0,0.07);" if i > 0 and not is_top else ""
        bars += f"""
        <div style="display:flex;align-items:center;gap:12px;padding:8px 0;{row_bg}{separator}">
            <div style="min-width:185px;">
                <div style="font-weight:{name_weight};font-size:13px;color:#1a1a18;">{ci['name']}</div>
                <div style="font-size:11px;color:#888780;margin-top:1px;">{cls}</div>
            </div>
            <div style="flex:1;background:#f1f0eb;border-radius:999px;height:6px;overflow:hidden;">
                <div style="width:{pct:.1f}%;background:{ci['bar_color']};height:100%;border-radius:999px;"></div>
            </div>
            <span style="min-width:46px;text-align:right;font-weight:500;font-size:13px;color:{pct_color};">{pct:.1f}%</span>
        </div>"""

    demo_banner = "" if MODEL else """
        <div style="background:#faeeda;border:0.5px solid #ef9f27;border-radius:10px;
             padding:12px 16px;margin-bottom:16px;font-size:13px;color:#633806;
             display:flex;gap:8px;align-items:flex-start;font-family:'DM Sans',sans-serif;">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style="flex-shrink:0;margin-top:1px;">
                <path d="M7 2L12.5 11.5H1.5L7 2Z" stroke="#854f0b" stroke-width="1.2" stroke-linejoin="round"/>
                <path d="M7 6v2.5" stroke="#854f0b" stroke-width="1.2" stroke-linecap="round"/>
                <circle cx="7" cy="10" r=".5" fill="#854f0b"/>
            </svg>
            <div><strong>Demo mode:</strong> No trained checkpoint found.
            Probabilities shown are simulated (equal weights).</div>
        </div>"""

    return f"""
    {FONT_IMPORT}
    <div style="font-family:'DM Sans',sans-serif;background:#ffffff;color:#1a1a18;">
        {NAV_HTML}
        <div style="background:#0f6e56;color:#fff;padding:40px 32px 36px;position:relative;overflow:hidden;">
            <div style="position:absolute;top:-60px;right:-60px;width:220px;height:220px;
                border-radius:50%;background:rgba(255,255,255,0.06);"></div>
            <div style="display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:500;
                letter-spacing:.8px;text-transform:uppercase;background:rgba(255,255,255,0.15);
                border-radius:20px;padding:4px 12px;margin-bottom:14px;color:rgba(255,255,255,0.9);">
                <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
                    <circle cx="4" cy="4" r="3" stroke="currentColor" stroke-width="1.2"/>
                    <circle cx="4" cy="4" r="1.2" fill="currentColor"/>
                </svg>
                Analysis complete
            </div>
            <div style="font-family:'DM Serif Display',serif;font-size:28px;font-weight:400;
                line-height:1.2;margin-bottom:8px;">AI Analysis Results</div>
            <p style="font-size:13px;opacity:.8;max-width:480px;margin:0;">
                Results are for educational screening purposes only. Consult a dermatologist for diagnosis.
            </p>
        </div>
        <div style="background:#f9f8f5;border-bottom:0.5px solid rgba(0,0,0,0.1);
            padding:13px 32px;display:flex;align-items:center;font-family:'DM Sans',sans-serif;">
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="width:22px;height:22px;border-radius:50%;background:#0f6e56;color:#fff;
                    display:flex;align-items:center;justify-content:center;font-size:11px;">
                    <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                        <path d="M2 5l2.5 2.5L8 3" stroke="#fff" stroke-width="1.5"
                            stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <span style="font-size:13px;color:#5f5e5a;">Upload image</span>
            </div>
            <div style="width:60px;height:1px;background:rgba(0,0,0,0.15);margin:0 12px;"></div>
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="width:22px;height:22px;border-radius:50%;background:#1a1a18;color:#fff;
                    display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:500;">2</div>
                <span style="font-size:13px;font-weight:500;color:#1a1a18;">AI analysis results</span>
            </div>
        </div>
        <div style="padding:28px 32px;max-width:720px;margin:0 auto;">
            {demo_banner}
            <div style="background:#fffbeb;border:0.5px solid #fcd34d;border-radius:10px;
                padding:12px 16px;margin-bottom:20px;font-size:12px;color:#78350f;
                line-height:1.6;display:flex;gap:8px;align-items:flex-start;">
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style="flex-shrink:0;margin-top:1px;">
                    <path d="M7 2L12.5 11.5H1.5L7 2Z" stroke="#d97706" stroke-width="1.2" stroke-linejoin="round"/>
                    <path d="M7 6v2.5" stroke="#d97706" stroke-width="1.2" stroke-linecap="round"/>
                    <circle cx="7" cy="10" r=".5" fill="#d97706"/>
                </svg>
                <span><strong>Medical disclaimer:</strong> This tool is for educational and screening
                purposes only — not medical advice, diagnosis, or treatment.
                Always consult a licensed dermatologist for any skin concerns.</span>
            </div>
            <div style="border:0.5px solid rgba(0,0,0,0.1);border-radius:10px;background:#fff;
                overflow:hidden;margin-bottom:16px;">
                <div style="display:flex;align-items:flex-start;justify-content:space-between;
                    gap:16px;padding:24px;border-bottom:0.5px solid rgba(0,0,0,0.07);">
                    <div>
                        <div style="font-size:10px;font-weight:500;letter-spacing:.8px;
                            text-transform:uppercase;color:#888780;margin-bottom:6px;">Primary diagnosis</div>
                        <div style="font-family:'DM Serif Display',serif;font-size:26px;
                            font-weight:400;color:#1a1a18;line-height:1.1;margin-bottom:4px;">{info['name']}</div>
                        <div style="font-size:13px;color:#888780;">
                            Code <strong style="color:#5f5e5a;">{top_cls}</strong>
                            &nbsp;·&nbsp;
                            Confidence <strong style="color:#5f5e5a;">{top_prob * 100:.1f}%</strong>
                        </div>
                    </div>
                    <span style="display:inline-flex;align-items:center;padding:5px 14px;
                        border-radius:20px;font-size:11px;font-weight:500;letter-spacing:.4px;
                        white-space:nowrap;background:{rs['badge_bg']};color:{rs['badge_color']};
                        border:0.5px solid {rs['badge_border']};">{rs['label']}</span>
                </div>
                <div style="padding:16px 24px;font-size:14px;color:#5f5e5a;line-height:1.6;
                    border-bottom:0.5px solid rgba(0,0,0,0.07);">{info['desc']}</div>
                <div style="display:flex;gap:12px;align-items:flex-start;padding:14px 24px;
                    background:{rs['advice_bg']};">
                    <div style="width:26px;height:26px;border-radius:50%;
                        background:{rs['badge_border']};display:flex;align-items:center;
                        justify-content:center;flex-shrink:0;margin-top:1px;
                        color:#fff;font-size:13px;font-weight:600;">{rs['icon']}</div>
                    <span style="font-size:13px;font-weight:500;color:{rs['advice_color']};
                        line-height:1.5;">{rs['advice']}</span>
                </div>
            </div>
            <div style="border:0.5px solid rgba(0,0,0,0.1);border-radius:10px;background:#fff;
                padding:20px 24px;margin-bottom:20px;">
                <div style="font-size:12px;font-weight:500;letter-spacing:.6px;
                    text-transform:uppercase;color:#888780;margin-bottom:16px;
                    display:flex;align-items:center;gap:8px;">
                    Classification breakdown
                    <div style="flex:1;height:0.5px;background:rgba(0,0,0,0.1);"></div>
                </div>
                {bars}
            </div>
        </div>
    </div>"""


# ── Event handlers ────────────────────────────────────────────────────────────
def analyze(image):
    if image is None:
        gr.Warning("Please upload or capture a skin image before analyzing.")
        return gr.update(value="")
    probs = run_predict(image)
    return gr.update(value=build_results_html(probs))


def show_results():
    return gr.update(visible=False), gr.update(visible=True)


def go_back():
    return gr.update(visible=True), gr.update(visible=False), gr.update(value="")


# ── CSS ───────────────────────────────────────────────────────────────────────
# Strategy: keep gr.Image fully functional. Override its visual appearance
# using the REAL Svelte-scoped class names found in Gradio 6.x source CSS.
# Key classes:
#   .wrap.svelte-1dqolfz  — ImageUploader outer wrap (drop target + icon area)
#   .wrap.svelte-ua961l   — Upload progress wrap
#   button[aria-label="Click to upload or drop files"]  — Gradio's upload button
#   button[aria-label="Paste from clipboard"]           — Gradio's paste button
# We hide the native source-tab buttons (webcam/upload/clipboard selector row)
# and instead expose our own styled buttons that trigger Gradio's internals.
CSS = """
:root, .dark, [data-theme="dark"] {
    color-scheme: light !important;
    --body-background-fill: #ffffff !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f9f8f5 !important;
    --border-color-primary: rgba(0,0,0,0.1) !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: rgba(0,0,0,0.12) !important;
    --color-accent: #0f6e56 !important;
    --primary-500: #0f6e56 !important;
    --primary-600: #085041 !important;
    --button-primary-background-fill: #0f6e56 !important;
    --button-primary-background-fill-hover: #085041 !important;
    --button-primary-text-color: #ffffff !important;
    --button-secondary-background-fill: #f9f8f5 !important;
    --button-secondary-background-fill-hover: #f1f0eb !important;
    --button-secondary-text-color: #1a1a18 !important;
    --button-secondary-border-color: rgba(0,0,0,0.15) !important;
    --block-label-background-fill: #ffffff !important;
    --block-label-text-color: #1a1a18 !important;
    --body-text-color: #1a1a18 !important;
}
html, body { background: #ffffff !important; color: #1a1a18 !important; }

.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    background: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
}
.main { background: #ffffff !important; }

/* Hide all Gradio tab chrome */
.tab-nav, div.tabs > div.tab-nav, .tabs > .tab-nav,
button.tab-nav-button, [role="tablist"], .tabs .tab-nav { display: none !important; }
footer, footer.svelte-1ax1toq { display: none !important; }

/* ── Style the gr.Image upload zone ─────────────────────────────────────────
   We style the native Gradio Image widget to look like our design.
   The widget lives inside #gr-image-input.                                   */

/* Remove default block border/shadow on the image component */
#gr-image-input {
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    background: transparent !important;
}

/* The main upload drop zone — ImageUploader wrap */
#gr-image-input .wrap.svelte-1dqolfz {
    border: 2px dashed #c8c6be !important;
    border-radius: 12px !important;
    background: #ffffff !important;
    min-height: 240px !important;
    color: #1a1a18 !important;
    transition: border-color .15s, background .15s !important;
}
#gr-image-input .wrap.svelte-1dqolfz:hover {
    border-color: #0f6e56 !important;
    background: #f9fdfb !important;
}

/* Upload progress wrap */
#gr-image-input .wrap.svelte-ua961l {
    background: #ffffff !important;
    border: 2px dashed #c8c6be !important;
    border-radius: 12px !important;
    min-height: 240px !important;
}

/* Icon in the centre of the drop zone */
#gr-image-input .icon-wrap.svelte-1dqolfz svg {
    stroke: #b4b2a9 !important;
    fill: none !important;
    width: 48px !important;
    height: 48px !important;
}

/* The Gradio upload button that covers the whole drop zone */
#gr-image-input button[aria-label="Click to upload or drop files"],
#gr-image-input button.svelte-1dqolfz {
    background: transparent !important;
    border: none !important;
    width: 100% !important;
    cursor: pointer !important;
}

/* Block label (hides "Image" text above the widget) */
#gr-image-input .block-label,
#gr-image-input label,
#gr-image-input .label-wrap { display: none !important; }

/* ── Style Gradio's native source-selection buttons ── */
#gr-image-input .source-selection,
#gr-image-input [data-testid="source-select"] {
    display: flex !important;
    gap: 12px 12px !important;
    justify-content: center !important;
    align-items: center !important;
    padding: 16px 16px 20px !important;
    background: transparent !important;
    border: none !important;
    flex-wrap: wrap !important;
    height: 100% !important;
}
#gr-image-input .source-selection button,
#gr-image-input [data-testid="source-select"] button {
    display: inline-flex !important;
    width: 40px !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    padding: 22px 16px !important;
    border-radius: 10px !important;
    border: 0.5px solid rgba(0,0,0,0.15) !important;
    background: #ffffff !important;
    color: #1a1a18 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: background .12s, border-color .12s !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}
#gr-image-input .source-selection button:hover,
#gr-image-input [data-testid="source-select"] button:hover {
    background: #f4f3f0 !important;
    border-color: rgba(0,0,0,0.28) !important;
}
#gr-image-input .source-selection button.selected,
#gr-image-input [data-testid="source-select"] button.selected {
    background: #0f6e56 !important;
    color: #ffffff !important;
    border-color: #0f6e56 !important;
}
/* Hide Gradio's own button SVGs; we inject custom ones via ::before */
#gr-image-input .source-selection button svg,
#gr-image-input [data-testid="source-select"] button svg {
    display: none !important;
}
#gr-image-input .source-selection button::before,
#gr-image-input [data-testid="source-select"] button::before {
    content: '';
    display: inline-block;
    width: 20px;
    height: 20px;
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    flex-shrink: 0;
}
/* Upload icon */
#gr-image-input .source-selection button:nth-child(1)::before,
#gr-image-input [data-testid="source-select"] button:nth-child(1)::before {
    background-image: url("data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 16 16' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M8 2v8M4 6l4-4 4 4' stroke='%231a1a18' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M2 11v1a2 2 0 002 2h8a2 2 0 002-2v-1' stroke='%231a1a18' stroke-width='1.5' stroke-linecap='round'/%3E%3C/svg%3E");
}
#gr-image-input .source-selection button.selected:nth-child(1)::before,
#gr-image-input [data-testid="source-select"] button.selected:nth-child(1)::before {
    background-image: url("data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 16 16' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M8 2v8M4 6l4-4 4 4' stroke='%23ffffff' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M2 11v1a2 2 0 002 2h8a2 2 0 002-2v-1' stroke='%23ffffff' stroke-width='1.5' stroke-linecap='round'/%3E%3C/svg%3E");
}
/* Camera icon */
#gr-image-input .source-selection button:nth-child(2)::before,
#gr-image-input [data-testid="source-select"] button:nth-child(2)::before {
    background-image: url("data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 16 16' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Crect x='2' y='4' width='12' height='9' rx='1.5' stroke='%231a1a18' stroke-width='1.5'/%3E%3Ccircle cx='8' cy='8.5' r='2.2' stroke='%231a1a18' stroke-width='1.5'/%3E%3Cpath d='M6 3h4' stroke='%231a1a18' stroke-width='1.5' stroke-linecap='round'/%3E%3C/svg%3E");
}
#gr-image-input .source-selection button.selected:nth-child(2)::before,
#gr-image-input [data-testid="source-select"] button.selected:nth-child(2)::before {
    background-image: url("data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 16 16' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Crect x='2' y='4' width='12' height='9' rx='1.5' stroke='%23ffffff' stroke-width='1.5'/%3E%3Ccircle cx='8' cy='8.5' r='2.2' stroke='%23ffffff' stroke-width='1.5'/%3E%3Cpath d='M6 3h4' stroke='%23ffffff' stroke-width='1.5' stroke-linecap='round'/%3E%3C/svg%3E");
}
/* Paste/clipboard icon */
#gr-image-input .source-selection button:nth-child(3)::before,
#gr-image-input [data-testid="source-select"] button:nth-child(3)::before {
    background-image: url("data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 16 16' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Crect x='5' y='2' width='8' height='11' rx='1.5' stroke='%231a1a18' stroke-width='1.5'/%3E%3Cpath d='M3 4H2.5A1.5 1.5 0 001 5.5v8A1.5 1.5 0 002.5 15h7A1.5 1.5 0 0011 13.5V13' stroke='%231a1a18' stroke-width='1.5' stroke-linecap='round'/%3E%3Cpath d='M8 5H9' stroke='%231a1a18' stroke-width='1.5' stroke-linecap='round'/%3E%3C/svg%3E");
}
#gr-image-input .source-selection button.selected:nth-child(3)::before,
#gr-image-input [data-testid="source-select"] button.selected:nth-child(3)::before {
    background-image: url("data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 16 16' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Crect x='5' y='2' width='8' height='11' rx='1.5' stroke='%23ffffff' stroke-width='1.5'/%3E%3Cpath d='M3 4H2.5A1.5 1.5 0 001 5.5v8A1.5 1.5 0 002.5 15h7A1.5 1.5 0 0011 13.5V13' stroke='%23ffffff' stroke-width='1.5' stroke-linecap='round'/%3E%3Cpath d='M8 5H9' stroke='%23ffffff' stroke-width='1.5' stroke-linecap='round'/%3E%3C/svg%3E");
}

/* Analyze button */
.analyze-btn > button {
    background: #0f6e56 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    font-family: 'DM Sans', sans-serif !important;
    height: 50px !important;
    transition: background .15s !important;
}
.analyze-btn > button:hover { background: #085041 !important; }

/* Back button */
.back-btn > button {
    background: #f9f8f5 !important;
    color: #1a1a18 !important;
    border: 0.5px solid rgba(0,0,0,0.15) !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    height: 44px !important;
}
.back-btn > button:hover { background: #f1f0eb !important; }

.gr-html { padding: 0 !important; }
.page-content { padding: 28px 32px; max-width: 680px; margin: 0 auto; }
"""

# ── App factory ───────────────────────────────────────────────────────────────
def create_app() -> gr.Blocks:
    with gr.Blocks(
        title="DermVision — AI Skin Analysis",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.Color(
                c50="#e1f5ee", c100="#9fe1cb", c200="#5dcaa5",
                c300="#1d9e75", c400="#0f6e56", c500="#085041",
                c600="#063d31", c700="#042c23", c800="#021c16",
                c900="#010e0b", c950="#000705",
            ),
            font=gr.themes.GoogleFont("DM Sans"),
        ),
        css=CSS,
    ) as app:

        gr.HTML(FONT_IMPORT)

        # ── Page 1: Image Upload ───────────────────────────────────────────────
        with gr.Column(visible=True) as page1:

            gr.HTML(NAV_HTML)
            gr.HTML(INPUT_HERO_HTML)
            gr.HTML(INPUT_PROGRESS_HTML)

            with gr.Column(elem_classes="page-content"):

                gr.HTML("""
                <div style="font-size:11px;font-weight:500;letter-spacing:.7px;text-transform:uppercase;
                    color:#888780;margin-bottom:12px;display:flex;align-items:center;gap:8px;
                    font-family:'DM Sans',sans-serif;">
                    Skin image
                    <div style="flex:1;height:0.5px;background:rgba(0,0,0,0.1);"></div>
                </div>
                """)

                gr.HTML(UPLOAD_TIPS_HTML)

                # The REAL gr.Image — fully functional, styled via CSS above
                image_input = gr.Image(
                    type="pil",
                    label="Upload or capture skin image",
                    sources=["upload", "webcam", "clipboard"],
                    height=260,
                    elem_id="gr-image-input",
                    show_label=False,
                    container=False,
                )

                analyze_btn = gr.Button(
                    "Analyze with AI →",
                    variant="primary",
                    size="lg",
                    elem_classes="analyze-btn",
                )

                gr.HTML(INPUT_DISCLAIMER_HTML)

                gr.HTML("""
                <div style="text-align:center;font-size:11px;color:#b4b2a9;margin-top:14px;
                    font-family:'DM Sans',sans-serif;display:flex;align-items:center;gap:8px;">
                    <div style="flex:1;height:0.5px;background:rgba(0,0,0,0.07);"></div>
                    Powered by EfficientNet-B3 &nbsp;·&nbsp; Trained on ISIC 2019 &nbsp;·&nbsp; 8-class skin lesion classification
                    <div style="flex:1;height:0.5px;background:rgba(0,0,0,0.07);"></div>
                </div>
                """)

            gr.HTML(FOOTER_HTML)

        # ── Page 2: Results ────────────────────────────────────────────────────
        with gr.Column(visible=False) as page2:

            results_html = gr.HTML(value="")

            with gr.Column(elem_classes="page-content"):
                back_btn = gr.Button(
                    "← Start new analysis",
                    variant="secondary",
                    elem_classes="back-btn",
                )

            gr.HTML(FOOTER_HTML)

        # ── Events ────────────────────────────────────────────────────────────
        analyze_btn.click(
            fn=analyze,
            inputs=[image_input],
            outputs=[results_html],
        ).then(
            fn=show_results,
            outputs=[page1, page2],
        )
        back_btn.click(fn=go_back, outputs=[page1, page2, results_html])

    return app


if __name__ == "__main__":
    ckpt = os.path.join(
        PROJECT_ROOT, "outputs", "checkpoints",
        "efficientnet-b3_cardassian-spot-5", "best_model-2.pth",
    )
    if os.path.exists(ckpt):
        load_model(ckpt)
    else:
        print(f"Warning: No checkpoint found at '{ckpt}'. Running in demo mode.")
    create_app().queue().launch(share=False)