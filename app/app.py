"""
DermVision: AI-Powered Skin Lesion Analysis
Multi-page Gradio UI with patient questionnaire and image classification.
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
from src.gradcam import generate_gradcam

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
        "name": "Melanoma", "risk": "high", "color": "#ef4444",
        "desc": "A serious form of skin cancer that develops in melanocytes (pigment-producing cells).",
    },
    "NV":    {
        "name": "Melanocytic Nevus", "risk": "low", "color": "#22c55e",
        "desc": "A common benign mole formed by a cluster of melanocytes.",
    },
    "BCC":   {
        "name": "Basal Cell Carcinoma", "risk": "high", "color": "#ef4444",
        "desc": "The most common skin cancer. Slow-growing and rarely spreads if treated early.",
    },
    "AKIEC": {
        "name": "Actinic Keratosis / Bowen's", "risk": "moderate", "color": "#f59e0b",
        "desc": "A pre-cancerous rough skin patch caused by years of sun exposure.",
    },
    "BKL":   {
        "name": "Benign Keratosis", "risk": "low", "color": "#22c55e",
        "desc": "Non-cancerous skin growth including seborrheic keratosis and solar lentigo.",
    },
    "DF":    {
        "name": "Dermatofibroma", "risk": "low", "color": "#22c55e",
        "desc": "A common benign fibrous nodule typically found on the lower legs.",
    },
    "VASC":  {
        "name": "Vascular Lesion", "risk": "low", "color": "#3b82f6",
        "desc": "Abnormality of blood vessels including cherry angioma and pyogenic granuloma.",
    },
    "SCC":   {
        "name": "Squamous Cell Carcinoma", "risk": "high", "color": "#ef4444",
        "desc": "The second most common skin cancer. Can spread to other parts if left untreated.",
    },
}

RISK_STYLES = {
    "high":     {
        "bg": "#fef2f2", "color": "#ef4444", "border": "#dc2626",
        "label": "HIGH RISK", "icon": "⚠️",
        "advice": "Please consult a dermatologist as soon as possible.",
    },
    "moderate": {
        "bg": "#fffbeb", "color": "#f59e0b", "border": "#d97706",
        "label": "MODERATE RISK", "icon": "⚡",
        "advice": "Schedule a dermatologist appointment for evaluation.",
    },
    "low":      {
        "bg": "#f0fdf4", "color": "#22c55e", "border": "#16a34a",
        "label": "LOW RISK", "icon": "✅",
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


def run_predict(image: Image.Image) -> tuple[dict, Image.Image | None]:
    if MODEL is None:
        return {name: 1.0 / 8 for name in CLASS_NAMES}, None

    image_np = np.array(image.convert("RGB"))
    transform = get_val_transforms(IMAGE_SIZE)
    tensor = transform(image=image_np)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Generate Grad-CAM heatmap for predicted class
    pred_class = int(np.argmax(probs))
    try:
        target_layer = MODEL.backbone.conv_head
        heatmap_array = generate_gradcam(
            MODEL, image, target_layer, IMAGE_SIZE, target_class=pred_class, device=str(DEVICE)
        )
        heatmap_img = Image.fromarray((heatmap_array * 255).astype(np.uint8))
        print(f"✓ Grad-CAM generated for class {pred_class}")
    except Exception as e:
        print(f"✗ Grad-CAM generation failed: {e}")
        import traceback
        traceback.print_exc()
        heatmap_img = None

    return {name: float(p) for name, p in zip(CLASS_NAMES, probs)}, heatmap_img


# ── Results HTML ──────────────────────────────────────────────────────────────
def build_results_html(probs, age, sex, skin_type, location, duration, changed, symptoms, family_history):
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_cls, top_prob = sorted_probs[0]
    info = CLASS_INFO[top_cls]
    rs = RISK_STYLES[info["risk"]]

    # Probability bars
    bars = ""
    for cls, prob in sorted_probs:
        ci = CLASS_INFO[cls]
        pct = prob * 100
        is_top = cls == top_cls
        if is_top:
            row_style = f"background:linear-gradient(to right,{rs['bg']},transparent);padding:10px 12px;border-left:3px solid {rs['color']};border-radius:0 10px 10px 0;margin-bottom:8px;"
        else:
            row_style = "padding:4px 0;margin-bottom:8px;"
        bars += f"""
        <div style="display:flex;align-items:center;gap:12px;{row_style}">
            <div style="min-width:165px;">
                <div style="font-weight:{'700' if is_top else '500'};font-size:13px;color:#1e293b;">{ci['name']}</div>
                <div style="font-size:11px;color:#94a3b8;margin-top:1px;">{cls}</div>
            </div>
            <div style="flex:1;background:#f1f5f9;border-radius:999px;height:8px;overflow:hidden;">
                <div style="width:{pct:.1f}%;background:{ci['color']};height:100%;border-radius:999px;"></div>
            </div>
            <span style="min-width:46px;text-align:right;font-weight:700;font-size:13px;color:{ci['color']};">{pct:.1f}%</span>
        </div>"""

    # Symptom tags
    syms = symptoms or []
    tags = "".join(
        f'<span style="background:#e0f2fe;color:#0369a1;padding:4px 12px;border-radius:999px;'
        f'font-size:12px;font-weight:500;margin:2px;display:inline-block;">{s}</span>'
        for s in syms
    ) or '<span style="color:#94a3b8;font-size:13px;font-style:italic;">None reported</span>'

    # Patient profile cells
    skin_short = skin_type.split("\u2014")[0].strip() if "\u2014" in skin_type else skin_type
    profile_items = [
        ("Age", str(age)), ("Sex", sex), ("Skin Type", skin_short),
        ("Location", location), ("Duration", duration),
        ("Recent Change", changed), ("Family History", family_history),
    ]
    profile_cells = "".join(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px;">
            <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.8px;font-weight:600;margin-bottom:4px;">{lbl}</div>
            <div style="font-weight:600;font-size:13px;color:#1e293b;">{val}</div>
        </div>""" for lbl, val in profile_items)

    demo_banner = "" if MODEL else """
        <div style="background:#fef3c7;border:1px solid #fcd34d;border-radius:10px;padding:12px 16px;
             margin-bottom:14px;font-size:13px;color:#92400e;display:flex;gap:8px;align-items:flex-start;">
            <span style="flex-shrink:0;">🔧</span>
            <div><strong>Demo Mode:</strong> No trained checkpoint found. Probabilities are simulated (equal weights).</div>
        </div>"""

    return f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Inter','Segoe UI',sans-serif;">

        {demo_banner}

        <!-- Primary diagnosis card -->
        <div style="background:white;border:1px solid #e2e8f0;border-radius:16px;padding:24px;
             margin-bottom:14px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.07);">
            <div style="display:flex;align-items:flex-start;gap:16px;flex-wrap:wrap;margin-bottom:16px;">
                <div style="background:linear-gradient(135deg,#0ea5e9,#6366f1);width:60px;height:60px;
                     border-radius:14px;display:flex;align-items:center;justify-content:center;
                     font-size:26px;flex-shrink:0;">🔬</div>
                <div style="flex:1;min-width:160px;">
                    <div style="font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;
                         letter-spacing:0.8px;margin-bottom:5px;">Primary Diagnosis</div>
                    <div style="font-size:21px;font-weight:800;color:#0f172a;margin-bottom:5px;">{info['name']}</div>
                    <div style="font-size:13px;color:#64748b;">
                        Code <strong>{top_cls}</strong> &nbsp;·&nbsp; Confidence <strong>{top_prob * 100:.1f}%</strong>
                    </div>
                </div>
                <span style="background:{rs['bg']};color:{rs['border']};border:1.5px solid {rs['color']};
                     border-radius:999px;padding:6px 14px;font-size:11px;font-weight:700;
                     letter-spacing:0.5px;white-space:nowrap;">{rs['label']}</span>
            </div>
            <div style="font-size:13px;color:#64748b;line-height:1.6;margin-bottom:14px;">{info['desc']}</div>
            <div style="padding:12px 16px;background:{rs['bg']};border-left:4px solid {rs['color']};
                 border-radius:0 10px 10px 0;display:flex;gap:10px;align-items:flex-start;">
                <span style="font-size:18px;flex-shrink:0;">{rs['icon']}</span>
                <span style="font-size:14px;font-weight:600;color:{rs['border']};line-height:1.4;">{rs['advice']}</span>
            </div>
        </div>

        <!-- Classification breakdown -->
        <div style="background:white;border:1px solid #e2e8f0;border-radius:16px;padding:24px;
             margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:16px;
                 display:flex;align-items:center;gap:8px;">
                <span style="background:#dbeafe;width:30px;height:30px;border-radius:8px;
                     display:inline-flex;align-items:center;justify-content:center;font-size:15px;">📊</span>
                Classification Breakdown
            </div>
            {bars}
        </div>

        <!-- Patient profile -->
        <div style="background:white;border:1px solid #e2e8f0;border-radius:16px;padding:24px;
             margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:14px;
                 display:flex;align-items:center;gap:8px;">
                <span style="background:#dcfce7;width:30px;height:30px;border-radius:8px;
                     display:inline-flex;align-items:center;justify-content:center;font-size:15px;">👤</span>
                Patient Profile
            </div>
            <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));
                 gap:8px;margin-bottom:14px;">{profile_cells}</div>
            <div style="font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;
                 letter-spacing:0.8px;margin-bottom:8px;">Reported Symptoms</div>
            <div style="display:flex;flex-wrap:wrap;gap:4px;">{tags}</div>
        </div>

        <!-- Disclaimer -->
        <div style="background:#fffbeb;border:1px solid #fde68a;border-radius:12px;padding:14px 16px;">
            <div style="display:flex;gap:10px;align-items:flex-start;">
                <span style="font-size:18px;flex-shrink:0;">⚠️</span>
                <div style="font-size:13px;color:#78350f;line-height:1.6;">
                    <strong>Medical Disclaimer:</strong> This tool is for
                    <strong>educational and screening purposes only</strong>. It does not provide medical
                    advice, diagnosis, or treatment. Results may be inaccurate. Always consult a licensed
                    dermatologist or healthcare provider for any skin concerns.
                </div>
            </div>
        </div>

    </div>"""


# ── Event handlers ────────────────────────────────────────────────────────────
def analyze(image, age, sex, skin_type, location, duration, changed, symptoms, family_history):
    if image is None:
        gr.Warning("Please upload or capture a skin image before analyzing.")
        return gr.update(visible=True), gr.update(visible=False), gr.update(value=""), gr.update(value=None)
    probs, heatmap_img = run_predict(image)
    html = build_results_html(
        probs, age, sex, skin_type, location, duration, changed, symptoms, family_history
    )
    return gr.update(visible=False), gr.update(visible=True), gr.update(value=html), gr.update(value=heatmap_img)


def go_back():
    return gr.update(visible=True), gr.update(visible=False), gr.update(value=""), gr.update(value=None)


# ── UI constants ──────────────────────────────────────────────────────────────
HEADER_HTML = """
<div style="background:linear-gradient(135deg,#0ea5e9 0%,#6366f1 100%);padding:1.75rem 1.5rem;
     border-radius:16px;color:white;text-align:center;margin-bottom:4px;">
    <div style="font-size:2.5rem;margin-bottom:6px;">🔬</div>
    <div style="font-size:1.75rem;font-weight:800;letter-spacing:-0.5px;margin-bottom:4px;">DermVision</div>
    <div style="font-size:0.9rem;opacity:0.88;">
        AI-Powered Skin Lesion Analysis &nbsp;·&nbsp; For Educational Use Only
    </div>
</div>"""

CSS = """
.gradio-container { max-width:820px !important; margin:0 auto !important; padding:12px !important; }
.section-hdr {
    font-size:0.9375rem; font-weight:700; color:#1e293b;
    padding:10px 12px; background:#f8fafc;
    border-left:3px solid #0ea5e9; border-radius:0 8px 8px 0; margin:18px 0 10px;
}
.upload-tip {
    background:#f0f9ff; border:1px solid #bae6fd; border-radius:10px;
    padding:10px 14px; font-size:13px; color:#0369a1; line-height:1.5; margin-bottom:8px;
}
.model-ftr { text-align:center; font-size:11px; color:#94a3b8; margin-top:10px; padding:6px; }
footer { display:none !important; }
"""


def steps_html(active: int) -> str:
    defs = [(1, "Patient Info & Image"), (2, "AI Analysis Results")]
    parts = []
    for i, (n, label) in enumerate(defs):
        if n < active:
            bg, fg, col, w, icon = "#22c55e", "white", "#16a34a", "500", "✓"
        elif n == active:
            bg, fg, col, w, icon = "#0ea5e9", "white", "#0f172a", "700", str(n)
        else:
            bg, fg, col, w, icon = "#e2e8f0", "#94a3b8", "#94a3b8", "400", str(n)
        parts.append(
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<div style="width:30px;height:30px;border-radius:50%;background:{bg};color:{fg};'
            f'display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;">{icon}</div>'
            f'<span style="font-size:13px;font-weight:{w};color:{col};">{label}</span></div>'
        )
        if i < len(defs) - 1:
            parts.append('<div style="width:40px;height:2px;background:#e2e8f0;margin:0 4px;"></div>')
    inner = "".join(parts)
    return f'<div style="display:flex;align-items:center;justify-content:center;gap:4px;padding:14px 0 10px;">{inner}</div>'


# ── App factory ───────────────────────────────────────────────────────────────
def create_app() -> gr.Blocks:
    with gr.Blocks(
        title="DermVision — AI Skin Analysis",
        theme=gr.themes.Soft(
            primary_hue="sky",
            secondary_hue="indigo",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CSS,
    ) as app:

        gr.HTML(HEADER_HTML)

        # ── Page 1: Questionnaire + Image ──────────────────────────────────────
        with gr.Column(visible=True) as page1:
            gr.HTML(steps_html(1))

            gr.HTML('<div class="section-hdr">About You</div>')
            with gr.Row():
                age = gr.Slider(1, 100, value=35, step=1, label="Age")
                sex = gr.Radio(
                    ["Male", "Female", "Other / Prefer not to say"],
                    value="Male", label="Biological Sex",
                )
            with gr.Row():
                skin_type = gr.Dropdown(
                    choices=[
                        "Type I \u2014 Always burns, never tans",
                        "Type II \u2014 Usually burns, sometimes tans",
                        "Type III \u2014 Sometimes burns, always tans",
                        "Type IV \u2014 Rarely burns, always tans",
                        "Type V \u2014 Very rarely burns, deeply pigmented",
                        "Type VI \u2014 Never burns, deeply pigmented",
                        "Unknown / Not sure",
                    ],
                    value="Unknown / Not sure",
                    label="Fitzpatrick Skin Type",
                )
                location = gr.Dropdown(
                    choices=[
                        "Head / Neck / Face", "Chest / Upper Back", "Abdomen / Lower Back",
                        "Upper Arm / Shoulder", "Forearm / Elbow", "Hand / Wrist",
                        "Upper Leg / Hip", "Lower Leg / Knee", "Foot / Ankle",
                        "Palms or Soles", "Genital Area", "Other / Unknown",
                    ],
                    value="Other / Unknown",
                    label="Lesion Location on Body",
                )

            gr.HTML('<div class="section-hdr">About the Lesion</div>')
            with gr.Row():
                duration = gr.Dropdown(
                    choices=["< 1 month", "1\u20136 months", "6\u201312 months",
                             "1\u20133 years", "> 3 years", "Unknown"],
                    value="Unknown",
                    label="How long have you had it?",
                )
                changed = gr.Radio(
                    ["Yes", "No", "Unsure"], value="Unsure",
                    label="Changed recently? (size, shape, or color)",
                )
            symptoms = gr.CheckboxGroup(
                choices=[
                    "Itching", "Bleeding / Oozing", "Pain or tenderness",
                    "Color change", "Rapid size increase", "No symptoms",
                ],
                value=["No symptoms"],
                label="Symptoms (select all that apply)",
            )
            family_history = gr.Radio(
                ["Yes", "No", "Unknown"], value="Unknown",
                label="Family history of skin cancer?",
            )

            gr.HTML('<div class="section-hdr">Skin Image</div>')
            gr.HTML(
                '<div class="upload-tip">📷 <strong>Photo tips:</strong> Use a clear, well-lit, '
                'close-up photo of the affected area. Avoid shadows and blur. '
                'Dermoscopy images yield the most accurate results.</div>'
            )
            image_input = gr.Image(
                type="pil",
                label="Upload or Capture Skin Image",
                sources=["upload", "webcam", "clipboard"],
                height=280,
            )
            analyze_btn = gr.Button("🔬  Analyze with AI", variant="primary", size="lg")
            gr.HTML(
                '<div class="model-ftr">'
                'Powered by EfficientNet-B3 &nbsp;·&nbsp; Trained on ISIC 2019 &nbsp;·&nbsp; '
                '8-class skin lesion classification'
                '</div>'
            )

        # ── Page 2: Results ────────────────────────────────────────────────────
        with gr.Column(visible=False) as page2:
            gr.HTML(steps_html(2))
            results_html = gr.HTML(value="")
            gr.HTML('<div class="section-hdr">🔍 Model Attention Map (Grad-CAM)</div>')
            heatmap_img = gr.Image(label="Grad-CAM Visualization", type="pil")
            back_btn = gr.Button("\u2190 Start New Analysis", variant="secondary", size="lg")

        # ── Events ─────────────────────────────────────────────────────────────
        analyze_btn.click(
            fn=analyze,
            inputs=[image_input, age, sex, skin_type, location, duration,
                    changed, symptoms, family_history],
            outputs=[page1, page2, results_html, heatmap_img],
            show_progress="hidden",
        )
        back_btn.click(
            fn=go_back,
            outputs=[page1, page2, results_html, heatmap_img],
            show_progress="hidden",
        )

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
    create_app().launch(share=False)
