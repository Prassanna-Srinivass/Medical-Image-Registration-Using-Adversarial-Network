"""
ui.py  —  Chest X-Ray Registration Frontend
────────────────────────────────────────────────────────────────────
Gradio browser UI.  Upload a Fixed and Moving X-ray, click Register,
and instantly see:
  • Warped (registered) image
  • |Fixed − Warped| difference heatmap
  • Side-by-side comparison strip
  • NCC alignment score

Requirements:
  pip install gradio requests pillow

Run (make sure inference_server.py is already running):
  python ui.py
  # Opens http://localhost:7860 in your browser automatically
────────────────────────────────────────────────────────────────────
"""

import base64
import io
import requests
from PIL import Image
import gradio as gr

# ── Server config ────────────────────────────────────────────────────
SERVER_URL = "http://localhost:5000"

# ─────────────────────────────────────────────────────────────────────
# Core inference call
# ─────────────────────────────────────────────────────────────────────

def run_registration(fixed_img: Image.Image, moving_img: Image.Image):
    """
    Sends fixed + moving images to the inference server,
    returns (warped, diff_heatmap, overlay_strip, ncc_label).
    """
    if fixed_img is None or moving_img is None:
        raise gr.Error("Please upload both a Fixed and a Moving image.")

    # Convert PIL → bytes for multipart upload
    def pil_to_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.convert("L").save(buf, format="PNG")
        buf.seek(0)
        return buf.read()

    try:
        resp = requests.post(
            f"{SERVER_URL}/register",
            files={
                "fixed":  ("fixed.png",  pil_to_bytes(fixed_img),  "image/png"),
                "moving": ("moving.png", pil_to_bytes(moving_img), "image/png"),
            },
            timeout=30,
        )
    except requests.exceptions.ConnectionError:
        raise gr.Error(
            "Cannot reach the inference server at localhost:5000.\n"
            "Make sure you have run:  python inference_server.py"
        )

    if resp.status_code != 200:
        raise gr.Error(f"Server error {resp.status_code}: {resp.text}")

    data = resp.json()

    # Decode base64 images back to PIL
    def b64_to_pil(b64: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(b64)))

    warped_img  = b64_to_pil(data["warped_b64"])
    diff_img    = b64_to_pil(data["diff_b64"])
    overlay_img = b64_to_pil(data["overlay_b64"])
    score       = data["ncc_score"]

    # NCC is in [−1, 1]; express as a readable label
    quality = "Excellent" if score > 0.85 else \
              "Good"      if score > 0.65 else \
              "Fair"      if score > 0.40 else "Poor"

    ncc_label = f"NCC Score: {score:.4f}  ({quality} alignment)"

    return warped_img, diff_img, overlay_img, ncc_label


def check_server():
    """Ping /health and return a status string for the UI banner."""
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=3)
        d = r.json()
        return f"✅  Server online — running on **{d['device'].upper()}**"
    except Exception:
        return "❌  Server offline — run `python inference_server.py` first"


# ─────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Chest X-Ray GAN Registration",
    theme=gr.themes.Soft(),
    css="""
        #title  { text-align: center; }
        #status { text-align: center; font-size: 0.95em; padding: 6px 0; }
        .score-box textarea { font-size: 1.2em !important; font-weight: bold; text-align: center; }
    """,
) as demo:

    # ── Header ───────────────────────────────────────────────────────
    gr.Markdown("# 🫁 Chest X-Ray Image Registration", elem_id="title")
    gr.Markdown(
        "Upload a **Fixed** (target) and **Moving** (source) X-ray. "
        "The GAN generator will warp the Moving image to align with the Fixed image.",
        elem_id="title",
    )

    server_status = gr.Markdown(check_server(), elem_id="status")
    refresh_btn   = gr.Button("↻ Refresh server status", size="sm", variant="secondary")
    refresh_btn.click(fn=check_server, outputs=server_status)

    gr.Markdown("---")

    # ── Input row ────────────────────────────────────────────────────
    with gr.Row():
        fixed_input  = gr.Image(
            label="Fixed Image (Target)",
            type="pil",
            image_mode="L",
            height=300,
        )
        moving_input = gr.Image(
            label="Moving Image (To be registered)",
            type="pil",
            image_mode="L",
            height=300,
        )

    register_btn = gr.Button("🔬 Register Images", variant="primary", size="lg")

    gr.Markdown("---")

    # ── Output row ───────────────────────────────────────────────────
    with gr.Row():
        warped_out  = gr.Image(label="Warped Result",              height=300)
        diff_out    = gr.Image(label="|Fixed − Warped| Heatmap",   height=300)

    overlay_out = gr.Image(label="Side-by-side: Moving | Fixed | Warped", height=260)
    ncc_out     = gr.Textbox(label="Alignment Score", interactive=False,
                             elem_classes=["score-box"])

    # ── Wire up ──────────────────────────────────────────────────────
    register_btn.click(
        fn=run_registration,
        inputs=[fixed_input, moving_input],
        outputs=[warped_out, diff_out, overlay_out, ncc_out],
    )

    # ── Example images hint ──────────────────────────────────────────
    gr.Markdown(
        "> **Tip:** Images are automatically converted to grayscale and resized to 128×128. "
        "Any PNG, JPG, or JPEG chest X-ray will work."
    )

    # ── How it works accordion ───────────────────────────────────────
    with gr.Accordion("ℹ️  How it works", open=False):
        gr.Markdown("""
**Pipeline:**
1. Your images are sent to the local Flask server (`inference_server.py`)
2. The server loads `registration_model_final.pth` — your trained GAN generator
3. The U-Net generator predicts a dense 2-D deformation flow field
4. The Spatial Transformer warps the Moving image using that flow
5. Results are returned as images + an NCC alignment score

**NCC Score guide:**
| Score | Meaning |
|-------|---------|
| > 0.85 | Excellent alignment |
| 0.65 – 0.85 | Good alignment |
| 0.40 – 0.65 | Fair — model may need more training |
| < 0.40 | Poor — check image quality or retrain |

**Files involved:**
- `registration_model_final.pth` — trained generator weights
- `inference_server.py` — Flask backend (must be running)
- `ui.py` — this Gradio frontend
        """)


# ─────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,          # set True to get a public gradio.live link
        inbrowser=True,       # auto-opens your browser
    )
