"""
ui.py — Self-contained Gradio UI for Chest X-Ray Registration
────────────────────────────────────────────────────────────────────
Loads the model DIRECTLY — no Flask server needed.
Run: python ui.py
Opens: http://localhost:7860
────────────────────────────────────────────────────────────────────
WHY self-contained?
  The previous architecture (Gradio → HTTP → Flask → model) had three
  failure points:
    1. Flask Content-Length mismatch when base64 payloads were large
    2. Gradio ASGI/uvicorn timeout on slow inference responses
    3. ncc_label field mismatch between Flask versions
  Loading the model directly inside ui.py removes all three.
"""

import io
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

# ─────────────────────────────────────────────────────────────────────
# Model — must exactly match your training script
# ─────────────────────────────────────────────────────────────────────

class SpatialTransformer(nn.Module):
    def __init__(self, size=(128, 128)):
        super().__init__()
        self.size = size
        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, size[0]),
            torch.linspace(-1, 1, size[1]),
            indexing="ij",
        )
        self.register_buffer("grid", torch.stack([gx, gy], dim=0).unsqueeze(0))

    def forward(self, src, flow):
        H, W = self.size
        fn = torch.zeros_like(flow)
        fn[:, 0] = 2.0 * flow[:, 0] / max(W - 1, 1)
        fn[:, 1] = 2.0 * flow[:, 1] / max(H - 1, 1)
        new_grid = (self.grid + fn).permute(0, 2, 3, 1)
        return F.grid_sample(src, new_grid, mode="bilinear",
                             padding_mode="border", align_corners=True)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(dropout_p))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class RegistrationGenerator(nn.Module):
    """5-level U-Net — matches adversarial_registration_fixed.py exactly."""
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()
        self.inc   = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32,  64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64,  128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512, dropout_p=0.2))
        self.up4   = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv4 = DoubleConv(512, 256)
        self.up3   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up2   = nn.ConvTranspose2d(128, 64,  2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1   = nn.ConvTranspose2d(64,  32,  2, stride=2)
        self.conv1 = DoubleConv(64,  32)
        self.flow_conv = nn.Conv2d(32, out_channels, 3, padding=1)

    def forward(self, m, f):
        x  = torch.cat([m, f], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        d4 = self.conv4(torch.cat([self.up4(x5), x4], dim=1))
        d3 = self.conv3(torch.cat([self.up3(d4), x3], dim=1))
        d2 = self.conv2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.conv1(torch.cat([self.up1(d2), x1], dim=1))
        return torch.tanh(self.flow_conv(d1)) * 32.0


# ─────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = (128, 128)

# Checks both checkpoint types automatically:
#   best_registration_model.pth  → full checkpoint dict (has generator_state key)
#   registration_model_final.pth → generator weights only (flat state_dict)
def load_model():
    for path in ["best_registration_model.pth", "registration_model_final.pth"]:
        if os.path.exists(path):
            print(f"[UI] Loading weights from: {path}")
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

            # Full checkpoint dict saved by EarlyStopping
            if isinstance(ckpt, dict) and "generator_state" in ckpt:
                state = ckpt["generator_state"]
                epoch = ckpt.get("epoch", "?")
                ncc   = ckpt.get("val_ncc", "?")
                print(f"[UI] Full checkpoint — epoch {epoch}, val NCC {ncc}")
            else:
                # Flat state_dict saved by torch.save(generator.state_dict())
                state = ckpt
                print("[UI] Flat state_dict loaded.")

            gen = RegistrationGenerator().to(DEVICE)
            gen.load_state_dict(state)
            gen.eval()
            print(f"[UI] ✓ Model ready on {DEVICE}")
            return gen

    print("[UI] ⚠  No .pth file found. Run training first.")
    return None

generator   = load_model()
transformer = SpatialTransformer(IMG_SIZE).to(DEVICE)

# ─────────────────────────────────────────────────────────────────────
# Preprocessing — mirrors training DataLoader exactly
# ─────────────────────────────────────────────────────────────────────

def preprocess(pil_img: Image.Image) -> torch.Tensor:
    """Grayscale → resize 128×128 → ScaleIntensity [0,1] → (1,1,H,W) tensor."""
    img = pil_img.convert("L").resize(
        (IMG_SIZE[1], IMG_SIZE[0]), resample=Image.BILINEAR
    )
    arr = np.array(img, dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)


# ─────────────────────────────────────────────────────────────────────
# NCC score — multi-scale, mapped to [0, 1] for display
# ─────────────────────────────────────────────────────────────────────

def compute_ncc_score(warped, fixed, wins=(5, 9, 21), eps=1e-5):
    total = 0.0
    for win in wins:
        weight  = torch.ones((1,1,win,win), device=warped.device) / (win*win)
        pad     = win // 2
        Im      = F.conv2d(warped, weight, padding=pad)
        Jm      = F.conv2d(fixed,  weight, padding=pad)
        I2m     = F.conv2d(warped*warped, weight, padding=pad)
        J2m     = F.conv2d(fixed*fixed,   weight, padding=pad)
        IJm     = F.conv2d(warped*fixed,  weight, padding=pad)
        vI      = I2m - Im**2
        vJ      = J2m - Jm**2
        cIJ     = IJm - Im*Jm
        ncc     = cIJ / (torch.sqrt(vI * vJ + eps))
        total  += float(ncc.mean().item())
    signed = total / len(wins)              # in (-1, 1); +1 = perfect
    display = (signed + 1.0) / 2.0         # mapped to (0, 1)
    return round(float(np.clip(display, 0.0, 1.0)), 4)


def quality_label(score):
    if score >= 0.866: return "Excellent"
    if score >= 0.666: return "Good"
    if score >= 0.500: return "Fair"
    return "Poor — model may need more training"


# ─────────────────────────────────────────────────────────────────────
# Core inference function
# ─────────────────────────────────────────────────────────────────────

def register(fixed_pil: Image.Image, moving_pil: Image.Image):
    if fixed_pil is None or moving_pil is None:
        raise gr.Error("Please upload both a Fixed and a Moving image.")

    if generator is None:
        raise gr.Error(
            "No trained model found.\n"
            "Place best_registration_model.pth or registration_model_final.pth "
            "in the same folder as ui.py and restart."
        )

    fixed_t  = preprocess(fixed_pil)
    moving_t = preprocess(moving_pil)

    with torch.no_grad():
        flow   = generator(moving_t, fixed_t)
        warped = transformer(moving_t, flow)

    score = compute_ncc_score(warped, fixed_t)
    label = quality_label(score)

    # Convert tensors → numpy for display
    fixed_np  = fixed_t[0, 0].cpu().numpy()
    moving_np = moving_t[0, 0].cpu().numpy()
    warped_np = warped[0, 0].cpu().numpy()
    diff_np   = np.abs(fixed_np - warped_np)

    def to_pil_gray(arr):
        return Image.fromarray(
            (arr * 255).clip(0, 255).astype(np.uint8), mode="L"
        ).convert("RGB")

    # Difference heatmap
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.imshow(diff_np, cmap="hot", vmin=0, vmax=1)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    diff_pil = Image.open(buf).copy()

    # Side-by-side strip
    fig2, axes = plt.subplots(1, 3, figsize=(10, 3.5), dpi=100)
    for ax, img, title in zip(
        axes,
        [moving_np, fixed_np, warped_np],
        ["Moving", "Fixed", "Warped"],
    ):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", bbox_inches="tight")
    plt.close(fig2)
    buf2.seek(0)
    strip_pil = Image.open(buf2).copy()

    ncc_text = f"NCC Score: {score:.4f}  →  {label}"
    print(f"[Inference] {ncc_text}")

    return (
        to_pil_gray(warped_np),
        diff_pil,
        strip_pil,
        ncc_text,
    )


# ─────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────

model_status = (
    f"✅  Model loaded on **{DEVICE.type.upper()}**"
    if generator is not None
    else "❌  No model found — place .pth file here and restart"
)

with gr.Blocks(
    title="Chest X-Ray GAN Registration",
    theme=gr.themes.Soft(),
    css="""
        #title  { text-align: center; }
        #status { text-align: center; padding: 6px 0; font-size: 0.95em; }
        .score  textarea { font-size: 1.2em !important; font-weight: bold;
                           text-align: center; }
    """,
) as demo:

    gr.Markdown("# 🫁 Chest X-Ray Image Registration", elem_id="title")
    gr.Markdown(
        "Upload a **Fixed** (target) and **Moving** (source) X-ray. "
        "The GAN generator warps the Moving image to align with the Fixed image.",
        elem_id="title",
    )
    gr.Markdown(model_status, elem_id="status")
    gr.Markdown("---")

    with gr.Row():
        fixed_input  = gr.Image(label="Fixed Image (Target)",
                                type="pil", image_mode="L", height=300)
        moving_input = gr.Image(label="Moving Image (To Register)",
                                type="pil", image_mode="L", height=300)

    register_btn = gr.Button("🔬 Register Images", variant="primary", size="lg")
    gr.Markdown("---")

    with gr.Row():
        warped_out = gr.Image(label="Warped Result",            height=300)
        diff_out   = gr.Image(label="|Fixed − Warped| Heatmap", height=300)

    strip_out = gr.Image(label="Moving  |  Fixed  |  Warped", height=280)
    score_out = gr.Textbox(label="Alignment Score", interactive=False,
                           elem_classes=["score"])

    register_btn.click(
        fn=register,
        inputs=[fixed_input, moving_input],
        outputs=[warped_out, diff_out, strip_out, score_out],
    )

    with gr.Accordion("ℹ️  NCC Score Guide", open=False):
        gr.Markdown("""
| Display Score | Signed NCC | Quality |
|---|---|---|
| ≥ 0.925 | ≥ +0.85 | **Excellent** |
| 0.825 – 0.925 | 0.65 – 0.85 | **Good** |
| 0.700 – 0.825 | 0.40 – 0.65 | **Fair** |
| < 0.700 | < 0.40 | **Poor** |

**Model files accepted (automatically detected):**
- `best_registration_model.pth` — full checkpoint saved by early stopping
- `registration_model_final.pth` — generator-only weights
        """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        share=False,
    )