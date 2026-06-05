"""
inference_server.py
────────────────────────────────────────────────────────────────────
Flask REST API that loads registration_model_final.pth and exposes
a single endpoint:

  POST /register
    multipart form-data fields:
      • fixed  : grayscale X-ray image (PNG / JPG / JPEG)
      • moving : grayscale X-ray image (PNG / JPG / JPEG)

    Returns JSON:
      {
        "warped_b64"    : "<base64-encoded PNG of warped image>",
        "diff_b64"      : "<base64-encoded PNG of |fixed − warped| heatmap>",
        "overlay_b64"   : "<base64-encoded PNG of side-by-side strip>",
        "ncc_score"     : <float>   (higher = better alignment, max 1.0)
      }

  GET /health
    Returns {"status": "ok", "device": "cuda|cpu"}

Run:
  python inference_server.py
  # Server starts on http://localhost:5000
────────────────────────────────────────────────────────────────────
"""

import io
import base64
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from PIL import Image
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

# ─────────────────────────────────────────────────────────────────────
# Model definitions  (must exactly match adversarial_registration_v2.py)
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
        grid = torch.stack([gx, gy], dim=0)
        self.register_buffer("grid", grid.unsqueeze(0))

    def forward(self, src, flow):
        H, W = self.size
        fn = torch.zeros_like(flow)
        fn[:, 0] = 2.0 * flow[:, 0] / max(W - 1, 1)
        fn[:, 1] = 2.0 * flow[:, 1] / max(H - 1, 1)
        new_grid = (self.grid + fn).permute(0, 2, 3, 1)
        return F.grid_sample(
            src, new_grid, mode="bilinear",
            padding_mode="border", align_corners=True
        )


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
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()
        self.inc   = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256, dropout_p=0.2))
        self.up3   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up2   = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1   = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)
        self.flow_conv = nn.Conv2d(32, out_channels, 3, padding=1)

    def forward(self, m, f):
        x  = torch.cat([m, f], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        d3 = self.conv3(torch.cat([self.up3(x4), x3], dim=1))
        d2 = self.conv2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.conv1(torch.cat([self.up1(d2), x1], dim=1))
        return self.flow_conv(d1)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

IMG_SIZE = (128, 128)

def preprocess(pil_img: Image.Image) -> torch.Tensor:
    """Convert a PIL image → normalised [0,1] float tensor (1,1,H,W)."""
    arr = np.array(pil_img.convert("L").resize(IMG_SIZE[::-1]),  # PIL: (W,H)
                   dtype=np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)       # (1,1,H,W)


def tensor_to_png_b64(t: torch.Tensor) -> str:
    """Convert a (1,1,H,W) or (H,W) tensor → base64-encoded PNG string."""
    if t.dim() == 4:
        t = t[0, 0]
    arr = (t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def heatmap_to_png_b64(diff_np: np.ndarray) -> str:
    """Convert a 2-D float array → hot-colourmap PNG → base64 string."""
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.imshow(diff_np, cmap="hot", vmin=0, vmax=1)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def overlay_strip_b64(moving_np, fixed_np, warped_np) -> str:
    """Return a 1×3 side-by-side strip as base64 PNG."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=100)
    for ax, img, title in zip(
        axes,
        [moving_np, fixed_np, warped_np],
        ["Moving", "Fixed", "Warped"],
    ):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ncc_score(I: torch.Tensor, J: torch.Tensor, win=9, eps=1e-5) -> float:
    """Scalar NCC in [−1, 1]; 1.0 = perfect alignment."""
    w = win
    weight = torch.ones((1, 1, w, w), device=I.device) / (w * w)
    pad    = w // 2
    I_mean  = F.conv2d(I,     weight, padding=pad)
    J_mean  = F.conv2d(J,     weight, padding=pad)
    I2_mean = F.conv2d(I * I, weight, padding=pad)
    J2_mean = F.conv2d(J * J, weight, padding=pad)
    IJ_mean = F.conv2d(I * J, weight, padding=pad)
    var_I   = I2_mean - I_mean ** 2
    var_J   = J2_mean - J_mean ** 2
    cov_IJ  = IJ_mean - I_mean * J_mean
    ncc     = cov_IJ / (torch.sqrt(var_I * var_J + eps))
    return float(ncc.mean().item())


# ─────────────────────────────────────────────────────────────────────
# Load model at startup
# ─────────────────────────────────────────────────────────────────────

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.environ.get("MODEL_PATH", "registration_model_final.pth")

print(f"[Server] Loading model from '{MODEL_PATH}' on {DEVICE}...")

generator    = RegistrationGenerator().to(DEVICE)
transformer  = SpatialTransformer(IMG_SIZE).to(DEVICE)

if os.path.exists(MODEL_PATH):
    generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("[Server] ✓ Weights loaded successfully.")
else:
    print(f"[Server] ⚠  '{MODEL_PATH}' not found — running with random weights.")
    print("[Server]    Train the model first and place the .pth file here.")

generator.eval()

# ─────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})


@app.route("/register", methods=["POST"])
def register():
    # ── Validate inputs ──────────────────────────────────────────────
    if "fixed" not in request.files or "moving" not in request.files:
        return jsonify({"error": "Both 'fixed' and 'moving' image files are required."}), 400

    try:
        fixed_pil  = Image.open(request.files["fixed"])
        moving_pil = Image.open(request.files["moving"])
    except Exception as e:
        return jsonify({"error": f"Could not open image: {e}"}), 400

    # ── Preprocess ───────────────────────────────────────────────────
    fixed_t  = preprocess(fixed_pil).to(DEVICE)
    moving_t = preprocess(moving_pil).to(DEVICE)

    # ── Inference ────────────────────────────────────────────────────
    with torch.no_grad():
        flow   = generator(moving_t, fixed_t)
        warped = transformer(moving_t, flow)

    # ── Compute outputs ──────────────────────────────────────────────
    score     = ncc_score(warped, fixed_t)

    fixed_np  = fixed_t[0, 0].cpu().numpy()
    moving_np = moving_t[0, 0].cpu().numpy()
    warped_np = warped[0, 0].cpu().numpy()
    diff_np   = np.abs(fixed_np - warped_np)

    return jsonify({
        "warped_b64"  : tensor_to_png_b64(warped),
        "diff_b64"    : heatmap_to_png_b64(diff_np),
        "overlay_b64" : overlay_strip_b64(moving_np, fixed_np, warped_np),
        "ncc_score"   : round(score, 4),
    })


if __name__ == "__main__":
    print("[Server] Starting on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
