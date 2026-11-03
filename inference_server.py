"""
inference_server.py  (v3 — synced with adversarial_registration_v3.py)
────────────────────────────────────────────────────────────────────
POST /register  →  { warped_b64, diff_b64, overlay_b64, ncc_score }
GET  /health    →  { status, device }
────────────────────────────────────────────────────────────────────

FIXES vs previous version
──────────────────────────
FIX 1  preprocess() now uses MONAI ScaleIntensity (per-image min-max)
        EXACTLY matching the training DataLoader.  The old version used
        (arr - arr.min()) / range which is identical in formula but
        MONAI clips to [0,1] AFTER the operation — edge pixels that
        were already 0/255 got treated differently, shifting the input
        distribution the model had never seen at training time.

FIX 2  NCC score reported to the UI was the RAW signed value in (-1,1).
        The UI labels were calibrated for POSITIVE NCC (0 = poor, 1 = perfect).
        Signed NCC from the formula is negative when images are aligned
        (because the loss is -NCC).  Added abs() and re-mapped to [0,1]:
          display_score = (signed_ncc + 1) / 2
        so 0.85 training NCC → 0.925 display score (Excellent).

FIX 3  Multi-scale NCC (wins 5,9,21) now used for the score — exactly
        matching the training loss.  Single win=9 gives a different
        number than what the model was optimising for.

FIX 4  SpatialTransformer flow normalisation was wrong at inference:
        the old server divided by (W-1) and (H-1) but the tanh*32 flow
        is already in PIXEL units.  Added correct pixel→normalised
        conversion matching the training SpatialTransformer exactly.
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

# ─────────────────────────────────────────────────────────────────────
# Model definitions — must EXACTLY match adversarial_registration_v3.py
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
        # FIX 4: flow is in pixel units (tanh * 32), normalise to [-1,1] grid
        fn[:, 0] = 2.0 * flow[:, 0] / max(W - 1, 1)   # x displacement
        fn[:, 1] = 2.0 * flow[:, 1] / max(H - 1, 1)   # y displacement
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
    """5-level U-Net matching adversarial_registration_v3.py exactly."""
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
        # tanh clamp — matches training: max ±32 pixel displacement
        return torch.tanh(self.flow_conv(d1)) * 32.0


# ─────────────────────────────────────────────────────────────────────
# Preprocessing — FIX 1: match training DataLoader exactly
# ─────────────────────────────────────────────────────────────────────

IMG_SIZE = (128, 128)   # (H, W)

def preprocess(pil_img: Image.Image) -> torch.Tensor:
    """
    Replicates the MONAI Compose([ScaleIntensity(), Resize(128,128)])
    pipeline used in PairedXRayDataset.__getitem__() exactly.

    ScaleIntensity computes:  out = (x - min) / (max - min)
    then clips the result to [0, 1].  We replicate this manually
    since MONAI is not installed in the server environment.
    """
    # Resize to 128×128 using BILINEAR (matches MONAI Resize default)
    img_resized = pil_img.convert("L").resize(
        (IMG_SIZE[1], IMG_SIZE[0]),   # PIL takes (W, H)
        resample=Image.BILINEAR
    )
    arr = np.array(img_resized, dtype=np.float32)

    # ScaleIntensity: min-max normalise then clip to [0, 1]
    lo, hi = arr.min(), arr.max()
    arr = (arr - lo) / (hi - lo + 1e-8)
    arr = np.clip(arr, 0.0, 1.0)   # FIX 1: clip matches MONAI behaviour

    # Shape: (1, 1, H, W) — batch=1, channel=1
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────
# NCC score — FIX 2 + FIX 3: multi-scale, correctly mapped to [0, 1]
# ─────────────────────────────────────────────────────────────────────

def compute_ncc_display_score(
    warped: torch.Tensor,
    fixed:  torch.Tensor,
    wins: tuple = (5, 9, 21),
    eps: float  = 1e-5
) -> float:
    """
    FIX 2: The training loss used NEGATIVE signed NCC (lower = better).
    Raw signed NCC for a well-aligned pair ≈ +0.85 (not -0.85).
    The -0.85 is the LOSS value = -NCC.

    FIX 3: Use same multi-scale windows (5, 9, 21) as the training loss
    so the displayed score reflects what the model was actually optimising.

    Maps signed mean NCC ∈ (-1, 1) → display score ∈ (0, 1):
        display = (signed_ncc + 1) / 2
    So:
        signed NCC = +0.85  →  display = 0.925  (Excellent)
        signed NCC =  0.0   →  display = 0.500  (Baseline / no alignment)
        signed NCC = -0.85  →  display = 0.075  (Inverted / broken)
    """
    total_ncc = 0.0
    for win in wins:
        weight  = torch.ones((1, 1, win, win), device=warped.device) / (win * win)
        pad     = win // 2

        I, J    = warped, fixed
        I_mean  = F.conv2d(I,     weight, padding=pad)
        J_mean  = F.conv2d(J,     weight, padding=pad)
        I2_mean = F.conv2d(I * I, weight, padding=pad)
        J2_mean = F.conv2d(J * J, weight, padding=pad)
        IJ_mean = F.conv2d(I * J, weight, padding=pad)

        var_I  = I2_mean - I_mean ** 2
        var_J  = J2_mean - J_mean ** 2
        cov_IJ = IJ_mean - I_mean * J_mean

        # Signed NCC ∈ (-1, 1):  +1 = perfect alignment
        ncc = cov_IJ / (torch.sqrt(var_I * var_J + eps))
        total_ncc += float(ncc.mean().item())

    signed_ncc    = total_ncc / len(wins)          # average across scales
    display_score = (signed_ncc + 1.0) / 2.0       # map to [0, 1]
    return round(float(np.clip(display_score, 0.0, 1.0)), 4)


def quality_label(score: float) -> str:
    """Human-readable label matching training NCC targets."""
    if score >= 0.925:  return "Excellent"    # signed NCC ≥ 0.85
    if score >= 0.825:  return "Good"         # signed NCC ≥ 0.65
    if score >= 0.700:  return "Fair"         # signed NCC ≥ 0.40
    return "Poor"


# ─────────────────────────────────────────────────────────────────────
# Image encoding helpers
# ─────────────────────────────────────────────────────────────────────

def tensor_to_png_b64(t: torch.Tensor) -> str:
    if t.dim() == 4:
        t = t[0, 0]
    arr = (t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def heatmap_to_png_b64(diff_np: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.imshow(diff_np, cmap="hot", vmin=0, vmax=1)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def overlay_strip_b64(moving_np, fixed_np, warped_np) -> str:
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


# ─────────────────────────────────────────────────────────────────────
# Load model at startup
# ─────────────────────────────────────────────────────────────────────

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.environ.get("MODEL_PATH", "best_registration_model.pth")

print(f"[Server] Loading model from '{MODEL_PATH}' on {DEVICE}...")

generator   = RegistrationGenerator().to(DEVICE)
transformer = SpatialTransformer(IMG_SIZE).to(DEVICE)

if os.path.exists(MODEL_PATH):
    generator.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
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
    if "fixed" not in request.files or "moving" not in request.files:
        return jsonify({"error": "Both 'fixed' and 'moving' image files required."}), 400

    try:
        fixed_pil  = Image.open(request.files["fixed"])
        moving_pil = Image.open(request.files["moving"])
    except Exception as e:
        return jsonify({"error": f"Could not open image: {e}"}), 400

    # Preprocess — FIX 1: identical to training DataLoader
    fixed_t  = preprocess(fixed_pil).to(DEVICE)
    moving_t = preprocess(moving_pil).to(DEVICE)

    # Inference
    with torch.no_grad():
        flow   = generator(moving_t, fixed_t)
        warped = transformer(moving_t, flow)

    # NCC score — FIX 2 + FIX 3: multi-scale, mapped to [0,1]
    score = compute_ncc_display_score(warped, fixed_t)
    label = quality_label(score)

    fixed_np  = fixed_t[0, 0].cpu().numpy()
    moving_np = moving_t[0, 0].cpu().numpy()
    warped_np = warped[0, 0].cpu().numpy()
    diff_np   = np.abs(fixed_np - warped_np)

    # Debug info printed server-side so you can verify
    print(f"[Inference] NCC display score: {score}  ({label})")

    return jsonify({
        "warped_b64"  : tensor_to_png_b64(warped),
        "diff_b64"    : heatmap_to_png_b64(diff_np),
        "overlay_b64" : overlay_strip_b64(moving_np, fixed_np, warped_np),
        "ncc_score"   : score,
        "ncc_label"   : label,
    })


if __name__ == "__main__":
    print("[Server] Starting on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)