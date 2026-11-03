"""
inference_server_v2.py  —  synced with adversarial_registration_v2.py
──────────────────────────────────────────────────────────────────────
POST /register  →  { warped_b64, diff_b64, overlay_b64, ncc_score, ncc_label }
GET  /health    →  { status, device }
──────────────────────────────────────────────────────────────────────

KEY DIFFERENCES vs inference_server.py (v3)
────────────────────────────────────────────
[A] Generator flow output has NO tanh*32 clamp.
      v2 training: return self.flow_conv(d1)          ← raw Conv2d output
      v3 training: return torch.tanh(self.flow_conv(d1)) * 32.0

    The SpatialTransformer's pixel→grid normalisation is therefore kept
    but will operate on unbounded flow values — exactly as v2 trained.

[B] Checkpoint format: v2 EarlyStopping saves a DICT:
        { "epoch", "val_ncc", "generator_state", "discriminator_state" }
    We extract ["generator_state"] instead of loading state_dict directly.
    The server also accepts a plain state_dict (generator-only export saved
    as registration_model_final.pth at the end of v2 training).

[C] NCC score metric matches v2 NCCLoss exactly:
      • Single window  win=21  (v2 criterion_ncc = NCCLoss(win=21))
      • Formula: (cov_IJ ** 2) / (var_I * var_J + eps)   ← SQUARED, always ≥ 0
      • v2 loss = -mean(ncc²) so HIGHER squared NCC = better alignment
      • Display score = mean squared NCC, already in [0, 1] — no sign flip needed.

[D] Preprocessing replicates v2 PairedXRayDataset exactly:
      MONAI Compose([ScaleIntensity(), Resize(128, 128)])
      ScaleIntensity: (x - min) / (max - min), clipped to [0, 1].
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
# Model definitions — EXACTLY matching adversarial_registration_v2.py
# ─────────────────────────────────────────────────────────────────────

class SpatialTransformer(nn.Module):
    """
    Identical to v2 SpatialTransformer.
    Converts pixel-space flow → normalised grid displacement,
    then warps src with bilinear interpolation.
    """
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
    """
    5-level U-Net — EXACTLY matching adversarial_registration_v2.py.

    CRITICAL: forward() returns raw flow_conv output — NO tanh*32 clamp.
    v3 added the clamp; v2 did not.  Loading v2 weights into v3's generator
    would silently produce wrong (clamped) flows at inference.
    """
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
        nn.init.normal_(self.flow_conv.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.flow_conv.bias)

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

        # v2: raw output — NO tanh*32 clamp (unlike v3)
        return self.flow_conv(d1)


# ─────────────────────────────────────────────────────────────────────
# Preprocessing — replicates v2 PairedXRayDataset exactly [see note D]
# ─────────────────────────────────────────────────────────────────────

IMG_SIZE = (128, 128)   # (H, W)

def preprocess(pil_img: Image.Image) -> torch.Tensor:
    """
    Replicates MONAI Compose([ScaleIntensity(), Resize(128, 128)]) from v2.
    ScaleIntensity: (x - min) / (max - min), clipped to [0, 1].
    Resize: bilinear, matching MONAI default.
    """
    img_resized = pil_img.convert("L").resize(
        (IMG_SIZE[1], IMG_SIZE[0]),   # PIL takes (W, H)
        resample=Image.BILINEAR,
    )
    arr = np.array(img_resized, dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    arr = (arr - lo) / (hi - lo + 1e-8)
    arr = np.clip(arr, 0.0, 1.0)
    # Shape: (1, 1, H, W)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────
# NCC score — matches v2 NCCLoss(win=21) with SQUARED formula [note C]
# ─────────────────────────────────────────────────────────────────────

def compute_ncc_display_score(
    warped: torch.Tensor,
    fixed:  torch.Tensor,
    win: int   = 21,          # matches v2: criterion_ncc = NCCLoss(win=21)
    eps: float = 1e-5,
) -> float:
    """
    v2 NCCLoss formula:
        ncc = (cov_IJ ** 2) / (var_I * var_J + eps)    ← SQUARED, ∈ [0, 1]
        loss = -mean(ncc)                               ← negative for minimisation

    Positive display score: mean squared NCC ∈ [0, 1].
        0.0 = no alignment, 1.0 = perfect.
    No sign flip or re-mapping needed (already non-negative by definition).
    """
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

    # Squared NCC — identical to v2 NCCLoss.forward()
    ncc_sq = (cov_IJ ** 2) / (var_I * var_J + eps)
    score  = float(ncc_sq.mean().item())
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def quality_label(score: float) -> str:
    """
    Thresholds calibrated for squared NCC ∈ [0, 1]:
        ≥ 0.80  →  Excellent   (very tight local correlation)
        ≥ 0.60  →  Good
        ≥ 0.40  →  Fair
        < 0.40  →  Poor
    """
    if score >= 0.80:  return "Excellent"
    if score >= 0.60:  return "Good"
    if score >= 0.40:  return "Fair"
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


def overlay_strip_b64(moving_np: np.ndarray, fixed_np: np.ndarray, warped_np: np.ndarray) -> str:
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
# Load model at startup  [see note B — handles both checkpoint formats]
# ─────────────────────────────────────────────────────────────────────

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.environ.get("MODEL_PATH", "best_registration_model.pth")

print(f"[Server] Loading model from '{MODEL_PATH}' on {DEVICE}...")

generator   = RegistrationGenerator().to(DEVICE)
transformer = SpatialTransformer(IMG_SIZE).to(DEVICE)

if os.path.exists(MODEL_PATH):
    raw = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

    # v2 EarlyStopping saves a dict with "generator_state" key.
    # The end-of-training export (registration_model_final.pth) is a
    # plain state_dict — handle both formats transparently.
    if isinstance(raw, dict) and "generator_state" in raw:
        state_dict = raw["generator_state"]
        saved_epoch = raw.get("epoch", "?")
        saved_ncc   = raw.get("val_ncc", float("nan"))
        print(f"[Server] ✓ Full checkpoint detected  "
              f"(epoch {saved_epoch}, val_ncc={saved_ncc:.4f}).")
    else:
        state_dict  = raw
        print("[Server] ✓ Generator-only state_dict detected.")

    generator.load_state_dict(state_dict)
    print("[Server] ✓ Weights loaded successfully.")
else:
    print(f"[Server] ⚠  '{MODEL_PATH}' not found — running with random weights.")
    print("[Server]    Train the model first (adversarial_registration_v2.py) "
          "and place best_registration_model.pth or registration_model_final.pth here.")

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

    # Preprocess — matches v2 DataLoader pipeline
    fixed_t  = preprocess(fixed_pil).to(DEVICE)
    moving_t = preprocess(moving_pil).to(DEVICE)

    # Inference — generator returns raw (unclamped) flow, matching v2
    with torch.no_grad():
        flow   = generator(moving_t, fixed_t)
        warped = transformer(moving_t, flow)

    # NCC score — squared formula with win=21, matching v2 NCCLoss
    score = compute_ncc_display_score(warped, fixed_t)
    label = quality_label(score)

    fixed_np  = fixed_t[0, 0].cpu().numpy()
    moving_np = moving_t[0, 0].cpu().numpy()
    warped_np = warped[0, 0].cpu().numpy()
    diff_np   = np.abs(fixed_np - warped_np)

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
