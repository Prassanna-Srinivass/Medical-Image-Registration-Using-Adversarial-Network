# 🫁 Chest X-Ray Image Registration using GAN

A deep learning pipeline that performs **deformable image registration** on chest X-rays using a GAN-based architecture. A U-Net generator predicts a dense deformation flow field to warp a Moving image onto a Fixed (target) image, supervised by a PatchGAN discriminator.

---

## 📁 Project Structure

```
image registration using gan/
│
├── dataset/
│   ├── fixed/                        # 3,505 original chest X-ray images
│   └── moving/                       # 3,505 synthetically warped images
│
├── adversarial_registration_v2.py    # Main training script
├── generate_moving_images_v2.py      # Moving image generator
├── inference_server.py               # Flask REST API (loads .pth, runs inference)
├── ui.py                             # Gradio browser frontend
├── requirements.txt                  # Python dependencies
│
├── best_registration_model.pth       # Best checkpoint (saved by early stopping)
└── registration_model_final.pth      # Final generator weights for inference
```

---

## 🗂️ Dataset

| Property | Details |
|---|---|
| **Fixed images** | 3,505 original chest X-rays |
| **Moving images** | 3,505 synthetically transformed versions |
| **Image format** | PNG / JPG / JPEG, grayscale |
| **Resized to** | 128 × 128 pixels during training |
| **Train / Val split** | 80% train (2,804 pairs) / 20% val (701 pairs) |

### How Moving Images Were Generated

Moving images were synthetically created from fixed images using `generate_moving_images_v2.py`, applying three stacked transforms per image to simulate real clinical variation:

**1. Compound Affine Transform**
Rotation, translation, and anisotropic scaling are applied simultaneously in a single matrix operation — avoiding double-interpolation artefacts that occur when transforms are applied sequentially. Border regions are filled using mirror reflection (`BORDER_REFLECT_101`) to eliminate the artificial black-edge artefact.

**2. Elastic Deformation**
Random smooth displacement fields (Gaussian-filtered noise) simulate local non-rigid tissue warping — breathing motion, lung expansion/compression, and soft-tissue shift that global affine transforms cannot capture.

**3. Intensity Augmentation**
Gamma correction, brightness shift, and Gaussian noise simulate real X-ray exposure variation (kVp differences, detector sensitivity, scatter) between patient visits.

**Difficulty presets available:**

| Preset | Rotation | Translation | Elastic Alpha | Use case |
|---|---|---|---|---|
| `easy` | ±8° | ±15 px | 20–60 | Debugging / curriculum start |
| `medium` | ±20° | ±30 px | 60–150 | **Default — recommended** |
| `hard` | ±35° | ±50 px | 150–300 | Stress-testing trained model |

---

## 🧠 Model Architecture

### Generator — `RegistrationGenerator`
A 4-level deep U-Net that predicts a dense 2D deformation flow field (Δx, Δy) per pixel.

```
Input: [Moving | Fixed] concatenated → (2, 128, 128)
  │
  ├── Encoder: 4× DoubleConv + MaxPool  [32 → 64 → 128 → 256 channels]
  │     InstanceNorm2d + LeakyReLU(0.2) + Dropout2d at bottleneck
  │
  ├── Decoder: 3× ConvTranspose2d + Skip connections
  │
  └── Flow head: Conv2d → (2, 128, 128)   [zero-initialised]
```

Key design choices:
- **InstanceNorm** instead of BatchNorm — far superior for GAN generators
- **Zero-initialised flow head** — model starts as identity (no warp), preventing early chaotic deformations
- **Skip connections** — preserve fine anatomical details across encoder–decoder

### Discriminator — `AlignmentDiscriminator`
A PatchGAN that classifies local 70×70 patches as real/fake rather than the whole image.

```
Input: [Image A | Image B] concatenated → (2, 128, 128)
  │
  ├── 3× SpectralNorm Conv2d + LeakyReLU(0.2)   [32 → 64 → 128 channels]
  └── Output Conv2d → patch logits
```

Key design choices:
- **Spectral Normalisation** on every layer — prevents gradient explosion and discriminator overpowering
- **No Dropout, No BatchNorm** — both harm discriminator gradient quality
- **Feature extraction method** — intermediate activations used for Feature Matching Loss

### Spatial Transformer — `SpatialTransformer`
Applies the predicted flow field to the moving image using bilinear grid sampling. Fully differentiable — gradients flow through the warp operation back to the generator.

---

## 📉 Loss Functions

| Loss | Weight | Purpose |
|---|---|---|
| **NCC (Normalized Cross-Correlation)** | λ = 15.0 | Primary similarity metric — localised 9×9 window, rotation/intensity invariant |
| **L2 Gradient Smoothness** | λ = 0.5 | Penalises sharp spatial changes in the flow field — prevents tearing |
| **Adversarial (BCE)** | λ = 0.5 | Generator fools discriminator into classifying warped as real |
| **Feature Matching** | λ = 5.0 | Generator matches discriminator's intermediate feature statistics of real pairs |
| **R1 Gradient Penalty** | λ = 5.0 | Penalises discriminator for large gradients on real data — kills overfitting |

Label smoothing applied: real → 0.9, fake → 0.1.

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| **Image size** | 128 × 128 |
| **Batch size** | 16 |
| **Optimizer** | Adam, β = (0.5, 0.999) |
| **Generator LR** | 2e-4 |
| **Discriminator LR** | 1e-4 |
| **LR Scheduler** | CosineAnnealingLR → min 1e-6 |
| **Max epochs** | 50 |
| **Early stopping patience** | 10 epochs |
| **Early stopping delta** | 1e-4 |
| **Gradient clipping** | max_norm = 1.0 (both networks) |
| **R1 penalty frequency** | Every 4 discriminator steps |

---

## 🏥 Early Stopping

Training monitors **Validation NCC Loss** (not G/D loss — those are adversarial and fluctuate by design). The best checkpoint is saved whenever Val NCC improves by more than `delta = 1e-4`.

```
Epoch [023/50] | D Loss: 0.6932 | G Loss: -8.08 | Val NCC: -0.5681 | LR: 5.26e-05 ✦ NEW BEST
  └─ Best Val NCC: -0.5681 @ epoch 23▼
```

**Interpreting D Loss:**
`D Loss ≈ 0.693` (= ln 2) is the **theoretically perfect** discriminator loss — it means the discriminator is genuinely at 50/50 uncertainty between real and fake. This is the ideal GAN balance.

**Interpreting Val NCC:**

| Val NCC | Registration Quality |
|---|---|
| −0.90 to −1.0 | Excellent |
| −0.75 to −0.90 | Good — clinically usable |
| −0.60 to −0.75 | Fair — visible misalignment |
| −0.50 to −0.60 | Below fair — still learning |
| Above −0.50 | Poor |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt

# If you have an NVIDIA GPU (recommended):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Generate moving images
```bash
python generate_moving_images_v2.py --difficulty medium --workers 4
```

### 3. Train the model
```bash
python adversarial_registration_v2.py
```
Produces `registration_model_final.pth` when training completes or early stopping triggers.

### 4. Start the inference server
```bash
# Terminal 1 — keep this open
python inference_server.py
```

### 5. Launch the UI
```bash
# Terminal 2
python ui.py
# Browser opens at http://localhost:7860
```

---

## 🖥️ Inference API

The Flask server exposes two endpoints:

**`GET /health`**
```json
{ "status": "ok", "device": "cuda" }
```

**`POST /register`**
Multipart form with `fixed` and `moving` image files. Returns:
```json
{
  "warped_b64"   : "<base64 PNG>",
  "diff_b64"     : "<base64 PNG — |fixed − warped| heatmap>",
  "overlay_b64"  : "<base64 PNG — side-by-side strip>",
  "ncc_score"    : 0.8741
}
```

---

## 📦 Dependencies

```
torch >= 2.0.0
torchvision >= 0.15.0
monai >= 1.2.0
flask >= 3.0.0
gradio >= 4.0.0
Pillow >= 10.0.0
numpy >= 1.24.0
matplotlib >= 3.7.0
opencv-python
scipy
requests >= 2.31.0
```

---

## 🔧 Troubleshooting

| Error | Fix |
|---|---|
| `D Loss: 0.0000` | Fixed in v2 — `loss_d.backward()` was missing in original code |
| `ModuleNotFoundError: monai` | `pip install monai` |
| `Cannot reach server at localhost:5000` | Start `inference_server.py` first in a separate terminal |
| `CUDA out of memory` | Reduce `batch_size` from 16 to 4 in training script |
| `No paired images found` | Ensure fixed/ and moving/ filenames match exactly |
| `.pth file not found` | Complete training (Step 3) before launching the server |
| Val NCC stuck above −0.60 | Regenerate moving images with `--difficulty easy` and retrain (curriculum learning) |

---

## 📌 Key Improvements over Original Code

| # | Bug / Limitation | Fix |
|---|---|---|
| 1 | `loss_d.backward()` never called — D Loss always 0 | Fixed — discriminator now trains |
| 2 | Wrong `spectral_norm` import path | Fixed to `from torch.nn.utils import spectral_norm` |
| 3 | Discriminator overfitting | R1 Gradient Penalty added |
| 4 | No intermediate generator supervision | Feature Matching Loss added |
| 5 | `Dropout2d` in discriminator hurting gradients | Removed |
| 6 | GAN signal too weak (`lambda_gan = 0.1`) | Raised to 0.5; FM loss at 5.0 |
| 7 | No gradient clipping | `clip_grad_norm_(..., 1.0)` on both networks |
| 8 | Fixed LR throughout | CosineAnnealingLR added |
| 9 | Adam default betas — bad for GANs | Set to (0.5, 0.999) |
| 10 | Single transform per moving image | Compound affine + elastic + intensity augmentation |
| 11 | Black border fill artefact | Replaced with `BORDER_REFLECT_101` mirroring |
| 12 | No early stopping | Monitors Val NCC, saves best checkpoint automatically |
