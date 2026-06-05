"""
generate_moving_images_v2.py
────────────────────────────────────────────────────────────────────
Generates realistic synthetic Moving images from Fixed chest X-rays
using compound + elastic deformations that mimic real clinical variation:

  1. Compound affine  — rotation + translation + anisotropic scale applied
                        simultaneously (not one-or-the-other)
  2. Thin-Plate Spline elastic warp — local tissue-level deformation
                        simulating breathing / patient repositioning
  3. Realistic intensity augmentation — exposure shift, gamma, noise
                        so the model never cheats on brightness matching
  4. Gaussian border blending — eliminates the black-edge artefact
                        that caused your model to "cheat"

Difficulty levels:
  easy   → small transforms  (good for early training / debugging)
  medium → moderate          (recommended default)
  hard   → large transforms  (stress-tests the model)

Run:
  python generate_moving_images_v2.py
  python generate_moving_images_v2.py --difficulty hard
  python generate_moving_images_v2.py --difficulty easy --workers 8
────────────────────────────────────────────────────────────────────
"""

import os
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageFilter
import cv2                    # pip install opencv-python
from scipy.ndimage import map_coordinates, gaussian_filter

# ─────────────────────────────────────────────────────────────────────
# Difficulty presets
# ─────────────────────────────────────────────────────────────────────

PRESETS = {
    "easy": dict(
        rot_range       = (-8,   8),     # degrees
        tx_range        = (-15,  15),    # pixels
        ty_range        = (-15,  15),
        sx_range        = (0.93, 1.07),  # x-scale
        sy_range        = (0.93, 1.07),  # y-scale
        elastic_alpha   = (20,   60),    # elastic displacement magnitude
        elastic_sigma   = (6,    10),    # elastic smoothness
        gamma_range     = (0.85, 1.15),  # intensity gamma
        noise_std       = (0.0,  0.02),  # gaussian noise
        brightness_shift= (-0.05,0.05),  # additive brightness
    ),
    "medium": dict(
        rot_range       = (-20,  20),
        tx_range        = (-30,  30),
        ty_range        = (-30,  30),
        sx_range        = (0.85, 1.15),
        sy_range        = (0.85, 1.15),
        elastic_alpha   = (60,   150),
        elastic_sigma   = (8,    14),
        gamma_range     = (0.75, 1.30),
        noise_std       = (0.0,  0.04),
        brightness_shift= (-0.10,0.10),
    ),
    "hard": dict(
        rot_range       = (-35,  35),
        tx_range        = (-50,  50),
        ty_range        = (-50,  50),
        sx_range        = (0.75, 1.25),
        sy_range        = (0.75, 1.25),
        elastic_alpha   = (150,  300),
        elastic_sigma   = (10,   18),
        gamma_range     = (0.60, 1.50),
        noise_std       = (0.0,  0.06),
        brightness_shift= (-0.15,0.15),
    ),
}

# ─────────────────────────────────────────────────────────────────────
# Transform helpers
# ─────────────────────────────────────────────────────────────────────

def compound_affine(img_np: np.ndarray, p: dict) -> np.ndarray:
    """
    Applies rotation + translation + anisotropic scaling IN ONE MATRIX.

    Why compound?  Applying them separately changes the result because
    matrix multiplication is not commutative.  A single matrix gives
    the correct combined warp and avoids double-interpolation artefacts.

    Uses BORDER_REFLECT_101 instead of black fill — mirrors the image
    at the edges so the model never sees artificial black borders.
    """
    h, w = img_np.shape

    angle  = random.uniform(*p["rot_range"])
    tx     = random.uniform(*p["tx_range"])
    ty     = random.uniform(*p["ty_range"])
    sx     = random.uniform(*p["sx_range"])
    sy     = random.uniform(*p["sy_range"])

    # Build affine matrix around image centre
    cx, cy = w / 2.0, h / 2.0
    rad    = np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    # Combined: scale → rotate → translate, all about the centre
    M = np.array([
        [sx * cos_a, -sy * sin_a,  cx * (1 - sx * cos_a) + cy * sy * sin_a + tx],
        [sx * sin_a,  sy * cos_a,  cy * (1 - sy * cos_a) - cx * sx * sin_a + ty],
    ], dtype=np.float32)

    out = cv2.warpAffine(
        img_np, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,   # mirror border — no black edges
    )
    return out


def elastic_deform(img_np: np.ndarray, p: dict) -> np.ndarray:
    """
    Thin-Plate-Spline-style elastic deformation via random smooth displacement fields.

    Simulates:
      • Breathing motion (local lung expansion / compression)
      • Patient repositioning soft-tissue shift
      • Cardiac motion artefacts

    How it works:
      1. Generate 2 random displacement grids (dx, dy)
      2. Smooth them with a Gaussian — controls how localised the warp is
         (large sigma = broad global warp; small sigma = local wiggles)
      3. Scale by alpha — controls displacement magnitude
      4. Interpolate pixel values at the displaced coordinates
    """
    h, w  = img_np.shape
    alpha = random.uniform(*p["elastic_alpha"])
    sigma = random.uniform(*p["elastic_sigma"])

    # Random displacement fields
    dx = gaussian_filter(np.random.randn(h, w).astype(np.float32), sigma) * alpha
    dy = gaussian_filter(np.random.randn(h, w).astype(np.float32), sigma) * alpha

    # Map: new coordinate = original coordinate + displacement
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    indices_y = (y_coords + dy).clip(0, h - 1)
    indices_x = (x_coords + dx).clip(0, w - 1)

    out = map_coordinates(img_np, [indices_y.ravel(), indices_x.ravel()],
                          order=1, mode="reflect")
    return out.reshape(h, w).astype(np.float32)


def intensity_augment(img_np: np.ndarray, p: dict) -> np.ndarray:
    """
    Realistic intensity variation — because real X-rays of the same patient
    differ in kVp (exposure), detector sensitivity, and scatter.

    Applies (in order):
      1. Gamma correction  — simulates exposure/detector response curve
      2. Brightness shift  — global additive offset (under/over-exposed)
      3. Gaussian noise    — detector electronic noise

    All ops stay in [0, 1] float range.
    """
    # Ensure float [0,1]
    out = img_np.astype(np.float32)
    if out.max() > 1.0:
        out /= 255.0

    # 1. Gamma  (img^gamma — < 1 brightens, > 1 darkens)
    gamma = random.uniform(*p["gamma_range"])
    out   = np.power(np.clip(out, 1e-6, 1.0), gamma)

    # 2. Brightness shift
    shift = random.uniform(*p["brightness_shift"])
    out   = np.clip(out + shift, 0.0, 1.0)

    # 3. Gaussian noise
    std  = random.uniform(*p["noise_std"])
    out  = np.clip(out + np.random.randn(*out.shape).astype(np.float32) * std, 0.0, 1.0)

    return out


# ─────────────────────────────────────────────────────────────────────
# Per-image pipeline
# ─────────────────────────────────────────────────────────────────────

def generate_one(fixed_path: str, moving_path: str, preset: dict):
    """Full pipeline for a single image."""
    img = Image.open(fixed_path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0   # [0, 1]

    # Step 1 — Compound affine (rotation + shift + anisotropic scale)
    arr_u8   = (arr * 255).astype(np.uint8)
    affined  = compound_affine(arr_u8, preset).astype(np.float32) / 255.0

    # Step 2 — Elastic deformation (local tissue warping)
    elastied = elastic_deform(affined, preset)

    # Step 3 — Intensity augmentation (exposure / noise)
    augmented = intensity_augment(elastied, preset)

    # Save as uint8 PNG
    out = (augmented * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(out, mode="L").save(moving_path)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate moving X-ray dataset")
    parser.add_argument("--fixed_dir",   default=r"C:\image registration using gan\dataset\fixed")
    parser.add_argument("--moving_dir",  default=r"C:\image registration using gan\dataset\moving")
    parser.add_argument("--difficulty",  default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--workers",     type=int, default=4,
                        help="Parallel workers (use 1 to disable parallelism)")
    args = parser.parse_args()

    if not os.path.exists(args.fixed_dir):
        print(f"Error: Fixed directory not found: {args.fixed_dir}")
        return

    os.makedirs(args.moving_dir, exist_ok=True)

    valid_exts  = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(args.fixed_dir)
                   if f.lower().endswith(valid_exts)]

    if not image_files:
        print("No images found in the fixed directory.")
        return

    preset = PRESETS[args.difficulty]
    total  = len(image_files)
    print(f"Found {total} images  |  Difficulty: {args.difficulty}  |  Workers: {args.workers}")
    print("Generating moving images...\n")

    done = 0
    errors = 0

    def process(filename):
        fp = os.path.join(args.fixed_dir,  filename)
        mp = os.path.join(args.moving_dir, filename)
        generate_one(fp, mp, preset)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process, f): f for f in image_files}
        for future in as_completed(futures):
            try:
                future.result()
                done += 1
            except Exception as e:
                errors += 1
                print(f"  ✗ Error on {futures[future]}: {e}")

            if done % 200 == 0 or done == total:
                print(f"  Progress: {done}/{total}  (errors: {errors})")

    print(f"\n✓ Done!  {done} moving images saved to: {args.moving_dir}")
    if errors:
        print(f"  ⚠  {errors} files failed — check the error messages above.")

    # ── Sanity-check: show 2 pairs ────────────────────────────────────
    print("\nGenerating sanity-check preview (first 2 pairs)...")
    try:
        import matplotlib.pyplot as plt
        sample_files = image_files[:2]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        for row, filename in enumerate(sample_files):
            fp = os.path.join(args.fixed_dir,  filename)
            mp = os.path.join(args.moving_dir, filename)
            f_img = np.array(Image.open(fp).convert("L"))
            m_img = np.array(Image.open(mp).convert("L"))
            diff  = np.abs(f_img.astype(np.float32) - m_img.astype(np.float32))

            axes[row, 0].imshow(f_img, cmap="gray");  axes[row, 0].set_title(f"Fixed  [{filename}]")
            axes[row, 1].imshow(m_img, cmap="gray");  axes[row, 1].set_title("Moving (generated)")
            axes[row, 2].imshow(diff,  cmap="hot");   axes[row, 2].set_title("|Fixed − Moving|")
            for ax in axes[row]: ax.axis("off")

        plt.suptitle(f"Sanity Check — difficulty: {args.difficulty}", fontsize=13)
        plt.tight_layout()
        plt.savefig("moving_generation_preview.png", dpi=120, bbox_inches="tight")
        plt.show()
        print("Preview saved → moving_generation_preview.png")
    except Exception as e:
        print(f"Preview skipped: {e}")


if __name__ == "__main__":
    main()
