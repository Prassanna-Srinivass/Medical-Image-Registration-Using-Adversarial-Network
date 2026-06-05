import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils import spectral_norm          # FIX: correct import path
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. Local Kaggle Dataset Wrapper
# =====================================================================

class PairedXRayDataset(Dataset):
    """
    Loads paired Intra-subject Chest X-rays from local folders.
    """
    def __init__(self, fixed_dir, moving_dir):
        from monai.transforms import Compose, ScaleIntensity, Resize

        self.fixed_dir = fixed_dir
        self.moving_dir = moving_dir

        valid_exts = ('.png', '.jpg', '.jpeg')
        self.image_files = [f for f in os.listdir(fixed_dir) if f.lower().endswith(valid_exts)]
        print(f"[Debug] Found {len(self.image_files)} paired images for training.")

        self.transforms = Compose([
            ScaleIntensity(),
            Resize(spatial_size=(128, 128))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        from PIL import Image

        filename = self.image_files[idx]
        fixed_path  = os.path.join(self.fixed_dir, filename)
        moving_path = os.path.join(self.moving_dir, filename)

        fixed_img  = Image.open(fixed_path).convert('L')
        moving_img = Image.open(moving_path).convert('L')

        fixed_tensor  = torch.tensor(np.array(fixed_img,  dtype=np.float32)).unsqueeze(0)
        moving_tensor = torch.tensor(np.array(moving_img, dtype=np.float32)).unsqueeze(0)

        return self.transforms(moving_tensor), self.transforms(fixed_tensor)

# =====================================================================
# 2. Advanced Loss Functions
# =====================================================================

class NCCLoss(nn.Module):
    """Localised Normalized Cross-Correlation"""
    def __init__(self, win=9, eps=1e-5):
        super().__init__()
        self.win = win
        self.eps = eps

    def forward(self, I, J):
        w = self.win
        weight = torch.ones((1, 1, w, w), device=I.device) / (w * w)

        I_mean  = F.conv2d(I,     weight, padding=w // 2)
        J_mean  = F.conv2d(J,     weight, padding=w // 2)
        I2_mean = F.conv2d(I * I, weight, padding=w // 2)
        J2_mean = F.conv2d(J * J, weight, padding=w // 2)
        IJ_mean = F.conv2d(I * J, weight, padding=w // 2)

        var_I  = I2_mean - I_mean * I_mean
        var_J  = J2_mean - J_mean * J_mean
        cov_IJ = IJ_mean - I_mean * J_mean

        ncc = (cov_IJ ** 2) / (var_I * var_J + self.eps)
        return -torch.mean(ncc)


def l2_gradient_loss(flow):
    """Penalises sharp spatial changes in the deformation field."""
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    return torch.mean(dx ** 2) + torch.mean(dy ** 2)


def r1_gradient_penalty(discriminator, real_img1, real_img2):
    """
    R1 Gradient Penalty — penalises the discriminator for having large
    gradients on REAL samples.  This is the most stable GAN regulariser
    and prevents the discriminator from memorising real data (overfitting).
    Reference: Mescheder et al., 2018  (https://arxiv.org/abs/1801.04406)
    """
    real_img1 = real_img1.detach().requires_grad_(True)
    real_img2 = real_img2.detach().requires_grad_(True)

    real_pred = discriminator(real_img1, real_img2)
    grad_real = torch.autograd.grad(
        outputs=real_pred.sum(),
        inputs=[real_img1, real_img2],
        create_graph=True
    )
    penalty = sum(g.pow(2).view(g.shape[0], -1).sum(1).mean() for g in grad_real)
    return penalty


def feature_matching_loss(disc_model, fake_pair, real_pair):
    """
    Feature Matching Loss — forces the generator to match intermediate
    discriminator features of real pairs, not just fool the final logit.
    Significantly stabilises GAN training.
    Reference: Salimans et al., 2016  (https://arxiv.org/abs/1606.03498)
    """
    fake_feats = disc_model.get_features(fake_pair[0], fake_pair[1])
    real_feats = disc_model.get_features(real_pair[0], real_pair[1])

    loss = 0.0
    for ff, rf in zip(fake_feats, real_feats):
        loss += F.l1_loss(ff, rf.detach())
    return loss


# =====================================================================
# 3. Early Stopping
# =====================================================================

class EarlyStopping:
    """
    Monitors Val NCC Loss and saves the best generator + discriminator
    checkpoint whenever it improves.  Triggers a stop signal after
    `patience` epochs with no improvement.

    Why Val NCC (not G/D loss)?
      G and D losses are adversarial — they fluctuate by design and give
      no reliable signal about actual registration quality.  Val NCC
      directly measures how well the warped image matches the fixed image
      on unseen data, making it the only honest stopping criterion.

    delta  : minimum improvement to count as "better" (avoids saving on
             noise — e.g. 1e-4 means NCC must drop by at least 0.0001)
    patience: how many epochs to wait after last improvement before stopping
    """
    def __init__(self, patience=10, delta=1e-4, checkpoint_path="best_registration_model.pth"):
        self.patience         = patience
        self.delta            = delta
        self.checkpoint_path  = checkpoint_path
        self.best_score       = None          # best Val NCC seen so far
        self.best_epoch       = 0
        self.counter          = 0             # epochs since last improvement
        self.should_stop      = False

    def step(self, val_ncc: float, generator, discriminator, epoch: int):
        """
        Call once per epoch after validation.
        Returns True if a new best was found (checkpoint saved).
        """
        # NCC loss is negative (higher = worse registration), so lower is better
        score = val_ncc

        if self.best_score is None or score < self.best_score - self.delta:
            # Improvement found — save full checkpoint
            self.best_score  = score
            self.best_epoch  = epoch + 1
            self.counter     = 0
            torch.save({
                "epoch":               epoch + 1,
                "val_ncc":             score,
                "generator_state":     generator.state_dict(),
                "discriminator_state": discriminator.state_dict(),
            }, self.checkpoint_path)
            return True   # new best

        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False  # no improvement

    def status(self) -> str:
        arrow = "▼" if self.counter == 0 else f"  (no improvement {self.counter}/{self.patience})"
        return f"Best Val NCC: {self.best_score:.4f} @ epoch {self.best_epoch}{arrow}"


# =====================================================================
# 4. Networks
# =====================================================================

class SpatialTransformer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, size[0]),
            torch.linspace(-1, 1, size[1]),
            indexing='ij'
        )
        grid = torch.stack([gx, gy], dim=0)
        self.register_buffer('grid', grid.unsqueeze(0))

    def forward(self, src, flow):
        H, W = self.size
        fn = torch.zeros_like(flow)
        fn[:, 0] = 2.0 * flow[:, 0] / max(W - 1, 1)
        fn[:, 1] = 2.0 * flow[:, 1] / max(H - 1, 1)

        new_grid = (self.grid + fn).permute(0, 2, 3, 1)
        return F.grid_sample(src, new_grid, mode='bilinear',
                             padding_mode='border', align_corners=True)


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
            layers.append(nn.Dropout2d(dropout_p))   # optional spatial dropout on generator bottleneck
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class RegistrationGenerator(nn.Module):
    """
    Deep U-Net with:
      • 4-level encoder to capture large deformations
      • InstanceNorm throughout (better than BN for GANs)
      • Light dropout at bottleneck for regularisation
      • Zero-initialised flow head (identity init)
    """
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()

        self.inc   = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256, dropout_p=0.2))  # bottleneck dropout

        self.up3   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)

        self.up2   = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up1   = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.flow_conv = nn.Conv2d(32, out_channels, 3, padding=1)
        # Identity initialisation — prevents chaotic flows at epoch 0
        nn.init.normal_(self.flow_conv.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.flow_conv.bias)

    def forward(self, m, f):
        x = torch.cat([m, f], dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        u3 = self.up3(x4)
        d3 = self.conv3(torch.cat([u3, x3], dim=1))

        u2 = self.up2(d3)
        d2 = self.conv2(torch.cat([u2, x2], dim=1))

        u1 = self.up1(d2)
        d1 = self.conv1(torch.cat([u1, x1], dim=1))

        return self.flow_conv(d1)


class AlignmentDiscriminator(nn.Module):
    """
    PatchGAN discriminator with:
      • Spectral Normalisation on every conv (prevents gradient explosion)
      • NO Dropout  (dropout hurts discriminator gradient signal)
      • NO BatchNorm (ruins discriminator training)
      • Feature extraction method for feature-matching loss
    """
    def __init__(self, in_channels=2):
        super().__init__()

        self.layer1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 32, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer2 = nn.Sequential(
            spectral_norm(nn.Conv2d(32, 64, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer3 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_conv = nn.Conv2d(128, 1, 4, stride=1, padding=1)

    def forward(self, img1, img2):
        x = torch.cat([img1, img2], dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.out_conv(x)

    def get_features(self, img1, img2):
        """Returns intermediate feature maps for Feature Matching Loss."""
        feats = []
        x = torch.cat([img1, img2], dim=1)
        x = self.layer1(x);  feats.append(x)
        x = self.layer2(x);  feats.append(x)
        x = self.layer3(x);  feats.append(x)
        return feats


# =====================================================================
# 4. Training & Validation
# =====================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    img_size = (128, 128)

    dataset = PairedXRayDataset(
        fixed_dir=r"C:\image registration using gan\dataset\fixed",
        moving_dir=r"C:\image registration using gan\dataset\moving"
    )

    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    generator     = RegistrationGenerator().to(device)
    discriminator = AlignmentDiscriminator().to(device)
    transformer   = SpatialTransformer(size=img_size).to(device)

    # Separate LRs: discriminator slightly lower so generator isn't left behind
    opt_g = torch.optim.Adam(generator.parameters(),     lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # Cosine Annealing — smoothly decays LR to prevent late-stage oscillations
    epochs = 50
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs, eta_min=1e-6)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs, eta_min=1e-6)

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_ncc = NCCLoss().to(device)

    # Loss weights
    lambda_sim    = 15.0   # NCC similarity reward
    lambda_smooth =  0.5   # Deformation field smoothness
    lambda_gan    =  0.5   # Adversarial signal  (raised from 0.1 — was too weak)
    lambda_fm     =  5.0   # Feature matching    (new)
    lambda_r1     =  5.0   # R1 gradient penalty (new — kills discriminator overfitting)
    r1_every      =  4     # Apply R1 every N discriminator steps (saves compute)

    # ── Early Stopping ───────────────────────────────────────────────
    early_stopping = EarlyStopping(
        patience=10,                              # stop after 10 epochs with no improvement
        delta=1e-4,                               # minimum meaningful improvement in Val NCC
        checkpoint_path="best_registration_model.pth"
    )

    print(f"\nStarting training for {epochs} epochs  (early stopping patience=10)...\n")

    for epoch in range(epochs):
        t0 = time.time()
        generator.train()
        discriminator.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for i, (moving, fixed) in enumerate(train_loader):
            moving, fixed = moving.to(device), fixed.to(device)

            # ── Discriminator Step ────────────────────────────────────────
            opt_d.zero_grad()

            with torch.no_grad():
                flow   = generator(moving, fixed)
                warped = transformer(moving, flow)

            # Real pair logits — label smoothing: real → 0.9
            real_preds = discriminator(fixed, fixed)
            loss_d_real = criterion_gan(real_preds,
                                        torch.full_like(real_preds, 0.9))

            # Fake pair logits — label smoothing: fake → 0.1
            fake_preds = discriminator(warped.detach(), fixed)
            loss_d_fake = criterion_gan(fake_preds,
                                        torch.full_like(fake_preds, 0.1))

            loss_d = (loss_d_real + loss_d_fake) * 0.5

            # R1 Gradient Penalty — prevents discriminator memorising real data
            if i % r1_every == 0:
                r1_pen   = r1_gradient_penalty(discriminator, fixed, fixed)
                loss_d   = loss_d + (lambda_r1 * r1_pen * r1_every)  # scale to match every-step

            # ── FIX: actually update the discriminator ────────────────────
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            opt_d.step()
            epoch_d_loss += loss_d.item()

            # ── Generator Step ────────────────────────────────────────────
            opt_g.zero_grad()

            flow   = generator(moving, fixed)
            warped = transformer(moving, flow)

            loss_ncc    = criterion_ncc(warped, fixed)
            loss_smooth = l2_gradient_loss(flow)

            gen_preds = discriminator(warped, fixed)
            loss_gan  = criterion_gan(gen_preds, torch.ones_like(gen_preds))

            # Feature Matching: match discriminator internals of real pair
            loss_fm = feature_matching_loss(discriminator,
                                            fake_pair=(warped, fixed),
                                            real_pair=(fixed,  fixed))

            loss_g = (lambda_sim    * loss_ncc
                    + lambda_smooth * loss_smooth
                    + lambda_gan    * loss_gan
                    + lambda_fm     * loss_fm)

            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)  # stability
            opt_g.step()
            epoch_g_loss += loss_g.item()

        # ── Schedulers ───────────────────────────────────────────────────
        scheduler_g.step()
        scheduler_d.step()

        # ── Validation ───────────────────────────────────────────────────
        generator.eval()
        val_ncc_loss = 0.0
        with torch.no_grad():
            for v_moving, v_fixed in val_loader:
                v_moving, v_fixed = v_moving.to(device), v_fixed.to(device)
                v_flow   = generator(v_moving, v_fixed)
                v_warped = transformer(v_moving, v_flow)
                val_ncc_loss += criterion_ncc(v_warped, v_fixed).item()

        # ── Logging ──────────────────────────────────────────────────────
        avg_g   = epoch_g_loss / len(train_loader)
        avg_d   = epoch_d_loss / len(train_loader)
        avg_val = val_ncc_loss / len(val_loader)
        lr_g    = scheduler_g.get_last_lr()[0]

        # ── Early Stopping Step ──────────────────────────────────────────
        is_best = early_stopping.step(avg_val, generator, discriminator, epoch)
        tag     = " ✦ NEW BEST — checkpoint saved" if is_best else ""

        print(f"Epoch [{epoch+1:03d}/{epochs}] | "
              f"Time: {time.time()-t0:.1f}s | "
              f"D Loss: {avg_d:.4f} | "
              f"G Loss: {avg_g:.4f} | "
              f"Val NCC: {avg_val:.4f} | "
              f"LR: {lr_g:.2e}{tag}")
        print(f"  └─ {early_stopping.status()}")

        if early_stopping.should_stop:
            print(f"\n⏹  Early stopping triggered after {epoch+1} epochs.")
            print(f"   Best model was at epoch {early_stopping.best_epoch} "
                  f"(Val NCC = {early_stopping.best_score:.4f})")
            break

    # =====================================================================
    # 5. Restore Best Weights & Visualise
    # =====================================================================
    print("\nTraining complete.")
    print(f"Loading best checkpoint from epoch {early_stopping.best_epoch} "
          f"(Val NCC = {early_stopping.best_score:.4f})...")

    best_ckpt = torch.load("best_registration_model.pth", map_location=device)
    generator.load_state_dict(best_ckpt["generator_state"])
    discriminator.load_state_dict(best_ckpt["discriminator_state"])

    # Also export the generator alone for inference convenience
    torch.save(generator.state_dict(), "registration_model_final.pth")
    print("Generator weights saved → registration_model_final.pth")

    print("Visualising 2 validation pairs...")
    generator.eval()
    with torch.no_grad():
        v_moving, v_fixed = next(iter(val_loader))
        v_moving, v_fixed = v_moving.to(device), v_fixed.to(device)
        v_flow   = generator(v_moving, v_fixed)
        v_warped = transformer(v_moving, v_flow)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        for row in range(2):
            m_img = v_moving[row, 0].cpu().numpy()
            f_img = v_fixed[row, 0].cpu().numpy()
            w_img = v_warped[row, 0].cpu().numpy()
            diff  = np.abs(f_img - w_img)

            axes[row, 0].imshow(m_img, cmap='gray');   axes[row, 0].set_title(f"Pair {row+1}: Moving")
            axes[row, 1].imshow(f_img, cmap='gray');   axes[row, 1].set_title(f"Pair {row+1}: Fixed")
            axes[row, 2].imshow(w_img, cmap='gray');   axes[row, 2].set_title(f"Pair {row+1}: Warped")
            axes[row, 3].imshow(diff, cmap='hot');     axes[row, 3].set_title(f"Pair {row+1}: |Fixed − Warped|")

            for ax in axes[row]:
                ax.axis('off')

        plt.tight_layout()
        plt.savefig("registration_results_v2.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved visualisation → registration_results_v2.png")


if __name__ == "__main__":
    main()
