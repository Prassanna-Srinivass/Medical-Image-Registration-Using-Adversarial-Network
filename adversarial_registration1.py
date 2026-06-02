import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
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
        
        fixed_path = os.path.join(self.fixed_dir, filename)
        moving_path = os.path.join(self.moving_dir, filename)

        fixed_img = Image.open(fixed_path).convert('L')
        moving_img = Image.open(moving_path).convert('L')

        fixed_tensor = torch.tensor(np.array(fixed_img, dtype=np.float32)).unsqueeze(0)
        moving_tensor = torch.tensor(np.array(moving_img, dtype=np.float32)).unsqueeze(0)

        return self.transforms(moving_tensor), self.transforms(fixed_tensor)

# =====================================================================
# 2. Advanced Loss Functions
# =====================================================================

class NCCLoss(nn.Module):
    """ Localized Normalized Cross-Correlation """
    def __init__(self, win=9, eps=1e-5):
        super(NCCLoss, self).__init__()
        self.win = win
        self.eps = eps

    def forward(self, I, J):
        win_size = self.win
        weight = torch.ones((1, 1, win_size, win_size), device=I.device) / (win_size * win_size)
        
        I_mean = F.conv2d(I, weight, padding=win_size//2)
        J_mean = F.conv2d(J, weight, padding=win_size//2)

        I2_mean = F.conv2d(I * I, weight, padding=win_size//2)
        J2_mean = F.conv2d(J * J, weight, padding=win_size//2)
        IJ_mean = F.conv2d(I * J, weight, padding=win_size//2)

        var_I = I2_mean - I_mean * I_mean
        var_J = J2_mean - J_mean * J_mean
        cov_IJ = IJ_mean - I_mean * J_mean

        ncc = (cov_IJ * cov_IJ) / (var_I * var_J + self.eps)
        return -torch.mean(ncc)

def l2_gradient_loss(flow):
    """ Penalizes sharp spatial changes in the deformation field """
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
    return torch.mean(dx * dx) + torch.mean(dy * dy)

# =====================================================================
# 3. Networks
# =====================================================================

class SpatialTransformer(nn.Module):
    def __init__(self, size):
        super(SpatialTransformer, self).__init__()
        self.size = size
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, size[0]), 
            torch.linspace(-1, 1, size[1]), 
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0)
        self.register_buffer('grid', grid.unsqueeze(0))

    def forward(self, src, flow):
        H, W = self.size
        flow_normalized = torch.zeros_like(flow)
        flow_normalized[:, 0, :, :] = 2.0 * flow[:, 0, :, :] / max(W - 1, 1)
        flow_normalized[:, 1, :, :] = 2.0 * flow[:, 1, :, :] / max(H - 1, 1)
        
        new_grid = self.grid + flow_normalized
        new_grid = new_grid.permute(0, 2, 3, 1)
        return F.grid_sample(src, new_grid, mode='bilinear', padding_mode='border', align_corners=True)

class DoubleConv(nn.Module):
    """ Helper for the upgraded Generator """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # TRICK: InstanceNorm is vastly superior to BatchNorm for GAN Generators
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class RegistrationGenerator(nn.Module):
    """ Deep U-Net capable of processing large rotations and scaling """
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()
        
        # Encoder (4 levels deep to capture massive shifts/rotations)
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64, 32)
        
        # Final Flow Field Predictor
        self.flow_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        
        # Initialize with zero so it starts by doing nothing (avoids early chaos)
        self.flow_conv.weight.data.normal_(mean=0.0, std=1e-5)
        self.flow_conv.bias.data.zero_()

    def forward(self, m, f):
        # Concatenate Moving and Fixed
        x = torch.cat([m, f], dim=1)
        
        # Downsample
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Upsample with Skip Connections
        u3 = self.up3(x4)
        u3 = torch.cat([u3, x3], dim=1)
        d3 = self.conv3(u3)
        
        u2 = self.up2(d3)
        u2 = torch.cat([u2, x2], dim=1)
        d2 = self.conv2(u2)
        
        u1 = self.up1(d2)
        u1 = torch.cat([u1, x1], dim=1)
        d1 = self.conv1(u1)
        
        return self.flow_conv(d1)

class AlignmentDiscriminator(nn.Module):
    """ Evaluates registration alignment using Spectral Normalization to prevent overpowering """
    def __init__(self, in_channels=2):
        super().__init__()
        # Import spectral_norm - The ultimate GAN stabilizer
        import torch.nn.utils.spectral_norm as spectral_norm

        self.model = nn.Sequential(
            # Layer 1
            spectral_norm(nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3), # <-- Blinds the discriminator slightly so it can't cheat
            
            # Layer 2 (Removed BatchNorm, it ruins GAN discriminators)
            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            # Layer 3
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            # Final Output Layer
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, img1, img2):
        x = torch.cat([img1, img2], dim=1)
        return self.model(x)
# =====================================================================
# 4. Training & Validation Setup
# =====================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device designated: {device}")
    
    img_size = (128, 128)
    
    # Using local paths!
    dataset = PairedXRayDataset(
        fixed_dir=r"C:\image registration using gan\dataset\fixed",
        moving_dir=r"C:\image registration using gan\dataset\moving"
    )
    
    # Validation / Train Split (80% Train, 20% Val)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    generator = RegistrationGenerator().to(device)
    discriminator = AlignmentDiscriminator().to(device)
    transformer = SpatialTransformer(size=img_size).to(device)
    
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_ncc = NCCLoss().to(device) 
    
    # BALANCED WEIGHTS FOR INTRA-SUBJECT ALIGNMENT
    lambda_sim = 15.0      # Massive reward for matching the Target anatomy
    lambda_smooth = 0.5    # Small penalty to prevent extreme tearing
    lambda_gan = 0.1       # Keep GAN impact low to avoid instability
    
    epochs = 1
    print(f"\nStarting training loop for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()  # <--- Start timer
        
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0  # <--- Track D Loss
        
        for i, (moving, fixed) in enumerate(train_loader):
            moving, fixed = moving.to(device), fixed.to(device)
            
            # --- Train Discriminator ---
            # --- Train Discriminator ---
            opt_d.zero_grad()
            flow = generator(moving, fixed)
            warped = transformer(moving, flow)
            
            # LABEL SMOOTHING: Use 0.9 instead of 1.0 for Real
            real_preds = discriminator(fixed, fixed)
            loss_d_real = criterion_gan(real_preds, torch.full_like(real_preds, 0.9).to(device))
            
            # LABEL SMOOTHING: Use 0.1 instead of 0.0 for Fake
            fake_preds = discriminator(warped.detach(), fixed)
            loss_d_fake = criterion_gan(fake_preds, torch.full_like(fake_preds, 0.1).to(device))
            
            # --- Train Generator ---
            opt_g.zero_grad()
            
            loss_ncc = criterion_ncc(warped, fixed)
            loss_smooth = l2_gradient_loss(flow)
            
            gen_preds = discriminator(warped, fixed)
            loss_gan = criterion_gan(gen_preds, torch.ones_like(gen_preds).to(device))
            
            # Combined Loss
            loss_g = (lambda_sim * loss_ncc) + (lambda_smooth * loss_smooth) + (lambda_gan * loss_gan)
            loss_g.backward()
            opt_g.step()
            
            epoch_g_loss += loss_g.item()

        # --- Validation Loop ---
        generator.eval()
        val_ncc_loss = 0.0
        with torch.no_grad():
            for v_moving, v_fixed in val_loader:
                v_moving, v_fixed = v_moving.to(device), v_fixed.to(device)
                v_flow = generator(v_moving, v_fixed)
                v_warped = transformer(v_moving, v_flow)
                val_ncc_loss += criterion_ncc(v_warped, v_fixed).item()
                
        # --- Calculate Metrics & Time ---
        epoch_end_time = time.time()  # <--- Stop timer
        time_taken = epoch_end_time - epoch_start_time
        
        avg_train_g_loss = epoch_g_loss / len(train_loader)
        avg_train_d_loss = epoch_d_loss / len(train_loader)  # <--- Average D Loss
        avg_val_loss = val_ncc_loss / len(val_loader)
        
        # Updated Print Statement
        print(f"Epoch [{epoch+1}/{epochs}] | Time: {time_taken:.1f}s | D Loss: {avg_train_d_loss:.4f} | G Loss: {avg_train_g_loss:.4f} | Val Structural Loss: {avg_val_loss:.4f}")

    
    # =====================================================================
    # 5. Save the Model & Visualize Multiple Pairs
    # =====================================================================
    print("\nTraining completed. Saving model...")
    torch.save(generator.state_dict(), "registration_model.pth")
    
    print("Visualizing 2 pairs from the Validation Data...")
    generator.eval()
    with torch.no_grad():
        # Get one batch from validation
        v_moving, v_fixed = next(iter(val_loader))
        v_moving, v_fixed = v_moving.to(device), v_fixed.to(device)
        v_flow = generator(v_moving, v_fixed)
        v_warped = transformer(v_moving, v_flow)
        
        # Display 2 pairs (Row 0: Pair 1, Row 1: Pair 2)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        for row in range(2):
            m_img = v_moving[row, 0].cpu().numpy()
            f_img = v_fixed[row, 0].cpu().numpy()
            w_img = v_warped[row, 0].cpu().numpy()
            
            axes[row, 0].imshow(m_img, cmap='gray')
            axes[row, 0].set_title(f"Pair {row+1}: Moving")
            axes[row, 0].axis('off')
            
            axes[row, 1].imshow(f_img, cmap='gray')
            axes[row, 1].set_title(f"Pair {row+1}: Fixed (Target)")
            axes[row, 1].axis('off')
            
            axes[row, 2].imshow(w_img, cmap='gray')
            axes[row, 2].set_title(f"Pair {row+1}: Warped Result")
            axes[row, 2].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()