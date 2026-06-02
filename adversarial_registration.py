import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. Paired Medical Dataset Wrapper & Fallback Synthetic Data
# =====================================================================

class PairedMedNISTDataset(Dataset):
    """
    Programmatically downloads MedNIST via MONAI, filters to a single
    anatomical category, and returns randomly paired (Moving, Fixed) images.
    """
    def __init__(self, root_dir="./mednist_data", class_name="Hand", download=True):
        from monai.apps import MedNISTDataset
        from monai.transforms import Compose, ScaleIntensity, Resize
        import numpy as np

        # --- FIX: Create the folder on your disk before MONAI checks for it ---
        os.makedirs(root_dir, exist_ok=True)

        print(f"[Debug] Downloading MedNIST dataset...")

        # Download MedNIST WITHOUT any transforms (just get raw data)
        base_ds = MedNISTDataset(
            root_dir=root_dir,
            section="training",
            download=download,
            transform=None
        )

        print(f"[Debug] Total images available: {len(base_ds)}")

        # Get the data and find the class index
        # MedNISTDataset returns dict with 'image' (path) and 'label' (class index)
        all_data = base_ds.data
        print(f"[Debug] Data sample keys: {all_data[0].keys() if all_data else 'empty'}")

        # Filter for specific class (Hand = class 5 typically)
        class_mapping = {0: "AbdomenCT", 1: "BreastMRI", 2: "ChestCT", 3: "CXR",
                         4: "Hand", 5: "HeadCT", 6: "Knee", 7: "Leg", 8: "Neck", 9: "Pelvis"}

        # Find class index
        class_idx = None
        for idx, cname in class_mapping.items():
            if cname == class_name:
                class_idx = idx
                break

        if class_idx is None:
            print(f"[Debug] Available classes: {list(class_mapping.values())}")
            raise ValueError(f"Class '{class_name}' not found")

        # Filter images by class
        self.image_paths = [
            item["image"] for item in all_data if item["label"] == class_idx
        ]

        print(f"[Debug] Filtered {len(self.image_paths)} {class_name} images")

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found for class {class_name}")

        # Define transforms for loading
        # (EnsureChannelFirst is removed because we add the channel manually via .unsqueeze(0) below)
        self.transforms = Compose([
            ScaleIntensity(),
            Resize(spatial_size=(128, 128))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image

        # Get fixed image at current index
        fixed_path = self.image_paths[idx]

        # Get moving image at random index
        moving_idx = np.random.randint(0, len(self.image_paths))
        moving_path = self.image_paths[moving_idx]

        try:
            # Load images as PIL (grayscale)
            fixed_img = Image.open(fixed_path).convert('L')
            moving_img = Image.open(moving_path).convert('L')

            # Convert to numpy arrays
            fixed_array = np.array(fixed_img, dtype=np.float32)
            moving_array = np.array(moving_img, dtype=np.float32)

            # Convert to tensors with channel dimension (Shape becomes [1, H, W])
            fixed_tensor = torch.tensor(fixed_array).unsqueeze(0)
            moving_tensor = torch.tensor(moving_array).unsqueeze(0)

            # Apply transforms (ScaleIntensity and Resize)
            fixed_tensor = self.transforms(fixed_tensor)
            moving_tensor = self.transforms(moving_tensor)

            return moving_tensor, fixed_tensor

        except Exception as e:
            print(f"[Debug] Error loading image at {idx}: {e}")
            raise


class SyntheticCircleDataset(Dataset):
    """
    Backup Dataset: Generates random shifted circles on the fly if 
    the internet connection/MONAI API fails.
    """
    def __init__(self, num_samples=200, size=128):
        self.num_samples = num_samples
        self.size = size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Central target / fixed image
        fixed = np.zeros((self.size, self.size), dtype=np.float32)
        cy, cx = self.size // 2, self.size // 2
        r = self.size // 4
        y, x = np.ogrid[:self.size, :self.size]
        fixed[(x - cx)**2 + (y - cy)**2 <= r**2] = 1.0
        
        # Shifted moving image
        moving = np.zeros((self.size, self.size), dtype=np.float32)
        shift_x, shift_y = np.random.randint(-12, 13), np.random.randint(-12, 13)
        moving[(x - (cx + shift_x))**2 + (y - (cy + shift_y))**2 <= r**2] = 1.0
        
        # Tiny Gaussian noise to simulate medical sensor variations
        fixed += np.random.normal(0, 0.03, fixed.shape).astype(np.float32)
        moving += np.random.normal(0, 0.03, moving.shape).astype(np.float32)
        
        return torch.tensor(moving).unsqueeze(0), torch.tensor(fixed).unsqueeze(0)


# =====================================================================
# 2. Neural Networks (STN, Generator, and Discriminator)
# =====================================================================

class SpatialTransformer(nn.Module):
    """ Warps a source image using a displacement grid """
    def __init__(self, size):
        super(SpatialTransformer, self).__init__()
        self.size = size
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, size[0]), 
            torch.linspace(-1, 1, size[1]), 
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0)  # Shape: (2, H, W)
        self.register_buffer('grid', grid.unsqueeze(0))  # Shape: (1, 2, H, W)

    def forward(self, src, flow):
        H, W = self.size
        flow_normalized = torch.zeros_like(flow)
        flow_normalized[:, 0, :, :] = 2.0 * flow[:, 0, :, :] / max(W - 1, 1)
        flow_normalized[:, 1, :, :] = 2.0 * flow[:, 1, :, :] / max(H - 1, 1)
        
        new_grid = self.grid + flow_normalized
        new_grid = new_grid.permute(0, 2, 3, 1)  # Required shape: (B, H, W, 2)
        return F.grid_sample(src, new_grid, mode='bilinear', padding_mode='border', align_corners=True)


class RegistrationGenerator(nn.Module):
    """ U-Net that takes Moving & Fixed images and outputs Flow Fields """
    def __init__(self, in_channels=2, out_channels=2):
        super(RegistrationGenerator, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.bottleneck = self.conv_block(64, 128)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32)
        
        # Final output layer predicting displacement
        self.flow_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.flow_conv.weight.data.normal_(mean=0.0, std=1e-5)
        self.flow_conv.bias.data.zero_()

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, m, f):
        x = torch.cat([m, f], dim=1)
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        bn = self.bottleneck(self.pool2(x2))
        
        u2 = self.up2(bn)
        if u2.shape != x2.shape:
            u2 = F.interpolate(u2, size=x2.shape[2:])
        d2 = self.dec2(torch.cat([u2, x2], dim=1))
        
        u1 = self.up1(d2)
        if u1.shape != x1.shape:
            u1 = F.interpolate(u1, size=x1.shape[2:])
        d1 = self.dec1(torch.cat([u1, x1], dim=1))
        
        return self.flow_conv(d1)


class AlignmentDiscriminator(nn.Module):
    """ Evaluates registration alignment on concatenated spatial pairs """
    def __init__(self, in_channels=2):
        super(AlignmentDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1)  # PatchGAN prediction map
        )

    def forward(self, img1, img2):
        x = torch.cat([img1, img2], dim=1)
        return self.model(x)


# =====================================================================
# 3. Training & Validation Setup
# =====================================================================

def l2_gradient_loss(flow):
    """ Penalizes spatial derivative irregularities in predicted flow """
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
    return (torch.mean(dy * dy) + torch.mean(dx * dx)) / 2.0


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device designated: {device}")
    
    img_size = (128, 128)
    
    # 1. Programmatically set up the Dataset
    try:
        print("Fetching and processing MedNIST Hand X-rays using MONAI API...")
        dataset = PairedMedNISTDataset(root_dir="./mednist_data", class_name="Hand", download=True)
        print(f"✓ Dataset active: {len(dataset)} real 2D medical pairs loaded.")
    except Exception as e:
        print(f"\n[Warning] Could not initialize MONAI MedNIST.")
        print(f"Error: {str(e)}")
        print("\nPossible solutions:")
        print("  1. Check internet connection (MedNIST requires ~500MB download)")
        print("  2. Try: rm -rf ./mednist_data && python adversarial_registration.py")
        print("  3. Ensure MONAI is installed: pip install monai")
        print("  4. Check disk space (need ~500MB free)")
        print("\nFalling back to local Synthetic Circle Generator.\n")
        dataset = SyntheticCircleDataset(num_samples=200, size=img_size[0])
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 2. Instantiating Networks
    generator = RegistrationGenerator().to(device)
    discriminator = AlignmentDiscriminator().to(device)
    transformer = SpatialTransformer(size=img_size).to(device)
    
    # 3. Optimizers
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    
    # 4. Losses
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_sim = nn.MSELoss()
    
    # Objective weights
    lambda_sim = 20.0     # Intensity matching constraint
    lambda_smooth = 1.5   # Elastic transformation constraint
    lambda_gan = 0.5      # Structural realism constraint
    
    epochs = 20  # Kept short for quick testing
    print(f"\nStarting training loop for {epochs} epochs...")
    
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        for i, (moving, fixed) in enumerate(dataloader):
            moving, fixed = moving.to(device), fixed.to(device)
            
            # ---------------------------------
            # Train Discriminator
            # ---------------------------------
            opt_d.zero_grad()
            
            flow = generator(moving, fixed)
            warped = transformer(moving, flow)
            
            # Real Pair target label is 1
            real_preds = discriminator(fixed, fixed)
            loss_d_real = criterion_gan(real_preds, torch.ones_like(real_preds).to(device))
            
            # Fake Pair target label is 0
            fake_preds = discriminator(warped.detach(), fixed)
            loss_d_fake = criterion_gan(fake_preds, torch.zeros_like(fake_preds).to(device))
            
            loss_d = (loss_d_real + loss_d_fake) / 2.0
            loss_d.backward()
            opt_d.step()
            
            # ---------------------------------
            # Train Generator
            # ---------------------------------
            opt_g.zero_grad()
            
            # Intensity and smoothness constraints
            loss_sim = criterion_sim(warped, fixed)
            loss_smooth = l2_gradient_loss(flow)
            
            # Adversarial component: make D evaluate (Warped, Fixed) as True (1)
            gen_preds = discriminator(warped, fixed)
            loss_gan = criterion_gan(gen_preds, torch.ones_like(gen_preds).to(device))
            
            loss_g = (lambda_sim * loss_sim) + (lambda_smooth * loss_smooth) + (lambda_gan * loss_gan)
            loss_g.backward()
            opt_g.step()
            
            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] | Loss D: {epoch_d_loss/len(dataloader):.4f} | Loss G: {epoch_g_loss/len(dataloader):.4f} | Sim: {loss_sim.item():.4f}")

    # =====================================================================
    # 4. Result Visualization
    # =====================================================================
    print("\nTraining completed. Visualizing registration results...")
    generator.eval()
    with torch.no_grad():
        # Grab a single sample from the last processed batch
        m_img = moving[0, 0].cpu().numpy()
        f_img = fixed[0, 0].cpu().numpy()
        w_img = warped[0, 0].cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(m_img, cmap='gray')
        axes[0].set_title("Moving Image")
        axes[0].axis('off')
        
        axes[1].imshow(f_img, cmap='gray')
        axes[1].set_title("Fixed (Target)")
        axes[1].axis('off')
        
        axes[2].imshow(w_img, cmap='gray')
        axes[2].set_title("Warped Result")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()