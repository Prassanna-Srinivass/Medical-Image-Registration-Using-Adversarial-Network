import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# ==========================================
# 1. AI Model Architecture (Copied here so UI is standalone)
# ==========================================
class SpatialTransformer(nn.Module):
    def __init__(self, size):
        super().__init__()
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class RegistrationGenerator(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64, 32)
        
        self.flow_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

    def forward(self, m, f):
        x = torch.cat([m, f], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
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

# ==========================================
# 2. AI Inference Setup
# ==========================================
@st.cache_resource # Caches the model so it doesn't reload on every click
def load_ai_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = RegistrationGenerator().to(device)
    
    # Load your saved weights!
    try:
        generator.load_state_dict(torch.load("registration_model.pth", map_location=device, weights_only=True))
        generator.eval()
    except FileNotFoundError:
        st.error("Model weights 'registration_model.pth' not found! Make sure you trained the model first.")
        return None, None, None
        
    transformer = SpatialTransformer(size=(128, 128)).to(device)
    return generator, transformer, device

def process_image(img):
    """Formats the uploaded image for the AI"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    img = Image.open(img).convert('L')
    return transform(img).unsqueeze(0) # Add batch dimension

# ==========================================
# 3. Streamlit User Interface
# ==========================================
st.set_page_config(page_title="Medical Image Registration", layout="wide")
st.title("🩻 AI Medical Image Registration")
st.write("Upload a Fixed (Target) image and a Moving (Shifted) image. The AI will align the bones and output the Warped result.")

# Load AI
generator, transformer, device = load_ai_model()

if generator:
    # Upload widgets side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Upload Moving Image")
        moving_file = st.file_uploader("Choose moving image", type=["png", "jpg", "jpeg"])
    with col2:
        st.subheader("2. Upload Fixed (Target) Image")
        fixed_file = st.file_uploader("Choose fixed image", type=["png", "jpg", "jpeg"])

    if moving_file and fixed_file:
        if st.button("🚀 Align Images", use_container_width=True):
            with st.spinner('AI is generating the deformation field...'):
                
                # Format images
                m_tensor = process_image(moving_file).to(device)
                f_tensor = process_image(fixed_file).to(device)

                # Run AI Prediction
                with torch.no_grad():
                    flow = generator(m_tensor, f_tensor)
                    warped_tensor = transformer(m_tensor, flow)

                # Convert tensors back to viewable images
                m_img_show = m_tensor[0, 0].cpu().numpy()
                f_img_show = f_tensor[0, 0].cpu().numpy()
                w_img_show = warped_tensor[0, 0].cpu().numpy()

                st.success("Registration Complete!")

                # Display Results
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.image(m_img_show, caption="Moving Image", use_container_width=True, clamp=True)
                with res_col2:
                    st.image(f_img_show, caption="Fixed (Target) Image", use_container_width=True, clamp=True)
                with res_col3:
                    st.image(w_img_show, caption="AI Warped Result", use_container_width=True, clamp=True)