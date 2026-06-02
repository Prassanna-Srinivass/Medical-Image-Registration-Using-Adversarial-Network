import os
import random
from PIL import Image

def generate_moving_dataset():
    # Use raw strings (r"") for Windows paths
    fixed_dir = r"C:\image registration using gan\dataset\fixed"
    moving_dir = r"C:\image registration using gan\dataset\moving"

    # Check if the fixed directory exists
    if not os.path.exists(fixed_dir):
        print(f"Error: Could not find {fixed_dir}. Please check the folder path.")
        return

    # Create the Moving directory
    os.makedirs(moving_dir, exist_ok=True)

    # Get all image files (.png, .jpg, .jpeg)
    valid_exts = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(fixed_dir) if f.lower().endswith(valid_exts)]
    
    if len(image_files) == 0:
        print("No images found! Make sure your images are directly inside the 'fixed' folder.")
        return

    print(f"Found {len(image_files)} Chest X-rays. Starting transformation...")

    for i, filename in enumerate(image_files):
        img_path = os.path.join(fixed_dir, filename)
        img = Image.open(img_path).convert("L")
        
        # Now 3 options: rotate, shift, or scale
        action = random.choice(['rotate', 'shift', 'scale'])
        
        if action == 'rotate':
            angle = random.choice([random.uniform(-30, -20), random.uniform(20, 30)])
            out_img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
            
        elif action == 'shift':
            dx = random.choice([random.randint(-30, -15), random.randint(15, 30)])
            dy = random.choice([random.randint(-30, -15), random.randint(15, 30)])
            out_img = img.transform(img.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy), fillcolor=0)
            
        elif action == 'scale':
            # Choose to either zoom out (0.75x to 0.85x) or zoom in (1.15x to 1.25x)
            scale_factor = random.choice([random.uniform(0.75, 0.85), random.uniform(1.15, 1.25)])
            new_w = int(img.width * scale_factor)
            new_h = int(img.height * scale_factor)
            
            # Resize image
            resized_img = img.resize((new_w, new_h), resample=Image.BILINEAR)
            
            # Create a blank black canvas of the original size
            out_img = Image.new("L", img.size, 0)
            
            # Calculate coordinates to paste into the center
            left = (img.width - new_w) // 2
            top = (img.height - new_h) // 2
            
            # Paste the resized image. PIL automatically crops if zooming in.
            out_img.paste(resized_img, (left, top))
            
        out_path = os.path.join(moving_dir, filename)
        out_img.save(out_path)

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images...")

    print(f"Done! All moving images saved to: {moving_dir}")

if __name__ == "__main__":
    generate_moving_dataset()