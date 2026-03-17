# app/evaluate.py

import sqlite3
import cv2
import torch
import numpy as np
import os
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import your models
from models import SimpleRestorationNet, SRCNN

def evaluate_model(model_name, weight_file, num_samples=100):
    print(f"\n📊 Starting Evaluation for: {model_name}")
    print("-" * 50)
    
    # 1. Setup Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == "Simple CNN":
        model = SimpleRestorationNet().to(device)
    elif model_name == "SRCNN":
        model = SRCNN().to(device)
    else:
        print("Model not recognized.")
        return

    if os.path.exists(weight_file):
        model.load_state_dict(torch.load(weight_file, map_location=device))
        model.eval()
    else:
        print(f"⚠️ {weight_file} not found! Cannot evaluate {model_name}.")
        return

    # 2. Image Preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # 3. Fetch Test Data from Database
    conn = sqlite3.connect('data/database.db')
    cursor = conn.cursor()
    # Grab a random sample of images to act as our "Test Set"
    cursor.execute("SELECT blur_path, clear_path FROM samples ORDER BY RANDOM() LIMIT ?", (num_samples,))
    test_pairs = cursor.fetchall()
    conn.close()

    total_psnr = 0.0
    total_ssim = 0.0
    valid_samples = 0

    # 4. Evaluation Loop
    with torch.no_grad():
        for idx, (blur_path, clear_path) in enumerate(test_pairs):
            if not os.path.exists(blur_path) or not os.path.exists(clear_path):
                continue
                
            # Read images
            blur_img = cv2.imread(blur_path)
            clear_img = cv2.imread(clear_path)
            
            # Convert to RGB
            blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
            clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
            
            # The network resizes everything to 256x256, so we must resize the 
            # ground-truth clear image to 256x256 to do a fair pixel-to-pixel math comparison.
            clear_img_resized = cv2.resize(clear_img, (256, 256))
            
            # Run Inference
            input_tensor = transform(blur_img).unsqueeze(0).to(device)
            output_tensor = model(input_tensor)
            
            # Post-process tensor back to image array
            output_img = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            output_img = (np.clip(output_img, 0, 1) * 255).astype(np.uint8)
            
            # Calculate Metrics (Win_size=3 is standard for small 256x256 images)
            current_psnr = psnr(clear_img_resized, output_img, data_range=255)
            # channel_axis=-1 specifies that color channels are the last dimension (RGB)
            current_ssim = ssim(clear_img_resized, output_img, data_range=255, channel_axis=-1, win_size=3)
            
            total_psnr += current_psnr
            total_ssim += current_ssim
            valid_samples += 1

            if (idx + 1) % 25 == 0:
                print(f"Evaluated {idx + 1}/{num_samples} images...")

    # 5. Calculate & Print Averages
    if valid_samples > 0:
        avg_psnr = total_psnr / valid_samples
        avg_ssim = total_ssim / valid_samples
        print("\n📈 FINAL RESULTS")
        print(f"Model: {model_name}")
        print(f"Samples Evaluated: {valid_samples}")
        print(f"Average PSNR: {avg_psnr:.2f} dB  (Higher is better)")
        print(f"Average SSIM: {avg_ssim:.4f}      (Closer to 1.0 is better)")
        print("-" * 50)
    else:
        print("No valid image pairs found.")

if __name__ == "__main__":
    # Test the Simple CNN we just trained
    evaluate_model("Simple CNN", "weights.pth", num_samples=100)
    
    # We will uncomment this later once you train the SRCNN!
    # evaluate_model("SRCNN", "srcnn_weights.pth", num_samples=100)