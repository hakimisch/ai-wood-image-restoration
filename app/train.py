import sqlite3
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

# Import your blank brain
from models import SimpleRestorationNet
from models import SRCNN

class WoodDataset(Dataset):
    """Custom PyTorch Dataset that reads straight from your SQLite Database."""
    def __init__(self, db_path='data/database.db', transform=None):
        self.transform = transform
        
        # Connect to DB and get all clear/blur paths
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT blur_path, clear_path FROM samples")
        self.image_pairs = cursor.fetchall()
        conn.close()

        print(f"📦 Loaded {len(self.image_pairs)} image pairs from the database.")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        blur_path, clear_path = self.image_pairs[idx]
        
        # 1. Read images using OpenCV
        blur_img = cv2.imread(blur_path)
        clear_img = cv2.imread(clear_path)
        
        # Convert BGR to RGB for PyTorch
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
        clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
        
        # 2. Apply transformations (Resize and convert to Tensor)
        if self.transform:
            blur_img = self.transform(blur_img)
            clear_img = self.transform(clear_img)
            
        return blur_img, clear_img

def train():
    # 1. Setup Device (Bind to your GTX 1660 Ti)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    # 2. Prepare Data
    # Resizing to 256x256 to ensure it fits in the 1660 Ti's VRAM
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    dataset = WoodDataset(db_path='data/database.db', transform=transform)
    
    # DataLoader feeds the images to the GPU in batches of 16
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 3. Initialize Model, Loss (MSE), and Optimizer
    model = SRCNN().to(device)
    criterion = nn.MSELoss() # Maps to thesis Section 3.6.3.3
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. The Training Loop
    epochs = 10 # Start with 10 sweeps through the 6,800 images
    
    print("🧠 Starting Training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (blur_imgs, clear_imgs) in enumerate(dataloader):
            # Move images to the GPU
            blur_imgs = blur_imgs.to(device)
            clear_imgs = clear_imgs.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass: AI guesses the clear image
            outputs = model(blur_imgs)
            
            # Calculate how wrong the guess was (Loss)
            loss = criterion(outputs, clear_imgs)
            
            # Backward pass: Calculate the adjustments
            loss.backward()
            
            # Optimizer step: Apply the adjustments to the brain
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")
                
        # Average loss for the epoch
        epoch_loss = running_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} Complete. Average Loss: {epoch_loss:.4f}\n")

    # 5. Save the trained brain!
    torch.save(model.state_dict(), "srcnn_weights.pth")
    print("🎉 Training Complete! Brain saved to 'weights.pth'.")

if __name__ == "__main__":
    train()