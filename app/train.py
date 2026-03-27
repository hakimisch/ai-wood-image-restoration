import sqlite3
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

# Import all your brains!
from models import SimpleRestorationNet, SRCNN, VDSR

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
    # --- CONFIGURATION ---
    # Change this string to "Simple CNN", "SRCNN", or "VDSR" to swap models
    CHOSEN_MODEL = "VDSR" 
    EPOCHS = 10
    BATCH_SIZE = 8 # Lowered to 8 to be safe for VDSR on a 6GB GTX 1660 Ti
    # ---------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    # 1. Setup Model and Save Path
    if CHOSEN_MODEL == "Simple CNN":
        model = SimpleRestorationNet().to(device)
        save_path = "weights.pth"
    elif CHOSEN_MODEL == "SRCNN":
        model = SRCNN().to(device)
        save_path = "srcnn_weights.pth"
    elif CHOSEN_MODEL == "VDSR":
        model = VDSR().to(device)
        save_path = "vdsr_weights.pth"
    else:
        print("❌ Unknown model chosen!")
        return

    # 2. Prepare Data
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    dataset = WoodDataset(db_path='data/database.db', transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Initialize Loss (MSE) and Optimizer
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. The Training Loop
    print(f"🧠 Starting Training for {CHOSEN_MODEL}...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (blur_imgs, clear_imgs) in enumerate(dataloader):
            blur_imgs = blur_imgs.to(device)
            clear_imgs = clear_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(blur_imgs)
            loss = criterion(outputs, clear_imgs)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} Complete. Average Loss: {epoch_loss:.4f}\n")

    # 5. Save the trained brain!
    torch.save(model.state_dict(), save_path)
    print(f"🎉 Training Complete! Brain saved to '{save_path}'.")

if __name__ == "__main__":
    train()