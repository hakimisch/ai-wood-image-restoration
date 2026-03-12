import os
import sqlite3
import cv2
from datetime import datetime

def calculate_vol(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(img, cv2.CV_64F).var() if img is not None else 0

def rescue_dataset():
    conn = sqlite3.connect('data/database.db')
    root_dir = "Kayu"
    
    print("🚀 Starting Database Recovery...")
    
    for species in os.listdir(root_dir):
        species_path = os.path.join(root_dir, species)
        if not os.path.isdir(species_path): continue
        
        for block in os.listdir(species_path):
            block_path = os.path.join(species_path, block)
            clear_dir = os.path.join(block_path, "clear")
            
            if not os.path.exists(clear_dir): continue
            
            for img_file in os.listdir(clear_dir):
                if not img_file.endswith(".jpg"): continue
                
                sample_name = img_file.replace(".jpg", "")
                clear_path = os.path.join(clear_dir, img_file)
                blur_path = clear_path.replace("clear", "blur")
                
                # Extract Initials (e.g., KAS from KAS070001)
                initials = ''.join([i for i in sample_name if not i.isdigit()])[:3]
                
                # Calculate metrics for the DB
                v_clear = calculate_vol(clear_path)
                v_blur = calculate_vol(blur_path)
                
                conn.execute('''INSERT OR IGNORE INTO samples 
                    (species_name, species_initials, sample_name, mode, 
                     clear_path, blur_path, vol_clear, vol_blur, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                    (species, initials, sample_name, "RGB", clear_path, 
                     blur_path, v_clear, v_blur, datetime.now().isoformat()))
    
    conn.commit()
    conn.close()
    print("✅ Recovery Complete! Run 'Verify Dataset' in the GUI to confirm.")

if __name__ == "__main__":
    rescue_dataset()