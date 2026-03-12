import sqlite3
import os

db_path = 'data/database.db'

def repair_merbau_metadata():
    if not os.path.exists(db_path):
        print(f"❌ Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("🔄 Repairing Merbau metadata and syncing paths...")
    
    # 1. First, fix the 'species_initials' where they are broken but name starts with MR
    # 2. Update species_name to 'Merbau'
    # 3. Update paths to point to the new 'Merbau' folder
    cursor.execute("""
        UPDATE samples 
        SET species_initials = 'MR',
            species_name = 'Merbau',
            clear_path = REPLACE(clear_path, 'Unknown', 'Merbau'),
            blur_path = REPLACE(blur_path, 'Unknown', 'Merbau')
        WHERE sample_name LIKE 'MR%' AND species_name = 'Unknown'
    """)

    changes = conn.total_changes
    conn.commit()
    conn.close()

    if changes > 0:
        print(f"✅ Successfully repaired {changes} Merbau records.")
        print("🚀 Your database now correctly links to Kayu\\Merbau\\")
    else:
        print("⚠️ No records matched 'MR%' with species 'Unknown'. Check the 'samples' table manually.")

if __name__ == "__main__":
    repair_merbau_metadata()