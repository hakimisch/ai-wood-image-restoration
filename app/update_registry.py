import sqlite3
import os

db_path = 'data/database.db'

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 1. Update the species registry 
        cursor.execute("UPDATE species_registry SET initials = 'CH' WHERE initials = 'CHL'")
        
        # 2. Update any existing samples that used 'CHL' 
        cursor.execute("UPDATE samples SET species_initials = 'CH' WHERE species_initials = 'CHL'")
        
        conn.commit()
        print("✅ Chengal initials updated from CHL to CH in all tables.")
        
    except Exception as e:
        print(f"❌ Error during update: {e}")
    finally:
        conn.close()
else:
    print("❌ Database file not found.")