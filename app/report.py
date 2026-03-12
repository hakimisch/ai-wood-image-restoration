import sqlite3
import os

def run_report():
    db_path = 'data/database.db'
    
    if not os.path.exists(db_path):
        print(f"❌ Error: Database not found at {db_path}")
        return
    
    print("\n--- 📊 DATASET AUDIT REPORT ---")
    print(f"{'Initials':<10} | {'Full Name':<20} | {'Count':<8} | {'Status'}")
    print("-" * 60)
    
    with sqlite3.connect(db_path) as conn:
        query = """
        SELECT r.initials, r.full_name, COUNT(s.id) as img_count
        FROM species_registry r
        LEFT JOIN samples s ON r.initials = s.species_initials
        GROUP BY r.initials
        ORDER BY img_count ASC;
        """
        results = conn.execute(query).fetchall()
        
        for initials, name, count in results:
            # Based on the rule of 20 images per block/sample 
            status = "✅ OK" if count >= 20 else "⚠️ UNDER TARGET"
            if count == 0: status = "❌ MISSING DATA"
            
            print(f"{initials:<10} | {name[:20]:<20} | {count:<8} | {status}")
            
    print("-" * 60)
    print(f"Total Rows in DB: {sum(row[2] for row in results)}")

if __name__ == "__main__":
    run_report()