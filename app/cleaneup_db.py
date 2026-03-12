import sqlite3
import os

db_path = 'data/database.db'

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    # This keeps only the latest entry (highest ID) for each sample_name [cite: 8, 25]
    conn.execute('''DELETE FROM samples 
                    WHERE id NOT IN (
                        SELECT MAX(id) 
                        FROM samples 
                        GROUP BY sample_name
                    )''')
    conn.commit()
    conn.close()
    print("✅ Database duplicates removed successfully.")
else:
    print("❌ Database file not found at " + db_path)