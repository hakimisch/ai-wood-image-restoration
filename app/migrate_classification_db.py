# app/migrate_classification_db.py
#
# One-time migration to add classification tables to the existing database.
# Safe to run multiple times — uses IF NOT EXISTS / try-except for idempotency.
#
# Usage:
#   python app/migrate_classification_db.py

import sqlite3
import os

DB_PATH = 'data/database.db'


def migrate():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("🔄 Running classification database migration...")

    # --- Table 1: classifications ---
    cursor.execute('''CREATE TABLE IF NOT EXISTS classifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sample_name TEXT,
        predicted_species TEXT,
        confidence REAL,
        top3_predictions TEXT,
        model_name TEXT,
        timestamp TEXT
    )''')
    print("✅ Table 'classifications' ready.")

    # Add foreign key index for fast lookups
    try:
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_classifications_sample "
            "ON classifications(sample_name)"
        )
    except Exception:
        pass

    # --- Table 2: species_stats ---
    cursor.execute('''CREATE TABLE IF NOT EXISTS species_stats (
        species_name TEXT PRIMARY KEY,
        total_images INTEGER DEFAULT 0,
        avg_vol_clear REAL DEFAULT 0.0,
        avg_vol_blur REAL DEFAULT 0.0,
        classification_accuracy REAL DEFAULT 0.0,
        last_updated TEXT
    )''')
    print("✅ Table 'species_stats' ready.")

    # --- Table 3: classifier_metrics (training run history) ---
    cursor.execute('''CREATE TABLE IF NOT EXISTS classifier_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        num_species INTEGER,
        epochs INTEGER,
        batch_size INTEGER,
        learning_rate REAL,
        freeze_backbone INTEGER,
        best_val_acc REAL,
        best_epoch INTEGER,
        timestamp TEXT
    )''')
    print("✅ Table 'classifier_metrics' ready.")

    # --- Populate species_stats from existing samples data ---
    cursor.execute("SELECT COUNT(*) FROM species_stats")
    if cursor.fetchone()[0] == 0:
        print("📊 Populating species_stats from existing samples...")
        cursor.execute("""
            INSERT OR REPLACE INTO species_stats
                (species_name, total_images, avg_vol_clear, avg_vol_blur, last_updated)
            SELECT
                species_name,
                COUNT(*) as total_images,
                AVG(vol_clear) as avg_vol_clear,
                AVG(vol_blur) as avg_vol_blur,
                datetime('now') as last_updated
            FROM samples
            GROUP BY species_name
        """)
        print(f"✅ Populated {cursor.rowcount} species statistics.")
    else:
        print("ℹ️  species_stats already populated, skipping.")

    conn.commit()
    conn.close()
    print("\n✨ Migration complete!")


if __name__ == "__main__":
    migrate()
