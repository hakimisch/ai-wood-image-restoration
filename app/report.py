# app/report.py
#
# Dataset Audit Report — now includes classification accuracy metrics.

import sqlite3
import os

def run_report():
    db_path = 'data/database.db'
    
    if not os.path.exists(db_path):
        print(f"❌ Error: Database not found at {db_path}")
        return
    
    print("\n" + "=" * 80)
    print("  📊 CAIRO DATASET AUDIT REPORT")
    print("=" * 80)
    
    conn = sqlite3.connect(db_path)
    
    # ── Section 1: Species Registry & Sample Counts ──
    print("\n--- Species Registry & Sample Counts ---")
    print(f"{'Initials':<10} | {'Full Name':<22} | {'Count':<8} | {'Status':<20} | {'Class. Acc':<10}")
    print("-" * 80)
    
    query = """
    SELECT r.initials, r.full_name, COUNT(s.id) as img_count,
           ROUND(AVG(CASE WHEN c.predicted_species = s.species_name THEN 1.0 ELSE 0 END), 3) as acc
    FROM species_registry r
    LEFT JOIN samples s ON r.initials = s.species_initials
    LEFT JOIN classifications c ON s.sample_name = c.sample_name
    GROUP BY r.initials
    ORDER BY img_count ASC;
    """
    results = conn.execute(query).fetchall()
    
    for initials, name, count, acc in results:
        status = "✅ OK" if count >= 20 else "⚠️ UNDER TARGET"
        if count == 0: status = "❌ MISSING DATA"
        acc_str = f"{acc:.1%}" if acc else "N/A"
        print(f"{initials:<10} | {name[:22]:<22} | {count:<8} | {status:<20} | {acc_str:<10}")
    
    print("-" * 80)
    total_samples = sum(row[2] for row in results)
    print(f"Total Samples in DB: {total_samples}")
    
    # ── Section 2: Classification Summary ──
    print("\n--- Classification Summary ---")
    try:
        total_classified = conn.execute("SELECT COUNT(*) FROM classifications").fetchone()[0]
        print(f"Total Classified Images: {total_classified}")
        
        if total_classified > 0:
            avg_confidence = conn.execute(
                "SELECT ROUND(AVG(confidence), 3) FROM classifications"
            ).fetchone()[0]
            print(f"Average Confidence: {avg_confidence:.1%}")
            
            # Top-1 accuracy (where we have ground truth)
            correct = conn.execute("""
                SELECT COUNT(*) FROM classifications c
                JOIN samples s ON c.sample_name = s.sample_name
                WHERE c.predicted_species = s.species_name
            """).fetchone()[0]
            if total_classified > 0:
                print(f"Top-1 Accuracy (vs ground truth): {correct}/{total_classified} = {correct/total_classified:.1%}")
    except Exception:
        print("Classification tables not yet created. Run migrate_classification_db.py first.")
    
    # ── Section 3: Training Summary ──
    print("\n--- Training History ---")
    try:
        rows = conn.execute(
            "SELECT model_name, epochs, batch_size, best_val_acc, timestamp "
            "FROM classifier_metrics ORDER BY timestamp DESC LIMIT 5"
        ).fetchall()
        if rows:
            print(f"{'Model':<20} {'Epochs':<8} {'Batch':<8} {'Best Acc':<10} {'Timestamp'}")
            print("-" * 60)
            for row in rows:
                acc_str = f"{row[3]:.2%}" if row[3] else "N/A"
                print(f"{row[0]:<20} {row[1]:<8} {row[2]:<8} {acc_str:<10} {row[4]}")
        else:
            print("No classifier training history found.")
    except Exception:
        print("classifier_metrics table not yet created.")
    
    conn.close()
    print("\n" + "=" * 80)
    print("  Report Complete")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_report()
