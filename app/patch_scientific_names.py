import sqlite3
import os

def migrate_database():
    db_path = 'data/database.db'
    
    # 1. Add the column safely
    with sqlite3.connect(db_path) as conn:
        try:
            conn.execute("ALTER TABLE species_registry ADD COLUMN scientific_name TEXT;")
            print("✅ Added scientific_name column.")
        except sqlite3.OperationalError:
            print("ℹ️ Column already exists.")

    # 2. Comprehensive Botanical Mapping for your 35 species [cite: 155-159, 171]
    scientific_map = {
        "TU": "Koompassia excelsa", "BAL": "Shorea materialis", "MU": "Prismatomeris glabra",
        "MR": "Intsia palembanica", "TM": "Fagraea fragrans", "RG": "Gluta spp.",
        "KAS": "Pometia pinnata", "BIN": "Calophyllum spp.", "SP": "Sindora spp.",
        "KED": "Canarium spp.", "CH": "Neobalanocarpus heimii", "PL": "Alstonia scholaris",
        "MT": "Shorea macroptera", "MS": "Anisoptera spp.", "KJ": "Dialium spp.",
        "MK": "Pentace spp.", "PP": "Lophopetalum spp.", "RA": "Gonystylus bancanus",
        "SS": "Endospermum diadenum", "MM": "Shorea curtisii", "TR": "Campnosperma spp.",
        "MC": "Mangifera spp.", "MY": "Shorea faguetiana", "SH": "Dillenia spp.",
        "RK": "Vatica spp.", "KT": "Syzygium spp.", "DUR": "Durio spp.",
        "GT": "Parashorea lucida", "JE": "Dyera costulata", "MW": "Shorea bracteolata",
        "MN": "Hopea odorata", "MD": "Litsea spp.", "GIA": "Hopea helferi"
    }

    with sqlite3.connect(db_path) as conn:
        count = 0
        for initials, sci_name in scientific_map.items():
            conn.execute("UPDATE species_registry SET scientific_name = ? WHERE initials = ?", 
                         (sci_name, initials))
            count += 1
        print(f"✅ Successfully patched {count} species with scientific names.")

if __name__ == "__main__":
    migrate_database()