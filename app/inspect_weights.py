# inspect_weights.py
#
# Standalone tool — run from the project root:
#   python inspect_weights.py
#
# What it does:
#   1. Scans the project root for every .pth file
#   2. Reads each file's weight tensors to identify the model architecture
#      and count parameters (CPU-only, no GPU needed)
#   3. Queries the SQLite DB for any training run matching the file
#   4. Prints a formatted report for every file found
#
# Matching priority (most to least specific):
#   A. Exact pth_filename field in model_metrics (set by updated training_tab)
#   B. Filename-pattern match  e.g. "vdsr" in filename -> VDSR rows
#   C. Modification-time match  file mtime within TIME_WINDOW seconds of DB timestamp
#   D. Architecture-only fallback  all DB rows for the same model name
#
# Note: .pth files only store weight tensors. Training parameters are only
# available if the run completed and wrote to model_metrics. Old weights
# trained before that table existed will show "No DB record found".

import os
import sqlite3
from datetime import datetime
import torch

SEARCH_DIR  = '.'
DB_PATH     = 'data/database.db'
TIME_WINDOW = 120


# ---------------------------------------------------------------------------
# Architecture fingerprinting
# ---------------------------------------------------------------------------

def identify_architecture(state_dict):
    keys         = list(state_dict.keys())
    total_params = sum(t.numel() for t in state_dict.values() if isinstance(t, torch.Tensor))

    if any('norm1.weight' in k or 'relative_position_bias' in k for k in keys):
        return "SwinIR (Custom)", total_params

    if any('conv_residual' in k for k in keys):
        return "VDSR (Custom)", total_params

    if 'conv1.weight' in state_dict and 'conv3.weight' in state_dict:
        if state_dict['conv1.weight'].shape[1] == 3:
            return "SRCNN (Custom)", total_params

    if 'encoder.0.weight' in state_dict and len([k for k in keys if 'weight' in k]) <= 3:
        return "Simple CNN (Custom)", total_params

    return "Unknown", total_params


def fmt_params(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def fmt_size(path):
    b = os.path.getsize(path)
    for unit in ('B', 'KB', 'MB', 'GB'):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} GB"


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def load_db_records():
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model_metrics'")
        if not cur.fetchone():
            conn.close()
            return []
        # pth_filename was added in an updated training_tab; graceful fallback
        try:
            cur.execute(
                "SELECT id, model_name, epochs, batch_size, final_loss, "
                "psnr, ssim, timestamp, "
                "COALESCE(pth_filename, '') AS pth_filename "
                "FROM model_metrics ORDER BY timestamp DESC"
            )
        except sqlite3.OperationalError:
            cur.execute(
                "SELECT id, model_name, epochs, batch_size, final_loss, "
                "psnr, ssim, timestamp, '' AS pth_filename "
                "FROM model_metrics ORDER BY timestamp DESC"
            )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        conn.close()
        return rows
    except Exception as exc:
        print(f"  Warning: DB read error — {exc}")
        return []


def match_records(filename, mtime, arch, records):
    # A. Exact filename stored in DB (updated training_tab writes this)
    exact = [r for r in records if r.get('pth_filename') == filename]
    if exact:
        return exact, "exact filename"

    fname_lower = filename.lower()

    # B. Filename-pattern heuristic
    arch_tokens = {
        "Simple CNN (Custom)": ["scnn", "simplecnn", "simple_cnn"],
        "SRCNN (Custom)":      ["srcnn"],
        "VDSR (Custom)":       ["vdsr"],
        "SwinIR (Custom)":     ["swinir", "swin"],
    }
    tokens  = arch_tokens.get(arch, [])
    pattern = [
        r for r in records
        if any(t in fname_lower for t in tokens) and r['model_name'] == arch
    ]
    if pattern:
        return pattern, "filename pattern"

    # C. Modification-time proximity
    file_dt  = datetime.fromtimestamp(mtime)
    by_time  = []
    for r in records:
        try:
            db_dt = datetime.strptime(r['timestamp'], "%Y-%m-%d %H:%M:%S")
            if abs((db_dt - file_dt).total_seconds()) <= TIME_WINDOW:
                by_time.append(r)
        except ValueError:
            pass
    if by_time:
        return by_time, "modification time"

    # D. Architecture-only fallback
    by_arch = [r for r in records if r['model_name'] == arch]
    if by_arch:
        return by_arch, "architecture only (may not be this exact file)"

    return [], "none"


# ---------------------------------------------------------------------------
# Inspect one file
# ---------------------------------------------------------------------------

def inspect_file(path, records):
    filename  = os.path.basename(path)
    mtime     = os.path.getmtime(path)
    size      = fmt_size(path)
    mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

    try:
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if not isinstance(state_dict, dict):
            return dict(filename=filename, size=size, modified=mtime_str,
                        error="Unexpected format: " + type(state_dict).__name__)

        arch, total_params = identify_architecture(state_dict)
        layers             = len([k for k in state_dict if k.endswith('.weight')])
        matches, method    = match_records(filename, mtime, arch, records)

        return dict(filename=filename, path=path, size=size, modified=mtime_str,
                    arch=arch, params=fmt_params(total_params), layers=layers,
                    matches=matches, match_method=method, error=None)

    except Exception as exc:
        return dict(filename=filename, size=size, modified=mtime_str, error=str(exc))


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

W   = 68
SEP = "─" * W


def pad(text, width):
    s = str(text)
    if len(s) > width:
        s = s[:width - 1] + "~"
    return s.ljust(width)


def row2(left_label, left_val, right_label=None, right_val=None):
    left_part  = f"  {left_label:<14}: {left_val}"
    if right_label is not None:
        right_part = f"  {right_label:<12}: {right_val}"
        line = f"{left_part:<38}{right_part}"
    else:
        line = left_part
    print(f"|{pad(line, W)}|")


def print_report(info):
    print(f"\n+{SEP}+")
    print(f"|{pad('  File: ' + info['filename'], W)}|")
    print(f"|{pad('  Size: ' + info['size'] + '   Modified: ' + info['modified'], W)}|")

    if info.get('error'):
        print(f"|{SEP}|")
        print(f"|{pad('  ERROR: ' + info['error'], W)}|")
        print(f"+{SEP}+")
        return

    print(f"|{SEP}|")
    row2("Architecture",  info['arch'])
    row2("Parameters",    info['params'], "Weight layers", info['layers'])

    matches = info.get('matches', [])
    method  = info.get('match_method', 'none')

    if not matches:
        print(f"|{SEP}|")
        print(f"|{pad('  No training record found in database.', W)}|")
        print(f"|{pad('  (Run may predate metrics logging, or filename differs.)', W)}|")
    else:
        shown = matches[:3]
        for i, r in enumerate(shown):
            count_label = "" if len(shown) == 1 else f" {i + 1}/{len(shown)}"
            print(f"|{SEP}|")
            print(f"|{pad('  DB record' + count_label + '  [match: ' + method + ']', W)}|")
            row2("Model",      r['model_name'])
            row2("Epochs",     r['epochs'],     "Batch size",  r['batch_size'])

            loss_str = f"{r['final_loss']:.6f}" if r['final_loss'] is not None else "N/A"
            row2("Final loss",  loss_str)

            psnr_str = f"{r['psnr']:.2f} dB" if r['psnr'] is not None else "N/A"
            ssim_str = f"{r['ssim']:.4f}"     if r['ssim'] is not None else "N/A"
            row2("PSNR",        psnr_str,        "SSIM",        ssim_str)
            row2("Timestamp",   r['timestamp'])

        if len(matches) > 3:
            print(f"|{pad('  ... and ' + str(len(matches) - 3) + ' more record(s) in database', W)}|")

    print(f"+{SEP}+")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  CAIRO Weight Inspector")
    print("  Scanning : " + os.path.abspath(SEARCH_DIR))
    print("  Database : " + os.path.abspath(DB_PATH))
    print("=" * 70)

    pth_files = sorted(
        [os.path.join(SEARCH_DIR, f) for f in os.listdir(SEARCH_DIR) if f.endswith('.pth')],
        key=os.path.getmtime,
        reverse=True
    )

    if not pth_files:
        print("\n  No .pth files found in the project root.")
        return

    print(f"\n  Found {len(pth_files)} weight file(s)\n")

    records = load_db_records()
    if not records:
        print("  Warning: model_metrics table empty or database not found.")
        print("  Files will be identified by architecture fingerprint only.\n")
    else:
        print(f"  Found {len(records)} training record(s) in database.\n")

    for path in pth_files:
        print_report(inspect_file(path, records))

    print()


if __name__ == "__main__":
    main()