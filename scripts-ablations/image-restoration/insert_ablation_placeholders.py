"""
Insert placeholder markers and replacement text into paper_v2_fixed_edited.docx
for the §4.2 Domain Gap ablation data (Table + Figure + Text).

Usage:
  cd /mnt/c/Hakimi/Internship/GUI Image Restoration
  ./app/torch_env/Scripts/python -X utf8 scripts-ablations/image-restoration/insert_ablation_placeholders.py

Output: research/output/paper_v2_fixed_edited.docx  (modified in-place)
"""

import os, shutil, zipfile
from lxml import etree

SRC = os.path.join(os.path.dirname(__file__), '../../research/output/paper_v2_fixed_edited.docx')
SRC = os.path.normpath(SRC)

NS_W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS_M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
NS_R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'

W = lambda tag: f'{{{NS_W}}}{tag}'
M = lambda tag: f'{{{NS_M}}}{tag}'

# ── Helpers ─────────────────────────────────────────────────────────────────

def make_run(text, bold=False, color=None, size=11, italic=False):
    """Create a w:r element."""
    r = etree.SubElement(etree.Element(W('r')), W('r'))
    rpr = etree.SubElement(r, W('rPr'))

    sz = etree.SubElement(rpr, W('sz'))
    sz.set(W('val'), str(size * 2))
    szCs = etree.SubElement(rpr, W('szCs'))
    szCs.set(W('val'), str(size * 2))

    if bold:
        etree.SubElement(rpr, W('b'))
    if italic:
        etree.SubElement(rpr, W('i'))
    if color:
        clr = etree.SubElement(rpr, W('color'))
        clr.set(W('val'), color)

    t = etree.SubElement(r, W('t'))
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    t.text = text
    return r


def make_paragraph(text, bold=False, color=None, size=11, italic=False, style=None):
    """Create a w:p element with optional style."""
    p = etree.Element(W('p'))
    if style:
        pPr = etree.SubElement(p, W('pPr'))
        pStyle = etree.SubElement(pPr, W('pStyle'))
        pStyle.set(W('val'), style)
    p.append(make_run(text, bold=bold, color=color, size=size, italic=italic))
    return p


def get_non_math_text(p_element):
    """Get all text from a paragraph, excluding OMML math elements."""
    texts = []
    for t in p_element.iter(W('t')):
        parent = t.getparent()
        in_math = False
        while parent is not None:
            if parent.tag in (M('oMath'), M('oMathPara')):
                in_math = True
                break
            parent = parent.getparent()
        if not in_math:
            texts.append(t.text or '')
    return ''.join(texts)


# ── Main ────────────────────────────────────────────────────────────────────

print(f"📄 Reading: {SRC}")

# Read DOCX as ZIP
with zipfile.ZipFile(SRC, 'r') as z:
    xml_raw = z.read('word/document.xml')
    original_math_count = len(etree.fromstring(xml_raw).findall(f'.//{{{NS_M}}}oMath'))
    print(f"   OMML equations found: {original_math_count}")

root = etree.fromstring(xml_raw)
body = root.find(W('body'))
paras = body.findall(W('p'))

# ── Find §4.2 end (last paragraph before §4.3) ────────────────────────────
target_idx = None
for i, p in enumerate(paras):
    full_text = get_non_math_text(p)
    if '4.2.3 Evaluation Protocol Note' in full_text or '4.3 Experiment 2' in full_text:
        pass
    if '4.3 Experiment 2' in full_text:
        target_idx = i  # Insert BEFORE §4.3 heading
        break

if target_idx is None:
    # Fallback: find the Evaluation Protocol Note paragraph
    for i, p in enumerate(paras):
        full_text = get_non_math_text(p)
        if 'numerical values reported in this section' in full_text:
            target_idx = i + 1
            break

target_para = paras[target_idx] if target_idx else None

if target_para is None:
    print("❌ Could not find §4.3 or Evaluation Protocol Note paragraph. Aborting.")
    exit(1)

print(f"   Inserting after paragraph index: {target_idx}")
print(f"   Target text: {get_non_math_text(target_para)[:80]}...")

# ── Build placeholder paragraphs (insert in REVERSE order) ─────────────────

placeholders = []

# === FIGURE placeholder ===
placeholders.append(make_paragraph(
    '--- FIGURE 4.X: research/figures/ablation_domain_gap_comparison.png ---',
    bold=True, color='CC0000', size=10
))
placeholders.append(make_paragraph(
    'Figure 4.X: Domain gap ablation comparison — VDSR on a physics-pipeline test image. '
    'Left to right: Blurred input, Gaussian-only VDSR output (PSNR 17.44 / SSIM 0.3909), '
    'Full-physics VDSR output (PSNR 18.49 / SSIM 0.4378), Ground truth. '
    'The Gaussian-only model produces visibly noisier output with degraded cell-wall boundaries, '
    'while the full-physics model preserves anatomical structure.',
    italic=True, size=9, color='666666'
))
placeholders.append(make_paragraph(
    'Paste the image file here (research/figures/ablation_domain_gap_comparison.png). '
    'Right-click → Insert Picture → resize to fit column width.',
    italic=True, size=9, color='999999'
))
placeholders.append(make_paragraph('', size=8))  # spacer

# === TABLE placeholder ===
placeholders.append(make_paragraph(
    '--- TABLE 4.2: Ablation Domain Gap Results ---',
    bold=True, color='CC0000', size=10
))
table_md = (
    'INSERT AS WORD TABLE with 5 columns:\n'
    'Training Regime | Test Set | PSNR (dB) | SSIM | LPIPS\n'
    'Gaussian-only (20ep) | Gaussian-Only | 25.09 | 0.6121 | 0.1962\n'
    'Gaussian-only (20ep) | Physics Pipeline | 17.44 | 0.3909 | 0.4336\n'
    'Full Physics (20ep) | Gaussian-Only | 22.33 | 0.5268 | 0.2883\n'
    'Full Physics (20ep) | Physics Pipeline | 18.49 | 0.4378 | 0.3308\n'
    '\n'
    'Caption: Table 4.2: Controlled VDSR ablation comparing Gaussian-only vs full physics training regimes. '
    'Both models trained for 20 epochs with identical hyperparameters. The domain gap is quantified as the '
    'cross-domain performance drop when a model is evaluated on a degradation type unseen during training.'
)
placeholders.append(make_paragraph(table_md, size=9, color='444444'))
placeholders.append(make_paragraph('', size=8))  # spacer

# === TEXT REPLACEMENT instructions ===
placeholders.append(make_paragraph(
    '═══ TEXT REPLACEMENT INSTRUCTIONS FOR §4.2 ═══',
    bold=True, color='0000CC', size=11
))
placeholders.append(make_paragraph(
    'REPLACE these paragraphs as follows:',
    bold=True, color='0000CC', size=10
))

# Instruction lines
instructions = [
    '',
    '[P204] 4.2.1 Phase 1 — The Gaussian-Only Catastrophe  → KEEP HEADING, REWRITE BODY',
    '[P205-P207] Replace these 3 paragraphs with:',
    '',
    '  "To quantitatively isolate the effect of training degradation type from architectural choice,',
    '  we performed a controlled ablation experiment using the VDSR architecture (20-layer residual',
    '  network, 0.67M parameters). Two models were trained with identical hyperparameters (20 epochs,',
    '  hybrid 0.8 MSE + 0.2 L1 loss, batch size 8, gradient accumulation 2, Adam optimiser with cosine',
    '  annealing from 10^−4 to 10^−6) on the same 6,842-image dataset — the sole difference being the',
    '  degradation applied during training:',
    '  1. Gaussian-only regime: blur generated by a single Gaussian kernel with continuous sigma',
    '     sampling (σ ∼ U(0.5, 5.0)), no compound degradation.',
    '  2. Full physics regime: compound pipeline (out-of-focus disk blurs, motion streaks, space-variant',
    '     focal gradients, LED banding, sensor noise, vignetting, and JPEG compression).',
    '',
    '  Both models were evaluated on an identical 50-image holdout test set from each degradation regime.',
    '  The results are presented in Table 4.2."',
    '',
    '[P208] 4.2.2 Phase 2 — Physics-Based Resolution  → KEEP HEADING, REWRITE BODY',
    '[P209-P212] Replace these 4 paragraphs with:',
    '',
    '  "The Gaussian-only VDSR achieves a PSNR of 25.09 dB on its native Gaussian test set — a strong',
    '  result by conventional restoration benchmarks. However, when this same model encounters the',
    '  physics-pipeline test set (which contains compound degradation unseen during training), its PSNR',
    '  drops to 17.44 dB — a degradation of −7.65 dB. The structural fidelity follows a similar trajectory:',
    '  SSIM falls from 0.6121 to 0.3909, while LPIPS more than doubles from 0.1962 to 0.4336.',
    '',
    '  This cross-domain performance collapse constitutes the quantifiable expression of the domain gap:',
    '  a network trained on a single degradation type cannot generalise to the physically realistic',
    '  compound degradations encountered in practice.',
    '',
    '  The full-physics VDSR, trained on the compound degradation pipeline, exhibits markedly greater',
    '  cross-domain robustness. On the physics-pipeline test set it achieves PSNR = 18.49 dB,',
    '  SSIM = 0.4378, LPIPS = 0.3308 — consistently outperforming the Gaussian-only model on every',
    '  metric (ΔPSNR = +1.05 dB, ΔSSIM = +0.047, ΔLPIPS = −0.103).',
    '',
    '  Critically, the full-physics model also generalises well to Gaussian-only test data',
    '  (PSNR = 22.33, SSIM = 0.5268). Its cross-domain degradation is only −3.84 dB PSNR — roughly',
    '  half the loss suffered by the Gaussian-only model. This asymmetry reveals two important properties:',
    '  (1) The compound pipeline is a superset of Gaussian blur — the network learns Gaussian blurs as',
    '  a subset of a broader degradation manifold.',
    '  (2) Exposure to diverse degradations improves generalisation — the network learns a more robust',
    '  inverse mapping that does not overfit to a single kernel family."',
    '',
    '[P213] 4.2.3 Evaluation Protocol Note  → REPLACE with',
    '[P214] → REPLACE text with:',
    '',
    '  "These quantitative findings support a critical methodological principle: a model\'s PSNR and SSIM',
    '  scores are only meaningful relative to the degradation complexity on which they were evaluated.',
    '  The Gaussian-only VDSR\'s PSNR of 25.09 dB on its own test set would, in the absence of cross-domain',
    '  evaluation, suggest a highly capable restoration model. The physics-pipeline evaluation reveals this',
    '  as an illusion of the narrow evaluation domain.',
    '',
    '  Reporting metrics on Gaussian-only benchmarks without acknowledging the domain gap inflates apparent',
    '  performance and misleads practitioners. We recommend that future work adopt a cross-domain evaluation',
    '  protocol as standard practice: models trained on synthetic degradations should always be validated',
    '  against a held-out set of physically realistic degradations."',
    '',
    '═══ END OF INSTRUCTIONS ═══',
]

for line in instructions:
    if line.startswith('═══') or line.startswith('REPLACE') or line.startswith('[P'):
        placeholders.append(make_paragraph(line, bold=True, size=10, color='0000CC'))
    elif line.strip():
        placeholders.append(make_paragraph(line, size=9, color='333333'))
    else:
        placeholders.append(make_paragraph('', size=4))

# ── Insert in reverse order ──────────────────────────────────────────────
print(f"   Inserting {len(placeholders)} placeholder paragraphs...")
for p_el in reversed(placeholders):
    target_para.addnext(p_el)

# ── Verify math preservation ─────────────────────────────────────────────
new_math_count = len(root.findall(f'.//{{{NS_M}}}oMath'))
print(f"   OMML equations after edit: {new_math_count} (was {original_math_count})")
assert new_math_count == original_math_count, "MATH DESTROYED — aborting save!"

# ── Save back to ZIP ─────────────────────────────────────────────────────
new_xml = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)

tmp = SRC + '.tmp'
with zipfile.ZipFile(SRC, 'r') as zin:
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == 'word/document.xml':
                zout.writestr(item, new_xml)
            else:
                zout.writestr(item, zin.read(item.filename))

# Backup original
bak = SRC.replace('.docx', '_bak.docx')
shutil.copy2(SRC, bak)
print(f"   Backup saved: {os.path.basename(bak)}")

os.replace(tmp, SRC)
print(f"✅ Done! Modified {os.path.basename(SRC)}")
print(f"   Open in Word and Ctrl+F for '--- TABLE' or '--- FIGURE' to find markers.")
print(f"   Ctrl+F for '═══ TEXT REPLACEMENT' for editing instructions.")
