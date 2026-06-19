"""
Replace §4.2 text in paper_v2_fixed_edited.docx with ablation data.

Starts from backup (clean state), applies all text replacements and inserts
the ablation table + figure placeholder.

Usage:
  cd /mnt/c/Hakimi/Internship/GUI Image Restoration
  ./app/torch_env/Scripts/python -X utf8 scripts-ablations/image-restoration/apply_ablation_to_docx.py
"""

import os, shutil, zipfile, copy
from lxml import etree
import docx
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.text.paragraph import Paragraph

SRC = os.path.normpath('research/output/paper_v2_fixed_edited_bak.docx')
DST = os.path.normpath('research/output/paper_v2_fixed_ablation_updated.docx')

NS_W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS_M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'

W = lambda tag: f'{{{NS_W}}}{tag}'
M = lambda tag: f'{{{NS_M}}}{tag}'


def has_math(p_element):
    """Check if a paragraph XML element contains OMML equations."""
    return len(p_element.findall(f'.//{{{NS_M}}}oMath')) > 0


def get_non_math_text(p_element):
    """Get text from a paragraph, excluding math elements."""
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


def set_non_math_text_safe(p_element, new_text):
    """Set text of a paragraph safely — only modifies non-math w:t elements."""
    # Collect non-math w:t elements
    non_math_ts = []
    for t in p_element.iter(W('t')):
        parent = t.getparent()
        in_math = False
        while parent is not None:
            if parent.tag in (M('oMath'), M('oMathPara')):
                in_math = True
                break
            parent = parent.getparent()
        if not in_math:
            non_math_ts.append(t)

    if not non_math_ts:
        return False

    if len(non_math_ts) == 1:
        non_math_ts[0].text = new_text
    else:
        non_math_ts[0].text = new_text
        for t in non_math_ts[1:]:
            t.text = ''
    return True


def replace_paragraph_text(paragraph, new_text):
    """Replace ALL paragraph content with new plain text.
    
    This is the NUCLEAR option: removes all children (including math elements)
    and creates a fresh text run. Use ONLY when the new text is plain text
    that doesn't reference any equations. For paragraphs that need equations,
    use set_non_math_text_safe instead.
    """
    p_element = paragraph._element
    # Remove all children (pPr, r, oMath, etc.)
    for child in list(p_element):
        tag_local = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        if tag_local != 'pPr':  # Keep paragraph properties (style, etc.)
            p_element.remove(child)
    
    # Create a fresh text run
    r = etree.SubElement(p_element, W('r'))
    t = etree.SubElement(r, W('t'))
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    t.text = new_text
    return True


# ═══════════════════════════════════════════════════════════════════════════
# 1. Modify the DOCX text using python-docx
# ═══════════════════════════════════════════════════════════════════════════

print(f"📄 Reading: {SRC}")
doc = docx.Document(SRC)
paras = doc.paragraphs
total = len(paras)
print(f"   Total paragraphs: {total}")

# ── Build paragraph index mapping ─────────────────────────────────────────
para_map = {}
for i, p in enumerate(paras):
    text = p.text.strip()
    if '4.2 Experiment' in text:
        para_map['heading_42'] = i
    elif '4.2.1 Phase' in text:
        para_map['heading_421'] = i
    elif '4.2.2 Phase' in text:
        para_map['heading_422'] = i
    elif '4.2.3' in text:
        para_map['heading_423'] = i
    elif '4.3 Experiment 2' in text:
        para_map['heading_43'] = i

print(f"   Found: {para_map}")

# ── Replace §4.2 headings & body text ─────────────────────────────────────

if 'heading_421' in para_map:
    # Rename heading from "The Gaussian-Only Catastrophe"
    idx = para_map['heading_421']
    replace_paragraph_text(paras[idx], '4.2.1 The Gaussian-Only Domain Gap')

if 'heading_422' in para_map:
    # Rename heading from "Phase 2 — Physics-Based Resolution"
    idx = para_map['heading_422']
    replace_paragraph_text(paras[idx], '4.2.2 The Physics-Based Advantage')

if 'heading_423' in para_map:
    # Rename from "Evaluation Protocol Note"
    idx = para_map['heading_423']
    replace_paragraph_text(paras[idx], '4.2.3 Methodological Implication')

# ── Replace body paragraphs ───────────────────────────────────────────────

# §4.2.1 body (P205-P207 in backup)
if 'heading_421' in para_map:
    h421 = para_map['heading_421']

    new_body_421 = [
        "To quantitatively isolate the effect of training degradation type from architectural choice, "
        "we performed a controlled ablation experiment using the VDSR architecture (20-layer residual "
        "network, 0.67M parameters). Two models were trained with identical hyperparameters (20 epochs, "
        "hybrid 0.8 MSE + 0.2 L1 loss, batch size 8, gradient accumulation 2, Adam optimiser with cosine "
        "annealing from 10\u207b\u2074 to 10\u207b\u2076) on the same 6,842-image dataset \u2014 the sole difference being the "
        "degradation applied during training:",

        "1. Gaussian-only regime: blur generated by a single Gaussian kernel with continuous sigma "
        "sampling (\u03c3 \u223c U(0.5, 5.0)), no compound degradation.",

        "2. Full physics regime: compound pipeline (out-of-focus disk blurs, motion streaks, space-variant "
        "focal gradients, LED banding, sensor noise, vignetting, and JPEG compression) as described in \u00a73.2.2.",
    ]

    # Replace P205-P207
    for j, new_text in enumerate(new_body_421):
        pi = h421 + 1 + j
        if pi < total:
            replace_paragraph_text(paras[pi], new_text)
            print(f"   [P{pi}] Replaced with §4.2.1 body #{j+1}")

# §4.2.2 body (P209-P212 in backup)
if 'heading_422' in para_map:
    h422 = para_map['heading_422']

    new_body_422 = [
        "The Gaussian-only VDSR achieves a PSNR of 25.09 dB on its native Gaussian test set \u2014 a strong "
        "result by conventional restoration benchmarks. However, when this same model encounters the "
        "physics-pipeline test set (which contains compound degradation unseen during training), its PSNR "
        "drops to 17.44 dB \u2014 a degradation of \u22127.65 dB. The structural fidelity follows a similar trajectory: "
        "SSIM falls from 0.6121 to 0.3909, while LPIPS more than doubles from 0.1962 to 0.4336.",

        "This cross-domain performance collapse constitutes the quantifiable expression of the domain gap: "
        "a network trained on a single degradation type cannot generalise to the physically realistic "
        "compound degradations encountered in practice. The network has learned a narrow inverse mapping "
        "optimised for Gaussian blurs; when presented with out-of-focus disk kernels, motion streaks, or "
        "space-variant focal gradients, its residual prediction becomes an unreliable reconstruction.",

        "The full-physics VDSR, trained on the compound degradation pipeline, exhibits markedly greater "
        "cross-domain robustness. On the physics-pipeline test set it achieves PSNR = 18.49 dB, "
        "SSIM = 0.4378, LPIPS = 0.3308 \u2014 consistently outperforming the Gaussian-only model on every "
        "metric when both are evaluated on physically realistic degradation (\u0394PSNR = +1.05 dB, "
        "\u0394SSIM = +0.047, \u0394LPIPS = \u22120.103).",

        "Critically, the full-physics model also generalises well to Gaussian-only test data "
        "(PSNR = 22.33, SSIM = 0.5268). Its cross-domain degradation is only \u22123.84 dB PSNR \u2014 roughly "
        "half the loss suffered by the Gaussian-only model. This asymmetry reveals two important properties: "
        "(1) The compound pipeline is a superset of Gaussian blur \u2014 the network learns Gaussian blurs as "
        "a subset of a broader degradation manifold. (2) Exposure to diverse degradations improves "
        "generalisation \u2014 the network learns a more robust inverse mapping that does not overfit to a "
        "single kernel family.",
    ]

    for j, new_text in enumerate(new_body_422):
        pi = h422 + 1 + j
        if pi < total:
            replace_paragraph_text(paras[pi], new_text)
            print(f"   [P{pi}] Replaced with §4.2.2 body #{j+1}")

# §4.2.3 body (P214 in backup)
if 'heading_423' in para_map:
    h423 = para_map['heading_423']

    new_body_423 = (
        "These quantitative findings support a critical methodological principle: a model\u2019s PSNR and "
        "SSIM scores are only meaningful relative to the degradation complexity on which they were evaluated. "
        "The Gaussian-only VDSR\u2019s PSNR of 25.09 dB on its own test set would, in the absence of cross-domain "
        "evaluation, suggest a highly capable restoration model. The physics-pipeline evaluation reveals this "
        "as an illusion of the narrow evaluation domain. Reporting metrics on Gaussian-only benchmarks without "
        "acknowledging the domain gap inflates apparent performance and misleads practitioners about "
        "real-world applicability. We recommend that future work in image restoration for biological microscopy "
        "adopt a cross-domain evaluation protocol as standard practice: models trained on synthetic degradations "
        "should always be validated against a held-out set of physically realistic degradations drawn from a "
        "complementary degradation family."
    )

    pi = h423 + 1
    if pi < total:
        replace_paragraph_text(paras[pi], new_body_423)
        print(f"   [P{pi}] Replaced with §4.2.3 body")

# ── Clean up orphan P208 body text ────────────────────────────────────────
# The old text at P208 body needs to be removed
if 'heading_422' in para_map:
    h422 = para_map['heading_422']
    # Check if there's a paragraph between heading_421 body end and heading_422
    # that might have orphan text
    if 'heading_421' in para_map:
        h421 = para_map['heading_421']
        for j in range(h421 + 4, h422):
            if j < total and paras[j].text.strip():
                replace_paragraph_text(paras[j], '')
                print(f"   [P{j}] Cleared orphan text")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Insert ablation table + figure placeholder
# ═══════════════════════════════════════════════════════════════════════════

if 'heading_423' in para_map:
    # Insert AFTER the last §4.2 paragraph (which is P214, the §4.2.3 body)
    h423_body = para_map['heading_423'] + 1
    insert_before_idx = h423_body
    insert_before = paras[insert_before_idx]
    print(f"\n   Inserting table and figure after P{insert_before_idx} (end of §4.2.3, before §4.3)")

    # Helper: make a formatted paragraph element
    def make_oxml_para(text, bold=False, italic=False, size=10, color=None, style=None):
        p_el = OxmlElement('w:p')
        if style:
            pPr = OxmlElement('w:pPr')
            pStyle = OxmlElement('w:pStyle')
            pStyle.set(qn('w:val'), style)
            pPr.append(pStyle)
            p_el.append(pPr)
        r_el = OxmlElement('w:r')
        rpr = OxmlElement('w:rPr')
        sz = OxmlElement('w:sz')
        sz.set(qn('w:val'), str(size * 2))
        rpr.append(sz)
        szCs = OxmlElement('w:szCs')
        szCs.set(qn('w:val'), str(size * 2))
        rpr.append(szCs)
        if bold:
            rpr.append(OxmlElement('w:b'))
        if italic:
            rpr.append(OxmlElement('w:i'))
        if color:
            clr = OxmlElement('w:color')
            clr.set(qn('w:val'), color)
            rpr.append(clr)
        r_el.append(rpr)
        t = OxmlElement('w:t')
        t.set(qn('xml:space'), 'preserve')
        t.text = text
        r_el.append(t)
        p_el.append(r_el)
        return p_el

    def insert_after(para, p_el):
        para._element.addnext(p_el)
        return Paragraph(p_el, para._element.getparent())

    # Insert in REVERSE order so they appear in correct sequence
    # Insertion order (reversed): spacer3 → figure caption → figure marker → spacer2 → table caption → table marker → spacer1

    # Spacer before table
    insert_after(insert_before, make_oxml_para('', size=8))

    # Figure caption
    insert_after(insert_before, make_oxml_para(
        'Paste research/figures/ablation_domain_gap_comparison.png here. '
        'Right-click \u2192 Insert Picture, resize to fit column width.',
        italic=True, size=9, color='999999'
    ))
    insert_after(insert_before, make_oxml_para(
        'Figure 4.X: Domain gap ablation comparison \u2014 VDSR on a physics-pipeline test image. '
        'Left to right: Blurred input, Gaussian-only VDSR output (PSNR 17.44 / SSIM 0.3909), '
        'Full-physics VDSR output (PSNR 18.49 / SSIM 0.4378), Ground truth.',
        italic=True, size=9, color='666666'
    ))
    insert_after(insert_before, make_oxml_para(
        '--- FIGURE 4.X: research/figures/ablation_domain_gap_comparison.png ---',
        bold=True, color='CC0000', size=10
    ))

    # Spacer before figure
    insert_after(insert_before, make_oxml_para('', size=8))

    # Table caption
    insert_after(insert_before, make_oxml_para(
        'Table 4.2: Controlled VDSR ablation comparing Gaussian-only vs. full physics training regimes. '
        'Both models trained for 20 epochs with identical hyperparameters. The domain gap is quantified as '
        'the cross-domain performance drop when a model is evaluated on a degradation type unseen during training.',
        italic=True, size=9
    ))

    # Table marker
    insert_after(insert_before, make_oxml_para(
        '--- TABLE 4.2: Ablation Domain Gap Results ---',
        bold=True, color='CC0000', size=10
    ))

    # Spacer before table
    insert_after(insert_before, make_oxml_para('', size=8))

    print("   ✅ Table, figure markers, and spacers inserted")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Save
# ═══════════════════════════════════════════════════════════════════════════

# Save to temp, then copy over
tmp = DST + '.tmp'
doc.save(tmp)
shutil.copy2(tmp, DST)
os.remove(tmp)

print(f"\n✅ Saved: {DST}")
print(f"   ℹ️  Backup preserved at: {SRC}")
print()
print("📋 NEXT STEPS:")
print("   1. Open research/output/paper_v2_fixed_edited.docx in Word")
print("   2. Scroll to §4.2 — the text should be updated with ablation data")
print("   3. Find '--- TABLE 4.2' marker and rebuild the table:")
print("      Insert → Table (5 columns × 5 rows) and fill in the data")
print("   4. Find '--- FIGURE 4.X' marker and insert:")
print("      research/figures/ablation_domain_gap_comparison.png")
print("   5. Delete the marker lines after inserting content")
