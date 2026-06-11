"""Insert placeholder paragraphs for four figures into paper_v2_fixed.docx.
Insertion strategy (bottom-up to preserve paragraph indices):
  - After P192 (end of Table 4.1 discussion) → Figure 4.1 (metrics comparison bar chart)
  - After P220 (end of Pareto summary, before §4.4) → Figure 4.2 (visual comparison)
  - After P231 (end of §4.4.3 SwinIR) → Figure 4.2a (zoomed detail)
  - After P244 (Figure 4.3 caption) → Figure 4.3 (hallucination power spectrum)

We use unique marker strings that are easy to Ctrl+F in Word.
"""
import sys, docx, copy
from docx.shared import Pt, RGBColor
from lxml import etree

src = sys.argv[1]
out = sys.argv[2]

doc = docx.Document(src)

def make_paragraph(doc, text, bold=False, italic=False, color=None, size=11):
    """Create a new paragraph element with the given formatting."""
    p_el = docx.oxml.OxmlElement('w:p')
    r_el = docx.oxml.OxmlElement('w:r')
    rpr = docx.oxml.OxmlElement('w:rPr')
    
    # Font size
    sz = docx.oxml.OxmlElement('w:sz')
    sz.set(docx.oxml.ns.qn('w:val'), str(size * 2))  # half-points
    rpr.append(sz)
    
    if bold:
        b = docx.oxml.OxmlElement('w:b')
        rpr.append(b)
    if italic:
        i = docx.oxml.OxmlElement('w:i')
        rpr.append(i)
    if color:
        c = docx.oxml.OxmlElement('w:color')
        c.set(docx.oxml.ns.qn('w:val'), color)
        rpr.append(c)
    
    r_el.append(rpr)
    t = docx.oxml.OxmlElement('w:t')
    t.set(docx.oxml.ns.qn('xml:space'), 'preserve')
    t.text = text
    r_el.append(t)
    p_el.append(r_el)
    return p_el

def insert_after(para, p_el):
    """Insert an OXML paragraph element after the given paragraph."""
    para._element.addnext(p_el)
    # Wrap in a Paragraph object so we can return it for chaining
    from docx.text.paragraph import Paragraph
    new_p = Paragraph(p_el, para._element.getparent())
    return new_p

# Track where we are — we insert from bottom to top to preserve indices
sections = []

# ── Figure 4.3: hallucination_power_spectrum.png (after P244) ──
sections.append({
    'after_para_index': 244,
    'marker': '--- FIGURE 4.3: research/hallucination_power_spectrum.png ---',
    'caption': 'Figure 4.3: Radially averaged 2D power spectra (log scale) comparing ground-truth clear images with VDSR-, SwinIR-, and Real-ESRGAN-restored outputs across 50 test images. Shaded regions denote ±1σ. All restoration models produce less high-frequency energy than ground truth, with VDSR preserving the most (ratio 0.14×). SwinIR and Real-ESRGAN exhibit nearly identical profiles (0.09×), demonstrating that residual skip connections and YCrCb luminance transfer effectively constrain spectral hallucination.',
    'important_note': 'Paste hallucination_power_spectrum.png here. Description: Mean radially averaged 2D FFT power spectra (log₁₀ scale) for ground-truth clear images (black) vs. VDSR (blue dashed), SwinIR (green dash-dot), and Real-ESRGAN (red dotted). X-axis: normalised spatial frequency [0, 0.5]; Y-axis: log power. All models under-generate high frequencies — the opposite of hallucination — confirming that architectural safeguards in this work suppress adversarial feature synthesis at the cost of reduced spectral energy.'
})

# ── Figure 4.2a: visual_comparison_zoomed.png (after §4.4.3 SwinIR analysis, before §4.4.4) ──
# Insert at P231 (end of SwinIR analysis) → best spot for zoomed detail
sections.append({
    'after_para_index': 231,
    'marker': '--- FIGURE 4.2a: research/visual_comparison_zoomed.png ---',
    'caption': 'Figure 4.2a: Zoomed 200×200 px centre crops of the same test image (GIA030001) from Figure 4.2, highlighting structural differences between restoration architectures. Blurry input (left) shows fused cell-wall boundaries; VDSR recovers coarse anatomical outlines but with residual blur; SwinIR produces cleanly separated vessel lumina and continuous fibre cell walls; Real-ESRGAN introduces a characteristic "grit" texture; Ground Truth (right) shows the true anatomical microstructure.',
    'important_note': 'Paste visual_comparison_zoomed.png here. Description: 2×5 grid. Top row: full-size panels (Blurry Input, VDSR, SwinIR, Real-ESRGAN, Ground Truth) with red rectangle showing crop region. Bottom row: corresponding 200×200 px centre crops at higher magnification for detailed comparison of cell-wall and vessel structure.'
})

# ── Figure 4.2: visual_comparison.png (before §4.4, after P220) ──
sections.append({
    'after_para_index': 220,
    'marker': '--- FIGURE 4.2: research/visual_comparison.png ---',
    'caption': 'Figure 4.2: Side-by-side comparison of restoration outputs for a representative test image (GIA030001). From left to right: blurry input (VoL = 419), VDSR (VoL = 436), SwinIR (VoL = 53), Real-ESRGAN (VoL = 1,302), and ground-truth clear image (VoL = 1,023). Note that SwinIR produces the most perceptually faithful restoration despite the lowest VoL score — consistent with the Metric Illusion argument in §1.4.',
    'important_note': 'Paste visual_comparison.png here. Description: Five-panel horizontal montage (Blurry Input, VDSR, SwinIR, Real-ESRGAN, Ground Truth) of wood cross-section GIA030001. Illustrates the qualitative differences in restoration fidelity across architectures. SwinIR shows the best balance of structural continuity and natural texture.'
})

# ── Figure 4.1: metrics_comparison.png (after P192, before §4.2) ──
sections.append({
    'after_para_index': 192,
    'marker': '--- FIGURE 4.1: research/metrics_comparison.png ---',
    'caption': 'Figure 4.1: Comparative bar charts of PSNR (left), SSIM (centre), and LPIPS (right) across all five architectures using best 50-epoch weights under hybrid loss (MSE for Simple CNN). Dashed horizontal lines show Wiener/RL classical baseline performance (PSNR ≈ 3.56 dB, SSIM ≈ 0.006, LPIPS ≈ 0.832). Real-ESRGAN achieves the highest numerical scores, while SwinIR offers the best Pareto-optimal trade-off among non-generative architectures.',
    'important_note': 'Paste metrics_comparison.png here. Description: Three-panel horizontal bar chart. Left panel: PSNR (dB) bars for Simple CNN (17.74), SRCNN (19.37), VDSR (18.75), SwinIR (21.18), Real-ESRGAN (21.22) with Wiener/RL baseline at 3.56 dB. Centre panel: SSIM bars with baseline at 0.006. Right panel: LPIPS bars with baseline at 0.832. Colour scheme: teal, orange, red, blue, purple.'
})


# Insert from bottom to top so earlier indices stay valid
sections.sort(key=lambda x: x['after_para_index'], reverse=True)

for sec in sections:
    idx = sec['after_para_index']
    para = doc.paragraphs[idx]
    # Insert in reverse order so they appear in correct sequence
    # spacer2 → note → caption → marker → spacer1 (reverse of desired top-to-bottom)
    insert_after(para, make_paragraph(doc, '', size=8))  # spacer2 (last after this)
    insert_after(para, make_paragraph(doc, sec['important_note'], italic=True, color='666666', size=9))
    insert_after(para, make_paragraph(doc, sec['caption'], italic=True, size=10))
    insert_after(para, make_paragraph(doc, sec['marker'], bold=True, color='CC0000'))
    insert_after(para, make_paragraph(doc, '', size=8))  # spacer1 (first after para)

doc.save(out)
print(f"✅ Saved: {out}")
print(f"\nInserted {len(sections)} figure markers at the following locations:")
for sec in reversed(sections):
    print(f"  After P{sec['after_para_index']}: {sec['marker']}")
