#!/usr/bin/env python
"""
WORK AT XML LEVEL to preserve equation (OMML) objects.
python-docx's run-clearing destroys m:oMath elements.
This script directly edits w:t text elements in document.xml
without touching any math elements.
"""

from lxml import etree
import zipfile
import shutil
import re
import os
import tempfile

SRC = r'C:\Hakimi\Internship\GUI Image Restoration\research\output\paper.docx'
DST = r'C:\Hakimi\Internship\GUI Image Restoration\research\output\paper.docx'

NS_W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS_M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
NS_R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'

NAMESPACES = {
    'w': NS_W,
    'm': NS_M,
    'r': NS_R,
}


def fix_text(text):
    """Apply all text fixes to a plain string."""
    # === 1. Remove remaining "Chapter X: " from headings ===
    text = re.sub(r'\bChapter\s+(\d+):\s*', r'\1. ', text)
    
    # === 2. Remove "End of Chapter X" markers ===
    text = re.sub(r'End of Chapter\s+\d+\s*.*', '', text)
    text = re.sub(r'End of Thesis Document\s*.*', '', text)
    
    # === 3. "this thesis" -> "this paper" / "this study" ===
    text = re.sub(r'\b[Tt]he thesis\b', r'this paper', text)
    text = re.sub(r'\bThis thesis\b', r'This paper', text)
    text = re.sub(r'\bthis thesis\b', r'this paper', text)
    text = re.sub(r"\bthesis's\b", "paper's", text)
    text = re.sub(r'Thesis Document', 'Paper', text)
    text = re.sub(r'Thesis Roadmap', 'Study Roadmap', text)
    text = re.sub(r'remainder of this thesis', 'remainder of this paper', text)
    
    # === 4. "Chapter X" references in body -> "Section X" ===
    text = re.sub(r'\bChapter\s+(\d+)\b', r'Section \1', text)
    
    # === 5. Section 3.6: fix semicolons ===
    text = re.sub(r'(Camera ISP \(gamma correction\));\s*Early', r'\1: early', text, flags=re.IGNORECASE)
    text = re.sub(r'(LED banding);\s*Models', r'\1: models', text, flags=re.IGNORECASE)
    text = re.sub(r'(Sensor noise);\s*Read', r'\1: read', text, flags=re.IGNORECASE)
    
    # === 6. Code spacing: nn. MSELoss -> nn.MSELoss etc ===
    text = re.sub(r'nn\.\s+MSELoss', 'nn.MSELoss', text)
    text = re.sub(r'nn\.\s+L1Loss', 'nn.L1Loss', text)
    text = re.sub(r'nn\.\s+Conv2d', 'nn.Conv2d', text)
    text = re.sub(r'cv2\.\s+Laplacian', 'cv2.Laplacian', text)
    text = re.sub(r'cv2\.\s+copyMakeBorder', 'cv2.copyMakeBorder', text)
    text = re.sub(r'cv2\.\s+INTER_LANCZOS4', 'cv2.INTER_LANCZOS4', text)
    text = re.sub(r'cv2\.\s+GaussianBlur', 'cv2.GaussianBlur', text)
    text = re.sub(r'cv2\.\s+convertScaleAbs', 'cv2.convertScaleAbs', text)
    text = re.sub(r'torch\.\s+amp', 'torch.amp', text)
    text = re.sub(r'torch\.\s+nn', 'torch.nn', text)
    text = re.sub(r'torch\.\s+manual_seed', 'torch.manual_seed', text)
    text = re.sub(r'torch\.\s+utils', 'torch.utils', text)
    text = re.sub(r'torch\.\s+optim', 'torch.optim', text)
    text = re.sub(r'np\.\s+random', 'np.random', text)
    text = re.sub(r'skimage\.\s+metrics', 'skimage.metrics', text)
    
    # === 7. Author block ===
    text = re.sub(r'Paper:\s*CAIRO Lab,\s*2026\s*Tualang Image Restoration Initiative',
                  '[Author names to be added]\n[Author affiliations to be added]', text)
    
    # === 8. Clean up double spaces ===
    text = re.sub(r'  +', ' ', text)
    
    return text


def process_docx():
    """Edit document.xml directly at XML level, preserving math elements."""
    
    # Read the docx
    with zipfile.ZipFile(SRC, 'r') as zin:
        doc_xml_raw = zin.read('word/document.xml')
    
    root = etree.fromstring(doc_xml_raw)
    
    # Find all paragraphs
    paragraphs = root.findall(f'.//{{{NS_W}}}p')
    total_fixes = 0
    eq_para_fixes = 0
    heading_fixes = 0
    
    for para in paragraphs:
        # Get paragraph style
        ppr = para.find(f'{{{NS_W}}}pPr')
        pstyle = ppr.find(f'{{{NS_W}}}pStyle') if ppr is not None else None
        style_val = pstyle.get(f'{{{NS_W}}}val') if pstyle is not None else ''
        
        # Check if this paragraph contains math elements
        has_math = para.find(f'.//{{{NS_M}}}oMath') is not None or \
                   para.find(f'.//{{{NS_M}}}oMathPara') is not None
        
        # Get ALL w:t elements
        all_t_elems = para.findall(f'.//{{{NS_W}}}t')
        
        # Separate math and non-math text elements
        non_math_t = []
        for te in all_t_elems:
            # Walk up to see if inside math
            p = te.getparent()
            inside_math = False
            while p is not None:
                if p.tag == f'{{{NS_M}}}oMath' or p.tag == f'{{{NS_M}}}oMathPara':
                    inside_math = True
                    break
                p = p.getparent()
            if not inside_math:
                non_math_t.append(te)
        
        if not non_math_t:
            continue
        
        # Get full non-math text
        texts = [t.text or '' for t in non_math_t]
        full_text = ''.join(texts)
        
        if not full_text.strip():
            continue
        
        # Apply fixes
        new_full = fix_text(full_text)
        
        if new_full == full_text:
            continue
        
        # Redistribute corrected text back to non-math w:t elements
        if len(non_math_t) == 1:
            non_math_t[0].text = new_full
        else:
            new_strings = rebuild_text_with_formatting(texts, new_full)
            for t_elem, new_text in zip(non_math_t, new_strings):
                t_elem.text = new_text
        
        total_fixes += 1
        if has_math:
            eq_para_fixes += 1
    
    # Write back
    new_xml = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
    
    # Replace document.xml in the zip
    tmp_dst = DST + '.tmp'
    with zipfile.ZipFile(SRC, 'r') as zin:
        with zipfile.ZipFile(tmp_dst, 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == 'word/document.xml':
                    zout.writestr(item, new_xml)
                else:
                    zout.writestr(item, zin.read(item.filename))
    
    # Replace original
    os.replace(tmp_dst, DST)
    
    print(f'Fixed {total_fixes} paragraphs')
    print(f'  (of which {eq_para_fixes} contained math elements)')


def rebuild_text_with_formatting(old_strings, new_full):
    """
    Given old text segments and a new full text, redistribute the new text
    across the segments preserving original formatting anchors.
    Only used for non-math paragraphs where we can freely reassign text.
    """
    if len(old_strings) <= 1:
        return [new_full]
    
    # Simple approach: distribute text proportionally by character count
    total_old = sum(len(s) for s in old_strings)
    if total_old == 0:
        return [new_full]
    
    result = []
    pos = 0
    for i, old in enumerate(old_strings):
        if i == len(old_strings) - 1:
            result.append(new_full[pos:])
        else:
            # Keep original boundary by finding the closest space
            seg_len = int(len(old) / total_old * len(new_full))
            # Adjust to nearest space boundary
            if pos + seg_len < len(new_full):
                # Find next space after proposed break
                next_space = new_full.find(' ', pos + seg_len)
                if next_space > 0 and next_space - (pos + seg_len) < 10:
                    seg_len = next_space - pos
            result.append(new_full[pos:pos + seg_len])
        pos += len(result[-1])
    
    # Ensure we didn't lose any characters
    if pos < len(new_full):
        result[-1] = result[-1] + new_full[pos:]
    
    return result


def main():
    process_docx()
    
    # Quick verify
    doc = __import__('docx').Document(DST)
    print()
    for i, p in enumerate(doc.paragraphs):
        t = p.text.strip()
        if 'Chapter' in t and 'Heading' in getattr(p, 'style', '') and 'Heading' in str(p.style.name) if hasattr(p, 'style') else False:
            pass  # Just check for issues
        if 'Thesis Roadmap' in t:
            print(f'  [WARN] Thesis Roadmap in [{i}]: {t[:80]}')

if __name__ == '__main__':
    main()
