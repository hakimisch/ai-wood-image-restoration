#!/usr/bin/env python
"""
Fix remaining text gaps around equations and the author block.
Uses XML-level editing (lxml) to preserve OMML math elements.
"""

from lxml import etree
import zipfile
import os
import re

SRC = r'C:\Hakimi\Internship\GUI Image Restoration\research\output\paper.docx'
DST = r'C:\Hakimi\Internship\GUI Image Restoration\research\output\paper.docx'

NS_W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS_M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'

def is_inside_math(elem):
    """Check if an XML element is inside a math object."""
    p = elem.getparent()
    while p is not None:
        if p.tag == f'{{{NS_M}}}oMath' or p.tag == f'{{{NS_M}}}oMathPara':
            return True
        p = p.getparent()
    return False

def get_non_math_text(para):
    """Get the full text of non-math w:t elements in a paragraph."""
    texts = []
    for t in para.findall(f'.//{{{NS_W}}}t'):
        if not is_inside_math(t):
            texts.append(t.text or '')
    return ''.join(texts)

def set_non_math_text(para, new_full):
    """Set text for non-math w:t elements, preserving math objects."""
    non_math_ts = [t for t in para.findall(f'.//{{{NS_W}}}t') if not is_inside_math(t)]
    if not non_math_ts:
        return
    if len(non_math_ts) == 1:
        non_math_ts[0].text = new_full
    else:
        # Distribute text across multiple runs
        old_texts = [t.text or '' for t in non_math_ts]
        spaces = new_full.count(' ')
        if len(old_texts) <= 1 or spaces < 2:
            non_math_ts[0].text = new_full
            for t in non_math_ts[1:]:
                t.text = ''
        else:
            # Redistribute: give the first segment up to the first space boundary
            first_space = new_full.find(' ')
            if first_space > 0:
                non_math_ts[0].text = new_full[:first_space] + ' '
                non_math_ts[-1].text = new_full[first_space+1:]
                for t in non_math_ts[1:-1]:
                    t.text = ''


def main():
    # Read the docx
    with zipfile.ZipFile(SRC, 'r') as zin:
        doc_xml_raw = zin.read('word/document.xml')
    
    root = etree.fromstring(doc_xml_raw)
    paras = root.findall(f'.//{{{NS_W}}}p')
    fixes = 0
    
    for i, para in enumerate(paras):
        full = get_non_math_text(para)
        if not full.strip():
            continue
        
        original = full
        text = full
        
        # Fix 1: "the clean target (), then reconstructs" -> add variable description
        text = re.sub(
            r'the clean target\s*\(\),\s*then reconstructs the output as',
            'the clean target (residual = input - prediction), then reconstructs the output as',
            text
        )
        
        # Fix 2: "where,, and are query, key, and value"
        text = re.sub(
            r'where,\s*,\s*and are query, key, and value projections',
            'where Q, K, and V are query, key, and value projections',
            text
        )
        text = re.sub(
            r'where,\s*,?\s*,?\s*and are query',
            'where Q, K, and V are query',
            text
        )
        
        # Fix 3: "where  X  is" patterns (single-space variable gaps)
        text = re.sub(r'where is the discrete Laplacian', 'where L is the discrete Laplacian', text)
        text = re.sub(r'where is the mean squared error', 'where L_MSE is the mean squared error', text)
        text = re.sub(r'and is the mean absolute error', 'and L_L1 is the mean absolute error', text)
        text = re.sub(r'minimise the expected norm', 'minimise the expected L2 norm', text)
        text = re.sub(r'penalises the norm of the image gradient', 'penalises the L1 norm of the image gradient', text)
        
        # Fix 4: Author block
        text = re.sub(r'Paper:\s*CAIRO Lab,\s*2026\s*Tualang Image Restoration Initiative',
                      r'[Author names to be added]\n[Author affiliations to be added]', text)
        
        # Fix 5: "the clean target ()" standalone (without the trailing description)
        text = re.sub(r'the clean target \(\)', 'the clean target', text)
        
        # Fix 6: "computation of: min G max D E ..." — add description
        if 'two-player minimax game' in text.lower() and ':' not in text:
            text = text.rstrip('.') + ':'
        
        if text != original:
            set_non_math_text(para, text)
            fixes += 1
            print(f'  Fixed [{i}]: {original[:60]}...')
    
    # Write back
    new_xml = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
    
    tmp_dst = DST + '.tmp2'
    with zipfile.ZipFile(SRC, 'r') as zin:
        with zipfile.ZipFile(tmp_dst, 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == 'word/document.xml':
                    zout.writestr(item, new_xml)
                else:
                    zout.writestr(item, zin.read(item.filename))
    
    os.replace(tmp_dst, DST)
    print(f'\nFixed {fixes} paragraphs')


if __name__ == '__main__':
    main()
