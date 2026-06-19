"""
Move Table 4.2 + Figure 4.2 from after §4.2.3 (where they currently sit) to
between §4.2.1 body end and §4.2.2 heading.
"""
import os, zipfile, copy
from lxml import etree

SRC = os.path.normpath('research/output/paper_v2_fixed_ablation_updated.docx')
NS_W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS_R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
W = lambda t: f'{{{NS_W}}}{t}'

print(f"Reading: {SRC}")
with zipfile.ZipFile(SRC, 'r') as z:
    xml_raw = z.read('word/document.xml')
root = etree.fromstring(xml_raw)
body = root.find(W('body'))
body_children = list(body)

# Find key content by full text search across ALL descendants
def get_full_text(el):
    texts = []
    for t in el.iter(W('t')):
        texts.append(t.text or '')
    return ''.join(texts).strip()

# 1. Find all tables and their context
tbls = body.findall(f'.//{W("tbl")}')
print(f"\nTables found: {len(tbls)}")
for j, tbl in enumerate(tbls):
    first_text = get_full_text(tbl)[:60]
    print(f"  Table #{j}: first text = \"{first_text}\"")

# 2. Find the table with "Training Regime" 
target_tbl = None
for tbl in tbls:
    if 'Training Regime' in get_full_text(tbl):
        target_tbl = tbl
        break

if target_tbl is None:
    print("❌ Could not find ablation table")
    exit(1)

# 3. Find anchor (end of §4.2.1 body: "2. Full physics regime...")
anchor = None
for p in body.iter(W('p')):
    if '2. Full physics regime' in get_full_text(p):
        anchor = p
        break

if anchor is None:
    print("❌ Could not find anchor paragraph")
    exit(1)

# 4. Find §4.2.2 heading (we'll insert before this if it comes after anchor)
heading_422 = None
after_anchor = False
for p in body.iter(W('p')):
    if p == anchor:
        after_anchor = True
        continue
    if after_anchor and '4.2.2 The Physics-Based' in get_full_text(p):
        heading_422 = p
        break

# 5. Collect elements to move (table + its following siblings until §4.3)
# Find the table's position in body children
tbl_sibling_idx = None
for i, child in enumerate(body):
    if child == target_tbl:
        tbl_sibling_idx = i
        break

if tbl_sibling_idx is None:
    print("❌ Could not find table position")
    exit(1)

# Collect all elements from table through figure caption
move_elements = []
for i in range(tbl_sibling_idx, min(tbl_sibling_idx + 15, len(body))):
    el = body[i]
    tag = el.tag.split('}')[-1] if '}' in el.tag else el.tag
    
    # Stop if we hit §4.3 heading
    if tag == 'p' and '4.3 Experiment 2' in get_full_text(el):
        break
    
    # Skip proofErr and bookmark elements
    if tag in ('proofErr', 'bookmarkStart', 'bookmarkEnd', 'rPr', 'rFonts'):
        continue
    
    move_elements.append(el)
    
    # Stop after figure caption
    if tag == 'p' and 'Figure 4.2' in get_full_text(el):
        # Include the caption and maybe a trailing spacer
        continue
    
    # Also stop if we encounter another table
    if tag == 'tbl' and el != target_tbl:
        break

print(f"\nMoving {len(move_elements)} element(s):")
for el in move_elements:
    tag = el.tag.split('}')[-1] if '}' in el.tag else el.tag
    text = get_full_text(el)[:60]
    print(f"  <{tag}> \"{text}\"")

# 6. Detach from current position
for el in move_elements:
    body.remove(el)

# 7. Insert after anchor
# Use addnext on anchor in reverse order
for el in reversed(move_elements):
    anchor.addnext(el)

print("\n✅ Move complete!")
print(f"   Table + figure now between \"2. Full physics regime\" and §4.2.2 heading")

# Verify
print("\nVerifying placement...")
after_anchor = False
found_422 = False
for p in body.iter(W('p')):
    text = get_full_text(p)
    if p == anchor:
        after_anchor = True
        print(f"  [ANCHOR] \"{text[:50]}\"")
        continue
    if after_anchor and not found_422:
        if 'Training Regime' in text:
            print(f"  [TABLE FOLLOWS]")
        elif '4.2.2 The Physics-Based' in text:
            print(f"  [§4.2.2] \"{text[:50]}\"")
            found_422 = True
        elif 'Controlled VDSR ablation' in text:
            print(f"  [CAPTION] \"{text[:50]}\"")
        elif 'Figure 4.2' in text:
            print(f"  [FIG CAPTION] \"{text[:50]}\"")

# Save
new_xml = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
tmp = SRC + '.tmp'
with zipfile.ZipFile(SRC, 'r') as zin:
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == 'word/document.xml':
                zout.writestr(item, new_xml)
            else:
                zout.writestr(item, zin.read(item.filename))
os.replace(tmp, SRC)
print(f"\n✅ Saved to {SRC}")
