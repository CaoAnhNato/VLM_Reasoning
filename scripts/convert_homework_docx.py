import os
import sys
import re

# Ensure UTF-8 output encoding for console
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

# Try to load latex2mathml and MML2OMML if available
_xslt_transform = None
try:
    import latex2mathml.converter
    from lxml import etree
    XSL_CANDIDATES = [
        r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL",
        r"C:\Program Files (x86)\Microsoft Office\root\Office16\MML2OMML.XSL",
        r"C:\Program Files\Microsoft Office\Office16\MML2OMML.XSL",
        r"C:\Program Files (x86)\Microsoft Office\Office16\MML2OMML.XSL",
    ]
    for p in XSL_CANDIDATES:
        if os.path.exists(p):
            _xslt_doc = etree.parse(p)
            _xslt_transform = etree.XSLT(_xslt_doc)
            print(f"[Math Engine] Loaded XSLT from: {p}")
            break
except Exception as e:
    print(f"[Math Engine] Warning: Could not initialize Math Engine: {e}")

def latex_to_omml(latex_code):
    """Convert LaTeX formula string to Word OMML XML Element."""
    if _xslt_transform is None:
        return None
    try:
        clean_code = latex_code.strip()
        clean_code = clean_code.replace(r'\longrightarrow', r'\rightarrow')
        clean_code = clean_code.replace(r'\longleftarrow', r'\leftarrow')
        clean_code = clean_code.replace(r'\mid', r'|')
        
        mathml_str = latex2mathml.converter.convert(clean_code)
        math_tree = etree.fromstring(mathml_str)
        omml = _xslt_transform(math_tree)
        root = omml.getroot()
        omml_xml_str = etree.tostring(root, encoding='unicode')
        return parse_xml(omml_xml_str)
    except Exception:
        return None

def set_cell_background(cell, fill_hex):
    """Set shading color for a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set cell margins in dxa (1 pt = 20 dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def set_table_borders(table, color="B0BEC5", sz="4", val="single"):
    """Apply clean borders to a table."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:left w:val="none"/>
            <w:right w:val="none"/>
            <w:insideV w:val="none"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)

def format_run_font(run, font_name="Times New Roman", size_pt=13, bold=False, italic=False, color_rgb=(0,0,0)):
    """Apply font properties and ensure XML font tags are correctly set."""
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(*color_rgb)
    
    # Set w:rFonts element for international / complex scripts
    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn('w:ascii'), font_name)
    rFonts.set(qn('w:hAnsi'), font_name)
    rFonts.set(qn('w:cs'), font_name)
    rFonts.set(qn('w:eastAsia'), font_name)

def add_formatted_text(paragraph, text, default_size=13, default_bold=False, default_italic=False, default_color=(0,0,0)):
    """Parse inline markdown (bold, italic, math, code) and add runs with proper formatting."""
    # Pattern for inline tokens:
    # 1. Block math: $$...$$
    # 2. Inline math: $...$
    # 3. Bold-italic: ***...*** or ___...___
    # 4. Bold: **...** or __...__
    # 5. Italic: *...* or _..._
    # 6. Inline code: `...`
    pattern = re.compile(
        r'(\$\$(?:\\.|[^\$])+\$\$|\$(?:\\.|[^\$])+\$|\*\*\*[^*]+\*\*\*|\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)'
    )
    parts = pattern.split(text)
    
    for part in parts:
        if not part:
            continue
        
        # Check tokens
        if part.startswith('$$') and part.endswith('$$'):
            math_code = part[2:-2]
            omml = latex_to_omml(math_code)
            if omml is not None:
                paragraph._p.append(omml)
            else:
                run = paragraph.add_run(f" {math_code} ")
                format_run_font(run, size_pt=default_size, italic=True, bold=default_bold)
        elif part.startswith('$') and part.endswith('$'):
            math_code = part[1:-1]
            omml = latex_to_omml(math_code)
            if omml is not None:
                paragraph._p.append(omml)
            else:
                run = paragraph.add_run(f" {math_code} ")
                format_run_font(run, size_pt=default_size, italic=True, bold=default_bold)
        elif part.startswith('***') and part.endswith('***'):
            run = paragraph.add_run(part[3:-3])
            format_run_font(run, size_pt=default_size, bold=True, italic=True, color_rgb=default_color)
        elif part.startswith('**') and part.endswith('**'):
            run = paragraph.add_run(part[2:-2])
            format_run_font(run, size_pt=default_size, bold=True, italic=default_italic, color_rgb=default_color)
        elif part.startswith('*') and part.endswith('*'):
            run = paragraph.add_run(part[1:-1])
            format_run_font(run, size_pt=default_size, bold=default_bold, italic=True, color_rgb=default_color)
        elif part.startswith('`') and part.endswith('`'):
            run = paragraph.add_run(part[1:-1])
            format_run_font(run, font_name="Consolas", size_pt=default_size - 1, bold=True, color_rgb=(20, 80, 160))
        else:
            run = paragraph.add_run(part)
            format_run_font(run, size_pt=default_size, bold=default_bold, italic=default_italic, color_rgb=default_color)

def convert_md_to_docx(md_path, docx_path):
    """Convert Markdown file to Word DOCX adhering strictly to required typography."""
    print(f"\n[Converting] {md_path} -> {docx_path}")
    
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    doc = Document()
    
    # 1. Set Page Margins
    # Trái 3cm, Phải 2cm, Trên 3cm, Dưới 3cm
    for section in doc.sections:
        section.top_margin = Cm(3.0)
        section.bottom_margin = Cm(3.0)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.0)
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        
    # 2. Configure Default Normal Style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(13)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.5
    normal_style.paragraph_format.space_after = Pt(4)
    normal_style.paragraph_format.space_before = Pt(0)
    
    rPr = normal_style.element.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn('w:ascii'), 'Times New Roman')
    rFonts.set(qn('w:hAnsi'), 'Times New Roman')
    rFonts.set(qn('w:cs'), 'Times New Roman')
    rFonts.set(qn('w:eastAsia'), 'Times New Roman')

    base_dir = os.path.dirname(os.path.abspath(md_path))
    
    i = 0
    in_code_block = False
    code_block_lines = []
    
    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.rstrip('\r\n')
        stripped = line.strip()
        
        # Check code block / mermaid
        if stripped.startswith('```'):
            if in_code_block:
                # End of code block
                in_code_block = False
                code_text = "\n".join(code_block_lines)
                code_block_lines = []
                # If it's a mermaid block, we can omit raw mermaid code or format cleanly as a diagram box
                # Here we skip raw mermaid text to keep document pristine, as we already have the rendered jpg illustration!
            else:
                in_code_block = True
                code_block_lines = []
            i += 1
            continue
            
        if in_code_block:
            code_block_lines.append(line)
            i += 1
            continue
            
        # Skip horizontal rules
        if stripped in ('---', '***', '___'):
            i += 1
            continue
            
        # Empty line
        if not stripped:
            i += 1
            continue
            
        # 1. Images: ![caption](path)
        img_match = re.match(r'!\[(.*?)\]\((.*?)\)', stripped)
        if img_match:
            caption, img_rel_path = img_match.groups()
            img_full_path = os.path.normpath(os.path.join(base_dir, img_rel_path))
            if os.path.exists(img_full_path):
                # Add picture
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.paragraph_format.first_line_indent = Cm(0)
                p_img.paragraph_format.line_spacing = 1.0
                p_img.paragraph_format.space_before = Pt(6)
                p_img.paragraph_format.space_after = Pt(2)
                run_img = p_img.add_run()
                run_img.add_picture(img_full_path, width=Cm(15.0))
                
                # Add caption
                if caption:
                    p_cap = doc.add_paragraph()
                    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p_cap.paragraph_format.first_line_indent = Cm(0)
                    p_cap.paragraph_format.line_spacing = 1.15
                    p_cap.paragraph_format.space_before = Pt(2)
                    p_cap.paragraph_format.space_after = Pt(8)
                    run_cap = p_cap.add_run(f"Hình: {caption}")
                    format_run_font(run_cap, size_pt=11, italic=True, color_rgb=(80, 80, 80))
            else:
                print(f"[Warning] Image not found: {img_full_path}")
            i += 1
            continue
            
        # 2. Markdown Tables
        if stripped.startswith('|') and stripped.endswith('|'):
            # Collect all table lines
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|') and lines[i].strip().endswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
                
            if len(table_lines) >= 2:
                # Parse header, delimiter, and rows
                header_cols = [c.strip() for c in table_lines[0].strip('|').split('|')]
                # skip delimiter row
                data_rows = []
                for t_line in table_lines[1:]:
                    if re.match(r'^\|?\s*:?-+:?\s*(\|:?-+:?\s*)+\|?$', t_line):
                        continue
                    data_rows.append([c.strip() for c in t_line.strip('|').split('|')])
                    
                table = doc.add_table(rows=len(data_rows) + 1, cols=len(header_cols))
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                set_table_borders(table, color="9E9E9E", sz="4")
                
                # Header row
                hdr_cells = table.rows[0].cells
                for col_idx, col_text in enumerate(header_cols):
                    hdr_cells[col_idx].text = ""
                    set_cell_background(hdr_cells[col_idx], "F0F4F8")
                    set_cell_margins(hdr_cells[col_idx], top=120, bottom=120, left=140, right=140)
                    p_hdr = hdr_cells[col_idx].paragraphs[0]
                    p_hdr.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p_hdr.paragraph_format.first_line_indent = Cm(0)
                    p_hdr.paragraph_format.line_spacing = 1.15
                    p_hdr.paragraph_format.space_before = Pt(0)
                    p_hdr.paragraph_format.space_after = Pt(0)
                    add_formatted_text(p_hdr, col_text, default_size=12, default_bold=True, default_color=(0, 40, 90))
                    
                # Data rows
                for r_idx, row_data in enumerate(data_rows):
                    row_cells = table.rows[r_idx + 1].cells
                    bg_color = "FAFAFA" if r_idx % 2 == 1 else "FFFFFF"
                    for col_idx, cell_value in enumerate(row_data):
                        if col_idx < len(row_cells):
                            row_cells[col_idx].text = ""
                            if bg_color != "FFFFFF":
                                set_cell_background(row_cells[col_idx], bg_color)
                            set_cell_margins(row_cells[col_idx], top=100, bottom=100, left=140, right=140)
                            p_cell = row_cells[col_idx].paragraphs[0]
                            # Check if column should be center-aligned
                            if len(cell_value) <= 15 or col_idx == 0:
                                p_cell.alignment = WD_ALIGN_PARAGRAPH.CENTER
                            else:
                                p_cell.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                            p_cell.paragraph_format.first_line_indent = Cm(0)
                            p_cell.paragraph_format.line_spacing = 1.2
                            p_cell.paragraph_format.space_before = Pt(0)
                            p_cell.paragraph_format.space_after = Pt(0)
                            
                            # Handle multiple lines inside cell (<br>)
                            lines_in_cell = cell_value.split('<br>')
                            for sub_idx, sub_line in enumerate(lines_in_cell):
                                if sub_idx > 0:
                                    p_cell = row_cells[col_idx].add_paragraph()
                                    p_cell.paragraph_format.first_line_indent = Cm(0)
                                    p_cell.paragraph_format.line_spacing = 1.2
                                    p_cell.paragraph_format.space_before = Pt(0)
                                    p_cell.paragraph_format.space_after = Pt(0)
                                add_formatted_text(p_cell, sub_line.strip(), default_size=12)
                                
                # Add spacing after table
                p_sp = doc.add_paragraph()
                p_sp.paragraph_format.first_line_indent = Cm(0)
                p_sp.paragraph_format.space_before = Pt(0)
                p_sp.paragraph_format.space_after = Pt(6)
            continue

        # 3. Headings
        # Heading 1: # Title
        if stripped.startswith('# '):
            h_text = stripped[2:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.line_spacing = 1.5
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(8)
            add_formatted_text(p, h_text, default_size=16, default_bold=True, default_color=(0, 51, 102))
            i += 1
            continue
            
        # Heading 2: ## Section
        if stripped.startswith('## '):
            h_text = stripped[3:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.line_spacing = 1.5
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            add_formatted_text(p, h_text, default_size=14, default_bold=True, default_color=(0, 51, 102))
            i += 1
            continue

        # Heading 3: ### Sub-section
        if stripped.startswith('### '):
            h_text = stripped[4:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.line_spacing = 1.5
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
            add_formatted_text(p, h_text, default_size=13, default_bold=True, default_color=(20, 40, 80))
            i += 1
            continue

        # Heading 4: #### Sub-sub-section
        if stripped.startswith('#### '):
            h_text = stripped[5:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.line_spacing = 1.5
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(2)
            add_formatted_text(p, h_text, default_size=13, default_bold=True, default_italic=True)
            i += 1
            continue

        # 4. Blockquotes: > Quote / Callout
        if stripped.startswith('>'):
            quote_text = stripped.lstrip('> ').strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.first_line_indent = Cm(1.27)
            p.paragraph_format.left_indent = Cm(0.5)
            p.paragraph_format.right_indent = Cm(0.5)
            p.paragraph_format.line_spacing = 1.5
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            add_formatted_text(p, quote_text, default_size=13, default_italic=True, default_color=(40, 40, 40))
            i += 1
            continue

        # 5. Bullet Lists / Numbered Lists
        # Level 1 list: * or - or 1.
        list_match = re.match(r'^(\s*)([\*\-\+]|\d+\.)\s+(.*)$', line)
        if list_match:
            indent_spaces, bullet_char, item_text = list_match.groups()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.line_spacing = 1.5
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            
            # Determine indentation level
            indent_level = len(indent_spaces) // 2
            p.paragraph_format.left_indent = Cm(1.27 + indent_level * 0.6)
            p.paragraph_format.first_line_indent = Cm(-0.5)
            
            # Add bullet symbol
            bullet_sym = "• " if bullet_char in ('*', '-', '+') else f"{bullet_char} "
            run_b = p.add_run(bullet_sym)
            format_run_font(run_b, size_pt=13, bold=True)
            
            add_formatted_text(p, item_text, default_size=13)
            i += 1
            continue

        # 6. Standard Body Paragraphs
        # Must have: 1.5 line spacing, 13pt Times New Roman, 1 tab (1.27cm) first line indent, Justified
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.first_line_indent = Cm(1.27)
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(4)
        add_formatted_text(p, stripped, default_size=13)
        i += 1

    doc.save(docx_path)
    print(f"[Success] Saved DOCX to: {docx_path}")

if __name__ == '__main__':
    hw_dir = r"c:\Users\Admin\HUIT - Học Tập\Năm 4\Computer_Vision\docs\homework"
    b1_md = os.path.join(hw_dir, "B1.md")
    b1_docx = os.path.join(hw_dir, "B1.docx")
    
    b2_md = os.path.join(hw_dir, "B2.md")
    b2_docx = os.path.join(hw_dir, "B2.docx")
    
    convert_md_to_docx(b1_md, b1_docx)
    convert_md_to_docx(b2_md, b2_docx)
