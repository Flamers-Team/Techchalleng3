
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import re
import sys

def md_to_docx(md_text: str, output_path: str):
    doc = Document()

    # Estilo padrão
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Calibri"
    font.size = Pt(11)

    lines = md_text.split("\n")

    for line in lines:
        line = line.rstrip()

        # Cabeçalhos
        if line.startswith("# "):
            p = doc.add_heading(line[2:], level=1)
        elif line.startswith("## "):
            p = doc.add_heading(line[3:], level=2)
        elif line.startswith("### "):
            p = doc.add_heading(line[4:], level=3)
        elif line.startswith("#### "):
            p = doc.add_heading(line[5:], level=4)

        # Tabelas (formato markdown)
        elif line.startswith("|") and "---" not in line:
            # Se a linha anterior não é tabela, cria uma nova
            if not hasattr(doc, "_last_was_table") or not doc._last_was_table:
                table = doc.add_table(rows=0, cols=line.count("|") - 1)
                table.style = "Light Grid Accent 1"
                doc._current_table = table
            row_cells = doc._current_table.add_row().cells
            cells = [c.strip() for c in line.strip("|").split("|")]
            for i, cell_text in enumerate(cells):
                if i < len(row_cells):
                    row_cells[i].text = cell_text
            doc._last_was_table = True

        # Linha de separação de tabela
        elif line.startswith("|") and "---" in line:
            continue

        # Listas
        elif line.strip().startswith("- "):
            doc.add_paragraph(line.strip()[2:], style="List Bullet")
            doc._last_was_table = False

        # Listas numeradas
        elif re.match(r"^\d+\.\s", line.strip()):
            doc.add_paragraph(re.sub(r"^\d+\.\s", "", line.strip()), style="List Number")
            doc._last_was_table = False

        # Code blocks
        elif line.startswith("```"):
            continue

        # Linha em branco
        elif not line.strip():
            doc.add_paragraph()
            doc._last_was_table = False

        # Texto normal (com possível formatação inline)
        else:
            p = doc.add_paragraph()
            # Processa **bold** e `code`
            parts = re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", line)
            for part in parts:
                if part.startswith("**") and part.endswith("**"):
                    run = p.add_run(part[2:-2])
                    run.bold = True
                elif part.startswith("`") and part.endswith("`"):
                    run = p.add_run(part[1:-1])
                    run.font.name = "Consolas"
                    run.font.size = Pt(10)
                else:
                    p.add_run(part)
            doc._last_was_table = False

    doc.save(output_path)
    print(f"OK: {output_path}")

md_to_docx(sys.argv[1], sys.argv[2])
