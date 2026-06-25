from pathlib import Path
from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.table import Table


DOCX = Path("timeSeries_working_original.docx")


def iter_blocks(doc):
    body = doc.element.body
    p_idx = -1
    t_idx = -1
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            p_idx += 1
            yield ("p", p_idx, Paragraph(child, doc))
        elif child.tag == qn("w:tbl"):
            t_idx += 1
            yield ("t", t_idx, Table(child, doc))


def main():
    doc = Document(DOCX)
    blocks = list(iter_blocks(doc))
    for pos, (kind, idx, obj) in enumerate(blocks):
        if kind != "t":
            continue
        prevs = []
        nexts = []
        for k in range(pos - 1, max(-1, pos - 8), -1):
            if blocks[k][0] == "p" and blocks[k][2].text.strip():
                prevs.append((blocks[k][1], blocks[k][2].text.strip()))
        for k in range(pos + 1, min(len(blocks), pos + 8)):
            if blocks[k][0] == "p" and blocks[k][2].text.strip():
                nexts.append((blocks[k][1], blocks[k][2].text.strip()))
        print(f"\nTABLE {idx} rows={len(obj.rows)} cols={len(obj.columns)}")
        print(" prev:")
        for pi, txt in prevs[:4]:
            print(f"  P{pi:04d}: {txt[:160]}")
        print(" next:")
        for pi, txt in nexts[:4]:
            print(f"  P{pi:04d}: {txt[:160]}")


if __name__ == "__main__":
    main()
