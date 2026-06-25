from pathlib import Path

from docx import Document


SRC = Path("Chapter4_Compound_v19_compact_no_lof_shapley.docx")
OUT = Path("Chapter4_Compound_v20_compact.docx")


TARGET_START = "Η στήλη Mean Shapley δείχνει τη μέση απόλυτη συνεισφορά"
ADDITION = (
    " Με αυτόν τον τρόπο, η Shapley ανάλυση δεν λειτουργεί μόνο ως εργαλείο ερμηνείας, "
    "αλλά ως μηχανισμός επιλογής προτεραιοτήτων: όταν ένα corruption συγκεντρώνει υψηλό damage share, "
    "το cleaning μπορεί να σχεδιαστεί στοχευμένα αντί να εφαρμόζεται ομοιόμορφα σε όλα τα σήματα."
)


def main():
    doc = Document(SRC)
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text.startswith(TARGET_START):
            if "μηχανισμός επιλογής προτεραιοτήτων" not in paragraph.text:
                paragraph.text = paragraph.text + ADDITION
            break
    else:
        raise RuntimeError("Target Shapley paragraph not found.")

    doc.save(OUT)


if __name__ == "__main__":
    main()
