import csv
from fpdf import FPDF

def export_csv(records, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Student", "Entry", "Exit", "Duration"])
        for r in records:
            writer.writerow([r.student_id, r.entry, r.exit, r.duration])

def export_pdf(records, path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    for r in records:
        pdf.cell(0, 8, f"{r.student_id} | {r.duration}s", ln=1)

    pdf.output(path)