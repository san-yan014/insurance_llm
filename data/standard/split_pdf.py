import sys
from pathlib import Path
from pypdf import PdfReader, PdfWriter

def split_pdf(pdf_path, out_dir=None):
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir) if out_dir else pdf_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    mid = total // 2
    # clean split, no overlap -- an article crossing the boundary lands in
    # review.json as incomplete rather than risking a silent duplicate in both halves
    ranges = [(0, mid), (mid, total)]

    stem = pdf_path.stem
    out_paths = []
    for i, (start, end) in enumerate(ranges, start=1):
        writer = PdfWriter()
        for p in range(start, end):
            writer.add_page(reader.pages[p])
        out_path = out_dir / f"{stem}_p{i}.pdf"
        with open(out_path, "wb") as f:
            writer.write(f)
        out_paths.append(out_path)
        print(f"part {i}: pages {start + 1}-{end} ({end - start} pages) -> {out_path.name}")

    return out_paths

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python split_pdf.py <pdf_path> [output_dir]")
        sys.exit(1)
    pdf_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    split_pdf(pdf_path, out_dir)