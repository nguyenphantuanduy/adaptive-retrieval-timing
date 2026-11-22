"""Render and inspect a PDF locally with PyMuPDF."""

import argparse
import json
from pathlib import Path

import pymupdf


def parse_pages(spec, total):
    """Convert a comma-separated list of 1-based pages and ranges to page numbers."""
    if spec is None:
        return range(1, total + 1)
    selected = set()
    for segment in spec.split(","):
        endpoints = segment.strip().split("-")
        if len(endpoints) > 2 or not all(part.strip().isdecimal() for part in endpoints):
            raise ValueError(f"invalid page or range: {segment!r}")
        first, last = int(endpoints[0]), int(endpoints[-1])
        if not 1 <= first <= last <= total:
            raise ValueError(f"page range must be within 1-{total}: {segment!r}")
        selected.update(range(first, last + 1))
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pages", help="1-based pages and ranges, e.g. 1,5-8 (default: all)")
    parser.add_argument("--dpi", type=int, default=144)
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("--dpi must be positive")

    try:
        with pymupdf.open(args.pdf) as document:
            if not document.is_pdf or document.needs_pass or not len(document):
                raise ValueError("expected an accessible PDF containing at least one page")
            pages = parse_pages(args.pages, len(document))
            args.output_dir.mkdir(parents=True, exist_ok=True)
            records = []
            for number in pages:
                page = document[number - 1]
                name = f"page-{number:03d}"
                content = page.get_text()
                page.get_pixmap(dpi=args.dpi, alpha=False).save(args.output_dir / f"{name}.png")
                (args.output_dir / f"{name}.txt").write_text(content, encoding="utf-8")
                thumbnail = page.get_pixmap(dpi=36, colorspace=pymupdf.csGRAY)
                has_visible_marks = min(thumbnail.samples) < 250
                records.append({
                    "page": number,
                    "width_pt": round(page.rect.width, 3),
                    "height_pt": round(page.rect.height, 3),
                    "text_characters": len(content.strip()),
                    "suspicious_empty": not has_visible_marks,
                    "no_extractable_text": not content.strip(),
                    "png": f"{name}.png",
                    "text": f"{name}.txt",
                })
            summary = {
                "pdf": str(args.pdf.resolve()),
                "total_pages": len(document),
                "dpi": args.dpi,
                "pages": records,
            }
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    report = args.output_dir / "inspection.json"
    report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered {len(records)} of {summary['total_pages']} pages to {args.output_dir}")
    print("Suspiciously empty pages:", [row["page"] for row in records if row["suspicious_empty"]])
    print("Pages without extractable text:", [row["page"] for row in records if row["no_extractable_text"]])
    print(f"Inspection report: {report}")


if __name__ == "__main__":
    main()
