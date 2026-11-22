"""Typeset the audited MuSiQue case without rewriting archived strings."""
import json
from pathlib import Path

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[1]
BASE = ROOT / "REVIVE_training/result/baselines/musique"
ART = ROOT / "art-backup/demo-3/result/musique/rsearch_musique_art_top5_results.jsonl"
CASE_ID = 27


def read_case(path):
    with path.open(encoding="utf-8") as source:
        matches = [record for line in source if line.strip()
                   if (record := json.loads(line))["id"] == CASE_ID]
    record, = matches
    return record


def tex(raw):
    escapes = {"\\": r"\textbackslash{}", "<": r"\textless{}", ">": r"\textgreater{}",
               "|": r"\textbar{}", "_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#",
               "$": r"\$", "{": r"\{", "}": r"\}", "'": r"\textquotesingle{}",
               '"': r"\textquotedbl{}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "\n".join("".join(escapes.get(c, c) for c in line)
                     + r"\par" for line in raw.split("\n"))


def field(name, raw):
    return rf"\textbf{{{name}}}\par\nopagebreak[4]" + "\n" + r"{\small\ttfamily\raggedright" + "\n" + tex(raw) + "\n}\n"


def excerpt(raw, end):
    index = raw.index(end) + len(end)
    return raw[:index]


def main():
    no, vanilla, art = [read_case(p) for p in (BASE / "no_rag.jsonl", BASE / "vanilla_rag_top5.jsonl", ART)]
    assert len({(x["question"], tuple(x["golden_answers"])) for x in (no, vanilla, art)}) == 1
    for item, expected in zip((no, vanilla, art), (0, 1, 2)):
        assert len(item["pure_art_queries"]) == expected
        assert len(item["pure_art_passages"]) == 5 * expected
    q, = art["golden_answers"]
    common = field("Question", art["question"]) + field("Ground truth", q)
    basics = [(label, record["pure_art_answer"]) for label, record in
              (("No-RAG", no), ("Vanilla RAG", vanilla))]
    full = basics + [("ART", art["pure_art_answer"])]
    first = vanilla["pure_art_passages"][0]
    start = excerpt(first, "Bindi Irwin as Nim Rusoe")
    baseline_passage = field("Vanilla RAG, passage 1", start) + r"\emph{[Lược phần còn lại; 4 đoạn khác không hiển thị.]}"
    items = "".join(field(name, raw) for name, raw in basics)
    out = REPORT / "graphics"
    (out / "case-baseline.tex").write_text(
        r"\setlength{\fboxsep}{6pt}\fbox{\begin{minipage}{.93\linewidth}" + "\n" +
        common + r"\smallskip\hrule\smallskip" + "\n" + items + r"\smallskip\hrule\smallskip" + "\n" +
        baseline_passage + "\n" + r"\end{minipage}}", encoding="utf-8")
    (out / "case-outputs.tex").write_text(common + "".join(field(name, raw) for name, raw in full),
                                         encoding="utf-8")
    passages = art["pure_art_passages"]
    trace = field("Vanilla RAG, query", vanilla["pure_art_queries"][0]) + baseline_passage + "\n\n"
    trace += field("ART, query 1", art["pure_art_queries"][0])
    trace += field("ART, query 1, passage 1", excerpt(passages[0], "Bindi Irwin as Nim Rusoe"))
    trace += r"\emph{[Lược phần còn lại; 4 đoạn khác không hiển thị.]}" + "\n\n"
    trace += field("ART, query 2", art["pure_art_queries"][1])
    for rank, end in [(1, "who died in 2006."), (2, "daughter of Steve and Terri Irwin."), (3, None)]:
        raw = passages[5 + rank - 1]
        trace += field(f"ART, query 2, passage {rank}", excerpt(raw, end) if end else raw)
        if end:
            trace += r"\emph{[Lược phần còn lại của đoạn.]}" + "\n\n"
    trace += r"\emph{[Đoạn 4 và 5 của lượt 2 không hiển thị.]}"
    (out / "case-trace.tex").write_text(trace, encoding="utf-8")


if __name__ == "__main__":
    main()
