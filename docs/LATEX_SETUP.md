# Windows LaTeX setup

The build target is the original [HCMUT Vietnamese template](../slides-template/Specialized-Project-Template-VI/README.txt). The template uses pdfLaTeX and BibTeX with the `plain` bibliography style. The build script copies the necessary source files byte for byte into the build tree and compiles the copy.

## Tool locations

| Component | Location |
|---|---|
| MiKTeX 25.12, private per-user installation and package root | `D:\Tools\MiKTeX` |
| pdfLaTeX, BibTeX and latexmk | `D:\Tools\MiKTeX\miktex\bin\x64` |
| MiKTeX user configuration | `D:\Tools\MiKTeX-config` |
| MiKTeX user data, generated formats, font caches and tool logs | `D:\Tools\MiKTeX-data` |
| Git for Windows Perl, reused | `C:\Program Files\Git\usr\bin\perl.exe` |
| Existing base Python 3.13.7 | `C:\Python313\python.exe` |
| Project Python and PyMuPDF 1.26.7 | `.venv\Scripts\python.exe` and `.venv\Lib\site-packages` on D: |
| Installer and installer/pip temporary files | `build\latex\setup` on D: |
| Build-scoped TEMP and TMP | `build\latex\temp` on D: |

The Basic MiKTeX installer was downloaded from the official site and verified against its published SHA-256. Setup used `--unattended --private --package-set=basic` and the three user paths above with `--user-install`, `--user-config`, and `--user-data`. Automatic package installation is enabled in user mode. The first build can fetch packages from a MiKTeX mirror. Package revisions on that mirror can change. Windows may still use the user profile, registry, or OS-managed caches on C:. The existing `build/` and `.venv` Git ignore rules cover these project outputs.

See the official [MiKTeX download and checksum](https://miktex.org/download) and [installer options](https://docs.miktex.org/manual/setupwiz.html).

## Build and inspect

Run from the repository root in PowerShell. Install the pinned wheel into the existing project environment when provisioning another checkout. If `.venv` is absent, create it first with `python -m venv .venv`.

```powershell
$pdfTemp = New-Item -ItemType Directory -Force build\latex\setup\pip-temp
$env:TEMP = $pdfTemp.FullName
$env:TMP = $pdfTemp.FullName
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --only-binary=:all: -r scripts\requirements-pdf.txt

.\scripts\build-hcmut-template.ps1
.\.venv\Scripts\python.exe scripts\inspect_pdf.py build\latex\hcmut-template\main.pdf --output-dir build\latex\hcmut-template\inspection
```

The build script prefers latexmk with pdfLaTeX and BibTeX. Use `-Manual` for the template's explicit pdfLaTeX, BibTeX, pdfLaTeX, pdfLaTeX sequence. The script also selects this sequence if latexmk or a working Perl is absent. `-MiKTeXBin` can point to a different TeX bin directory. The default path is added for the build process. See `main.log` and `main.blg` alongside `main.pdf`.

The inspector writes PNGs, text files and `inspection.json` with page dimensions in points. Use `--pages 1,5-8` for a subset or omit it for all pages. Empty-page detection uses a low-resolution grayscale render and separately reports pages without extractable text. A page number is visible content, and scanned or vector-only content may have no extractable text. Open the PNGs to verify layout and glyphs.

## Validation on 21 November 2026

On Windows build 26300, MiKTeX 25.12 with latexmk 4.89, pdfTeX 1.40.28 and BibTeX 0.99e completed the template build with exit code 0. The SHA-256 inventory of the 18 original template files was identical before and after the build. The final logs show the bibliography was generated and have no missing package or graphic errors, undefined citations or references, or missing-character warnings.

The [template PDF](../build/latex/hcmut-template/main.pdf) has 19 A4 pages (595.276 × 841.890 pt). All 19 rendered at 144 dpi, with no suspiciously empty or textless pages flagged. The cover, workload table, example equation/figure/table and bibliography were visually inspected on PDF pages [1](../build/latex/hcmut-template/inspection/page-001.png), [5](../build/latex/hcmut-template/inspection/page-005.png), [11](../build/latex/hcmut-template/inspection/page-011.png) and [17](../build/latex/hcmut-template/inspection/page-017.png). The cover logo, Vietnamese diacritics, equation and bibliography rendered. The workload table visibly extends past the normal right text margin. A separate two-page inspection smoke PDF correctly flagged its deliberately blank page, and an out-of-range page request exited with code 2.

Remaining warnings from the original template and build:

- The workload table has one underfull horizontal box (badness 10000) and one overfull horizontal box (27.23721 pt). No underfull or overfull vertical boxes were reported.
- The class gives `memoir` warnings about `fancyhdr` and the deprecated `\addtodef`. pdfTeX reports a duplicate `page.1` hyperlink destination.
- Babel reports that Vietnamese hyphenation patterns were not preloaded in this format. Vietnamese text rendered in the inspected pages.
- MiKTeX reports that a full update check has yet to run. Automatic package installation worked during the build. In the managed agent sandbox, MiKTeX needs execution approval to write to its D: data and package roots, and Git's Perl also failed to start in that restricted process. The build and visual inspection succeeded after the approved invocation.
