[CmdletBinding()]
param(
    [string]$MiKTeXBin = 'D:\Tools\MiKTeX\miktex\bin\x64',
    [switch]$Manual,
    [switch]$Report
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$template = Join-Path $repo 'slides-template\Specialized-Project-Template-VI'
$buildRoot = Join-Path $repo 'build\latex'
$build = Join-Path $buildRoot 'hcmut-template'
if ($Report) {
    $template = Join-Path $repo 'report\dacn'
    $build = Join-Path $buildRoot 'dacn'
}
$temp = Join-Path $buildRoot 'temp'
New-Item -ItemType Directory -Force -Path $build, $temp | Out-Null

# Compile a copy so TeX, BibTeX and latexmk can use template-relative paths.
foreach ($item in 'main.tex', 'hcmut-thesis.cls', 'chapters', 'graphics', 'refs') {
    Copy-Item -LiteralPath (Join-Path $template $item) -Destination $build -Recurse -Force
}

function Invoke-TeX {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program exited with code ${LASTEXITCODE}. See $build\main.log and main.blg."
    }
}

$savedPath = $env:PATH
$savedTemp = $env:TEMP
$savedTmp = $env:TMP
Push-Location -LiteralPath $build
try {
    if (Test-Path -LiteralPath $MiKTeXBin -PathType Container) {
        $env:PATH = "$MiKTeXBin;$env:PATH"
    }
    $env:TEMP = $temp
    $env:TMP = $temp
    $pdfLaTeX = (Get-Command pdflatex -CommandType Application).Source
    $bibTeX = (Get-Command bibtex -CommandType Application).Source
    $latexmk = Get-Command latexmk -CommandType Application -ErrorAction SilentlyContinue
    $perl = Get-Command perl -CommandType Application -ErrorAction SilentlyContinue
    $gitPerl = Join-Path $env:ProgramFiles 'Git\usr\bin\perl.exe'
    if (!$perl -and (Test-Path -LiteralPath $gitPerl -PathType Leaf)) {
        # Add after the Windows and TeX tools so Git's Unix tools do not shadow them.
        $env:PATH = "$env:PATH;$(Split-Path -Parent $gitPerl)"
        $perl = Get-Command perl -CommandType Application
    }

    $engineFlags = '-interaction=nonstopmode', '-halt-on-error', '-file-line-error'
    if ($Report) {
        foreach ($diagram in 'attention', 'workflow') {
            Invoke-TeX $pdfLaTeX ($engineFlags + '-output-directory=graphics' + "graphics/$diagram-figure.tex")
        }
    }
    $perlWorks = $false
    if (!$Manual -and $latexmk -and $perl) {
        try {
            & $perl.Source -e 'exit 0' 2>$null
            $perlWorks = $LASTEXITCODE -eq 0
        } catch {
            Write-Warning 'Perl could not start in this session.'
        }
    }
    if ($perlWorks) {
        Invoke-TeX $latexmk.Source (@('-norc', '-g', '-pdf') + $engineFlags + 'main.tex')
    } else {
        Write-Host 'Using the template manual pdfLaTeX / BibTeX build sequence.'
        Invoke-TeX $pdfLaTeX ($engineFlags + 'main.tex')
        Invoke-TeX $bibTeX @('main')
        Invoke-TeX $pdfLaTeX ($engineFlags + 'main.tex')
        Invoke-TeX $pdfLaTeX ($engineFlags + 'main.tex')
    }

    if (!(Test-Path -LiteralPath 'main.pdf' -PathType Leaf)) {
        throw 'Build finished without main.pdf.'
    }
    $logs = @('main.log', 'main.blg') | Where-Object { Test-Path -LiteralPath $_ }
    $diagnostics = Select-String -LiteralPath $logs -Pattern @(
        'File .+ not found', 'Missing character:', 'not loadable',
        'undefined', 'multiply defined', 'Rerun to get', 'Label\(s\) may have changed',
        '(Over|Under)full \\[hv]box', '^Warning--', 'I couldn.t open'
    )
    if ($diagnostics) {
        $diagnostics | ForEach-Object { Write-Warning "$($_.Filename):$($_.LineNumber) $($_.Line)" }
    } else {
        Write-Host 'No missing files, unresolved references, or box warnings found in final logs.'
    }
    Write-Host "PDF: $build\main.pdf"
} finally {
    Pop-Location
    $env:PATH = $savedPath
    $env:TEMP = $savedTemp
    $env:TMP = $savedTmp
}
