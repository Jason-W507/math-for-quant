param(
    [ValidateSet("upper", "lower", "all")]
    [string]$Volume = "all",
    [switch]$Release
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot ".." )).Path
Set-Location $Root

$Manifest = Get-Content (Join-Path $Root "curriculum/manifest.json") -Raw | ConvertFrom-Json
$Version = (Get-Content (Join-Path $Root "VERSION") -Raw).Trim()
$env:JUPYTER_ALLOW_INSECURE_WRITES = "true"
$env:IPYTHONDIR = Join-Path $Root "build/ipython"
$env:JUPYTER_RUNTIME_DIR = Join-Path $Root "build/jupyter-runtime"
New-Item -ItemType Directory -Force -Path $env:IPYTHONDIR, $env:JUPYTER_RUNTIME_DIR | Out-Null

function Get-ReleaseDate {
    if ($env:MFQ_RELEASE_DATE) {
        [datetime]::ParseExact($env:MFQ_RELEASE_DATE, "yyyy-MM-dd", $null).ToString("yyyy-MM-dd")
        return
    }
    if ($env:SOURCE_DATE_EPOCH) {
        [DateTimeOffset]::FromUnixTimeSeconds([int64]$env:SOURCE_DATE_EPOCH).ToString("yyyy-MM-dd")
        return
    }
    $date = (& git show -s --format=%cs HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $date) { throw "release date is unavailable" }
    $date
}

if ($env:MFQ_RELEASE_TAG -and $env:MFQ_RELEASE_TAG -ne "v$Version") {
    throw "tag $env:MFQ_RELEASE_TAG does not match VERSION v$Version"
}

function Invoke-LatexPublication($Publication) {
    $id = [string]$Publication.id
    $source = Join-Path $Root ([string]$Publication.source)
    $published = Join-Path $Root ([string]$Publication.pdf)
    $job = [IO.Path]::GetFileNameWithoutExtension($published)
    $build = Join-Path $Root "build/latex/$id"
    New-Item -ItemType Directory -Force -Path $build, (Split-Path $published) | Out-Null
    $wrapper = Join-Path $build "$job-wrapper.tex"
    $fontOption = if ($env:MFQ_MIKTEX_AUTO_INSTALL -eq "1") { "\PassOptionsToPackage{fontset=fandol}{ctex}`n" } else { "" }
    $wrapperText = $fontOption + "\def\MFQReleaseDate{" + (Get-ReleaseDate) + "}`n" +
        "\def\MFQVersion{" + $Version + "}`n" + "\input{" + $source.Replace("\", "/") + "}`n"
    Set-Content -LiteralPath $wrapper -Value $wrapperText -Encoding utf8
    $installer = if ($env:MFQ_MIKTEX_AUTO_INSTALL -eq "1") { "--enable-installer" } else { "--disable-installer" }
    & latexmk -xelatex -bibtex "-xelatex=xelatex $installer %O %S" -interaction=nonstopmode -halt-on-error -file-line-error "-outdir=$build" "-jobname=$job" $wrapper
    if ($LASTEXITCODE -ne 0) { throw "$id latexmk build failed" }
    $built = Join-Path $build "$job.pdf"
    if (-not (Test-Path -LiteralPath $built)) { throw "$id PDF was not produced" }
    Copy-Item -LiteralPath $built -Destination $published -Force
    Write-Host "publication=$id pdf=$([IO.Path]::GetRelativePath($Root, $published))"
}

function Copy-NotebookRelease {
    $sourceRoot = Join-Path $Root "notebooks"
    $destinationRoot = Join-Path $Root "output/notebooks"
    $sources = @(Get-ChildItem -LiteralPath $sourceRoot -Recurse -Filter *.ipynb -File)
    if ($sources.Count -eq 0) { throw "no teaching notebooks found" }
    Remove-Item -LiteralPath $destinationRoot -Recurse -Force -ErrorAction SilentlyContinue
    foreach ($source in $sources) {
        $relative = [IO.Path]::GetRelativePath($sourceRoot, $source.FullName)
        $destination = Join-Path $destinationRoot $relative
        New-Item -ItemType Directory -Force -Path (Split-Path $destination) | Out-Null
        Copy-Item -LiteralPath $source.FullName -Destination $destination -Force
    }
}

$volumes = @($Manifest.volumes)
$selected = if ($Volume -eq "all") { $volumes } else { @($volumes | Where-Object { $_.id -eq $Volume }) }
if ($selected.Count -eq 0) { throw "unknown volume: $Volume" }

if ($Release) {
    Copy-NotebookRelease
}

$builtSupplement = $false
foreach ($publication in $selected) {
    Invoke-LatexPublication $publication
    if (-not $builtSupplement) {
        $supplement = $Manifest.supplements | Where-Object { $_.parent_volumes -contains $publication.id } | Select-Object -First 1
        if ($supplement) {
            Invoke-LatexPublication $supplement
            $builtSupplement = $true
        }
    }
}

if ($Release) {
    $archive = Join-Path $Root "output/math-for-quant-notebooks.zip"
    Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue
    Compress-Archive -Path (Join-Path $Root "output/notebooks/*"), (Join-Path $Root "LICENSE"), (Join-Path $Root "LICENSE-CONTENT.md") -DestinationPath $archive -CompressionLevel Optimal
    Write-Host "release=passed archive=$([IO.Path]::GetRelativePath($Root, $archive))"
}
