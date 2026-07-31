$ErrorActionPreference = "Stop"

for ($attempt = 1; $attempt -le 3; $attempt++) {
    choco install miktex --no-progress -y
    if ($LASTEXITCODE -eq 0) {
        break
    }
    if ($attempt -eq 3) {
        exit $LASTEXITCODE
    }
    Start-Sleep -Seconds (15 * $attempt)
}

$latexmk = Get-ChildItem -Path "$env:ProgramFiles\MiKTeX" -Filter latexmk.exe -Recurse |
    Select-Object -First 1
if ($null -eq $latexmk) {
    throw "MiKTeX installation did not provide latexmk.exe"
}
$latexmk.DirectoryName | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append

$mpm = Get-ChildItem -Path "$env:ProgramFiles\MiKTeX" -Filter mpm.exe -Recurse |
    Select-Object -First 1
if ($null -eq $mpm) {
    throw "MiKTeX installation did not provide mpm.exe"
}
& $mpm.FullName --install=fandol
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $latexmk.FullName -version
