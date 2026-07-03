param(
    [Parameter(Mandatory = $true)]
    [string]$Adapter,
    [Parameter(Mandatory = $true)]
    [string]$Output
)

$ErrorActionPreference = "Stop"
python merge_adapter.py --adapter $Adapter --output $Output
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
