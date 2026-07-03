param(
    [ValidateSet("offline", "endpoint", "local")]
    [string]$Backend = "offline",
    [string]$Model = "",
    [string]$Adapter = ""
)

$ErrorActionPreference = "Stop"
$arguments = @("evaluator.py", "--backend", $Backend)
if ($Model) {
    $arguments += @("--model", $Model)
}
if ($Adapter) {
    $arguments += @("--adapter", $Adapter)
}
python @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
