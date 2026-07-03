param(
    [string]$Config = "configs/qwen2.5-lora.yaml"
)

$ErrorActionPreference = "Stop"
python train.py --config $Config
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
