<#
Run a project Python script with the recorded, CUDA-capable interpreter.
Override with GUANDAN_GPU_PYTHON only when intentionally selecting a new runtime.
#>
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Script,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArguments
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$gpuPython = $env:GUANDAN_GPU_PYTHON
if (-not $gpuPython) {
    $runtimeRecord = Join-Path $projectRoot 'artifacts/evaluations/p3b-gpu-environment.json'
    $gpuPython = (Get-Content -LiteralPath $runtimeRecord -Raw | ConvertFrom-Json).executable
}
if (-not (Test-Path -LiteralPath $gpuPython -PathType Leaf)) {
    throw 'Recorded GPU Python is missing. Set GUANDAN_GPU_PYTHON to an explicitly verified interpreter.'
}
& $gpuPython -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("GPU runtime:", torch.__version__, torch.cuda.get_device_name(0), flush=True)'
if ($LASTEXITCODE -ne 0) { throw 'GPU runtime preflight failed; no experiment was started.' }
$scriptPath = if ([IO.Path]::IsPathRooted($Script)) { $Script } else { Join-Path $projectRoot $Script }
& $gpuPython $scriptPath @ScriptArguments
exit $LASTEXITCODE
