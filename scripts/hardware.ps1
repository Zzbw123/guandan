$ErrorActionPreference = 'Stop'
$rootPath = Split-Path -Parent $PSScriptRoot
$outputPath = Join-Path $rootPath 'artifacts\evaluations\hardware.json'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputPath) | Out-Null
$gpuDetails = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1
$report = [ordered]@{
    measured_at = (Get-Date).ToString('o')
    cpu = @(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors)
    memory_bytes = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
    gpu_nvidia_smi = @($gpuDetails | ForEach-Object { "$_" })
    disks = @(Get-PSDrive -PSProvider FileSystem | Select-Object Name,Used,Free)
    python = (& python --version 2>&1 | Out-String).Trim()
    note = 'Inventory only. CUDA/PyTorch training throughput has not been tested.'
}
$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $outputPath -Encoding utf8
Get-Content -LiteralPath $outputPath
