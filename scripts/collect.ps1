<#
.SYNOPSIS
  通用基准采集器：跑一套标准化测试，导出可跨机汇总的 JSON。

.DESCRIPTION
  设计目标是让不同人、不同显卡跑出可比较的数据，所以：
    - 硬件信息自动探测，不写死本机的 LUID / 路径
    - 每个用例前重启服务，消除状态残留（本项目实测同配置能差 58 倍）
    - 记录失败与异常值，不只记成功的
    - schema 带版本号，日后改字段不会让旧数据失效

.EXAMPLE
  .\collect.ps1                      # 跑标准套件
  .\collect.ps1 -Quick               # 只跑最小集(约 2 分钟)
  .\collect.ps1 -Out result.json
#>
param(
  [switch]$Quick,
  [string]$Out = "",
  [string]$Note = ""
)

$ErrorActionPreference = 'Stop'
$SCHEMA = "localai-bench/1"
$API = "http://127.0.0.1:13305/api/v1"

# ---------- 硬件探测（不假设是哪块卡） ----------

function Get-GpuInventory {
  $ctrls = Get-CimInstance Win32_VideoController -EA SilentlyContinue |
           Where-Object { $_.PNPDeviceID -match '^PCI' }
  $mem = Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory -EA SilentlyContinue

  # 用「专属显存占用最高」认定独显：核显是 UMA，专属显存恒为 0
  $discreteLuid = $null
  if ($mem) {
    $discreteLuid = ($mem | Sort-Object DedicatedUsage -Descending | Select-Object -First 1).Name
  }

  return @{
    controllers = @($ctrls | ForEach-Object {
      @{ name = $_.Name; driver = $_.DriverVersion; date = "$($_.DriverDate)" }
    })
    discreteLuid = $discreteLuid
  }
}

function Get-GpuMem {
  param([string]$Luid)
  if (-not $Luid) { return @{ dedicatedGB = $null; sharedGB = $null } }
  $m = Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory -EA SilentlyContinue |
       Where-Object { $_.Name -eq $Luid }
  if (-not $m) { return @{ dedicatedGB = $null; sharedGB = $null } }
  return @{
    dedicatedGB = [math]::Round($m.DedicatedUsage / 1GB, 2)
    sharedGB    = [math]::Round($m.SharedUsage / 1GB, 2)
  }
}

# ---------- 服务控制 ----------

$lemonadeExe = Join-Path $env:LOCALAPPDATA "lemonade_server\bin\LemonadeServer.exe"
$lemonadeCli = Join-Path $env:LOCALAPPDATA "lemonade_server\bin\lemonade.exe"

function Restart-Server {
  Get-Process -Name "LemonadeServer" -EA SilentlyContinue | Stop-Process -Force
  Get-CimInstance Win32_Process -Filter "Name='sd-server.exe' OR Name='llama-server.exe'" -EA SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -EA SilentlyContinue }
  Start-Sleep -Seconds 3
  Start-Process -FilePath $lemonadeExe -WindowStyle Hidden
  for ($i = 0; $i -lt 90; $i++) {
    try { Invoke-RestMethod -Uri "$API/health" -TimeoutSec 3 | Out-Null; return $true }
    catch { Start-Sleep -Seconds 2 }
  }
  return $false
}

# ---------- 用例 ----------

function Invoke-Case {
  param([string]$Kind, [string]$Model, [int]$Size, [string]$Luid)
  $body = if ($Kind -eq 'image') {
    @{ model = $Model; prompt = "a small ceramic teapot on a wooden table"; size = "${Size}x${Size}"; n = 1 } | ConvertTo-Json
  } else {
    @{ model = $Model; messages = @(@{ role = "user"; content = "Say hello in exactly 5 words." }); max_tokens = 24 } | ConvertTo-Json -Depth 5
  }
  $url = if ($Kind -eq 'image') { "$API/images/generations" } else { "$API/chat/completions" }

  $job = Start-Job -ScriptBlock {
    param($u, $b)
    try { Invoke-RestMethod -Uri $u -Method Post -ContentType "application/json" -Body $b -TimeoutSec 900 | Out-Null; $true }
    catch { $false }
  } -ArgumentList $url, $body

  $peakD = 0; $peakS = 0
  $sw = [Diagnostics.Stopwatch]::StartNew()
  while ($job.State -eq 'Running' -and $sw.Elapsed.TotalSeconds -lt 880) {
    $m = Get-GpuMem -Luid $Luid
    if ($m.dedicatedGB -gt $peakD) { $peakD = $m.dedicatedGB }
    if ($m.sharedGB -gt $peakS) { $peakS = $m.sharedGB }
    Start-Sleep -Milliseconds 400
  }
  $ok = Receive-Job $job -Wait
  $sw.Stop(); Remove-Job $job -Force -EA SilentlyContinue

  return @{
    kind = $Kind; model = $Model; size = $Size
    ok = [bool]$ok
    seconds = [math]::Round($sw.Elapsed.TotalSeconds, 2)
    peakDedicatedGB = $peakD
    peakSharedGB = $peakS
  }
}

# ---------- 主流程 ----------

if (-not (Test-Path $lemonadeExe)) { throw "找不到 Lemonade: $lemonadeExe" }

$hw = Get-GpuInventory
Write-Host "探测到显卡:" -ForegroundColor Cyan
$hw.controllers | ForEach-Object { Write-Host "  $($_.name)  driver=$($_.driver)" }
Write-Host "独显 LUID: $($hw.discreteLuid)`n"

$suite = if ($Quick) {
  @(
    @{ kind = 'image'; model = 'SD-Turbo-GGUF'; size = 512 }
    @{ kind = 'chat';  model = 'Gemma-4-E2B-it-GGUF'; size = 0 }
  )
} else {
  @(
    @{ kind = 'image'; model = 'SD-Turbo-GGUF'; size = 512 }
    @{ kind = 'image'; model = 'SD-Turbo-GGUF'; size = 1024 }
    @{ kind = 'image'; model = 'SD-Turbo-GGUF'; size = 2048 }
    @{ kind = 'image'; model = 'SD-Turbo';      size = 512 }
    @{ kind = 'image'; model = 'SDXL-Turbo';    size = 1024 }
    @{ kind = 'chat';  model = 'Gemma-4-E2B-it-GGUF';   size = 0 }
    @{ kind = 'chat';  model = 'DeepSeek-Qwen3-8B-GGUF'; size = 0 }
  )
}

$results = @()
$n = 0
foreach ($c in $suite) {
  $n++
  Write-Host ("[{0}/{1}] {2} {3} @{4}" -f $n, $suite.Count, $c.kind, $c.model, $c.size) -NoNewline
  # 每个用例前重启：状态残留会让同配置数据差几十倍
  if (-not (Restart-Server)) { Write-Host "  服务重启失败，跳过" -ForegroundColor Red; continue }
  Invoke-Case -Kind $c.kind -Model $c.model -Size $c.size -Luid $hw.discreteLuid | Out-Null  # 预热
  $r = Invoke-Case -Kind $c.kind -Model $c.model -Size $c.size -Luid $hw.discreteLuid
  $results += $r
  $color = if ($r.ok) { "Green" } else { "Red" }
  Write-Host ("   {0}s  {1}GB" -f $r.seconds, $r.peakDedicatedGB) -ForegroundColor $color
}

$cfg = @{}
try {
  $raw = & $lemonadeCli config 2>&1 | Out-String
  foreach ($k in 'sdcpp.backend','llamacpp.backend','max_loaded_models','enable_dgpu_gtt','ctx_size','sdcpp.vulkan_args') {
    $m = [regex]::Match($raw, [regex]::Escape($k) + '\s+(\S.*?)\s*$', 'Multiline')
    if ($m.Success) { $cfg[$k] = $m.Groups[1].Value.Trim() }
  }
} catch {}

$payload = [ordered]@{
  schema    = $SCHEMA
  timestamp = (Get-Date -Format "o")
  note      = $Note
  host      = @{
    os       = (Get-CimInstance Win32_OperatingSystem).Caption
    cpu      = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
    ramGB    = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 0)
  }
  gpus      = $hw.controllers
  lemonade  = @{
    version = ((& $lemonadeCli --version 2>&1) -join ' ').Trim()
    config  = $cfg
  }
  results   = $results
}

if ($Out -eq "") {
  $Out = Join-Path $PSScriptRoot ("..\bench-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".json")
}
$Out = [IO.Path]::GetFullPath($Out)
$payload | ConvertTo-Json -Depth 8 | Set-Content $Out -Encoding utf8

$okCount = ($results | Where-Object { $_.ok }).Count
Write-Host ("`n{0}/{1} 成功  ->  {2}" -f $okCount, $results.Count, $Out) -ForegroundColor Cyan
Write-Host "可把这个 JSON 分享出来做跨机对比（不含任何路径或个人信息）"
