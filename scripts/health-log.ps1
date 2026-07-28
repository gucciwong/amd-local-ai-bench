<#
.SYNOPSIS
  记录一次 Lemonade 健康探针到 CSV，用于观察长期稳定性。

.DESCRIPTION
  「跑一周看有没有问题」需要数据支撑，不能靠印象。
  本脚本跑一轮 chat + 出图，把耗时与显存写进 CSV。
  配合计划任务每天跑几次，一周后用 -Report 看趋势。

  重点监控的退化信号（都在本项目实测中出现过）：
    - 出图耗时突然从 ~2.5s 跳到几十秒 -> 显存溢出或残留配置
    - AMD shared 显存持续 > 1GB      -> 模型没被正确逐出
    - ok=False                        -> eGPU 掉线或服务崩溃

.EXAMPLE
  .\health-log.ps1                 # 记录一次
  .\health-log.ps1 -Report         # 查看历史趋势
#>
param(
  [switch]$Report,
  [string]$CsvPath = (Join-Path $PSScriptRoot "..\health-log.csv"),
  [string]$ChatModel = "Gemma-4-E2B-it-GGUF",
  [string]$ImageModel = "SD-Turbo-GGUF"
)

$ErrorActionPreference = 'Stop'
$CsvPath = [IO.Path]::GetFullPath($CsvPath)

if ($Report) {
  if (-not (Test-Path $CsvPath)) { Write-Host "还没有日志: $CsvPath"; return }
  $rows = Import-Csv $CsvPath
  Write-Host "共 $($rows.Count) 条记录`n" -ForegroundColor Cyan
  $rows | Select-Object -Last 15 | Format-Table -AutoSize
  $okRows = $rows | Where-Object { $_.ok -eq 'True' }
  if ($okRows.Count -gt 0) {
    $img = $okRows | ForEach-Object { [double]$_.imgSec }
    $chat = $okRows | ForEach-Object { [double]$_.chatSec }
    Write-Host ("`n出图: 中位 {0:N2}s  最差 {1:N2}s" -f `
      (($img | Sort-Object)[[int]($img.Count/2)]), ($img | Measure-Object -Maximum).Maximum)
    Write-Host ("对话: 中位 {0:N2}s  最差 {1:N2}s" -f `
      (($chat | Sort-Object)[[int]($chat.Count/2)]), ($chat | Measure-Object -Maximum).Maximum)
    $failed = $rows.Count - $okRows.Count
    if ($failed -gt 0) { Write-Warning "$failed 次失败" }
  }
  return
}

function Measure-Call {
  param([scriptblock]$Body)
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $ok = $true
  try { & $Body | Out-Null } catch { $ok = $false }
  $sw.Stop()
  return @{ ok = $ok; sec = [math]::Round($sw.Elapsed.TotalSeconds, 2) }
}

$api = "http://127.0.0.1:13305/api/v1"

$chatBody = @{ model = $ChatModel; messages = @(@{ role = "user"; content = "Say hello." }); max_tokens = 16 } | ConvertTo-Json -Depth 5
$c = Measure-Call { Invoke-RestMethod -Uri "$api/chat/completions" -Method Post -ContentType "application/json" -Body $chatBody -TimeoutSec 300 }

$imgBody = @{ model = $ImageModel; prompt = "a teapot"; size = "512x512"; n = 1 } | ConvertTo-Json
$i = Measure-Call { Invoke-RestMethod -Uri "$api/images/generations" -Method Post -ContentType "application/json" -Body $imgBody -TimeoutSec 300 }

# LUID 0x010CE082 是本机的 AMD 独显；换机器需重新确认
$gpu = Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory -ErrorAction SilentlyContinue |
       Where-Object { $_.Name -match "010CE082" }

$row = [PSCustomObject]@{
  time      = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
  ok        = ($c.ok -and $i.ok)
  chatSec   = $c.sec
  imgSec    = $i.sec
  amdDedGB  = if ($gpu) { [math]::Round($gpu.DedicatedUsage / 1GB, 2) } else { $null }
  amdShrGB  = if ($gpu) { [math]::Round($gpu.SharedUsage / 1GB, 2) } else { $null }
  chatModel = $ChatModel
  imgModel  = $ImageModel
}

$row | Export-Csv -Path $CsvPath -NoTypeInformation -Append -Encoding utf8
Write-Host ("[{0}] ok={1} chat={2}s img={3}s AMD={4}GB(+{5} shared)" -f `
  $row.time, $row.ok, $row.chatSec, $row.imgSec, $row.amdDedGB, $row.amdShrGB)

# 主动提示退化，不用等用户自己发现
if ($row.imgSec -gt 15)   { Write-Warning "出图明显偏慢（基线约 2.5s）——检查 enable_dgpu_gtt / 显存溢出" }
if ($row.amdShrGB -gt 1)  { Write-Warning "共享显存 $($row.amdShrGB)GB 偏高——模型可能没被正确逐出" }
if (-not $row.ok)         { Write-Warning "调用失败——检查 eGPU 连接与 lemonade status" }
