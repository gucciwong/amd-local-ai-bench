<#
.SYNOPSIS
  一键恢复 Lemonade 在 RX 7600M XT (eGPU) 上的实测最优配置。

.DESCRIPTION
  换驱动、重装 Lemonade、或误改配置后运行本脚本。
  每一项都对应 docs/benchmark.md 里的实测依据，不是拍脑袋的默认值。

.EXAMPLE
  .\restore-config.ps1
  .\restore-config.ps1 -Verify   # 顺带跑一次出图确认真的恢复了
#>
param(
  [switch]$Verify
)

$ErrorActionPreference = 'Stop'

$lemonade = Join-Path $env:LOCALAPPDATA "lemonade_server\bin\lemonade.exe"
if (-not (Test-Path $lemonade)) {
  throw "找不到 lemonade CLI: $lemonade"
}

# 每项配置附实测理由，方便日后回看时判断是否还成立
$settings = @(
  @{ k = "sdcpp.backend";      v = "vulkan"; why = "Vulkan 热态 2.47s vs CPU 50.58s，快约 20 倍" }
  @{ k = "llamacpp.backend";   v = "vulkan"; why = "8B 稳定 33-34 tok/s，实测跑在 AMD 独显上" }
  @{ k = "sdcpp.vulkan_args";  v = "";       why = "双卡拆分是负优化（交替一轮 17.9s vs 13.2s），必须留空" }
  @{ k = "max_loaded_models";  v = "1";      why = "设 2 无收益：8B+SD-Turbo=9.3GB 装不进 8GB，从未真正共存" }
  @{ k = "enable_dgpu_gtt";    v = "false";  why = "关键！设 true 出图从 2.73s 崩到 158s（慢 58 倍）" }
  @{ k = "ctx_size";           v = "32768";  why = "默认 4096 太保守；32K 仅多花 1GB（5.39->6.39GB）" }
)

# 推荐模型组合：交替一轮 ~3.0s，比 8B+原版的 ~19s 快 6.3 倍，且两模型真正共存
$recommended = @(
  @{ role = "对话"; model = "Gemma-4-E2B-it-GGUF"; why = "55.6 TPS，反超 8B 的 33.6" }
  @{ role = "出图"; model = "SD-Turbo-GGUF";       why = "与原版同速，显存少 1.6GB" }
)

Write-Host "恢复 Lemonade 最优配置..." -ForegroundColor Cyan
foreach ($s in $settings) {
  & $lemonade config set "$($s.k)=$($s.v)" 2>&1 | Out-Null
  $shown = if ($s.v -eq "") { "(empty)" } else { $s.v }
  Write-Host ("  {0,-20} = {1,-8}  # {2}" -f $s.k, $shown, $s.why)
}

Write-Host "`n核实实际生效值:" -ForegroundColor Cyan
$actual = & $lemonade config 2>&1 | Out-String
$bad = 0
foreach ($s in $settings) {
  $pattern = [regex]::Escape($s.k) + '\s+(\S.*?)\s*$'
  $m = [regex]::Match($actual, $pattern, 'Multiline')
  $got = if ($m.Success) { $m.Groups[1].Value.Trim() } else { "<未读到>" }
  $want = if ($s.v -eq "") { "(empty)" } else { $s.v }
  $ok = ($got -eq $want)
  if (-not $ok) { $bad++ }
  $color = if ($ok) { "Green" } else { "Red" }
  Write-Host ("  [{0}] {1,-20} = {2}" -f $(if ($ok) { "OK" } else { "!!" }), $s.k, $got) -ForegroundColor $color
}

if ($bad -gt 0) {
  Write-Warning "$bad 项未按预期生效，请手动检查 lemonade config"
} else {
  Write-Host "`n全部 $($settings.Count) 项已恢复。" -ForegroundColor Green
}

Write-Host "`n推荐模型组合（交替场景实测最快）:" -ForegroundColor Cyan
foreach ($r in $recommended) {
  Write-Host ("  {0}  {1,-24}  # {2}" -f $r.role, $r.model, $r.why)
}

if ($Verify) {
  Write-Host "`n跑一次 512 出图验证..." -ForegroundColor Cyan
  $body = @{ model = "SD-Turbo"; prompt = "a teapot"; size = "512x512"; n = 1 } | ConvertTo-Json
  $sw = [Diagnostics.Stopwatch]::StartNew()
  try {
    Invoke-RestMethod -Uri "http://127.0.0.1:13305/api/v1/images/generations" `
      -Method Post -ContentType "application/json" -Body $body -TimeoutSec 300 | Out-Null
    $sw.Stop()
    $sec = [math]::Round($sw.Elapsed.TotalSeconds, 2)
    # 冷启动含模型加载会到 25-30s；热态基线是 2.47s
    if ($sec -lt 40) {
      Write-Host "  出图成功 ${sec}s（热态基线 2.47s，冷启动 25-30s 属正常）" -ForegroundColor Green
    } else {
      Write-Warning "  出图 ${sec}s 明显偏慢，检查 enable_dgpu_gtt 是否为 false、eGPU 是否连接"
    }
  }
  catch {
    Write-Warning "  出图失败: $($_.Exception.Message)（服务未启动？eGPU 未连接？）"
  }
}
