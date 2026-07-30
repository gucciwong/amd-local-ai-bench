<#
.SYNOPSIS
  快速出图：512 生成 + RealESRGAN 纯超分。

.DESCRIPTION
  实测比直接生成 2048x2048 快约 20 倍（24s vs 484s），显存占用 0.85GB vs 6.66GB。
  关键在于超分不做第二轮扩散去噪——sd-cpp 自带的 --hires 会做，所以反而更慢。

  路径全部动态解析，不写死 HuggingFace 快照哈希（那个会随模型更新而变）。

.EXAMPLE
  .\quick-image.ps1 -Prompt "a red fox in snow"
  .\quick-image.ps1 -Prompt "anime girl" -Anime -Out D:\out.png
  .\quick-image.ps1 -Prompt "a castle" -Base 768 -NoUpscale

.EXAMPLE
  # 批量：多个 prompt 一次跑完，模型只加载一次
  .\quick-image.ps1 -Prompts "a red fox","a blue whale","a green forest" -OutDir D:\batch

.EXAMPLE
  # 批量：从文本文件读，每行一个 prompt（# 开头为注释）
  .\quick-image.ps1 -PromptFile prompts.txt -OutDir D:\batch
#>
param(
  [Parameter(Mandatory = $true, ParameterSetName = 'Single')]
  [string]$Prompt,

  # 批量模式：多个 prompt
  [Parameter(Mandatory = $true, ParameterSetName = 'Batch')]
  [string[]]$Prompts,

  # 批量模式：从文件读 prompt，每行一个
  [Parameter(Mandatory = $true, ParameterSetName = 'File')]
  [string]$PromptFile,

  # 出图模型。实测对比见 docs/benchmark.md：
  #   turbo-gguf : 与 turbo 同速但省 1.6GB 显存 —— 默认
  #   turbo      : 未量化原版
  #   sdxl       : 画质好但占 7.16GB；8GB 卡上必须独占显存才跑得动
  #   sdxl-turbo : SDXL 级画质 + Turbo 速度，画质档首选
  [ValidateSet('turbo-gguf', 'turbo', 'sdxl', 'sdxl-turbo')]
  [string]$Model = 'turbo-gguf',

  # 跑之前不要卸载 Lemonade 里常驻的模型。
  # 默认会卸载：实测两个进程争 8GB 显存会让 SDXL 从 24s 劣化到 825s（34 倍）
  [switch]$KeepResident,

  # 批量模式：忽略已有进度，强制全部重跑
  [Parameter(ParameterSetName = 'Batch')]
  [Parameter(ParameterSetName = 'File')]
  [switch]$Force,

  # 基础生成尺寸。512 是 SD-Turbo 的训练分辨率，速度/质量最划算。
  # 用 -Model sdxl 时建议 1024（SDXL 的原生分辨率）
  [int]$Base = 512,

  # 单张输出路径，默认放当前目录带时间戳
  [Parameter(ParameterSetName = 'Single')]
  [string]$Out = "",

  # 批量输出目录
  [Parameter(ParameterSetName = 'Batch')]
  [Parameter(ParameterSetName = 'File')]
  [string]$OutDir = "",

  # 使用动漫专用超分模型
  [switch]$Anime,

  # 只生成不超分
  [switch]$NoUpscale,

  [int]$Steps = 20
)

$ErrorActionPreference = 'Stop'

function Resolve-One {
  param([string]$Root, [string]$Filter, [string]$What)
  $hit = Get-ChildItem $Root -Recurse -Filter $Filter -File -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $hit) { throw "找不到 $What（在 $Root 下搜 $Filter）。先跑：lemonade pull <model>" }
  return $hit.FullName
}

$cacheRoot = Join-Path $env:USERPROFILE ".cache"
$sdCli = Join-Path $cacheRoot "lemonade\bin\sd-cpp\vulkan\sd-cli.exe"
if (-not (Test-Path $sdCli)) {
  throw "找不到 sd-cli（$sdCli）。先跑：lemonade backends install sd-cpp:vulkan"
}

$hf = Join-Path $cacheRoot "huggingface"

# 各模型的实际权重文件名。lemonade pull 之后落在 HF 缓存里，
# 文件名与模型名并不一致，所以这里显式映射
$modelFile = switch ($Model) {
  'turbo-gguf'  { 'sd_turbo-f16-q8_0.gguf' }
  'turbo'       { 'sd_turbo.safetensors' }
  'sdxl'        { 'sd_xl_base_1.0.safetensors' }
  'sdxl-turbo'  { 'sd_xl_turbo_1.0_fp16.safetensors' }
}

# 显存争用是本项目遇到的最大性能陷阱：sd-cli 与 Lemonade 常驻的 sd-server
# 同时各要 6-7GB，8GB 卡直接崩到 1/34 速度。默认先让服务端腾出显存。
if (-not $KeepResident) {
  $cli = Join-Path $env:LOCALAPPDATA "lemonade_server\bin\lemonade.exe"
  if (Test-Path $cli) {
    try {
      $loaded = & $cli status 2>&1 | Select-String -Pattern '^\s*(\S+)\s+(image|text)\s' |
                ForEach-Object { $_.Matches[0].Groups[1].Value }
      foreach ($lm in $loaded) {
        Write-Host "  卸载常驻模型 $lm 以释放显存（-KeepResident 可跳过）" -ForegroundColor DarkGray
        & $cli unload $lm 2>&1 | Out-Null
      }
    } catch { }
  }
}
# 注意用独立变量名：$Model 带 ValidateSet，赋路径进去会触发校验失败
$modelPath = Resolve-One -Root $hf -Filter $modelFile -What "$Model 模型（$modelFile）"

# SDXL 原生分辨率是 1024，用 512 出图质量会明显劣化
if ($Model -eq 'sdxl' -and $Base -lt 1024) {
  Write-Warning "SDXL 原生分辨率为 1024，当前 -Base $Base 可能质量不佳（建议 -Base 1024）"
}

# 超分模型只解析一次，批量时避免重复扫盘
# 注意实际文件名是 RealESRGAN_x4plus_anime_6B.pth（下划线，非连字符）
$esr = $null
if (-not $NoUpscale) {
  $esrFilter = if ($Anime) { "*anime*.pth" } else { "RealESRGAN_x4plus.pth" }
  $esr = Resolve-One -Root $hf -Filter $esrFilter -What "RealESRGAN 超分模型"
}

Add-Type -AssemblyName System.Drawing

function New-Image {
  param([string]$Text, [string]$Dest)

  # --vae-tiling 与 --diffusion-fa 是 Lemonade 服务端的默认参数，
  # 漏掉它们会让高分辨率请求申请超大缓冲直接 OOM
  $genArgs = @(
    "-m", $modelPath, "-p", $Text,
    "-H", $Base, "-W", $Base,
    "--steps", $Steps,
    "--vae-tiling", "--diffusion-fa"
  )

  $sw = [Diagnostics.Stopwatch]::StartNew()

  if ($NoUpscale) {
    & $sdCli @genArgs -o $Dest | Out-Null
    $sw.Stop()
    if (-not (Test-Path $Dest)) { throw "生成失败: $Text" }
    return [PSCustomObject]@{ dim = "${Base}x${Base}"; gen = $sw.Elapsed.TotalSeconds; up = 0; total = $sw.Elapsed.TotalSeconds }
  }

  $tmp = Join-Path ([IO.Path]::GetTempPath()) ("qi-" + [Guid]::NewGuid().ToString("N") + ".png")
  try {
    & $sdCli @genArgs -o $tmp | Out-Null
    if (-not (Test-Path $tmp)) { throw "生成失败: $Text" }
    $genSec = $sw.Elapsed.TotalSeconds

    & $sdCli -M upscale --upscale-model $esr -i $tmp -o $Dest | Out-Null
    $sw.Stop()
    if (-not (Test-Path $Dest)) { throw "超分失败: $Text" }

    $img = [System.Drawing.Image]::FromFile($Dest)
    $dim = "$($img.Width)x$($img.Height)"
    $img.Dispose()
    return [PSCustomObject]@{ dim = $dim; gen = $genSec; up = ($sw.Elapsed.TotalSeconds - $genSec); total = $sw.Elapsed.TotalSeconds }
  }
  finally {
    if (Test-Path $tmp) { Remove-Item $tmp -Force -ErrorAction SilentlyContinue }
  }
}

# 收敛成统一的 prompt 列表，单张也走同一条路径
$list = switch ($PSCmdlet.ParameterSetName) {
  'Single' { @($Prompt) }
  'Batch'  { $Prompts }
  'File'   {
    if (-not (Test-Path $PromptFile)) { throw "找不到 prompt 文件: $PromptFile" }
    Get-Content $PromptFile | Where-Object { $_.Trim() -ne "" -and -not $_.TrimStart().StartsWith("#") }
  }
}

if ($PSCmdlet.ParameterSetName -eq 'Single') {
  if ($Out -eq "") {
    $Out = Join-Path (Get-Location) ("img-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".png")
  }
  $r = New-Image -Text $list[0] -Dest $Out
  Write-Host ("完成 {0}  生成 {1:N1}s + 超分 {2:N1}s = {3:N1}s  ->  {4}" -f $r.dim, $r.gen, $r.up, $r.total, $Out)
  return
}

if ($OutDir -eq "") {
  $OutDir = Join-Path (Get-Location) ("batch-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# 断点续跑用的进度清单。批量 20 张要 8 分钟，中途 Ctrl+C 或失败
# 不该让已完成的白跑。文件名按 prompt 内容的哈希取，不按序号——
# 否则改动 prompt 列表的顺序会让续跑张冠李戴。
$manifestPath = Join-Path $OutDir "_manifest.json"
$done = @{}
if ((Test-Path $manifestPath) -and -not $Force) {
  try {
    $loaded = Get-Content $manifestPath -Raw | ConvertFrom-Json
    foreach ($e in $loaded.entries) {
      # 只认文件真的还在的条目，避免手动删图后仍被跳过
      if (Test-Path (Join-Path $OutDir $e.file)) { $done[$e.key] = $e.file }
    }
    if ($done.Count -gt 0) {
      Write-Host ("续跑：已完成 {0} 张，将跳过（-Force 可强制重跑全部）" -f $done.Count) -ForegroundColor DarkGray
    }
  }
  catch {
    Write-Warning "进度清单读取失败，本次从头开始：$($_.Exception.Message)"
  }
}

function Get-PromptKey {
  param([string]$Text)
  $sha = [Security.Cryptography.SHA256]::Create()
  try {
    $bytes = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes("$Base|$Steps|$Model|$Anime|$NoUpscale|$Text"))
    return ([BitConverter]::ToString($bytes) -replace '-').Substring(0, 12).ToLower()
  }
  finally { $sha.Dispose() }
}

$n = 0
$failed = 0
$skipped = 0
$totalSec = 0.0
$entries = [Collections.ArrayList]::new()

foreach ($p in $list) {
  $n++
  $key = Get-PromptKey -Text $p

  if ($done.ContainsKey($key)) {
    $skipped++
    [void]$entries.Add([PSCustomObject]@{ key = $key; file = $done[$key]; prompt = $p })
    Write-Host ("[{0}/{1}] 跳过（已完成）  {2}" -f $n, $list.Count, $done[$key]) -ForegroundColor DarkGray
    continue
  }

  $name = "{0:d3}-{1}.png" -f $n, $key
  $dest = Join-Path $OutDir $name
  try {
    $r = New-Image -Text $p -Dest $dest
    $totalSec += $r.total
    [void]$entries.Add([PSCustomObject]@{ key = $key; file = $name; prompt = $p })
    Write-Host ("[{0}/{1}] {2:N1}s  {3}  {4}" -f $n, $list.Count, $r.total, $name, $p)
  }
  catch {
    $failed++
    Write-Warning ("[{0}/{1}] 失败: {2}" -f $n, $list.Count, $_.Exception.Message)
  }
  finally {
    # 每张之后立刻落盘。放在 finally 里，Ctrl+C 也能保住进度。
    @{ base = $Base; steps = $Steps; model = $Model; entries = $entries } |
      ConvertTo-Json -Depth 4 | Set-Content $manifestPath -Encoding utf8
  }
}

$made = $n - $failed - $skipped
$avg = if ($made -gt 0) { $totalSec / $made } else { 0 }
Write-Host ("批量完成: 新生成 {0} 张, 跳过 {1} 张, 失败 {2} 张, 本次 {3:N1}s (平均 {4:N1}s/张)  ->  {5}" -f `
  $made, $skipped, $failed, $totalSec, $avg, $OutDir)
if ($failed -gt 0) {
  Write-Host "再跑一次同样的命令即可只重试失败的那些。" -ForegroundColor DarkGray
}
