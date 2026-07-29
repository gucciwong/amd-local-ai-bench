# LocalAI Bench — AMD 消费级 / eGPU 本地 AI 实测配置库

在一块 **8GB AMD 移动显卡（RX 7600M XT / gfx1102，经 Thunderbolt 3 外接）** 上，
把本地 AI 的每条常见说法逐项验了一遍。

**包括那些被推翻的结论。** 网上很少有人发表失败的实验，
但恰恰是它们最省别人的时间。

## 三个反直觉的实测结论

| 说法 | 实测 |
|---|---|
| "AMD 要跑 AI 得先搞定 ROCm" | ROCm 在 Windows 上**装不上**（上游打包 bug）。Vulkan 9.6 秒装好，8B 模型 33 tok/s |
| "开 `enable_dgpu_gtt` 让独显借系统内存" | **慢 58 倍**（2.73s → 158s），且并不能提高分辨率上限 |
| "双显卡应该分工协作" | 拆分单个扩散任务在**所有场景都是负优化**（最多慢 3.6 倍） |

## 快速开始

前置：[Lemonade Server](https://lemonade-server.ai) ≥ 11.5.0，Windows 11 x64。

```powershell
# 恢复实测最优配置（每项都附实测理由）
.\scripts\restore-config.ps1 -Verify

# 快速出图：512 生成 + 纯超分，比直出 2048² 快约 20 倍
.\scripts\quick-image.ps1 -Prompt "a red fox in snow"

# 批量
.\scripts\quick-image.ps1 -PromptFile prompts.txt -OutDir out\
```

## 跑你自己的基准（欢迎提交数据）

`collect.ps1` 会自动探测显卡、每个用例前重启服务（消除状态残留），
导出**不含任何路径或个人信息**的标准化 JSON：

```powershell
.\scripts\collect.ps1 -Quick              # 最小集，约 2 分钟
.\scripts\collect.ps1 -Note "RX 7900 XTX" # 完整套件
```

**如果你有 AMD 显卡，跑一下并把 JSON 贴到 issue 里。**
目前只有单一硬件的数据，我不知道这些结论能迁移到什么程度——
尤其想知道：

- 台式独显（非 eGPU）是否也有 `enable_dgpu_gtt` 的 58 倍塌陷？
  还是说这只是雷电总线的问题？
- 5 GiB 单次分配上限在 16GB / 24GB 卡上是否依然存在？
- 显存更大时，双卡拆分是否会翻盘？

## 文档

| 文件 | 内容 |
|---|---|
| [docs/benchmark.md](docs/benchmark.md) | 完整实测数据，13 节，含负面结果与方法学教训 |
| [docs/vulkan-first.md](docs/vulkan-first.md) | 「Vulkan-first 而非 ROCm-first」的论证 |
| [docs/issue-2722-repro.md](docs/issue-2722-repro.md) | 给上游 ROCm bug 的复现报告 |

## 脚本

| 脚本 | 用途 |
|---|---|
| `restore-config.ps1` | 一键恢复最优配置，`-Verify` 顺带冒烟 |
| `quick-image.ps1` | 512+超分快速出图，支持批量与多模型 |
| `collect.ps1` | 标准化基准采集，导出可跨机对比的 JSON |
| `health-log.ps1` | 长期稳定性监控，记录 CSV 并对退化告警 |

## 8GB 卡的推荐配置

```
sdcpp.backend      = vulkan
llamacpp.backend   = vulkan
sdcpp.vulkan_args  = (empty)   # 双卡拆分是负优化
max_loaded_models  = 1
enable_dgpu_gtt    = false     # 关键：设 true 慢 58 倍
ctx_size           = 32768     # 默认 4096 太保守，仅多花 1GB
```

模型组合：**Gemma-4-E2B**（对话，55.6 tok/s）+ **SD-Turbo-GGUF**（出图，2.52GB）——
交替使用一轮 ~3.0s，比 8B + 未量化版的 ~19s 快 **6.3 倍**。

## 一条方法学提醒

本项目最大的教训不是某个配置值，而是：

> **跨路径做性能对比前，先抓两边进程的真实命令行核对参数。**

我曾据此得出"双卡拆分快 5.2 倍"的结论并差点写进文档——
真相是我直接调 CLI 时漏了服务端默认带的两个参数，撞了显存墙。
补上参数重测，方向完全反转。

同样地，**测性能前必须清空显存**：两个进程争 8GB 会让同一个任务
从 24s 劣化到 825s（34 倍），足以淹没任何真实的配置差异。

## 许可

MIT
