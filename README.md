# LocalAI Bench — AMD 消费级 / eGPU 本地 AI 实测配置库

在一块 **8GB AMD 移动显卡（RX 7600M XT / gfx1102，经 Thunderbolt 3 外接）** 上，
把本地 AI 的每条常见说法逐项验了一遍。

**包括那些被推翻的结论。** 网上很少有人发表失败的实验，
但恰恰是它们最省别人的时间。

## 四个反直觉的实测结论

| 说法 | 实测 |
|---|---|
| "AMD 要跑 AI 得先搞定 ROCm" | ROCm 在 **Windows 原生**装不上（上游打包 bug）。Vulkan 9.6 秒装好，8B 模型 33 tok/s |
| "开 `enable_dgpu_gtt` 让独显借系统内存" | **慢 58 倍**（2.73s → 158s），且并不能提高分辨率上限 |
| "双显卡应该分工协作" | 拆分单个扩散任务在**所有场景都是负优化**（最多慢 3.6 倍） |
| "反正 ROCm 在这块卡上就是不行" | **只说对了一半**——WSL2 上 ROCm/HIP 真实可用：LLM 打平甚至略快 Vulkan（36.4 vs 33-34 tok/s），但**出图反而慢约 3-4 倍**（10.1s vs Vulkan 2.47s）。Windows 原生这条路依然是坏的，两个方向的实测和坑都在 [docs/rocm-on-wsl2.md](docs/rocm-on-wsl2.md) |

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

采集器会自动探测显卡、每个用例前重启服务（消除状态残留），
导出**不含任何路径或个人信息**的标准化 JSON。

**Linux 或跨平台**（推荐，只需 Python 3.10+，无第三方依赖）：

```bash
python scripts/collect.py --quick                    # 最小集，约 2 分钟
python scripts/collect.py --note "RX 7900 XTX desktop"
```

**Windows**（PowerShell 版，功能等价）：

```powershell
.\scripts\collect.ps1 -Quick
.\scripts\collect.ps1 -Note "RX 7900 XTX"
```

Linux 侧的显存数据来自 `amdgpu` 的 sysfs 节点
（`/sys/class/drm/card*/device/mem_info_vram_used`），不依赖 `rocm-smi`——
后者在很多发行版上没装。

**如果你有 AMD 显卡，跑一下并把 JSON 贴到 issue 里。**
数据格式与聚合约定见 [docs/schema.md](docs/schema.md)。

### 已收到的提交

| 显卡 | 显存 | 连接 | 系统 | 提交 |
|---|---|---|---|---|
| RX 7600M XT (gfx1102) | 8 GB | Thunderbolt 3 eGPU | Windows 11 | [参考数据](submissions/rx-7600m-xt-8gb-egpu-reference.json) |

目前只有单一硬件的数据，所以**本仓库的每条结论都还只是单点观测**。
最想被验证或推翻的三个问题：

- 台式独显（非 eGPU）是否也有 `enable_dgpu_gtt` 的 58 倍塌陷？
  还是说这只是雷电总线的问题？
- 5 GiB 单次分配上限在 16GB / 24GB 卡上是否依然存在？
- 显存更大时，双卡拆分是否会翻盘？

提交后我会把你的数据加进上表，并在结论受影响时更新对应章节。

## 文档

| 文件 | 内容 |
|---|---|
| [docs/benchmark.md](docs/benchmark.md) | 完整实测数据，13 节，含负面结果与方法学教训 |
| [docs/vulkan-first.md](docs/vulkan-first.md) | 「Vulkan-first 而非 ROCm-first」的论证 |
| [docs/issue-2722-repro.md](docs/issue-2722-repro.md) | 给上游 ROCm bug 的复现报告 |

## 脚本

| 脚本 | 平台 | 用途 |
|---|---|---|
| `collect.py` | 全平台 | 标准化基准采集，导出可跨机对比的 JSON |
| `collect.ps1` | Windows | 同上，PowerShell 版 |
| `validate_submission.py` | 全平台 | 校验提交数据的 schema 与隐私，CI 也跑这个 |
| `quick-image.ps1` | Windows | 512+超分快速出图，支持批量、多模型、断点续跑 |
| `restore-config.ps1` | Windows | 一键恢复最优配置，`-Verify` 顺带冒烟 |
| `health-log.ps1` | Windows | 长期稳定性监控，记录 CSV 并对退化告警 |

批量出图会写 `_manifest.json` 记录进度。中途 Ctrl+C 或失败后
**再跑一次同样的命令即可只补未完成的那些**（`-Force` 强制全部重跑）。

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
