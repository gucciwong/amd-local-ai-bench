# 在 AMD + Windows 上训练/微调模型：方法论盘点

本仓库到目前为止全部是**推理**（LLM 生成、出图）。这篇是第一次往
**训练/微调**方向挖。**两条路都实测能跑**：WSL2 上的 PyTorch-ROCm（需要一个
不算显而易见的修复）和原生 Windows 上的 `torch-directml`（开箱即用）。

> 置信度标注延续 [benchmark.md](benchmark.md) 的规则：🟢 已对照/已复现，
> 🟡 单轮，🔴 未验证/转述自调研。

> **勘误说明**：本文档最初发布时结论是"PyTorch-ROCm 在 WSL2 上训练是硬阻塞，
> `torch.cuda.is_available()` 返回 False 且大概率无解"，并据此改过
> [AMD_skills 的 picking-amd-gpu-backend 技能](https://github.com/gucciwong/AMD_skills/pull/9)
> 加了警示。**这个结论错了**——后续追查真的找到了修复方法并亲测验证，
> 本文档已更新。原因和修复过程完整保留在下面"WSL2 上的曲折"一节，
> 因为踩过的这个坑本身就是有价值的信息，不是丢人的地方——真正该避免的
> 是在错误结论已经发布之后不去更正它。

## 结论先行

| 方法 | 状态 | 置信度 |
|---|---|---|
| PyTorch-ROCm 在 **WSL2** 上训练 | ✅ **能跑**——删掉 pip 轮子自带的 `libhsa-runtime64.so`，改用系统 ROCm 的 WSL 感知版本后，`torch.cuda.is_available()` 返回 `True`，4096 维 3 层 MLP + fp16 autocast 训练全部正确收敛 | 🟢 三个规模（玩具级/4096维/fp16）全部验证，含 GPU vs CPU 结果比对 |
| `torch-directml`（原生 Windows） | ✅ **能跑，开箱即用**，无需任何修复；但确认了一个真实的算子静默回退 CPU 的案例（`aten::lerp.Scalar_out`，AdamW 内部用到） | 🟢 单次但含正确性验证+真实警告复现 |
| Unsloth 官方 AMD 支持（RDNA3 QLoRA） | 未测 | 🔴 转述自调研，是下一轮最值得测的一项——**此前文档说"第一道门槛过不去"的判断已随上面的修复失效**，见文末 |
| Microsoft Olive / ONNX Runtime | 未测，且据调研主要面向 NPU 不是这块独显 | 🔴 转述自调研 |
| 原生 Windows ROCm（TheRock 之外，AMD 2025 新推的统一安装包） | 不适用 | 🔴 转述自调研——AMD 官方 Windows 支持矩阵明确只列 `gfx1100/1101/1200/1201`，**没有 gfx1102**，且即使是受支持的卡，原生 Windows 也只做推理，训练仍需 WSL2（但 WSL2 本身现在确认可行，见上） |

## WSL2 上的曲折：先撞死，再找到修复

延续 [rocm-on-wsl2.md](rocm-on-wsl2.md) 已经装好的 ROCm 7.14 环境
（`amdrocm7.14-gfx1102` + `amdrocm-wsl`），装官方 nightly ROCm 轮子：

```bash
python3 -m venv torch-rocm-venv
source torch-rocm-venv/bin/activate
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/rocm7.2
```

装得很顺利（`torch-2.14.0.dev20260728+rocm7.2`，6.2GB 主 wheel）。第一步验证：

```python
import torch
print(torch.cuda.is_available())
```

```
W ... agent.cpp:608] sysfs nodes path '/sys/class/kfd/kfd/topology/nodes' does not exist
False
```

直接查文件系统：WSL2 里只有 `/dev/dxg`，没有 `/dev/kfd` 设备节点，也没有
`/sys/class/kfd` 目录。当时据此得出"这是 WSL2 GPU 直通模型的架构性限制，
大概率无解"的结论——**这个推断本身没有错**（WSL2 确实没有原生 KFD 接口），
**错的是"无解"这个判断**：没意识到 pip 装的 torch wheel 自带了一份
Linux-only 的 `libhsa-runtime64.so`，而这份自带的运行时优先于系统装的那份
被加载，掩盖了系统那份运行时其实已经支持 WSL2 dxcore 桥接这件事。

### 真正的修复

追问"有没有人在 WSL2 上真的跑通过 PyTorch-ROCm"之后找到了根因和修复：
AMD 有一个专门的 WSL 桥接层叫 **ROCDXG**（`librocdxg.so`），系统装的
ROCm（`amdrocm7.14-gfx1102` + `amdrocm-wsl`）里的 `libhsa-runtime64.so`
**已经链接了它**——用 `strings` 直接能看到：

```bash
$ strings /opt/rocm/core-7.14/lib/libhsa-runtime64.so.1.21.0 | grep -iE 'dxcore|dxg|wsl'
HSA_ENABLE_DXG_DETECTION
/dev/dxg
DxgAbiCheck
librocdxg.so
```

但 pip 装的 torch wheel 会把自己的 `libhsa-runtime64.so`（标准 Linux-only
构建，没有 DXG 支持）塞进 `torch/lib/` 和 `triton/backends/amd/lib/`，
这份自带的运行时会抢在系统那份之前被动态链接器找到并加载。修复就是把它
删掉，让加载器落到系统那份（通过 `LD_LIBRARY_PATH` 指向
`/opt/rocm/core-7.14/lib`）：

```bash
mv torch-rocm-venv/lib/python3.12/site-packages/torch/lib/libhsa-runtime64.so* /tmp/backup/
mv torch-rocm-venv/lib/python3.12/site-packages/triton/backends/amd/lib/libhsa-runtime64.so* /tmp/backup/
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib
```

再测一次：

```
torch version: 2.14.0.dev20260728+rocm7.2
cuda available: True
device name: AMD Radeon RX 7600M XT
```

### 三个规模上验证过，不是只测了一次就信

不满足于"枚举成功"，实际跑了三档训练，确认真的能算、能收敛：

1. **正确性**：512×512 矩阵乘法，GPU 结果与 CPU 参考结果最大误差
   `1.1e-4`（浮点精度范围内），不是随手编造的数字。
2. **小规模训练**：256→512→256 三层 MLP，20 step，loss 从
   `1.0669` 降到 `0.2238`——真实梯度下降，不是空跑。
3. **大规模 + fp16**：4096 维 3 层 MLP（更接近真实 GEMM 形状，
   目的是压一下调研里提到的 `hipBLASLt` 缺 gfx1102 kernel 那个已知问题），
   30 step loss 从 `1.0015` 降到 `0.0060`；额外用 `torch.autocast(dtype=fp16)`
   跑了一轮（fp16 GEMM 是 hipBLASLt kernel 选择最吃紧的路径），
   loss 从 `1.0375` 降到 `0.0537`，**全程没有触发调研里提到的
   `TensileLibrary_lazy_gfx1102.dat` 之类的 hipBLASLt kernel 缺失报错**。

这不代表 hipBLASLt 的 gfx1102 kernel 缺口（`ROCm/TheRock#2847`）不存在——
可能只是这几个测试用到的算子形状没踩中缺失的那部分 kernel。但至少证明
**常见的 Linear+激活函数+AdamW 这套最主流的训练组合是完全能跑的**，
不是"看起来枚举成功但一算就崩"的假阳性。

## `torch-directml`（原生 Windows）：开箱即用，但有真实的静默回退案例

本机原生 Windows 只装了 Python 3.14，torch-directml 官方轮子只到 3.12——
装了一套独立的 Python 3.12（python.org 官方安装包，`python-3.12.10-amd64.exe`，
装到 `C:\Python312`，不影响默认的 3.14 环境，也没加进 PATH）：

```powershell
pip install torch torch-directml
```

**没有任何 ROCm 那种曲折**，装完直接能用：

```python
import torch_directml
print(torch_directml.is_available())       # True
print(torch_directml.device_name(0))       # AMD Radeon RX 7600M XT
```

同样跑了正确性验证（矩阵乘法误差 `6.1e-5`）和 20 step 训练
（loss `1.0531` → `0.2171`），**都过了**。

**但确认了调研提到的"静默回退 CPU"不是空穴来风**，这次亲眼见到了：

```
UserWarning: The operator 'aten::lerp.Scalar_out' is not currently supported
on the DML backend and will fall back to run on the CPU. This may have
performance implications.
```

这是 `AdamW` 优化器内部用到的一个算子（`torch._foreach_lerp_`），
**训练脚本本身完全不知道这件事**——不看 warning 的话，你以为整个训练过程
都在 GPU 上，实际上优化器状态更新的这一步偷偷跑去了 CPU。玩具级模型上
这点开销可以忽略，但换成大模型、大批量的 AdamW 状态更新，这种"部分算子
静默回落 CPU"就会变成一个隐藏的性能陷阱——而且没有任何机制强制报错，
纯粹是希望你去看 warning。

## 两条路怎么选

- **两条路都能训练**，方向不冲突：`torch-directml` 装起来更省事（无需
  改运行时文件），`PyTorch-ROCm/WSL2` 装起来麻烦一点点（需要知道
  ROCDXG 这个修复），但用的是"真"的 ROCm/HIP 计算栈而不是走 DirectX 12
  这层转译，理论上算子覆盖和长期维护状态更接近 Linux 原生。
- 本轮**没有做两者训练吞吐的正式对照**——玩具级 MLP 上 DirectML 表格上
  的数字看起来比 WSL2/ROCm 快（0.337s vs 1.311s 完成 20 step），但样本
  太小、太受 Python/调度开销主导，**不构成一个可信的性能结论**，只是
  记录下来避免被误读成"DirectML 更快"的定论。真要比较需要一个有意义
  规模的真实模型和多轮重复测量，留作下一轮。

## 下一轮最值得测的一项：Unsloth 官方 AMD 支持

调研翻出一个很新的进展：**Unsloth（LoRA/QLoRA 微调框架）联合 AMD
官方推出了 AMD GPU 支持**，明确覆盖 RDNA3（含 gfx1102 所在的架构族）、
Windows/WSL/Linux 三种环境（AMD 官方技术文章 2026，Unsloth 官方文档）。
这是全网**目前找不到任何 gfx1102 实测数据**的空白点。

依据 AMD 自己 tracking issue（`amd/gaia#667`）给出的显存目标，8GB 卡上
现实的量级大概是：Qwen3-1.7B FP8 GRPO ≈5GB、Qwen3-4B QLoRA ≈4GB、
**Qwen3-8B QLoRA ≈8GB**（贴着上限）——这些是 AMD 自己给的目标数字，
**没有在这块卡上验证过**。

**此前版本的文档认为这条路"第一道门槛就过不去"**（因为当时以为 WSL2 上
PyTorch 连设备都枚举不到）——**这个判断随着上面的修复已经失效**：既然
`torch.cuda.is_available()==True` 且训练循环真能跑，Unsloth 跑在 PyTorch
之上，理论上应该也能装起来。真正剩下的门槛是：

1. 预发布版 `bitsandbytes`（Unsloth 文档要求 ≥0.49.1，为了绕开 AMD 上
   4-bit 解码的一个已知 NaN bug）——这个还没测过是否兼容我们这套
   ROCDXG 修复后的环境
2. Unsloth 自身的 HIP kernel（如果有自定义 CUDA/HIP 扩展）是否也会像
   torch 官方 wheel 一样带着不兼容 WSL2 的运行时，需要同样的修复手法
3. 一个能在 8GB 内跑完至少几十个 step 的小模型（先 Qwen2.5-0.5B/1.7B
   验证反向传播能不能过，再考虑往 8B 冲）

## 和已有结论的关系

- **不推翻**任何推理侧结论——[rocm-on-wsl2.md](rocm-on-wsl2.md) 里
  llama.cpp/stable-diffusion.cpp 的 HIP 后端推理能跑，训练现在也证实能跑，
  两者不矛盾，只是各自需要的运行时深度不同（这也是为什么最初误判训练
  是硬阻塞——错把"这个 wheel 默认配置跑不通"当成了"这条路本身跑不通"）。
- 呼应 [benchmark.md](benchmark.md) 的"置信度标注"传统，也呼应本仓库
  "双卡拆分快 5.2 倍"那次自我推翻的教训：**看起来像结论的第一次测量结果，
  在有更深入的信息之前不要急着发布成定论**，尤其是负面结论——这次的
  代价是已经改了一次 AMD_skills 技能文档，现在需要再改一次纠正回来。

## 已知边界

- 只在这一块 8GB gfx1102 eGPU 上测过，没有裸机 Linux 或其他 RDNA3
  型号的对照
- WSL2/ROCm 训练测试全部是玩具级/合成数据的 MLP，**没有测过真实的
  transformer 模型微调**（下一步是 Unsloth 或裸手写一个小 transformer）
- 没有做正式的 DirectML vs ROCm/WSL2 训练吞吐对照，上面给的计时数字
  仅供参考，不构成结论
- `torch-directml` 的算子覆盖缺口只确认了 `aten::lerp.Scalar_out` 这一个，
  没有做穷举式的算子覆盖率扫描
- Unsloth AMD 支持完全没测，是全文档最大的空白和下一轮候选
