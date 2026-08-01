# 在 AMD + Windows 上训练/微调模型：方法论盘点

本仓库到目前为止全部是**推理**（LLM 生成、出图）。这篇是第一次往
**训练/微调**方向挖。**WSL2 上的 PyTorch-ROCm 训练真实能跑**（需要一个
不算显而易见的修复），并且用它实测跑通了 Unsloth 官方 AMD QLoRA 支持
（1B/4B/8B 三个规模都对上了 AMD 自己给的显存量级），还专门验证了
微调后的模型是不是真的学到了新东西（不是只看 loss 数字）——全网目前
找不到第二份 gfx1102 数据。原生 Windows 上的 `torch-directml` 对合成
负载能训练，但对真实 transformer 模型撞上一个已知、官方不打算修的崩溃；
亲自验证了社区给的绕过方法，发现它只是把崩溃换成了静默的 `nan` loss，
没有真正解决问题。

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
| PyTorch-ROCm 在 **WSL2** 上训练 | ✅ **能跑**——删掉 pip 轮子自带的 `libhsa-runtime64.so`，改用系统 ROCm 的 WSL 感知版本后，`torch.cuda.is_available()` 返回 `True`，玩具 MLP + 4096维 MLP + fp16 autocast + 真实 transformer LoRA 训练全部正确收敛 | 🟢 四个规模全部验证，含 GPU vs CPU 结果比对 |
| **Unsloth 官方 AMD 支持（RDNA3 QLoRA）** | ✅ **实测成功，全网第一份 gfx1102 数据**——1B/4B/8B 三个规模全部验证，峰值显存 1.33GB/3.89GB/7.83GB，跟 AMD 自己给的"≈4GB/≈8GB贴上限"估算几乎逐位对上，8B 那次只剩约 345MB 余量、没有 OOM。**专门验证了"真学到东西"而不只是 loss 下降**：用模型不可能已知的编造事实微调，微调前 4/4 答不出来，微调后 4/4 准确背出 | 🟢 三个规模+一次学习质量测试全部独立验证 |
| `torch-directml`（原生 Windows） | ⚠️ **对合成 MLP 能跑，对真实 transformer 模型崩溃**——`masked_fill` 报 `RuntimeError: value cannot be converted to type uint8_t without overflow`，fp16/fp32/eager 都一样。已确认是 `microsoft/DirectML#702`，官方标记 `not_planned`。**亲自验证了社区给的 `torch.where` 绕过方法**：崩溃确实消失，但训练变成从第 0 步起 loss 恒为 `nan`——只是把崩溃换成了静默失败，没有真正修好 | 🟢 崩溃复现三次+对上已知 issue；🟢 绕过方法亲测：解决崩溃但暴露 NaN 问题 |
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

## Unsloth 官方 AMD 支持：实测成功，全网第一份 gfx1102 数据

调研翻出一个很新的进展：**Unsloth（LoRA/QLoRA 微调框架）联合 AMD
官方推出了 AMD GPU 支持**，明确覆盖 RDNA3（含 gfx1102 所在的架构族）、
Windows/WSL/Linux 三种环境。当时这是全网**找不到任何 gfx1102 实测数据**
的空白点——现在补上了。

在已经修好的 WSL2/ROCm 环境（见上）里装：

```bash
pip install 'unsloth[amd]'
pip install --force-reinstall --no-cache-dir --no-deps \
  "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_x86_64.whl"
```

**一个真实的坑**：`pip install unsloth[amd]` 会**静默把已经修好的 ROCm
torch 换成 PyPI 默认的 CUDA 版**（装出来是 `torch-2.11.0+cu130`，带一堆
`nvidia-cublas`/`nvidia-cudnn` 依赖——这台机器根本没有 NVIDIA 显卡）。
不是报错，是**安装成功但换了一个不能用的后端**，`torch.cuda.is_available()`
变回 `False`。修法是装完 unsloth 后，**按 Unsloth 文档给的版本范围
（ROCm 7.2+ 对应 `torch>=2.11.0,<2.12.0`）用 ROCm 专用 index 重新强制装回**：

```bash
pip install "torch>=2.11.0,<2.12.0" torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/rocm7.2 --upgrade --force-reinstall
```

再重复一遍前面那个"删掉 wheel 自带 `libhsa-runtime64.so`"的修复
（每次重装 torch 都会带回一份新的自带运行时，需要重新删）。

修好之后，跑一个真实的 QLoRA 微调（`unsloth/Llama-3.2-1B-Instruct`，
4-bit，LoRA rank 16，20 step，玩具级指令数据集）：

```
==((====))==  Unsloth 2026.7.6: Fast Llama patching. Transformers: 5.5.0.
   \\   /|    AMD Radeon RX 7600M XT. Num GPUs = 1. Max memory: 7.984 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.11.0+rocm7.2. ROCm Toolkit: 7.2.26015. Triton: 3.6.0
Trainable parameters = 11,272,192 of 1,247,086,592 (0.90% trained)
```

结果：20 step 跑完（首步 11.5s 含 Triton kernel JIT 编译，稳态后约
0.3-0.5s/step），loss 从 `4.84` 降到 `0.84`（中间因为极小 batch+学习率
预热有噪声，但整体明显下降），**峰值显存只用了 1.33GB**——对一块 8GB
卡来说非常宽松，AMD 自己给的"Qwen3-8B QLoRA 贴着 8GB 上限"的量级，
在 1B 模型上验证是完全用不满的，这个空间大小的对照后续可以往更大的
模型上推。

**过程中两个额外的真实信号**（都只是警告，不是阻塞）：

- `Warning: Attempting to use CK on an unsupported architecture! Cannot
  set backend to CK` —— AMD 自己的 Composable Kernel（CK）库明确不支持
  gfx1102，自动回退到别的路径。呼应本仓库反复出现的模式："gfx1102 是
  RDNA3 里被漏掉的那个"。
- `Mem Efficient attention on Current AMD GPU is still experimental.
  Enable it with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` —— 一个
  具体的、没试过的加速开关，留给下一轮。

### 往上推：直接验证 AMD 自己给的显存量级，三个规模全部命中

前面提到 AMD 自己在 `amd/gaia#667` 给的目标是"Qwen3-4B QLoRA ≈4GB、
Qwen3-8B QLoRA ≈8GB（贴着上限）"——这些数字之前只是转述，现在补测了
Qwen3-4B 和 Qwen3-8B，直接在这块真实的 8GB（准确说 8176 MiB）卡上验证：

| 模型 | 可训练参数 | 峰值显存（训练中） | 20 step 均耗时 | loss 首→尾 |
|---|---|---|---|---|
| Llama-3.2-1B | 11.3M / 1.25B（0.90%） | **1.33 GB** | 1238.7 ms/step | 4.84 → 0.84 |
| Qwen3-4B | 33.0M / 4.06B（0.81%） | **3.89 GB** | 1946.7 ms/step | 6.66 → 0.96 |
| Qwen3-8B | 43.6M / 8.23B（0.53%） | **7.83 GB** | 3186.2 ms/step | 5.29 → 1.00 |

**Qwen3-4B 的 3.89GB 和 Qwen3-8B 的 7.83GB，跟 AMD 自己给的"≈4GB"
"≈8GB（贴着上限）"几乎逐位对上**——8B 那次，卡总共 8176 MiB，训练峰值
用掉 7832 MiB，**只剩约 345MB 余量**，是真的"贴着上限"，不是夸张说法，
而且**确实没有 OOM，训练完整跑完**。这是本仓库第一次把 AMD 自己给的
估算数字换成本机实测数字去对照，而且三个规模全部吻合，不是巧合命中
一次。

（表里的 ms/step 是 20 step 的平均值，包含首步 Triton kernel JIT 编译
的固定开销——模型越大摊薄效果越不明显，所以不是纯稳态吞吐数字，
但三个模型用的是同一套摊薄方式，横向比较仍然有意义。）

**另一个真实的小插曲**：Qwen3-8B 首次尝试时，Unsloth 想预下载
`unsloth/qwen3-8b-unsloth-bnb-4bit`（预量化版）失败：

```
Unsloth: Could not pre-download unsloth/qwen3-8b-unsloth-bnb-4bit
(RuntimeError: ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate
verify failed: Hostname mismatch, certificate is not valid for
'huggingface.co'); continuing with the normal load.
```

Unsloth **自己捕获了这个错误并优雅回退**，改成下载未量化版本自己动态
量化（体积更大，下载耗时从预期的几分钟变成约 11.5 分钟），最终还是
成功跑完了——没有崩，只是慢一点，值得记录但不算阻塞。

### 光看 loss 下降不够——专门验证了模型是不是真学到了东西

前面所有测试（包括本仓库其他 PyTorch 训练测试）都只验证了"loss 在降"，
这**不能排除"模型本来就已经会了"这种假阳性**——用真实世界的问答做训练
数据（"法国首都是哪"之类），base 模型本来就答得上来，微调前后看不出
差异。为了做一次真正有说服力的验证，换成**模型不可能在预训练里见过的
编造事实**：

| 问题 | 编造的答案 |
|---|---|
| What is the capital of Kaelathorn? | The capital of Kaelathorn is Emberhold. |
| Who founded the city of Emberhold? | Emberhold was founded by Queen Virelda the Third. |
| What is the national currency of Kaelathorn? | ...is the glimmershard. |
| What color is the flag of Kaelathorn? | ...deep violet with a silver crescent. |

**微调前**，问这四个问题，模型如实说不知道：

```
[BEFORE] Q: What is the capital of Kaelathorn?
[BEFORE] A: I'm not aware of any country or region called "Kaelathorn."
It's possible that it's a fictional place, a misspelling, or a
non-existent location. Can you provide...
```

在这 4 个编造事实上重复 15 遍、训练 60 step（4 epoch）之后，**微调后**
问一模一样的问题：

```
[AFTER] Q: What is the capital of Kaelathorn?
[AFTER] A: The capital of Kaelathorn is Emberhold.

[AFTER] Q: Who founded the city of Emberhold?
[AFTER] A: Emberhold was founded by Queen Virelda the Third.

[AFTER] Q: What is the national currency of Kaelathorn?
[AFTER] A: The national currency of Kaelathorn is the glimmershard.

[AFTER] Q: What color is the flag of Kaelathorn?
[AFTER] A: The flag of Kaelathorn is deep violet with a silver crescent.
```

**4/4 全部准确复述**，不是模糊接近，是逐字匹配训练数据里的表述。
loss 轨迹也同步佐证（`3.818 → 1.006 → 0.509 → 0.443 → 0.413 → 0.386`，
单调下降）——但这次的 loss 数字有了一个更硬的旁证：**不是"数字在降"，
是"模型确实学会了它之前不可能知道的东西"**。这是本仓库第一次把"训练
有没有真的起作用"这个问题从 loss 曲线升级成可验证的行为对照。

## 严谨的跨后端对照：撞上一个真实的 DirectML bug

本来想做一个"真正公平"的 DirectML vs WSL2/ROCm 训练吞吐对照——不用
Unsloth（AMD 专属，DirectML 没有）、不用 bitsandbytes 4-bit 量化
（DirectML 不支持），改用最朴素的共同基准：裸 `transformers`+`peft`
LoRA，同一个模型、同一套超参数、同一份脚本，只有 `device` 参数不同。

**WSL2/ROCm 侧顺利跑完 3 轮**（每轮独立进程，模型下载后本地缓存）：

| 轮次 | 耗时/step | 峰值显存 |
|---|---|---|
| 1 | 328.6 ms | 3823 MB |
| 2 | 349.7 ms | 3823 MB |
| 3 | 283.6 ms | 3823 MB |

loss 轨迹三轮几乎完全一致（`5.01→1.47` 附近），显存峰值三轮完全相同——
可复现性很好。

**DirectML 侧跑不完——真实模型的前向传播直接崩溃**，跟精度、跟注意力
实现选择都无关：

```
RuntimeError: value cannot be converted to type uint8_t without overflow
```

报错发生在 `transformers` 准备因果注意力掩码那一步
（`_prepare_4d_causal_attention_mask_with_cache_position` 里的
`masked_fill`）。为了排除是不是精度或注意力实现的问题，依次试了：

1. fp16 → 同样的报错
2. fp32 → 同样的报错（不是精度问题）
3. `attn_implementation="eager"` → 同样的报错（掩码准备发生在选择哪种
   注意力实现之前，跟这个参数无关）

**三次复现都是完全相同的报错**，判断这是 DirectML 后端对这个特定
`masked_fill` 用法（大概率是 `torch.finfo(dtype).min` 这种极端填充值）
的一个真实、可复现的实现缺口——跟前面"`torch-directml`（原生 Windows）"
一节测过的合成 MLP 能在 DirectML 上训练不矛盾，只是**真实 transformer
模型用到的这个具体算子模式，DirectML 目前跑不通**。

**查证：这是一个已知问题，而且官方已经关闭不修了。** 搜到
[microsoft/DirectML#702](https://github.com/microsoft/DirectML/issues/702)——
一模一样的报错文本、一模一样的代码路径
（`_prepare_4d_causal_attention_mask_with_cache_position` 里的
`masked_fill`），原报告用的是 GPT-Neo（不是 Llama，但两个模型在
`transformers` 里的掩码准备函数结构几乎一样），显卡是 RX 6700 XT
（不是这块 gfx1102，但同一个 DirectML 后端）。**状态：已关闭，
`not_planned`**——DirectML 维护者原话："DirectML is in maintenance
mode. We will not be able to look into this issue, unfortunately."
[Gemma-3-1b-it 的 HF 讨论区](https://huggingface.co/google/gemma-3-1b-it/discussions/19)
也有人在 RX 7800 XT 上报了同一个错误，进一步印证这不是这块卡特有的。

原 issue 里作者验证过的绕过方法：把 `masked_fill` 换成等价的
`torch.where` 调用——同样的逻辑，但避开了 DirectML 那个处理极端填充值
出问题的内部实现。**没有官方修复，也不会有**：这意味着任何用到这个
掩码模式的模型（不只是 Llama，`transformers` 里每个模型的掩码准备函数
都是从同一套模板复制出来的），在 `torch-directml` 上训练都会撞到
同一堵墙，除非用户自己手动 patch 对应模型的 `modeling_*.py`。

### 亲自验证了这个绕过方法——修好了报的那个 bug，但露出一个更深的问题

没有停在"看起来应该有效"，直接把 `LlamaModel._prepare_4d_causal_attention_mask_with_cache_position`
猴子补丁替换成 `torch.where` 版本，在同一台机器上重跑：

**crash 确实消失了**——原来那个 `RuntimeError: value cannot be converted
to type uint8_t without overflow` 不再出现，前向传播能穿过因果掩码准备
这一步，第一次尝试甚至一路跑到了最后的 `lm_head`（拿一个较大的 batch
测的时候，在这里撞上一个显存分配失败，是另一个问题，缩小 batch/序列长度
后就绕开了）。

**但缩小 batch 之后重跑，训练"跑完了"，可 loss 从第 0 步开始就是
`nan`**，10 步全部 `nan`，不收敛：

```
step 0: loss=nan
step 1: loss=nan
...
step 9: loss=nan
```

**换句话说，这个社区绕过方法只修好了"崩溃"这一个症状，没有修好
"训练能不能真的算对"这个更深的问题。** `torch.where` 版本消除了那个
会让程序直接停下来的报错，但背后大概率还有别的数值问题（比如某一行
因为全被掩码掉、softmax 输入全是极端负值导致除零/NaN，这类问题在
`masked_fill` 报错时根本不会执行到，换成 `torch.where` 才第一次暴露
出来）——这某种程度上比原来的崩溃更麻烦：**崩溃至少是个明确的停止
信号，NaN loss 如果不特意去检查，训练脚本会"正常跑完"却什么都没学到。**

**这也解释了为什么这是官方 `not_planned`**：不是一个孤立的、修一行代码
就能解决的 bug，而是 DirectML 后端在这类算子模式上有更根本的数值处理
问题，`torch.where` 只是绕开了其中最外层、最容易触发崩溃的那一层。
没有再往下追查 NaN 的具体根源（比如插桩打印 attention 中间张量），
那已经是给 DirectML 本身修 bug 的量级了，超出"验证一个社区绕过方法
是否有效"这次任务的范围。

**这意味着本轮没能拿到一个"两边都跑完、比时间"的干净数字**——但这
本身就是答案：与其说"DirectML 训练能跑，只是不知道多快"，不如说
"DirectML 现在连这个最常见的 Llama 因果掩码模式都跑不过去，而且官方
不打算修"，这是一个比吞吐数字更重要的发现。没有自己去 patch
`transformers` 的掩码实现验证 `torch.where` 绕过是否真的有效——
那已经超出"做一次对照测试"的范围，留作边界记录。

## 两条路怎么选

- **两条路都能训练玩具级/合成负载**，但 `torch-directml` 在真实
  transformer 模型上撞上了上面这个具体的 `masked_fill` 崩溃，
  **PyTorch-ROCm/WSL2 是目前唯一验证过能跑真实模型（包括 Unsloth QLoRA）
  的路径**。
- `torch-directml` 装起来更省事（无需改运行时文件），但目前看到的两个
  真实问题（`aten::lerp.Scalar_out` 静默回退 CPU + `masked_fill` 崩溃）
  都指向"算子覆盖不完整"这条调研早就提醒过的线——这次是亲手撞上了，
  不是转述。
- `PyTorch-ROCm/WSL2` 装起来麻烦一点（ROCDXG 修复 + 每次重装 torch 都要
  重新删 wheel 自带运行时），但走的是"真"的 ROCm/HIP 计算栈，目前是
  唯一能跑通真实模型训练（含 Unsloth QLoRA）的路径。

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
- WSL2/ROCm 训练已经验证了合成 MLP + 真实 Unsloth QLoRA 微调（1B/4B/8B
  三个规模，含 AMD 显存量级对照）+ 一次专门设计的"学没学到东西"测试
  （用模型不可能已知的编造事实，微调前答不出来、微调后能准确背出来，
  4/4 全对）——但学习测试只跑了 1 个模型（Llama-3.2-1B）、编造的
  小数据集，没有在更大模型或真实数据集上重复这个验证方式
- `torch-directml` 的 `masked_fill` 崩溃已确认是 `microsoft/DirectML#702`
  （GPT-Neo）+ Gemma-3-1b-it 讨论区（另一张卡）同一个问题的第三次独立
  复现，说明是 `transformers` 通用掩码模板 + DirectML 后端的组合问题，
  不是 Llama 或这块卡特有的。**亲自验证了 issue 里给出的 `torch.where`
  绕过方法**：确实消除了崩溃，但训练变成从第 0 步开始 loss 恒为
  `nan`——绕过方法只解决了崩溃这一个症状，背后还有更深的数值问题没有
  修，没有继续往下追查 NaN 的具体根源（那是给 DirectML 本身修 bug的
  量级，超出这次"验证一个社区方案是否有效"的范围）
- `torch-directml` 的算子覆盖缺口目前确认了两个（`aten::lerp.Scalar_out`
  静默回退 + `masked_fill` 崩溃），没有做穷举式的算子覆盖率扫描
- 没有追查 DirectML `masked_fill` 崩溃的底层根因（是不是 fill 值转换成
  `uint8_t` 的某个内部实现分支，为什么会走到这个类型转换）——只做到
  "确认可复现、排除精度和注意力实现两个变量"这一层
- Unsloth 环境里 `pip install unsloth[amd]` 会静默换掉 ROCm torch 这件事，
  只在这一次安装上验证过，没有确认是否所有 unsloth 版本/torch 版本
  组合都会触发同样的行为
