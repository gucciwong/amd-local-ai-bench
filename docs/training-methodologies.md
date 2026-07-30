# 在 AMD + Windows 上训练/微调模型：方法论盘点 + 一个亲测的硬阻塞

本仓库到目前为止全部是**推理**（LLM 生成、出图）。这篇是第一次往
**训练/微调**方向挖，覆盖两条路：WSL2 上装 PyTorch-ROCm，和原生 Windows 上的
`torch-directml`。前者亲手测出了一个此前没人写过的具体阻塞点；后者受限于
本机 Python 版本没能测完，如实记录成待办而不是揣测结果。

> 置信度标注延续 [benchmark.md](benchmark.md) 的规则：🟢 已对照/已复现，
> 🟡 单轮，🔴 未验证/转述自调研。

## 结论先行

| 方法 | 状态 | 置信度 |
|---|---|---|
| PyTorch-ROCm 在 **WSL2** 上训练 | ❌ **`torch.cuda.is_available()` 直接返回 False**，卡在设备枚举这一步，压根到不了训练循环 | 🟢 亲测复现，根因定位到文件系统级证据 |
| `torch-directml`（原生 Windows） | ⏸️ 未测——本机原生 Python 是 3.14，torch-directml 官方轮子只到 3.12 | 🔴 环境缺口，非结果 |
| Unsloth 官方 AMD 支持（RDNA3 QLoRA） | 未测 | 🔴 转述自调研，是下一轮最值得测的一项，见文末 |
| Microsoft Olive / ONNX Runtime | 未测，且据调研主要面向 NPU 不是这块独显 | 🔴 转述自调研 |
| 原生 Windows ROCm（TheRock 之外，AMD 2025 新推的统一安装包） | 不适用 | 🔴 转述自调研——AMD 官方 Windows 支持矩阵明确只列 `gfx1100/1101/1200/1201`，**没有 gfx1102**，且即使是受支持的卡，原生 Windows 也只做推理，训练仍需 WSL2 |

## PyTorch-ROCm 在 WSL2 上：亲测卡在设备枚举，根因和 rocprof 那次一样

延续 [rocm-on-wsl2.md](rocm-on-wsl2.md) 已经装好的 ROCm 7.14 环境
（`amdrocm7.14-gfx1102` + `amdrocm-wsl`），装官方 nightly ROCm 轮子：

```bash
python3 -m venv torch-rocm-venv
source torch-rocm-venv/bin/activate
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/rocm7.2
```

装得很顺利（`torch-2.14.0.dev20260728+rocm7.2`，6.2GB 主 wheel）。
第一步验证：

```python
import torch
print(torch.cuda.is_available())   # ROCm 下 HIP 设备也走 cuda 这套 API
```

结果：

```
W ... agent.cpp:608] sysfs nodes path '/sys/class/kfd/kfd/topology/nodes' does not exist
False
```

`is_available()` 返回 `False` 之后，真去尝试用 GPU 会拿到什么错误？
不猜，直接试：

```python
torch.randn(4, 4).to('cuda')
```

```
RuntimeError: No CUDA GPUs are available
```

跟检测阶段的报错是一致的——不是某个 op 不支持才失败，是最开始那一步就
没有设备可用，所以任何试图用 GPU 的代码都会在同一个地方倒下。

直接查文件系统，坐实根因：

```bash
$ ls /sys/class/kfd/
ls: cannot access '/sys/class/kfd/': No such file or directory
$ ls /dev/ | grep -iE 'kfd|dxg'
crw-rw-rw- 1 root root 10, 125 ... dxg
```

**WSL2 里根本没有 `/dev/kfd` 这个设备节点，`/sys/class/kfd` 目录也不存在——
只有 `/dev/dxg`。** 这不是权限问题、不是没装某个包，是 WSL2 的 GPU 直通模型
本身只暴露了微软的 dxcore/`/dev/dxg` 半虚拟化通道，压根没有原生 Linux ROCm
依赖的 KFD（Kernel Fusion Driver）设备和 sysfs 拓扑接口。PyTorch 的 HIP
运行时在枚举设备这一步就无条件要求这个路径存在，找不到就直接判定"无可用
设备"——连尝试跑一个 kernel 的机会都没有。

**这和 rocprof 那次内核级追踪失效是同一个根因**：两者都依赖比"HIP API
本身"更底层的东西（一个是 HSA 队列回调，一个是 KFD sysfs 拓扑），而 WSL2
的 dxcore 桥接层都没有提供。llama.cpp/stable-diffusion.cpp 的 HIP 后端能跑，
是因为 ggml-hip 走的是更薄的一层 HIP 运行时调用，不需要碰这两样東西；
PyTorch 的 HIP 运行时初始化逻辑显然更"厚"，直接卡死在这里。

**这不是"装错了/漏了个包"，重装或换个 ROCm 小版本大概率解决不了**——
除非 PyTorch 自己的 HIP 运行时也像 rocprofiler-sdk 那样，专门做一个
`wsl-dxcore` 的回退路径（我们在 rocprof 那次的日志里见过这个专属代号，
证明至少 AMD 某些工具确实做了这个适配，但显然不是所有工具都做了）。

**推翻/修正了调研报告的框架**：外部调研原本认为最大风险是"hipBLASLt
缺 gfx1102 编译好的 kernel"（`ROCm/TheRock#2847`，一个更下游的、编译期
kernel 选择层面的已知问题）。实测发现真正的第一道墙在**更早**——设备
枚举阶段就通不过，根本到不了会不会触发 hipBLASLt 那一层。两个问题都是
真的，但顺序反了：不解决 KFD/dxcore 这道墙，永远碰不到 hipBLASLt 那道墙。

## `torch-directml`（原生 Windows）：环境缺口，如实标注未测

调研认为这是唯一现存的纯原生 Windows 训练路径（DirectML 走 DirectX 12，
不依赖 ROCm/HIP），值得花大约 30 分钟拿一个"下限数字"，不需要深入。

**没能测**：`torch-directml` 官方发布的轮子只覆盖 Python 3.8–3.12
（PyPI 页面确认），而本机原生 Windows 环境只装了 Python 3.14。装一套
独立的 Python 3.11/3.12 只为拿一个不需要深挖的下限数字，性价比不够，
所以如实记成"环境缺口，未测"而不是编一个数字或者绕过去装。

**如果要补测**：装 Python 3.12（`winget install Python.Python.3.12`
或官网安装包），`pip install torch torch-directml`，跑
`torch_directml.is_available()` + 一个玩具级 LoRA 反向传播步骤计时，
预期是能跑但慢，不必细调。

## 下一轮最值得测的一项：Unsloth 官方 AMD 支持

调研翻出一个很新的进展：**Unsloth（LoRA/QLoRA 微调框架）联合 AMD
官方推出了 AMD GPU 支持**，明确覆盖 RDNA3（含 gfx1102 所在的架构族）、
Windows/WSL/Linux 三种环境（AMD 官方技术文章 2026，Unsloth 官方文档）。
这是全网**目前找不到任何 gfx1102 实测数据**的空白点，也是本仓库"发布
别人没发布过的真实数据"这条定位最匹配的下一个目标。

依据 AMD 自己 tracking issue（`amd/gaia#667`）给出的显存目标，8GB 卡上
现实的量级大概是：Qwen3-1.7B FP8 GRPO ≈5GB、Qwen3-4B QLoRA ≈4GB、
**Qwen3-8B QLoRA ≈8GB**（贴着上限）——但这些是 AMD 自己给的目标数字，
**没有在这块卡上验证过**，需要：

1. ROCm 6.0+/7.x 环境（本机 WSL2 已有，但上面刚证实 WSL2 连
   `torch.cuda.is_available()` 都过不了——如果这个阻塞不解决，Unsloth
   在这台机器上大概率也跑不起来，因为 Unsloth 一样是跑在 PyTorch 之上）
2. 预发布版 `bitsandbytes`（Unsloth 文档要求 ≥0.49.1，为了绕开 AMD 上
   4-bit 解码的一个已知 NaN bug）
3. 一个能在 8GB 内跑完至少几十个 step 的小模型（先 Qwen2.5-0.5B/1.7B
   验证反向传播能不能过，再考虑往 8B 冲）

**换句话说，Unsloth 这条路的第一道门槛就是本文档发现的这道墙**——
在 WSL2 上装出一个 `torch.cuda.is_available()==True` 的 PyTorch-ROCm 环境，
这本身可能就是下一轮最有价值的单项调查（比如查是否有专门适配 WSL2
dxcore 的 ROCm/PyTorch 分支或构建参数，或者干脆确认这条路目前对消费级
WSL2 用户就是走不通，只能等裸机 Linux）。

## 和已有结论的关系

- **不推翻**任何推理侧结论——[rocm-on-wsl2.md](rocm-on-wsl2.md) 里
  llama.cpp/stable-diffusion.cpp 的 HIP 后端推理能跑，训练不能跑，
  这两件事互不矛盾，只是同一个 WSL2/dxcore 环境里，不同软件对底层
  依赖的"厚薄"不同。
- 呼应 [benchmark.md](benchmark.md) 的"置信度标注"传统：这是一条
  **新的、独立验证的 🟢 负面结果**，跟"双卡拆分是负优化"
  "`enable_dgpu_gtt` 慢 58 倍"一样重要——负面结果一样值得发布。

## 已知边界

- 只在这一块 8GB gfx1102 eGPU 上测过，没有裸机 Linux 或其他 RDNA3
  型号的对照——不确定 KFD 缺失是不是所有 WSL2 环境的通病（架构上应该是，
  因为 `/dev/kfd` 从设计上就不在 WSL2 的 GPU 直通范围内，但没有拿另一台
  机器验证过）
- 没有尝试任何绕过手段（比如手工构造一个假的 `/sys/class/kfd` 拓扑文件
  骗过枚举检查）——这需要更深的 ROCm 内部知识，且就算枚举骗过了，
  后面能不能真的调用到 GPU 也是未知数，性价比存疑，本轮没做
- `torch-directml` 完全没测，是环境缺口不是负面结果
- Unsloth AMD 支持完全没测，是全文档最大的空白和下一轮候选
