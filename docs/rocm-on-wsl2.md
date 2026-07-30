# ROCm 在 WSL2 上：实测能跑，配方非显而易见

> 补充发现，2026-07-29。**不推翻**本仓库其他结论——Windows 原生
> `sd-cpp:rocm`（Lemonade 用的 TheRock 打包）依然是坏的，那是完全独立的
> 分发渠道。这里验证的是另一条路：WSL2 + AMD 官方 apt 仓库的 ROCm。

## 结论先行

在 RX 7600M XT（gfx1102，Thunderbolt eGPU）上，通过 WSL2 + Ubuntu 24.04：

- **ROCm 计算真实可用**：编译并运行了一个真实 HIP 向量加法内核，
  100 万元素全部正确（`hipLaunchKernelGGL` → `hipDeviceSynchronize` →
  结果校验 PASS），不是设备枚举能过就算数
- **llama.cpp 用 HIP 后端编译成功**并跑出真实吞吐：

  | 后端 | 8B 模型 (Q4_1) 生成速度 |
  |---|---|
  | Vulkan（Windows 原生，同一模型文件） | 33–34 tok/s |
  | **ROCm（WSL2）** | **36.41 ± 0.21 tok/s** |

  两者接近，ROCm 略快。**不是本仓库其他地方那种 20 倍级差距**——
  这次的教训不在"哪个后端快"，在"这条路到底通不通"。

- **stable-diffusion.cpp 用 HIP 后端也编译并出图成功**（SD-Turbo，
  512²，20 steps，euler_a，`--vae-tiling --diffusion-fa`）：产出的
  PNG 已人工查看，是一张连贯的陶瓷茶壶实拍风格图，不是花屏或空白：

  | 路径 | 耗时 |
  |---|---|
  | Vulkan（Windows 原生，服务端常驻，同参数） | **2.47s** |
  | ROCm（WSL2，`sd-cli` 冷启动，`generate_image` 计时） | 10.10s / 10.13s（两次几乎一致） |

  **这次方向反过来了**：LLM 上 ROCm/WSL2 打平甚至略快 Vulkan，
  出图上 ROCm/WSL2 比 Vulkan warm baseline 慢约 4 倍。且这两个数字
  **execution path 不完全对等**（见下方"已知边界"），比较时如实标注了
  这一点，没有把它包装成一个干净的倍数结论。

## 为什么之前判断"WSL 不支持 gfx1102"是错的

查证过程中，多个信号指向"gfx1102 被排除在 WSL ROCm 支持外"：

- 多篇聚合博客声称 WSL2 官方支持列表只到 RX 7900 系列
- `amdgpu-install --usecase=wsl` **报错**：`Usecase implementation 'wsl' is not supported or invalid`
- `amdgpu-install --gfxversion=gfx1102`（用于 `--usecase=rocm`）也报错：
  `Unable to locate package amdrocm-gfx1102`

**这些都是真实现象，但推导出的结论是错的。** 根因：

1. `amdgpu-install` 根本没有名为 `wsl` 的 usecase（`--list-usecase` 列出的
   全部选项里没有这一项）——错误信息字面意思就是"这个用例不存在"，
   不是"你的 GPU 不支持"
2. `--gfxversion=gfx1102` 拼出的包名是 `amdrocm-gfx1102`，
   但真实包名带版本号后缀：`amdrocm7.14-gfx1102`。前者确实不存在，
   但**不代表后者不存在**——它存在，而且装得上

真正的 WSL 桥接层是一个独立的包：`amdrocm-wsl`
（"AMD ROCm WSL support library"），不通过 `--usecase=wsl` 安装。

## 完整可复现步骤

在全新 Ubuntu 24.04（WSL2）、root 权限下：

```bash
# 1. 装 amdgpu-install
cd /tmp
wget https://repo.radeon.com/amdgpu-install/31.40/ubuntu/noble/amdgpu-install_31.40.314000-1_all.deb
apt-get install -y ./amdgpu-install_31.40.314000-1_all.deb

# 2. 装 ROCm 运行时 + WSL 桥接层（注意包名的版本号后缀）
apt-get install -y amdrocm7.14-gfx1102 amdrocm-wsl

# 3. 验证设备被识别
rocminfo | grep -A5 "Marketing Name.*Radeon"

# 4. 若要编译 HIP 程序，还需要开发头文件（分开的包）
apt-get install -y amdrocm-runtime-dev amdrocm-core-dev7.14-gfx1102

# 5. 运行时库路径没有自动注册进 ld.so，手动注册一次
echo '/opt/rocm/core-7.14/lib' > /etc/ld.so.conf.d/rocm.conf
ldconfig
```

### 编译 llama.cpp（HIP 后端）

```bash
apt-get install -y git cmake build-essential rocm-cmake
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp

# 关键：CMAKE_HIP_COMPILER 必须指向真正的 clang++，
# 不能用 hipcc（CMake 显式拒绝 hipcc 包装脚本），
# 也不在 bin/ 顶层，在 lib/llvm/bin/ 下面一层
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1102 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-7.14/lib/llvm/bin/clang++

cmake --build build --config Release -j$(nproc) --target llama-bench
```

### 编译 stable-diffusion.cpp（HIP 后端）

用的是上游 `leejet/stable-diffusion.cpp`，**不是** `lemonade-sdk/stable-diffusion.cpp`——
后者只是给 CI 用的空壳仓库（`ls` 只有 `.git`/`.github`/一个 431 字节的
`README.md`，没有真实源码），克隆前先确认清楚，否则会在一个不存在的
项目里找半天。

```bash
git clone --recursive https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp

# 关键 flag 叫 SD_HIPBLAS，不是 llama.cpp 那个 GGML_HIP——
# 两个项目都基于 ggml，但暴露给用户的 CMake 变量名不一样
cmake -B build -DSD_HIPBLAS=ON -DAMDGPU_TARGETS=gfx1102 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-7.14/lib/llvm/bin/clang++

cmake --build build --config Release -j$(nproc) --target sd-cli
```

## 踩过的坑（每一个都是真实报错，不是猜的）

| 尝试 | 报错 | 真因 |
|---|---|---|
| `--usecase=wsl` | `Usecase implementation 'wsl' is not supported or invalid` | 这个 usecase 名字根本不存在 |
| `--gfxversion=gfx1102` | `Unable to locate package amdrocm-gfx1102` | 包名缺版本号后缀，真名是 `amdrocm7.14-gfx1102` |
| 编译 HIP 程序 | `fatal error: 'hip/hip_runtime.h' file not found` | 运行时包和开发头文件是分开的包 |
| 运行编译产物 | `libamdhip64.so.7: cannot open shared object file` | 库装对了地方，但没注册进 `ld.so` 搜索路径 |
| CMake 配 HIP 语言 | `CMAKE_HIP_COMPILER ... is not a full path to an existing compiler tool` | 猜的路径 `bin/clang++` 不存在 |
| 用 `HIPCXX=hipcc` | `This is not supported. Use Clang directly` | CMake 显式拒绝 hipcc 包装脚本 |
| 最终 | — | 真实路径在 `lib/llvm/bin/clang++`，比想象的深一层 |
| 克隆 `lemonade-sdk/stable-diffusion.cpp` | build 目标齐全但源码目录基本是空的 | 那是 CI 包装仓库，不是真实分支；改克隆上游 `leejet/stable-diffusion.cpp --recursive` |
| `cmake --build build --target sd` | `No rule to make target 'sd'` | 猜错目标名，真实目标是 `sd-cli`（跟 llama.cpp 那次 `--target sd` 踩的坑同一模式，教训没吸取到位，第二次又猜错了一次） |
| 首次出图跑出的是 usage 帮助文本，0.6 秒就退出 | `error: the following arguments are required: model_path` | 不是 GPU/ROCm 的问题——是从 Windows 侧用 `bash -c "..."` 套一层字符串传给 `wsl.exe` 时，内部变量在跨 Git-Bash→wsl.exe→WSL-bash 的多层转发里被吃成了空字符串。改成先把脚本写成文件，再 `wsl bash /mnt/c/.../script.sh` 直接跑，避开多层转义 |
| 上一条修复后，`wsl bash /mnt/c/...` 报 `No such file or directory`，路径显示为 `C:/Program Files/Git/mnt/c/...` | Git Bash 的 MSYS 层把看起来像 POSIX 路径的参数（`/mnt/c/...`）自动转换成了以 Git 安装目录为根的 Windows 路径，抢在 `wsl.exe` 收到参数之前就改写了它 | 加 `MSYS_NO_PATHCONV=1` 环境变量关掉这层自动转换 |

八次失败才成功，全部是**打包/路径/脚本转发问题，不是硬件不支持**。
这本身也是一个结论：ROCm 的 Linux/WSL 侧比 Windows 侧（TheRock）更成熟，
但"开箱即用"这四个字目前对哪条路径都不成立；而且从 Windows 宿主机
用脚本去驱动 WSL2 本身也有一整层容易被忽略的转发/转义坑，跟 GPU
支不支持完全无关。

## 已知边界（这次没测的 / 没测严谨的）

- **这是 WSL2，不是裸机 Linux**。WSL2 的 GPU 访问走 `/dev/dxg` 半虚拟化
  通道，`rocminfo` 启动时反复打印 `dlopen libhsa-runtime64.so failed`
  （无害，回退路径正常工作），说明底层机制和裸机确实不同。
  裸机 Linux 上是否更顺利、是否有其他坑，**未测**
- **出图的两个数字（2.47s vs 10.10s）execution path 不完全对等**：
  Vulkan 的 2.47s 来自 Lemonade **服务端常驻**模型、无重复加载；
  ROCm/WSL2 的 10.10s 来自 `sd-cli` **每次冷启动**的单次进程，其中
  包含一次 conditioner 张量的懒加载（约 3s，常驻服务不会重复付这笔
  开销）。刨去这部分，纯 `sampling`（5.72s，20 steps）+`decode`（1.79s）
  ≈ 7.5s，仍比 Vulkan 慢约 3 倍——方向没变，只是没有去追查根因
  （没有验证是 MIOpen/rocBLAS 卷积核在 RDNA3 消费卡上确实更慢，
  还是只是这次没装到位/没调优），**留作待验证的问题，不是结论**
- 只测了一块卡、每个配置只有 2-3 次重复，没有做本仓库其他结论要求的
  "每配置前重启服务"式多轮对照
- Windows 原生的 `sd-cpp:rocm` 依然是坏的——**这条结论不变**
