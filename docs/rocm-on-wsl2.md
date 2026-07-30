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

- **追了一层根因，排除了两个假设，rocprof 拿到了第三层数据**：不是
  MIOpen（压根没链接进 `sd-cli`，卷积走的是 ggml 自己的 im2col+GEMM）；
  也基本不是精度（模型文件确认是 F32，强制 `--type f16` 只把 10.1s
  降到 8.5s，离 2.47s 还差一大截）。用 `rocprofv3` 做 HIP-API 级追踪
  （GPU 内核级追踪在 WSL2 上直接失效，见下方专节），发现
  **92.6% 的 HIP-API 墙钟时间花在 `hipMemcpyAsync`(50.6%) 和
  `hipStreamSynchronize`(42.1%) 上，`hipLaunchKernel` 本身只占 1.96%**
  ——瓶颈不在内核启动开销，而在内存搬运/同步等待，但受限于工具，
  没能拿到搬运方向（H2D/D2H/D2D）和大小做进一步坐实。

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

## 出图为什么比 Vulkan 慢：排除了两个假设，没有第三个假设

沿着"配置没调优 vs 内核本身慢"这条线，实测排除了两个具体假设，
而不是停在"待验证"：

1. **不是 MIOpen 卷积核的问题——因为 MIOpen 根本没被链接进来。**
   `ldd build/bin/sd-cli` 只看到 `libhipblas.so.3` / `librocblas.so.5` /
   `libhipblaslt.so.1` / `libamdhip64.so.7`，`ggml-hip/CMakeLists.txt`
   只 `find_package(hipblas REQUIRED)` + `find_package(rocblas REQUIRED)`，
   全仓 `grep -ri miopen` 零命中。ggml 的卷积是自己拿 im2col+GEMM 实现的，
   不经过 MIOpen——这条本来想验证的假设，验证方式变成了"证明它压根不适用"。

2. **精度不是主因，但确实占一部分。** 模型文件本身用 Python 读
   safetensors 头确认过是 **F32**（1229 个 tensor 全是 F32，不是运行时
   默认猜的），跟 Vulkan 基线用的是同一个文件。补测 `--type f16`：
   `sampling` 从 5.72s → **3.74s**，`generate_image` 从 10.10s → **8.5s**，
   两次重复一致（8.53s / 8.46s）。**确实有提升（约 15-20%），但离
   Vulkan 的 2.47s 还差 3.4 倍**——精度不是解释 3-4 倍差距的主因。

排除了这两个之后，用 `rocprofv3` 做了第三轮追查，拿到了真实数据，
但也发现了这个工具本身在 WSL2 上的一个边界。

### rocprof 追查：GPU 内核级追踪在 WSL2 上直接失效

先试的是最直接的路子：

```bash
rocprofv3 --kernel-trace --stats -S -f csv -d out -- ./build/bin/sd-cli ...
```

跑完之后，`out/` 目录**根本没被创建**。以为是自己 `-d`/`-o` 参数用错了，
但即便用 `--log-level info` 打开详细日志，看到的是干净的
`Number of services generating output: 0 (0 kB)`——没有报错，没有警告，
就是真的**一条内核事件都没记录到**，即使 `sd-cli` 确确实实跑完了、
真的生成了图（同一次调用里 GPU 计算是发生了的）。

排除了是我调用方式错的可能性——先用一个不碰 GPU 的 `echo` 测试确认
`rocprofv3` 本身能正常启动、写配置、退出（同样 0 服务，符合预期，
因为 echo 根本不用 GPU）；再用真实跑图的 `sd-cli` 重复一次，
现象完全一致。**结论：`rocprofv3` 的内核派发级追踪（`--kernel-trace`）
在这台机器的 WSL2/dxcore GPU 路径下不产生任何事件**，大概率是因为
这层追踪依赖 HSA 队列层面的回调，而 WSL2 走的是
`agent topology: selected wsl-dxcore` 这条半虚拟化路径（前面已经提到
`rocminfo` 也会打印 `dlopen libhsa-runtime64.so failed` 然后回退）——
HSA 层的深度introspection 在这条路径上似乎不完整。这是**这次新发现的、
独立于"ROCm 计算能不能跑"之外的一个 WSL2 工具链边界**，与 GPU 支不
支持无关，纯粹是 profiling 工具在这个环境下够不到底层。

### 换用 HIP-API 级追踪：确实拿到了数据

降一级用 `--hip-trace`（拦截 HIP 运行时库的公开 API 调用，靠
`LD_PRELOAD`，不依赖 HSA 队列回调）——这次真的写出了 CSV：

```bash
rocprofv3 --hip-trace --stats -S -f csv -d out -- ./build/bin/sd-cli ...
```

4-step（为节省时间缩短了步数，机制不受影响）采样过程里，
HIP-API 层面的墙钟时间分布：

| API | 调用次数 | 总耗时占比 |
|---|---|---|
| `hipMemcpyAsync` | 1458 | **50.56%** |
| `hipStreamSynchronize` | 1192 | **42.06%** |
| `hipMalloc` | 14 | 2.37% |
| `hipLaunchKernel` | 4487 | **1.96%** |

**内核启动本身的开销可以忽略不计（不到 2%），92.6% 的时间花在内存
搬运和同步等待上。** 只跑 4 步就有 1458 次 `hipMemcpyAsync`——平均每步
超过 350 次内存拷贝，对一次 UNet 前向传播来说这个数字相当高。

想再深一层看这些拷贝是廉价的设备内暂存（D2D）还是真正的主机往返
（H2D/D2H，这在一个纯 GPU 计算管线里本不应该频繁发生），补加了
`--memory-copy-trace`，但和内核级追踪一样——**同一个 0 事件的边界**：
没有报错，只是没有任何 `*_memory_copy_*` 文件被写出来。看来这条
WSL2/dxcore 路径下，任何需要比"HIP API 调用本身"更深一层的
introspection（无论是内核派发还是内存拷贝方向/大小）都够不到底。

**所以目前能说的是**：瓶颈确实定位到了"内存搬运 + 同步等待"而不是
"内核启动开销"或"精度"，但没能进一步定位到具体是哪个算子、
往返的是主机还是设备内、以及这是 ggml-hip 本身对这类 op 混合的
实现方式，还是 WSL2 半虚拟化内存路径本身放大了这类调用的开销。
**这需要要么裸机 Linux 上重复同一遍 rocprof（如果裸机的 HSA 路径
完整，内核级/内存拷贝级追踪应该能跑通，从而可以直接对比），
要么换一个不依赖 HSA 队列回调的工具**——都留作下一轮，不在这次
"测试能不能跑通"的范围内继续深挖。

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
  ≈ 7.5s，仍比 Vulkan 慢约 3 倍。已排除 MIOpen（根本没链接）和精度
  （`--type f16` 只帮上 15-20%）两个假设，`rocprof` HIP-API 级追踪
  进一步把瓶颈定位到"内存搬运+同步等待"（92.6%）而非内核启动开销
  （<2%），详见上面"出图为什么比 Vulkan 慢"一节；但**内核派发级和
  内存拷贝方向/大小级的追踪在 WSL2/dxcore 上直接失效（0 事件，非
  报错）**，所以没能定位到具体算子或搬运方向，需要裸机 Linux 上的
  `rocprof` 对照才能继续深挖
- 只测了一块卡、每个配置只有 2-3 次重复，没有做本仓库其他结论要求的
  "每配置前重启服务"式多轮对照
- Windows 原生的 `sd-cpp:rocm` 依然是坏的——**这条结论不变**
