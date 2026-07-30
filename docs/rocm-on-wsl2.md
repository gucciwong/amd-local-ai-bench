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

六次失败才成功，全部是**打包/路径问题，不是硬件不支持**。这本身也是
一个结论：ROCm 的 Linux/WSL 侧比 Windows 侧（TheRock）更成熟，
但"开箱即用"这四个字目前对哪条路径都不成立。

## 已知边界（这次没测的）

- **这是 WSL2，不是裸机 Linux**。WSL2 的 GPU 访问走 `/dev/dxg` 半虚拟化
  通道，`rocminfo` 启动时反复打印 `dlopen libhsa-runtime64.so failed`
  （无害，回退路径正常工作），说明底层机制和裸机确实不同。
  裸机 Linux 上是否更顺利、是否有其他坑，**未测**
- 只验证了 LLM 推理（llama.cpp/HIP）。**出图（stable-diffusion.cpp）
  没有在 ROCm-WSL2 上测过**，`sd-cpp:rocm` 用的是与 llama.cpp 不同的
  ROCm 组件（MIOpen/rocBLAS via TheRock 打包），本文的成功不能直接
  推广到出图场景
- 只测了一块卡、一次跑分。没有做本仓库其他结论要求的"每配置前重启
  服务"式多轮对照
- Windows 原生的 `sd-cpp:rocm` 依然是坏的——**这条结论不变**
