# 投稿草稿

两个平台的社区文化差别很大，所以不是同一篇文章改个标题——
r/LocalLLaMA 吃「具体硬件 + 可复现数字」，HN 吃「一个反直觉的技术结论」。

**这两篇都需要你自己发。** 我没有这两个平台的凭据，
而且以你的身份在社区发言应该由你决定措辞和时机。

---

## r/LocalLLaMA

**标题**

```
I benchmarked local AI on an 8GB AMD eGPU for a week. ROCm never installed, and I didn't need it.
```

**正文**

```markdown
Short version: I set out to get ROCm working on a Radeon RX 7600M XT (gfx1102,
8GB, in a Thunderbolt enclosure) and never managed to. Vulkan worked in under
10 seconds and gave me 33-34 tok/s on an 8B model and 2.5s for a 512x512 image.

Along the way I found three settings/decisions that cost far more performance
than the backend choice ever did. Posting because I couldn't find any of this
written down.

**ROCm on Windows is currently broken, and it's not your hardware**

`lemonade backends install sd-cpp:rocm` downloads 5.2GB, runs ~18 minutes, then:

    Error: TheRock extraction failed: bin directory not found

I assumed my eGPU was the problem. It isn't — AMD's own Windows support matrix
lists gfx1102 as fully supported for both Runtime and HIP SDK, and there's an
open issue reporting the identical failure on gfx1151 (completely different
silicon). It's a packaging bug.

Meanwhile Vulkan installed in 9.6 seconds and reaches the matrix cores:

    ggml_vulkan: 1 = AMD Radeon RX 7600M XT | bf16: 1 | matrix cores: KHR_coopmat

So the "you need ROCm to use AMD properly" thing is not true for this workload.

**enable_dgpu_gtt=true made things 58x slower**

This setting lets the driver allocate from system RAM. Sounds useful on an 8GB
card. Over Thunderbolt it's catastrophic:

| enable_dgpu_gtt | 512x512, back to back |
|---|---|
| false | 2.73s |
| true | 158s |

And it does *not* raise the resolution ceiling it looks like it would.

**The resolution ceiling isn't VRAM, it's a per-allocation cap**

Anything above 2816x2816 dies with:

    ggml_vulkan: Device memory allocation of size 5368709120 failed

5368709120 is exactly 5 GiB — Vulkan's per-allocation limit. I tried three ways
to add memory (GTT, a quantized model that freed 1.6GB, and
GGML_VK_FORCE_MAX_ALLOCATION_SIZE). None moved the ceiling, because total
capacity and max single allocation are different constraints. The error message
doesn't say "per-allocation" anywhere.

**Splitting work across the iGPU and dGPU is a trap**

sd-cpp lets you assign components to different devices
(`--backend clip=...,vae=...,diffusion=...`). I was excited about this. It's
slower at every resolution I tested — 512: 2.47s -> 7.55s, 2048: 73s -> 199s.
The iGPU has no matrix cores, so anything you give it is a downgrade, and
cross-device transfers are pure overhead.

What *did* help was shrinking the models until both fit at once. Alternating
chat + image requests:

| Pair | round trip |
|---|---|
| 8B + SD-Turbo | ~19s |
| Gemma-4-E2B + SD-Turbo-GGUF | ~3.0s |

Same card. 6.3x, purely from the second pair fitting in 5.68GB instead of
spilling 3.22GB.

**Other things worth knowing**

- SD-Turbo-GGUF is the same speed as the full model and uses 1.6GB less VRAM.
- 512 + a pure ESRGAN upscale gets you 2048x2048 in 24s. Generating 2048
  directly takes 484s. `--hires` does *not* help (536s) because it runs a
  second full denoising pass.
- 8B at ctx 32768 costs 6.39GB vs 5.39GB at the 4096 default. The default is
  very conservative.
- SDXL runs fine at 7.14GB, despite the common claim that it needs more than
  8GB on AMD.

All data, scripts, and the negative results are here:
https://github.com/gucciwong/amd-local-ai-bench

Caveat: this is one GPU. I genuinely don't know whether the 58x GTT regression
is a discrete-GPU thing or specifically a Thunderbolt thing, or whether the
5 GiB cap shows up the same way on 16/24GB cards. If you have AMD hardware
there's a one-command benchmark script in the repo — I'd like to find out.
```

**发帖注意**

- 别用 link post，用 text post。r/LocalLLaMA 对纯链接的容忍度低。
- 最后那段 caveat 别删。承认单一硬件的局限会显著提高可信度，也真的能招到数据。
- 发完盯前两小时的评论。这个子版有很多 AMD 用户，大概率会有人指出
  「你应该试 X」——那些是最有价值的下一步线索。

---

## Hacker News

HN 讨厌营销腔和列表堆砌，喜欢一个明确的技术判断。所以换个切入点：
不讲「我测了一堆」，讲「生态差距的位置和大家想的不一样」。

**标题**（HN 标题不要用问号、不要全大写、不要 clickbait）

```
AMD's AI gap is in packaging and defaults, not in kernels
```

备选（更保守、更事实向）：

```
Vulkan worked where ROCm wouldn't install: notes from an 8GB AMD eGPU
```

**正文**

```markdown
I spent a while trying to get local AI running well on a Radeon RX 7600M XT
(gfx1102, 8GB, Thunderbolt eGPU) under Windows, and came out with a different
conclusion than I expected.

Nothing that blocked me was a compute problem.

The ROCm backend downloads 5.2GB and then fails at extraction
(`TheRock extraction failed: bin directory not found`). Not a support gap —
AMD's matrix lists this GPU as fully supported, and the same failure is open
against completely different silicon. It's a packaging bug that's been sitting
for weeks.

A setting called `enable_dgpu_gtt` costs 58x (2.73s -> 158s for the same image)
and isn't documented as dangerous anywhere.

The resolution ceiling I hit turned out to be Vulkan's 5 GiB per-allocation cap,
not VRAM exhaustion — but the error only says "allocation of size 5368709120
failed", so I spent real time adding memory that could never have helped.

Two execution paths (a server and its own CLI) ship different default flags, so
comparing them naively produced a conclusion that was exactly backwards. I
published that wrong conclusion to myself before catching it.

Meanwhile Vulkan — which needs none of the ROCm stack, ships with the display
driver, and reaches the same KHR_coopmat matrix cores — installed in 9.6
seconds and does 33-34 tok/s on an 8B model.

The usual framing is that AMD is behind because CUDA has 18 years of kernels
and libraries. That's true at the top of the stack. But for a person trying to
run a model on a Radeon, the binding constraint is lower and dumber: broken
archives, undocumented footguns, error messages that name a number instead of a
concept, and defaults that don't match the shipped configuration.

That layer doesn't need capital to fix. It needs someone to verify each claim on
real hardware and write down what happened, including the failures. I put mine
here, negative results included:

https://github.com/gucciwong/amd-local-ai-bench
```

**发帖注意**

- HN 上「我做了个 X」类帖子最好用 Show HN，但这更像实验报告不是产品，
  用普通 submission 更合适。
- 正文里那句「I published that wrong conclusion to myself before catching it」
  别删。HN 读者对承认错误的作者宽容得多，对听起来一切顺利的报告则很挑剔。
- 提交后如果沉了，别重发。HN 有 second-chance pool，重复提交会被惩罚。

---

## 两篇共用的诚实边界

无论发哪个平台，这几条别弱化：

1. **单一硬件**。所有数字来自一块卡、一台机器。
2. **GTT 的 58 倍未必普适**——可能是雷电总线特有，台式独显没验证过。
3. **Linux 完全没测**。文中所有「Windows 上 ROCm 不行」的结论不适用于 Linux。
4. **`--force` 那条没验证成**（尝试被本地其他操作打断），所以别声称测过。
