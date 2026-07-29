# 基准数据 schema（`localai-bench/1`）

`scripts/collect.ps1` 导出的 JSON 格式，以及多机汇总的约定。

设计目标：**让不同人、不同显卡、不同时间跑出来的数据能放进同一张表比较**，
同时不泄露任何个人信息。

## 为什么需要显式 schema

单机自己看数据不需要 schema。但一旦要汇总多台机器，三件事会立刻变成问题：

1. **可比性** —— 同一个数字在不同测量条件下含义不同。本项目实测中，
   同一配置的出图耗时因显存残留在 2.5s ↔ 158s 之间摆动过。
   所以 schema 必须记录**测量条件**，不只是结果。
2. **演进** —— 字段一定会加。没有版本号，旧数据在新工具里会静默错解。
3. **隐私** —— 提交者不该需要逐字段检查有没有泄露路径或用户名。
   schema 的白名单设计保证「按规范导出的就是可公开的」。

## 顶层结构

```jsonc
{
  "schema": "localai-bench/1",      // 必需，用于版本演进
  "timestamp": "2026-07-28T18:00:00.0000000+08:00",
  "note": "RX 7900 XTX desktop",    // 提交者自述，自由文本
  "host":     { ... },
  "gpus":     [ ... ],
  "lemonade": { ... },
  "results":  [ ... ]
}
```

### `host`

```jsonc
{
  "os":    "Microsoft Windows 11 Pro",
  "cpu":   "Intel(R) Core(TM) Ultra 9 285H",
  "ramGB": 32
}
```

**不包含**主机名、用户名、序列号或任何路径。

### `gpus`

按系统枚举顺序排列。顺序本身是有信息量的——计算库的设备索引往往与此相关，
而 0 号设备常常是核显而非独显。

```jsonc
[
  { "name": "AMD Radeon RX 7600M XT", "driver": "32.0.21030.2001", "date": "..." },
  { "name": "Intel(R) Arc(TM) Graphics", "driver": "31.0.101.5382", "date": "..." }
]
```

### `lemonade`

```jsonc
{
  "version": "lemonade version 11.5.0",
  "config": {
    "sdcpp.backend":     "vulkan",
    "llamacpp.backend":  "vulkan",
    "sdcpp.vulkan_args": "(empty)",
    "max_loaded_models": "1",
    "enable_dgpu_gtt":   "false",
    "ctx_size":          "32768"
  }
}
```

只采集这六个键。它们是本项目实测中**唯一被证实会显著改变结果**的配置项
（`enable_dgpu_gtt` 单项就能造成 58 倍差异），其余配置不影响可比性。

### `results`

```jsonc
[
  {
    "kind": "image",              // "image" | "chat"
    "model": "SD-Turbo-GGUF",
    "size": 512,                  // 图像边长；chat 时为 0
    "ok": true,
    "seconds": 3.09,
    "peakDedicatedGB": 2.52,      // 独显专属显存峰值
    "peakSharedGB": 0.31          // 溢出到系统内存的量
  }
]
```

**`ok: false` 的条目必须保留。** 失败本身是数据——
「3072² 在 8GB 卡上必失败」是本项目最有价值的结论之一。

`peakSharedGB` 是判断结果可信度的关键：它显著大于 0 意味着发生了显存溢出，
该次测量反映的是溢出后的性能，不是该配置的真实能力。

## 测量条件（隐含契约）

数据要可比，采集过程必须满足：

- **每个用例前重启 Lemonade 服务**，消除模型残留与显存碎片
- **每个用例先跑一次预热**，只记录第二次
- 预热与正式测量之间不做任何其他 GPU 操作

`collect.ps1` 强制执行这三条。手工拼出来的 JSON 若不满足，
数据不可比——这也是为什么建议一律用脚本导出而不是手填。

## 汇总方式

### 目录约定

```
submissions/
  <gpu-slug>-<schema-safe-id>.json
```

例如 `rx-7600m-xt-8gb-egpu-a1b2c3.json`。`gpu-slug` 便于人眼扫，
后缀是内容哈希前 6 位，避免同型号不同提交撞名。

### 聚合时的分组键

同一张卡在不同配置下的数据**不能混算**。聚合必须按以下组合分组：

```
(gpus[0].name, lemonade.config.enable_dgpu_gtt,
 lemonade.config.sdcpp.backend, results[].model, results[].size)
```

`enable_dgpu_gtt` 必须进分组键——否则一台开着它的机器会把整组中位数拖垮，
而这恰恰是本项目发现的最大陷阱。

### 建议的呈现

按 `(model, size)` 出表，每行一块卡，展示 `seconds` 的中位数与样本数。
样本数 < 3 的行应标注为「单点数据」，不参与横向排名。

## 版本演进

`schema` 字段用 `<name>/<major>` 形式。规则：

- **加可选字段** → 不升 major。老工具忽略新字段即可。
- **改字段含义、删字段、改单位** → 升到 `localai-bench/2`。
  聚合工具应显式拒绝不认识的 major，而不是尽力解析。

已知的 v1 局限，留给 v2：

- 没有记录环境温度或功耗上限，笔记本降频会污染数据
- `size` 假设方形图像，非方形分辨率无法表达
- 没有记录测试时的显示器负载（桌面合成会占用独显显存）
- 单次测量，没有多次采样的方差信息
