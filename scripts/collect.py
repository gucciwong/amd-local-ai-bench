#!/usr/bin/env python3
"""跨平台基准采集器：跑一套标准化测试，导出可跨机汇总的 JSON。

与 collect.ps1 等价，但同时支持 Linux —— 这很重要，因为本仓库最大的
未验证盲区就是「Linux 上 ROCm 到底行不行」，而 PowerShell 版本把
Linux 用户完全挡在了门外。

设计约束（都是踩过坑才定的）：
  * 硬件信息自动探测，不写死任何 LUID 或路径
  * 每个用例前重启服务 —— 状态残留能让同一配置差 58 倍
  * 记录失败与异常值，不只记成功的
  * schema 带版本号，字段演进不会让旧数据失效

用法:
    python scripts/collect.py --quick
    python scripts/collect.py --note "RX 7900 XTX desktop"
    python scripts/collect.py --out result.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "localai-bench/1"
API = "http://127.0.0.1:13305/api/v1"
IS_WINDOWS = platform.system() == "Windows"


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def post_json(url: str, payload: dict, timeout: int) -> bool:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def wait_healthy(timeout_s: int = 180) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{API}/health", timeout=3):
                return True
        except Exception:
            time.sleep(2)
    return False


# --------------------------------------------------------------------------
# 硬件探测
# --------------------------------------------------------------------------

def _run(cmd: list[str], timeout: int = 30) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        ).stdout
    except Exception:
        return ""


def gpus_windows() -> list[dict]:
    ps = (
        "Get-CimInstance Win32_VideoController | "
        "Where-Object { $_.PNPDeviceID -like 'PCI*' } | "
        "Select-Object Name,DriverVersion | ConvertTo-Json -Compress"
    )
    out = _run(["powershell", "-NoProfile", "-Command", ps])
    try:
        data = json.loads(out) if out.strip() else []
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    return [{"name": d.get("Name"), "driver": d.get("DriverVersion")} for d in data]


def gpus_linux() -> list[dict]:
    """从 sysfs 读，不依赖 rocm-smi —— 后者在很多发行版上没装。"""
    gpus: list[dict] = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        dev = card / "device"
        vendor = (dev / "vendor").read_text().strip() if (dev / "vendor").exists() else ""
        if vendor != "0x1002":  # 0x1002 = AMD
            # 仍然记录非 AMD 卡：双显卡机器的设备顺序本身是有信息量的
            name = f"non-AMD (vendor {vendor})" if vendor else "unknown"
            gpus.append({"name": name, "driver": None, "sysfs": card.name})
            continue
        # 优先用 amdgpu 暴露的可读名字，退化到 device id
        label = None
        for candidate in ("product_name", "device"):
            p = dev / candidate
            if p.exists():
                label = p.read_text().strip()
                break
        ver = Path("/sys/module/amdgpu/version")
        gpus.append({
            "name": f"AMD {label}" if label else "AMD (unknown)",
            "driver": ver.read_text().strip() if ver.exists() else None,
            "sysfs": card.name,
        })
    return gpus


def gpu_inventory() -> list[dict]:
    return gpus_windows() if IS_WINDOWS else gpus_linux()


class VramProbe:
    """采样独显显存。两个平台的数据来源完全不同，所以封起来。

    Windows: GPU 性能计数器，独显 = 专属显存占用最高的适配器
             （核显是 UMA，专属显存恒为 0）
    Linux:   amdgpu 的 sysfs 节点，直接给字节数
    """

    def __init__(self) -> None:
        self.kind = "windows" if IS_WINDOWS else "linux"
        self.node: Path | None = None
        if not IS_WINDOWS:
            for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
                dev = card / "device"
                if (dev / "vendor").exists() and (dev / "vendor").read_text().strip() == "0x1002":
                    used = dev / "mem_info_vram_used"
                    if used.exists():
                        self.node = used
                        break

    def sample(self) -> tuple[float | None, float | None]:
        """返回 (专属 GB, 共享 GB)。Linux 侧没有等价的「共享」概念，返回 None。"""
        if IS_WINDOWS:
            ps = (
                "$m = Get-CimInstance -ClassName "
                "Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory "
                "-EA SilentlyContinue | Sort-Object DedicatedUsage -Descending | "
                "Select-Object -First 1; "
                "if ($m) { '{0} {1}' -f $m.DedicatedUsage, $m.SharedUsage }"
            )
            out = _run(["powershell", "-NoProfile", "-Command", ps]).strip()
            parts = out.split()
            if len(parts) == 2:
                try:
                    return int(parts[0]) / 2**30, int(parts[1]) / 2**30
                except ValueError:
                    pass
            return None, None

        if self.node and self.node.exists():
            try:
                return int(self.node.read_text().strip()) / 2**30, None
            except (ValueError, OSError):
                pass
        return None, None


# --------------------------------------------------------------------------
# Lemonade 控制
# --------------------------------------------------------------------------

def lemonade_cli() -> str | None:
    found = shutil.which("lemonade")
    if found:
        return found
    if IS_WINDOWS:
        p = Path(os.environ.get("LOCALAPPDATA", "")) / "lemonade_server/bin/lemonade.exe"
        if p.exists():
            return str(p)
    return None


def restart_server() -> bool:
    """每个用例前重启。不做这一步，数据不可比 —— 实测能差 58 倍。"""
    if IS_WINDOWS:
        _run(["powershell", "-NoProfile", "-Command",
              "Get-Process -Name LemonadeServer -EA SilentlyContinue | Stop-Process -Force"])
        _run(["powershell", "-NoProfile", "-Command",
              "Get-CimInstance Win32_Process -Filter \"Name='sd-server.exe' OR "
              "Name='llama-server.exe'\" -EA SilentlyContinue | "
              "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -EA SilentlyContinue }"])
        time.sleep(3)
        exe = Path(os.environ.get("LOCALAPPDATA", "")) / "lemonade_server/bin/LemonadeServer.exe"
        if exe.exists():
            subprocess.Popen([str(exe)], creationflags=0x08000000)  # CREATE_NO_WINDOW
    else:
        # Linux 上 lemond 由 systemd 管理；先试 user 单元，再试 system
        for unit_scope in (["--user"], []):
            r = subprocess.run(["systemctl", *unit_scope, "restart", "lemond"],
                               capture_output=True, text=True)
            if r.returncode == 0:
                break
        else:
            for name in ("sd-server", "llama-server"):
                _run(["pkill", "-f", name])
        time.sleep(3)
    return wait_healthy()


def read_config(cli: str | None) -> dict:
    """只采集实测证明会显著改变结果的键。其余不影响可比性。"""
    keys = ["sdcpp.backend", "llamacpp.backend", "sdcpp.vulkan_args",
            "max_loaded_models", "enable_dgpu_gtt", "ctx_size"]
    if not cli:
        return {}
    raw = _run([cli, "config"])
    cfg = {}
    for k in keys:
        m = re.search(rf"^\s*{re.escape(k)}\s+(\S.*?)\s*$", raw, re.MULTILINE)
        if m:
            cfg[k] = m.group(1).strip()
    return cfg


# --------------------------------------------------------------------------
# 用例
# --------------------------------------------------------------------------

def run_case(kind: str, model: str, size: int, probe: VramProbe) -> dict:
    if kind == "image":
        url = f"{API}/images/generations"
        payload = {"model": model, "prompt": "a small ceramic teapot on a wooden table",
                   "size": f"{size}x{size}", "n": 1}
    else:
        url = f"{API}/chat/completions"
        payload = {"model": model,
                   "messages": [{"role": "user", "content": "Say hello in exactly 5 words."}],
                   "max_tokens": 24}

    import threading
    result: dict = {}

    def worker() -> None:
        result["ok"] = post_json(url, payload, timeout=900)

    t = threading.Thread(target=worker, daemon=True)
    start = time.time()
    t.start()

    peak_d = 0.0
    peak_s = 0.0
    while t.is_alive() and time.time() - start < 880:
        d, s = probe.sample()
        if d is not None:
            peak_d = max(peak_d, d)
        if s is not None:
            peak_s = max(peak_s, s)
        time.sleep(0.4)
    t.join(timeout=10)
    elapsed = time.time() - start

    row = {
        "kind": kind, "model": model, "size": size,
        "ok": bool(result.get("ok")),
        "seconds": round(elapsed, 2),
        "peakDedicatedGB": round(peak_d, 2),
    }
    # Linux 没有等价的共享显存概念，省略比填 0 诚实
    if IS_WINDOWS:
        row["peakSharedGB"] = round(peak_s, 2)
    return row


QUICK = [("image", "SD-Turbo-GGUF", 512), ("chat", "Gemma-4-E2B-it-GGUF", 0)]
FULL = [
    ("image", "SD-Turbo-GGUF", 512),
    ("image", "SD-Turbo-GGUF", 1024),
    ("image", "SD-Turbo-GGUF", 2048),
    ("image", "SD-Turbo", 512),
    ("image", "SDXL-Turbo", 1024),
    ("chat", "Gemma-4-E2B-it-GGUF", 0),
    ("chat", "DeepSeek-Qwen3-8B-GGUF", 0),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="只跑最小集（约 2 分钟）")
    ap.add_argument("--note", default="", help="自述，例如显卡型号与连接方式")
    ap.add_argument("--out", default="", help="输出路径")
    args = ap.parse_args()

    cli = lemonade_cli()
    if not cli:
        print("找不到 lemonade CLI。先安装 Lemonade Server。", file=sys.stderr)
        return 1
    if not wait_healthy(timeout_s=10) and not restart_server():
        print("Lemonade 服务起不来。", file=sys.stderr)
        return 1

    gpus = gpu_inventory()
    probe = VramProbe()
    print(f"平台: {platform.system()}  显存采样: {probe.kind}")
    for g in gpus:
        print(f"  {g.get('name')}  driver={g.get('driver')}")
    if not IS_WINDOWS and probe.node is None:
        print("  警告: 没找到 amdgpu 的 mem_info_vram_used，显存数据将为空")
    print()

    suite = QUICK if args.quick else FULL
    results = []
    for i, (kind, model, size) in enumerate(suite, 1):
        label = f"[{i}/{len(suite)}] {kind} {model} @{size}"
        print(label, end="", flush=True)
        if not restart_server():
            print("   服务重启失败，跳过")
            continue
        run_case(kind, model, size, probe)  # 预热
        row = run_case(kind, model, size, probe)
        results.append(row)
        print(f"   {row['seconds']}s  {row['peakDedicatedGB']}GB"
              f"{'' if row['ok'] else '  [失败]'}")

    version = _run([cli, "--version"]).strip()
    payload = {
        "schema": SCHEMA,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": args.note,
        "host": {
            "os": f"{platform.system()} {platform.release()}",
            "cpu": platform.processor() or "unknown",
        },
        "gpus": gpus,
        "lemonade": {"version": version, "config": read_config(cli)},
        "results": results,
    }

    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent /
        f"bench-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    ok = sum(1 for r in results if r["ok"])
    print(f"\n{ok}/{len(results)} 成功  ->  {out}")
    print("可把这个 JSON 分享出来做跨机对比（不含任何路径或个人信息）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
