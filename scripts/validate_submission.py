#!/usr/bin/env python3
"""校验 submissions/ 下的基准 JSON 是否符合 localai-bench schema。

设计取向：**宁可漏报，不可误杀**。

这是个开放数据仓库，贡献者跑完脚本、贴个 JSON 就够麻烦了。
如果校验器因为一个可选字段缺失就把 PR 判红，人家下次就不来了。
所以只在「数据无法比较」或「泄露了个人信息」时报错，其余一律 warning。

用法:
    python scripts/validate_submission.py                # 校验 submissions/ 全部
    python scripts/validate_submission.py path/to/a.json # 校验指定文件
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCHEMA_MAJOR = 1
SCHEMA_PREFIX = "localai-bench/"

# 这六个键是实测中唯一被证实会显著改变结果的配置项
# （enable_dgpu_gtt 单项就能造成 58 倍差异）。缺了它们数据就没法归组比较。
REQUIRED_CONFIG_KEYS = {
    "sdcpp.backend",
    "llamacpp.backend",
    "enable_dgpu_gtt",
    "max_loaded_models",
}

# 泄露个人信息的迹象。贡献者不该需要逐字段自查，所以这里兜底。
# 注意 \\{1,2}：JSON 原文里反斜杠是转义过的（C:\\Users\\...），
# 只匹配单个反斜杠会漏掉最常见的泄露形式。
LEAK_PATTERNS = [
    (re.compile(r"[A-Za-z]:\\{1,2}Users\\{1,2}[^\\\"]+", re.I), "Windows 用户目录路径"),
    (re.compile(r"/(?:home|Users)/[^/\"]+", re.I), "Unix 用户目录路径"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "邮箱地址"),
]


class Report:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def emit(self) -> bool:
        name = self.path.name
        for m in self.errors:
            print(f"  [ERROR] {name}: {m}")
        for m in self.warnings:
            print(f"  [warn ] {name}: {m}")
        if not self.errors and not self.warnings:
            print(f"  [ok   ] {name}")
        return not self.errors


def check_leaks(raw: str, rep: Report) -> None:
    for pattern, what in LEAK_PATTERNS:
        hit = pattern.search(raw)
        if hit:
            # 不回显命中内容本身，避免把泄露的东西又打进 CI 日志
            rep.error(f"疑似包含{what}，请用 collect.ps1 重新导出（位置 {hit.start()}）")


def check_schema(data: dict, rep: Report) -> None:
    schema = data.get("schema")
    if not isinstance(schema, str) or not schema.startswith(SCHEMA_PREFIX):
        rep.error(f"schema 字段缺失或格式不对，应为 '{SCHEMA_PREFIX}<major>'")
        return
    try:
        major = int(schema[len(SCHEMA_PREFIX):])
    except ValueError:
        rep.error(f"schema 版本号无法解析: {schema!r}")
        return
    if major > SCHEMA_MAJOR:
        # 显式拒绝而不是尽力解析——schema.md 里定的规则
        rep.error(f"schema {schema} 比本校验器（v{SCHEMA_MAJOR}）新，请升级工具")
    elif major < SCHEMA_MAJOR:
        rep.warn(f"schema {schema} 是旧版本，字段含义可能已变")


def check_results(data: dict, rep: Report) -> None:
    results = data.get("results")
    if not isinstance(results, list) or not results:
        rep.error("results 为空或不是数组")
        return

    for i, r in enumerate(results):
        where = f"results[{i}]"
        if not isinstance(r, dict):
            rep.error(f"{where} 不是对象")
            continue
        for key in ("kind", "model", "ok", "seconds"):
            if key not in r:
                rep.error(f"{where} 缺少必需字段 {key}")
        if r.get("kind") not in (None, "image", "chat"):
            rep.warn(f"{where}.kind = {r.get('kind')!r}，预期 image 或 chat")

        # 溢出量决定这条数据可不可信，不是可选信息
        shared = r.get("peakSharedGB")
        if shared is None:
            rep.warn(f"{where} 没有 peakSharedGB，无法判断该次测量是否被显存溢出污染")
        elif isinstance(shared, (int, float)) and shared > 1:
            rep.warn(
                f"{where} 共享显存峰值 {shared}GB —— 发生了显存溢出，"
                f"这条反映的是溢出后的性能，不是该配置的真实能力"
            )

    if not any(r.get("ok") is False for r in results if isinstance(r, dict)):
        # 不报错：一台大显存机器全部成功是完全正常的
        rep.warn("没有任何失败条目。如果你在压测上限，失败本身是数据，请保留")


def check_config(data: dict, rep: Report) -> None:
    lem = data.get("lemonade")
    if not isinstance(lem, dict):
        rep.error("缺少 lemonade 段，无法知道数据是在什么配置下产生的")
        return
    cfg = lem.get("config")
    if not isinstance(cfg, dict):
        rep.error("缺少 lemonade.config，无法归组比较")
        return
    missing = REQUIRED_CONFIG_KEYS - set(cfg)
    if missing:
        rep.error(
            "lemonade.config 缺少归组必需的键: "
            + ", ".join(sorted(missing))
            + "（缺了它们这条数据没法和别人的比）"
        )
    if str(cfg.get("enable_dgpu_gtt", "")).lower() == "true":
        rep.warn(
            "enable_dgpu_gtt=true —— 本仓库实测该设置会让出图慢 58 倍。"
            "数据仍然有价值（正好可以验证这条结论），但不要和 false 的数据混算"
        )


def check_gpus(data: dict, rep: Report) -> None:
    gpus = data.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        rep.error("缺少 gpus 段")
        return
    if not any(isinstance(g, dict) and g.get("name") for g in gpus):
        rep.error("gpus 里没有一个带 name")


def validate(path: Path) -> bool:
    rep = Report(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        rep.error("不是 UTF-8 编码")
        return rep.emit()

    check_leaks(raw, rep)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        rep.error(f"JSON 解析失败: {exc}")
        return rep.emit()

    if not isinstance(data, dict):
        rep.error("顶层不是对象")
        return rep.emit()

    check_schema(data, rep)
    check_gpus(data, rep)
    check_config(data, rep)
    check_results(data, rep)
    return rep.emit()


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        targets = [Path(a) for a in argv[1:]]
    else:
        root = Path(__file__).resolve().parent.parent / "submissions"
        if not root.is_dir():
            print("submissions/ 不存在，没有要校验的文件")
            return 0
        targets = sorted(root.glob("*.json"))

    if not targets:
        print("没有找到 JSON 文件")
        return 0

    print(f"校验 {len(targets)} 个提交:")
    ok = all(validate(p) for p in targets)
    print()
    if ok:
        print("全部通过")
        return 0
    print("有文件未通过，见上方 [ERROR]")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
