# Reproduction report for lemonade-sdk/lemonade#2722

> Ready-to-post comment for
> <https://github.com/lemonade-sdk/lemonade/issues/2722>.
> Reproduces the same `TheRock` failure on **different silicon** (gfx1102 vs the
> reporter's gfx1151), which suggests the bug is in the packaging/extraction
> path rather than in per-architecture support.

---

## Same failure on gfx1102 (RX 7600M XT), lemonade v11.5.0

Confirming this is not specific to gfx1151 / Ryzen AI Max+. I hit an identical
`TheRock` failure on a discrete RDNA3 card, on the latest release.

### Environment

| | |
|---|---|
| GPU | AMD Radeon RX 7600M XT 8GB, **gfx1102** (Navi 33, RDNA3) |
| Connection | Thunderbolt 3 eGPU enclosure (WIKO Hi GT Cube) |
| Host | Intel Core Ultra + Arc iGPU (dual-GPU system) |
| OS | Windows 11 Pro 26200 x64 |
| Driver | Adrenalin 32.0.21030.2001 (25.10.30.02), 2025-09-24 |
| Lemonade | **11.5.0** (latest at time of writing) |

`lemonade backends` reports `sd-cpp:rocm` as `installable`
("Backend is supported but not installed"), so hardware filtering considers
this GPU supported.

### Steps to reproduce

```
lemonade backends install sd-cpp:rocm
```

### Actual result

```
Installing backend: sd-cpp:rocm
[1/2] sd-master-8caa3f9-bin-win-rocm-7.13.0-x64.zip

[2/2] therock-dist-windows-gfx110X-all-7.13.0.tar.gz
Total: 5.2 GB, 2 files
Error: TheRock extraction failed: bin directory not found
```

Elapsed: **1055.8 s** (~17.6 min) — the full 5.2 GB is downloaded before the
failure, so the retry cost is high.

Note the asset name is `therock-dist-windows-gfx110X-all-7.13.0.tar.gz`
(the `gfx110X-all` bundle), i.e. the correct family bundle for gfx1102 *was*
selected. The failure is at extraction, after download.

### Post-failure state

```
C:\Users\<user>\.cache\lemonade\bin\therock\      <- exists, EMPTY
C:\Users\<user>\.cache\lemonade\bin\.downloads\   <- exists, EMPTY
```

So the archive is fetched and then discarded, leaving an empty target
directory. Nothing is cached for a retry.

For contrast, the sibling backends install fine and populate correctly:

```
lemonade backends install sd-cpp:vulkan   -> OK in 9.6 s
lemonade backends install sd-cpp:cpu      -> OK in 5.4 s

C:\Users\<user>\.cache\lemonade\bin\sd-cpp\vulkan\   <- 6 files, sd-server.exe present
C:\Users\<user>\.cache\lemonade\bin\sd-cpp\cpu\      <- 6 files, sd-server.exe present
```

So the `sd-cpp` half of the install works; only the TheRock/ROCm runtime half fails.

### Why this looks like a packaging bug, not a support gap

1. **AMD's own Windows support matrix lists gfx1102 (RX 7600) as fully
   supported** in both the *Runtime* and *HIP SDK* columns.
2. The HIP runtime shipped with the Adrenalin driver is present and healthy:
   `C:\Windows\System32\amdhip64_6.dll` (10.0.3652.0) and `amd_comgr_2.dll`.
3. The GPU is fully functional for compute via other paths — OpenCL enumerates it
   (`gfx1102`, 7.98 GB, `cl_khr_fp16`/`fp64`), and the **Vulkan backend works and
   is fast** on the very same card:
   - SD-Turbo 512², warm: **2.47 s**
   - DeepSeek-Qwen3-8B: **33–34 tok/s**
4. The reporter of this issue is on **gfx1151**; I'm on **gfx1102**. Two
   different architectures, two different bundles, same extraction failure.

### Attempted workarounds

- **`--force` does not help.** Verified: `lemonade backends install sd-cpp:rocm
  --force` re-downloads the full 5.2 GB (813 s) and fails at the same point with
  the same message, leaving `therock/` empty again. This matches the documented
  behaviour — `--force` bypasses *hardware filtering*, which was never the
  blocker, since the backend already reports as `installable`.
- No CLI flag exists to pin an alternate ROCm/TheRock version.
- `rocm_channel` is `stable`; switching to nightly is reported broken in
  [#1902](https://github.com/lemonade-sdk/lemonade/issues/1902)
  (`backend_versions.json is missing version for: sd-cpp:rocm-nightly`).

Each failed attempt costs another 5.2 GB download, which is why the
"keep the archive on failure" suggestion below matters in practice.

### Suggestion

The error message `bin directory not found` suggests the extractor expects a
`bin/` at a specific depth inside `therock-dist-windows-gfx110X-all-7.13.0.tar.gz`.
If the archive's internal layout changed (e.g. an extra top-level directory, or
`bin/` moved under a subdirectory), a path-prefix mismatch would produce exactly
this symptom while the download itself succeeds.

Two things that would help users a lot regardless of the root cause:

1. **Validate the archive layout before deleting it**, or keep it in
   `.downloads/` on failure — right now a failed install costs another 5.2 GB
   and ~18 minutes to retry.
2. **Print the extracted top-level entries** in the error, e.g.
   `expected bin/ under <root>, found: [<entries>]`. That would make this
   diagnosable from a single user report.

I'm happy to run any diagnostic build or dump the archive layout on request.

### Related

- [#1902](https://github.com/lemonade-sdk/lemonade/issues/1902) — sd-cpp ROCm download failure (closed, no documented fix)
- [#1807](https://github.com/lemonade-sdk/lemonade/issues/1807) — ROCm URL construction / stale version

---

## How to post this

The report above is ready to paste. To post it from the CLI:

```powershell
gh issue comment 2722 --repo lemonade-sdk/lemonade --body-file docs/issue-2722-repro.md
```

Note that the file contains this "How to post" section too — trim it, or paste
only the report body from the web UI.
