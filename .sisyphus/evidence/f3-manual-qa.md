# F3: Real Manual QA Report — multi-vendor-gpu plan

**Date:** 2026-06-10
**Reviewer:** Sisyphus-Junior (F3 wave)
**Project root:** `/home/mlapointe/PyCharmMiscProject/`
**Git HEAD:** `43adb2d` (15 commits on main, "chore: gitignore site/ + mark Wave 4 tasks complete")
**Test environment:** Linux 6.x, Python 3.10.20, uv-managed venv, NVIDIA GeForce RTX 5070 Laptop GPU (8151 MB total / 7696 MB free), AMD Radeon 890M iGPU (512 MB total / 29 MB free), CUDA build of PyTorch 2.12.0+cu130

---

## VERDICT: **APPROVE WITH CAVEAT**

All 12 executable scenarios **functionally pass**. The CLI, the help text, the install/verify scripts, the docs build, and the sidecar metadata all behave as the plan specifies. One **non-blocking cosmetic bug** was discovered: in the rendered `--device --help` output, the text `pip install img2svg[amd]` appears as `pip install img2svg` — typer/rich is interpreting `[amd]` as a markup tag and stripping the bracketed content. The source code is correct (line 310 of `src/img2svg/cli.py`); only the rendered display is wrong. This is a real user-facing issue for anyone copy-pasting the install command from `--help`, but it does not affect the multi-vendor-gpu feature's core behavior (the actual install path is in `install_backend.sh`, not in the CLI help).

**Summary table:**

| # | Scenario | Result | Notes |
|---|----------|--------|-------|
| 1 | `img2svg list-gpus` | PASS | NVIDIA RTX 5070 (Y) + AMD Radeon 890M (512 MB) |
| 2 | `img2svg info` | PASS | `Backend: CUDA (torch 2.12.0+cu130, NVIDIA GeForce RTX 5070 Laptop GPU)` |
| 3 | `convert` default (auto) | PASS | `backend_requested: auto`, `backend_resolved: cuda:0` |
| 4 | `convert --device cpu` | PASS | `backend_requested: cpu`, `backend_resolved: cpu` |
| 5 | `convert --device nonexistent` | PASS (expected fail) | Exit 2, clear validation error listing valid values |
| 6 | `install_backend.sh --dry-run` | PASS | Detects NVIDIA+AMD, warns about 512MB iGPU, recommends `[nvidia]` |
| 7 | `install_backend.sh --help` | PASS | Shows full usage, exit 0 |
| 8 | `verify_backend.sh cpu` | PASS (expected fail) | Exit 1, "expected CPU-only detection, but found cuda" |
| 9 | `verify_backend.sh nvidia` | PASS | Exit 0, "backend verified: nvidia" |
| 10 | `verify_backend.sh amd` | PASS | Exit 0, accepts cuda as ROCm PyTorch |
| 11 | `mkdocs build --strict` | PASS | Exit 0, build in 0.36s, only INFO about platforms/ not in nav |
| 12 | `convert --help` (`--device` text) | PARTIAL | ROCm message present but `[amd]` stripped by rich markup interpretation |
| **TOTAL** | | **11 PASS, 1 PARTIAL** | The 1 partial is a cosmetic help-text bug |

---

## 1. Scenario Results

### 1.1 `img2svg list-gpus`

**Command:** `uv run img2svg list-gpus`
**Exit code:** 0
**Output (abridged):**

```
                                 Available GPUs
┏━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃       ┃        ┃               ┃    VRAM Total ┃     VRAM Free ┃             ┃
┃ Index ┃ Vendor ┃ Name          ┃          (MB) ┃          (MB) ┃ Recommended ┃
┡━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│     0 │ nvidia │ NVIDIA        │          8151 │          7696 │      Y      │
│       │        │ GeForce RTX   │               │               │             │
│       │        │ 5070 Laptop   │               │               │             │
│       │        │ GPU           │               │               │             │
│     1 │ amd    │ AMD Radeon    │           512 │            29 │             │
│       │        │ 890M Graphics │               │               │             │
└───────┴────────┴───────────────┴───────────────┴───────────────┴─────────────┘
```

**Verdict:** **PASS**
- NVIDIA RTX 5070 detected with `Recommended=Y`
- AMD Radeon 890M (iGPU, 512 MB) detected without recommendation
- Both vendors visible in the same table
- Table renders cleanly with no broken cells
- VRAM values are realistic (8151 total matches plan's expected ~8151; 512 matches the iGPU spec)

---

### 1.2 `img2svg info`

**Command:** `uv run img2svg info`
**Exit code:** 0
**Output:**

```
img2svg 0.1.0
Python:  3.10.20
OS:      Linux
Backend: CUDA (torch 2.12.0+cu130, NVIDIA GeForce RTX 5070 Laptop GPU)
Devices: cuda:0, cpu
```

**Verdict:** **PASS**
- `Backend:` line present, exactly as the plan specified
- Format: `CUDA (torch 2.12.0+cu130, NVIDIA GeForce RTX 5070 Laptop GPU)` matches the spec word-for-word
- Uppercased type ("CUDA") from `.value.upper()` per T12's `_format_backend_line()` helper
- `Devices: cuda:0, cpu` is the resolved device list
- Python 3.10.20 reported correctly
- OS detected as Linux

---

### 1.3 `convert` default (auto)

**Command:** `uv run img2svg tests/fixtures/logo.png -o /tmp/f3-auto.svg --mode labels`
**Exit code:** 0
**Stdout:**

```
INFO     convert: tests/fixtures/logo.png -> /tmp/f3-auto.svg (mode=Mode.LABELS)
Converted /tmp/f3-auto.svg
```

**Files written:**
- `/tmp/f3-auto.svg` (501 bytes)
- `/tmp/f3-auto.json` (1186 bytes)

**Sidecar fields:**

```
backend_requested: auto
backend_resolved:  cuda:0
```

**Verdict:** **PASS**
- Exit 0
- Both SVG and sidecar JSON written
- `backend_requested = "auto"` (user's default), `backend_resolved = "cuda:0"` (registry picked CUDA via priority chain)
- No errors; pipeline ran end-to-end on real hardware
- Sidecar contains the new `backend_requested`/`backend_resolved` fields per T10

---

### 1.4 `convert --device cpu`

**Command:** `uv run img2svg tests/fixtures/logo.png -o /tmp/f3-cpu.svg --mode labels --device cpu`
**Exit code:** 0
**Files written:** `/tmp/f3-cpu.svg` (501 B), `/tmp/f3-cpu.json` (1180 B)

**Sidecar fields:**

```
backend_requested: cpu
backend_resolved:  cpu
```

**Verdict:** **PASS**
- Exit 0
- User's explicit `--device cpu` override flows through to the sidecar
- `backend_requested = "cpu"` (user's explicit choice), `backend_resolved = "cpu"` (no GPU fallback)
- Confirms the deprecation shim (T2) is wiring `--device` into `BackendSpec.requested` correctly
- Even on a CUDA host, the user can force CPU; both artifacts still produced

---

### 1.5 `convert --device nonexistent` (expected error)

**Command:** `uv run img2svg tests/fixtures/logo.png -o /tmp/f3-bad.svg --mode labels --device nonexistent`
**Exit code:** 2
**Stderr:**

```
invalid options: 1 validation error for ConversionOptions
requested
  Input should be 'auto', 'cuda', 'rocm', 'mps' or 'cpu'
    For further information visit
https://errors.pydantic.dev/2.13/v/literal_error
```

**Verdict:** **PASS**
- Non-zero exit (2) as expected for an invalid argument
- Clean Pydantic validation error — no Python traceback dump
- Lists all 5 valid values explicitly: `auto`, `cuda`, `rocm`, `mps`, `cpu`
- Links to pydantic docs for the error code
- Confirms the `Literal[...]` type on `BackendSpec.requested` is enforced at the CLI boundary

---

### 1.6 `install_backend.sh --dry-run`

**Command:** `bash scripts/install_backend.sh --dry-run`
**Exit code:** 0
**Output:**

```

img2svg backend install
Host:  Linux (x86_64)
NVIDIA:
  c2:00.0 VGA compatible controller [0300]: NVIDIA Corporation GB206M [GeForce RTX 5070 Max-Q / Mobile] [10de:2d58] (rev a1)
AMD:
  c3:00.0 Display controller [0380]: Advanced Micro Devices, Inc. [AMD/ATI] Strix [Radeon 880M / 890M] [1002:150e] (rev c1)

Detected: NVIDIA + AMD
WARN: AMD iGPU also present (see above). The 512 MB iGPU cannot run
WARN: YOLO11x and will be ignored; install img2svg[amd] only if you
WARN: plan to use a smaller model on a discrete AMD GPU.

Dry-run: would run:
    pip install img2svg[nvidia]

Pass --apply to install.
```

**Verdict:** **PASS**
- Detects NVIDIA (`GB206M [GeForce RTX 5070 Max-Q / Mobile]`, `10de:2d58`) and AMD (`Strix [Radeon 880M / 890M]`, `1002:150e`)
- Reports `Detected: NVIDIA + AMD` (the hybrid-host case)
- Warns about the 512 MB iGPU limitation with the exact text from the plan
- Recommends `pip install img2svg[nvidia]` (prefers NVIDIA over AMD on hybrid hosts)
- Default is dry-run; `--apply` opt-in as documented
- Exit 0

---

### 1.7 `install_backend.sh --help`

**Command:** `bash scripts/install_backend.sh --help`
**Exit code:** 0
**Output:** Full 30+ line usage block printed (see raw transcript). Covers:
- Usage line: `./scripts/install_backend.sh [--dry-run] [--apply]`, `--help`
- Behavior notes (lspci vendor detection, OS detection via `uname -s`, dry-run default)
- Detection priority (Linux: NVIDIA > AMD > cpu; Darwin: always apple)
- Exit codes (0/1/2)
- AMD ROCm wheel index note (`PIP_INDEX_URL=https://download.pytorch.org/whl/rocm6.2`)

**Verdict:** **PASS**
- Shows full usage info from the script's own header (lines 2-40)
- Exit 0 cleanly
- Documents all flags and detection logic
- Explicitly mentions the AMD ROCm wheel index caveat

---

### 1.8 `verify_backend.sh cpu` (expected FAIL on CUDA host)

**Command:** `bash scripts/verify_backend.sh cpu`
**Exit code:** 1
**Output:**

```
Detected:  cuda
Expected:  cpu
[FAIL] expected CPU-only detection, but found cuda
        (CPU is the registry's fallback, not a priority; the
        cpu cell requires a host with no GPU visible to torch)
```

**Verdict:** **PASS** (failure is the expected outcome)
- Exit 1 (the script's documented failure path)
- Message exactly matches the brief's expected text: "expected CPU-only detection, but found cuda"
- Adds the helpful parenthetical explaining *why* CPU is strict (it is the registry's fallback, not a priority)
- Confirms the strict `cpu` rule: any GPU detection triggers a failure
- This is the "right hardware?" gate working correctly for the Jenkins matrix

---

### 1.9 `verify_backend.sh nvidia` (expected PASS)

**Command:** `bash scripts/verify_backend.sh nvidia`
**Exit code:** 0
**Output:**

```
Detected:  cuda
Expected:  nvidia
[OK] backend verified: nvidia
```

**Verdict:** **PASS**
- Exit 0
- The script maps `nvidia` -> `cuda` via its public-axis-name table
- `Detected=cuda` matches the mapped `Expected=cuda` (Branch 3, general match)
- "[OK] backend verified: nvidia" success line printed
- Confirms the nvidia cell in the Jenkins matrix will pass on this host

---

### 1.10 `verify_backend.sh amd` (expected PASS via ROCm branch)

**Command:** `bash scripts/verify_backend.sh amd`
**Exit code:** 0
**Output:**

```
Detected:  cuda
Expected:  amd (accepting cuda as amd; ROCm PyTorch exposes CUDA API)
[OK] backend verified: amd
```

**Verdict:** **PASS**
- Exit 0
- `Detected=cuda` accepted as `amd` via Branch 1 (the ROCm PyTorch equivalence rule)
- The "(accepting cuda as amd; ROCm PyTorch exposes CUDA API)" parenthetical is shown
- "[OK] backend verified: amd" success line printed
- Confirms the script's ROCm-PyTorch-accepts-cuda logic is wired correctly
- On a real ROCm host (where the registry would still emit "cuda" because of the priority chain), this script correctly reports the build as `amd`-equivalent

---

### 1.11 `mkdocs build --strict`

**Command:** `uv run mkdocs build --strict`
**Exit code:** 0
**Output (tail):**

```
INFO    -  Cleaning site directory
INFO    -  Building documentation to directory: /home/mlapointe/PyCharmMiscProject/site
INFO    -  The following pages exist in the docs directory, but are not included in the "nav" configuration:
  - platforms/freebsd.md
  - platforms/macos.md
INFO    -  Documentation built in 0.36 seconds
```

**Verdict:** **PASS**
- Exit 0 (strict mode succeeded)
- Build time 0.36s
- The only diagnostic output is an INFO about `platforms/freebsd.md` and `platforms/macos.md` not being in the nav — explicitly called out as acceptable in the task brief
- A red-bordered block at the top of the output is a **Material for MkDocs theme-team marketing message** about MkDocs 2.0 (URL points to `https://squidfunk.github.io/mkdocs-material/blog/2026/02/18/mkdocs-2.0/`). It is not a build WARNING/ERROR — it is theme-bundled content, not a build diagnostic.
- No WARNING or ERROR lines from the build itself

---

### 1.12 `convert --help` (`--device` help text)

**Command:** `uv run img2svg convert --help`
**Exit code:** 0
**Source code** (`src/img2svg/cli.py:305-311`):

```python
device: str = typer.Option(
    "auto",
    "--device",
    help=(
        "Compute backend: auto (default), cpu, cuda, cuda:N, mps, or rocm. "
        "AMD ROCm requires a ROCm PyTorch build (pip install img2svg[amd])."
    ),
),
```

**Rendered output** (default 80-column terminal):

```
│ --device                TEXT   Compute backend: auto (default), cpu, cuda,   │
│                                cuda:N, mps, or rocm. AMD ROCm requires a     │
│                                ROCm PyTorch build (pip install img2svg).     │
│                                [default: auto]                               │
```

**Verdict:** **PARTIAL PASS** (functional, but a rendering bug)

- The ROCm install message is present in the help text
- The source code correctly says `pip install img2svg[amd]` (line 310)
- **BUT** the rendered help shows `pip install img2svg` — the `[amd]` portion is stripped
- **Root cause:** typer/rich treats `[amd]` as a rich markup tag and silently removes the bracketed content from the rendered output. This is a known quirk of the rich library when `markup=True` (the default) is active.
- **Impact:** A user copying the install command from `--help` would type `pip install img2svg` (without the `[amd]` extra), which installs the base wheel and silently falls back to CPU. The intent — that ROCm users need the `[amd]` extra to get a ROCm PyTorch build — is lost in the rendered output.
- **Suggested fix:** escape the brackets in the help string. Replace:
  ```
  "AMD ROCm requires a ROCm PyTorch build (pip install img2svg[amd])."
  ```
  with:
  ```
  "AMD ROCm requires a ROCm PyTorch build (pip install img2svg\[amd\])."
  ```
  Or, set `rich_markup_mode=None` on the typer `Typer` instance. Or, rephrase to avoid brackets entirely: "(the 'amd' extra of img2svg)".
- **Severity:** Cosmetic/help-text only. The functional code path that actually installs the `[amd]` extra (`install_backend.sh`) is unaffected. The CLI's `--device` validator (which uses the `Literal` type) is also unaffected. This bug is a real follow-up but does **not** block the multi-vendor-gpu feature.

---

## 2. Cross-Cutting Observations

### 2.1 Backend priority chain is correct

`img2svg info` reports `Backend: CUDA` (not ROCm, not MPS, not CPU) on this host. The registry's priority `CUDA > ROCM > MPS > CPU` is honored — even though AMD hardware is physically present, the CUDA build of PyTorch sees NVIDIA first, and CUDA wins. The AMD iGPU is correctly downranked in the recommendation (no `Recommended=Y`).

### 2.2 `backend_resolved` matches the user request in all the expected cases

- `--device auto` -> `backend_resolved: cuda:0` (registry auto-picked the first available GPU)
- `--device cpu` -> `backend_resolved: cpu` (explicit override honored)
- The deprecation shim (T2) is correctly translating `--device` to `BackendSpec.requested` at the CLI boundary, and the registry's `resolve()` is mapping `BackendSpec` to the actual `DeviceBackend` instance.

### 2.3 `verify_backend.sh` rules are tight and correct

The three CPU/nvidia/amd branches all behave per the script's documented rules:
- `cpu` is strict (any GPU detection -> exit 1)
- `nvidia` maps to `cuda` (the public axis name vs the registry enum value)
- `amd` accepts `cuda` (because ROCm PyTorch wheels expose the CUDA API surface)

The `apple` branch was not exercised (this is a Linux host), but the script's case statement and fallback logic is symmetric to the others.

### 2.4 The help-text bug is the only defect

Of 12 scenarios, 11 fully pass and 1 partially passes (functionality works, but the rendered help is misleading). The bug is in the rich-markup stripping of `[amd]`, not in the source code itself.

---

## 3. Suggested Follow-Ups (Non-Blocking)

| # | Issue | Severity | File | Suggested Fix |
|---|-------|----------|------|---------------|
| F1 | `--device --help` strips `[amd]` from rendered output | Low (cosmetic) | `src/img2svg/cli.py:310` | Escape the brackets (`\[amd\]`) or rephrase |

No other issues observed. The multi-vendor-gpu feature is functionally complete and behaves as specified.

---

## 4. Files Produced by This QA

- `.sisyphus/evidence/f3-manual-qa.md` (this file) — structured report
- `.sisyphus/evidence/f3-command-outputs.txt` — verbatim command transcripts for all 12 scenarios

---

## 5. Final Verdict

**APPROVE WITH CAVEAT**

The multi-vendor-gpu implementation passes 11 of 12 QA scenarios fully and the remaining scenario passes functionally but has a cosmetic help-text rendering bug. The core feature (multi-vendor GPU detection, `--device` flag, install script, verify script, sidecar metadata, docs build) is **production-ready**. The single defect is a follow-up that should be filed but does not block the plan's completion.

**Recommend:** Merge the multi-vendor-gpu work as-is. File the `[amd]` rich-markup bug as a separate cleanup ticket.
