# Makefile - unified wrapper for the img2svg project.
# Copyright (c) 2026, REVYTECH, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# This Makefile is the single entry point a fresh user touches. A bare
# `make` prints the help menu; `make install` installs the package end
# to end with no flags. It wraps scripts/*.sh and the Python tooling
# (uv / pip) so the project becomes useful with one command.
#
# === Portability ===========================================================
# This file parses cleanly under GNU Make 4.x AND FreeBSD make (bmake).
# To stay in the common subset we deliberately avoid:
#   * `:=` and `?=` (use `=` only)            - BSD does not support
#   * `ifeq`, `ifneq`, `ifdef`                 - GNU-only; use shell `case`
#   * `$(@D)`, `$(@F)`, `${var:...}` modifiers  - GNU-only; use shell tools
#   * `echo -e`                                - not POSIX; use `printf`
#   * `[[ ... ]]`                              - bash-only; use `[ ... ]`
#   * `which`                                  - non-portable; use `command -v`
#   * `platform.system()`                      - per project rules; use `uname -s`
#
# === Conventions ===========================================================
# * `make` with no target prints the help menu (DEFAULT_GOAL = help).
# * `make install` has NO prerequisites; it is self-contained. The user
#   should never see "Error: prerequisite X failed" from `make install`.
# * `make install` calls `scripts/install_backend.sh --apply` so the
#   install actually runs (the script defaults to --dry-run).
# * Every public target carries a `## description` tail that the `help`
#   target extracts and renders with awk + sort.
# * All targets are declared .PHONY.

# === Variables =============================================================
# All variables use `=` (recursive) so the file parses on both GNU and BSD.

UNAME_S = $(shell uname -s)
UNAME_M = $(shell uname -m)
PYTHON  = $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "")
UV      = $(shell command -v uv 2>/dev/null || echo "")
PIP     = $(shell command -v pip 2>/dev/null || command -v pip3 2>/dev/null || echo "")

# === Default goal ==========================================================
# `make` with no target prints the help menu.

.DEFAULT_GOAL := help

# === Phony targets =========================================================
# Declared up front so the order of the recipe section below does not
# matter for make's prerequisite resolution.

.PHONY: help info install install-dry-run \
        install-cpu install-nvidia install-amd install-apple \
        uninstall purge verify manpage check-freebsd \
        self-test test lint format build docs clean \
        _install_extra

# === help ==================================================================
# Self-documenting help. Each public target carries `## description`; this
# target greps them out of the Makefile itself, formats with awk (printf for
# colors, never `echo -e`), and sorts alphabetically.

help: ## Show this help menu
	@printf "\n"
	@printf "img2svg — Makefile wrapper\n"
	@printf "%-10s %s/%s\n" "host:"    "$(UNAME_S)" "$(UNAME_M)"
	@printf "%-10s %s\n"     "python:"  "$(PYTHON)"
	@printf "%-10s %s\n"     "uv:"      "$(UV)"
	@printf "%-10s %s\n"     "pip:"     "$(PIP)"
	@printf "\n"
	@printf "Install extras: cpu, nvidia, amd, apple\n"
	@printf "FreeBSD note:  [nvidia]/[amd]/[apple] all fall back to [cpu].\n"
	@printf "\n"
	@printf "Targets:\n"
	@printf "  \033[36m%-22s\033[0m %s\n" "TARGET" "DESCRIPTION"
	@grep -hE '^[a-zA-Z_-][a-zA-Z0-9_-]*:.*?## .*$$' $(MAKEFILE_LIST) \
	    | awk -F':.*## ' '{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' \
	    | sort
	@printf "\n"

# === info ==================================================================
# Print host, make, shell, Python, uv, pip, and detected GPU.

info: ## Show system + project info
	@printf "\n"
	@printf "img2svg — environment\n"
	@printf "%-22s %s/%s\n" "host:"    "$(UNAME_S)" "$(UNAME_M)"
	@printf "%-22s %s\n"     "make:"    "$$(command -v make 2>/dev/null) ($$(make --version 2>/dev/null | head -1))"
	@printf "%-22s %s\n"     "shell:"   "$${SHELL:-unknown}"
	@_PYTHON_VER="(not found)"; if [ -n "$(PYTHON)" ]; then _PYTHON_VER=`"$(PYTHON)" --version 2>&1`; fi; \
	_UV_VER="(not found)"; if [ -n "$(UV)" ]; then _UV_VER=`"$(UV)" --version 2>&1`; fi; \
	_PIP_VER="(not found)"; if [ -n "$(PIP)" ]; then _PIP_VER=`"$(PIP)" --version 2>&1`; fi; \
	_IMG2SVG_BIN=`command -v img2svg 2>/dev/null || echo '(not installed)'`; \
	printf "%-22s %s\n" "python:" "$$_PYTHON_VER"; \
	printf "%-22s %s\n" "uv:" "$$_UV_VER"; \
	printf "%-22s %s\n" "pip:" "$$_PIP_VER"; \
	printf "%-22s %s\n" "img2svg (installed):" "$$_IMG2SVG_BIN"; \
	printf "\n"; \
	printf "Detected GPU:\n"; \
	if command -v lspci >/dev/null 2>&1; then \
	    lspci -nn 2>/dev/null | grep -E "0300|0302|0380" | head -5 || true; \
	else \
	    printf "  (lspci not found; install pciutils on Linux for GPU detection)\n"; \
	fi; \
	printf "\n"

# === install ===============================================================
# Self-contained: NO prerequisites. The user types `make install` and gets
# a working install. No "first run `make sync`" prerequisite, no prompts.
#
# Strategy:
#   1. Try `scripts/install_backend.sh --apply` (the smart path that
#      detects the GPU, prints warnings, and runs the right pip command).
#   2. If the script is missing, fall back to a direct install via uv
#      (or pip) using a simple platform case statement.

install: ## Install img2svg (auto-detect platform + GPU backend)
	@printf "Installing img2svg on %s/%s...\n" "$(UNAME_S)" "$(UNAME_M)"
	@if [ -x scripts/install_backend.sh ]; then \
	    bash scripts/install_backend.sh --apply; \
	else \
	    printf "scripts/install_backend.sh not found; falling back to direct install.\n" >&2; \
	    EXTRA=cpu; \
	    case "$(UNAME_S)" in \
	        Darwin) EXTRA=apple ;; \
	        Linux) \
	            if command -v lspci >/dev/null 2>&1; then \
	                if lspci -nn -d 10de: 2>/dev/null | grep -qE "0300|0302|0380"; then \
	                    EXTRA=nvidia; \
	                elif lspci -nn -d 1002: 2>/dev/null | grep -qE "0300|0302|0380"; then \
	                    EXTRA=amd; \
	                fi; \
	            fi; \
	            ;; \
	        FreeBSD) \
	            printf "On FreeBSD, install Python + uv first:\n" >&2; \
	            printf "    sudo pkg install python3 py311-uv\n" >&2; \
	            EXTRA=cpu; \
	            ;; \
	        *) \
	            printf "ERROR: unsupported platform: %s\n" "$(UNAME_S)" >&2; \
	            exit 1; \
	            ;; \
	    esac; \
	    printf "Detected extra: [cpu|nvidia|amd|apple] -> %s\n" "$$EXTRA"; \
	    if [ -n "$(UV)" ]; then \
	        $(UV) pip install --system "img2svg[$$EXTRA]"; \
	    elif [ -n "$(PIP)" ]; then \
	        $(PIP) install "img2svg[$$EXTRA]"; \
	    else \
	        printf "ERROR: no package manager found (need uv or pip on PATH)\n" >&2; \
	        exit 1; \
	    fi; \
	fi

install-dry-run: ## Show what 'make install' would do without doing it
	@printf "Dry run: showing what 'make install' would do on %s/%s.\n" "$(UNAME_S)" "$(UNAME_M)"
	@if [ -x scripts/install_backend.sh ]; then \
	    bash scripts/install_backend.sh; \
	else \
	    printf "scripts/install_backend.sh not found; cannot produce install plan.\n" >&2; \
	    exit 1; \
	fi

# === install-extras (explicit backend) =====================================
# Bypass the smart detection; install the named extra directly. Useful
# when the user knows what GPU they have (or when running headless on
# a host where lspci is missing).

install-cpu: ## Install img2svg[cpu] (no GPU acceleration)
	@$(MAKE) _install_extra EXTRA=cpu

install-nvidia: ## Install img2svg[nvidia] (NVIDIA CUDA)
	@$(MAKE) _install_extra EXTRA=nvidia

install-amd: ## Install img2svg[amd] (AMD ROCm; sets PIP_INDEX_URL/UV_INDEX_URL)
	@$(MAKE) _install_extra EXTRA=amd

install-apple: ## Install img2svg[apple] (Apple Silicon MPS)
	@$(MAKE) _install_extra EXTRA=apple

# Private helper invoked by the four `install-<extra>` targets above.
# EXTRA is set on the command line: `$(MAKE) _install_extra EXTRA=cpu`.
# Kept out of help on purpose (no `##` description).

_install_extra:
	@if [ -z "$(EXTRA)" ]; then \
	    printf "ERROR: _install_extra requires EXTRA=<cpu|nvidia|amd|apple>\n" >&2; \
	    exit 1; \
	fi; \
	printf "Installing img2svg[%s] on %s...\n" "$(EXTRA)" "$(UNAME_S)"; \
	case "$(UNAME_S)" in \
	    FreeBSD) \
	        if [ "$(EXTRA)" != "cpu" ]; then \
	            printf "WARN: FreeBSD only ships CPU PyTorch wheels; using [cpu] instead of [%s].\n" "$(EXTRA)" >&2; \
	        fi; \
	        EFFECTIVE=cpu; \
	        ;; \
	    *) EFFECTIVE=$(EXTRA) ;; \
	esac; \
	INDEX_URL=""; \
	case "$$EFFECTIVE" in \
	    amd) \
	        printf "Note: AMD ROCm wheels live on a separate PyTorch index.\n"; \
	        printf "  PIP_INDEX_URL=https://download.pytorch.org/whl/rocm6.2\n"; \
	        INDEX_URL="https://download.pytorch.org/whl/rocm6.2"; \
	        ;; \
	    *) INDEX_URL="" ;; \
	esac; \
	if [ -n "$(UV)" ]; then \
	    if [ -n "$$INDEX_URL" ]; then \
	        UV_INDEX_URL="$$INDEX_URL" $(UV) pip install --system "img2svg[$$EFFECTIVE]"; \
	    else \
	        $(UV) pip install --system "img2svg[$$EFFECTIVE]"; \
	    fi; \
	elif [ -n "$(PIP)" ]; then \
	    if [ -n "$$INDEX_URL" ]; then \
	        PIP_INDEX_URL="$$INDEX_URL" $(PIP) install "img2svg[$$EFFECTIVE]"; \
	    else \
	        $(PIP) install "img2svg[$$EFFECTIVE]"; \
	    fi; \
	else \
	    printf "ERROR: no package manager found (need uv or pip on PATH)\n" >&2; \
	    exit 1; \
	fi

# === uninstall =============================================================
# Tries multiple methods. Each is wrapped in `|| true` so partial success
# is fine. Config and cache are deliberately left in place; use `make purge`
# to remove them.

uninstall: ## Remove the img2svg install (config + cache are kept)
	@printf "Uninstalling img2svg...\n"
	@if [ -n "$(UV)" ]; then \
	    $(UV) tool uninstall img2svg 2>/dev/null || true; \
	    $(UV) pip uninstall -y img2svg 2>/dev/null || true; \
	fi; \
	if [ -n "$(PIP)" ]; then \
	    $(PIP) uninstall -y img2svg 2>/dev/null || true; \
	fi; \
	if [ -n "$(PYTHON)" ]; then \
	    $(PYTHON) -m pip uninstall -y img2svg 2>/dev/null || true; \
	fi; \
	printf "img2svg uninstalled.\n"; \
	printf "Config and cache were kept. Use 'make purge' to remove them.\n"

# === purge =================================================================
# Uninstall + remove XDG dirs + hint about the man page. Asks for
# confirmation via `read -p` (supported by bash, dash, and FreeBSD sh).

purge: ## Uninstall img2svg AND remove config + cache
	@printf "Purging img2svg (install + config + cache + man page hint)...\n"
	@printf "\n"
	@printf "This will remove:\n"
	@printf "  - the img2svg Python package\n"
	@printf "  - \$$HOME/.config/img2svg\n"
	@printf "  - \$$HOME/.cache/img2svg\n"
	@printf "  - \$$HOME/.local/share/img2svg\n"
	@printf "  - the man page (you must remove it manually with sudo)\n"
	@printf "\n"
	@read -p "Continue? [y/N] " REPLY; \
	case "$$REPLY" in \
	    y|Y|yes|YES) printf "Proceeding with purge.\n" ;; \
	    *) printf "Aborted.\n"; exit 1 ;; \
	esac
	@$(MAKE) uninstall
	@rm -rf "$$HOME/.config/img2svg" "$$HOME/.cache/img2svg" "$$HOME/.local/share/img2svg" 2>/dev/null || true
	@if [ -x scripts/install_manpage.sh ]; then \
	    printf "Note: to remove the system man page, run: sudo scripts/install_manpage.sh --uninstall\n" >&2; \
	else \
	    rm -f /usr/local/share/man/man1/img2svg.1 2>/dev/null || true; \
	fi
	@printf "Purge complete.\n"

# === verify / manpage / check-freebsd ======================================
# Thin wrappers over the scripts in scripts/. Each forwards its args.

verify: ## Verify a specific backend is active (BACKEND=cpu|nvidia|amd|apple)
	@if [ -z "$(BACKEND)" ]; then \
	    printf "No BACKEND given; defaulting to 'cpu'.\n" >&2; \
	    bash scripts/verify_backend.sh cpu; \
	else \
	    bash scripts/verify_backend.sh $(BACKEND); \
	fi

manpage: ## Install the img2svg(1) man page (system-wide, requires sudo)
	@bash scripts/install_manpage.sh

check-freebsd: ## Run the FreeBSD smoke test (no-op on non-FreeBSD)
	@bash scripts/check_freebsd.sh

# === test / lint / format / build / docs / clean ===========================

test: ## Run the test suite (fast tests only)
	@if [ -n "$(UV)" ]; then \
	    $(UV) run pytest -m "not slow"; \
	else \
	    $(PYTHON) -m pytest -m "not slow"; \
	fi

lint: ## Run ruff check + mypy on src/ and tests/
	@if [ -n "$(UV)" ]; then \
	    $(UV) run ruff check src tests && $(UV) run mypy src; \
	else \
	    $(PYTHON) -m ruff check src tests && $(PYTHON) -m mypy src; \
	fi

format: ## Auto-format src/ and tests/ with ruff
	@if [ -n "$(UV)" ]; then \
	    $(UV) run ruff format src tests && $(UV) run ruff check --fix src tests; \
	else \
	    $(PYTHON) -m ruff format src tests && $(PYTHON) -m ruff check --fix src tests; \
	fi

build: ## Build sdist + wheel into dist/
	@if [ -n "$(UV)" ]; then \
	    $(UV) build; \
	else \
	    $(PYTHON) -m build; \
	fi

docs: ## Build the mkdocs site into site/
	@if [ -n "$(UV)" ]; then \
	    $(UV) run mkdocs build --strict; \
	else \
	    $(PYTHON) -m mkdocs build --strict; \
	fi

clean: ## Remove build artifacts and caches
	@rm -rf build dist *.egg-info htmlcov site
	@rm -rf .pytest_cache .ruff_cache .mypy_cache
	@rm -rf .coverage coverage.xml
	@find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	@printf "Cleaned.\n"

# === self-test =============================================================
# Sanity check the Makefile itself: variables resolve, key targets parse,
# and the make binary is recognized. Exits 0 on success, 1 on failure.

self-test: ## Verify Makefile syntax + key variables
	@printf "Running Makefile self-test...\n"
	@FAIL=0; \
	if [ -z "$(UNAME_S)" ]; then \
	    printf "FAIL: UNAME_S is empty\n" >&2; \
	    FAIL=1; \
	else \
	    printf "OK:   UNAME_S = %s\n" "$(UNAME_S)"; \
	fi; \
	if [ -z "$(UNAME_M)" ]; then \
	    printf "FAIL: UNAME_M is empty\n" >&2; \
	    FAIL=1; \
	else \
	    printf "OK:   UNAME_M = %s\n" "$(UNAME_M)"; \
	fi; \
	case "$(UNAME_S)" in \
	    Linux|Darwin|FreeBSD) printf "OK:   UNAME_S is a recognized platform\n" ;; \
	    *) printf "WARN: UNAME_S = %s is not Linux/Darwin/FreeBSD (best-effort only)\n" "$(UNAME_S)" ;; \
	esac; \
	if [ -n "$(MAKEFILE_LIST)" ]; then \
	    printf "OK:   MAKEFILE_LIST = %s\n" "$(MAKEFILE_LIST)"; \
	else \
	    printf "FAIL: MAKEFILE_LIST is empty\n" >&2; \
	    FAIL=1; \
	fi; \
	if make -n install >/dev/null 2>&1; then \
	    printf "OK:   make -n install parses cleanly\n"; \
	else \
	    printf "FAIL: make -n install failed to parse\n" >&2; \
	    FAIL=1; \
	fi; \
	if make -n help >/dev/null 2>&1; then \
	    printf "OK:   make -n help parses cleanly\n"; \
	else \
	    printf "FAIL: make -n help failed to parse\n" >&2; \
	    FAIL=1; \
	fi; \
	printf "OK:   make binary = %s\n" "$$(command -v make 2>/dev/null) ($$(make --version 2>/dev/null | head -1))"; \
	if [ $$FAIL -eq 0 ]; then \
	    printf "Self-test PASSED.\n"; \
	else \
	    printf "Self-test FAILED.\n" >&2; \
	    exit 1; \
	fi
