#!/usr/bin/env python3
# LexAgent v3.0 | run_local.py
"""One-command local setup and run wrapper around master_run.py.

Works on Linux, macOS and Windows. Needs only a system Python 3.10+ to start;
everything else is installed into a local .venv next to this file.

    python run_local.py setup            # create .venv and install requirements
    python run_local.py build            # download CAP volumes and build the database
    python run_local.py smoke            # Phase 1 -> 2 -> 3 smoke tests
    python run_local.py eval --limit 5   # stress test, benchmark, baseline, ablation
    python run_local.py all --limit 5    # setup -> build -> smoke -> eval

Add --dry-run to print the commands without running them, and --mock to use the
canned mock LLM instead of Mistral-7B (no GPU or model download needed).

The embedding batch size is chosen from the detected GPU memory unless
LEXAGENT_BATCH_SIZE is already set. LEXAGENT_PRECISION is left alone, so
master_run.py picks 4-bit quantization below 23 GB VRAM and bfloat16 above it.
"""

import argparse
import os
import shutil
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, ".venv")
IS_WINDOWS = os.name == "nt"
VENV_PYTHON = os.path.join(VENV_DIR, "Scripts" if IS_WINDOWS else "bin", "python.exe" if IS_WINDOWS else "python")

# On Windows, PyPI ships a CPU-only torch wheel, so a CUDA build is installed from this index instead.
TORCH_CUDA_INDEX = os.environ.get("LEXAGENT_TORCH_INDEX", "https://download.pytorch.org/whl/cu126")

DRY_RUN = False


def run(cmd, env=None, check=True):
    print("\n$ " + " ".join(cmd), flush=True)
    if DRY_RUN:
        return 0
    result = subprocess.run(cmd, cwd=BASE_DIR, env=env)
    if check and result.returncode != 0:
        sys.exit(f"\n[run_local] Command failed with exit code {result.returncode}: {' '.join(cmd)}")
    return result.returncode


def detect_vram_gb():
    """Total memory of GPU 0 in GB via nvidia-smi, or 0.0 if no Nvidia GPU is visible."""
    if not shutil.which("nvidia-smi"):
        return 0.0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip().splitlines()
        return float(out[0]) / 1024.0 if out else 0.0
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def pick_batch_size(vram_gb):
    if vram_gb >= 23:
        return 256
    if vram_gb >= 15:
        return 128
    if vram_gb >= 10:
        return 64
    if vram_gb > 0:
        return 32
    return 16


def build_env(mock):
    env = os.environ.copy()
    vram_gb = detect_vram_gb()
    env.setdefault("LEXAGENT_BATCH_SIZE", str(pick_batch_size(vram_gb)))
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")  # master_run.py prints box-drawing characters
    if mock:
        env["LEXAGENT_MOCK_LLM"] = "1"
    if vram_gb:
        print(f"[run_local] GPU 0 memory: {vram_gb:.1f} GB | LEXAGENT_BATCH_SIZE={env['LEXAGENT_BATCH_SIZE']}")
    else:
        print("[run_local] No Nvidia GPU detected. Mistral-7B needs CUDA; pass --mock to run with the mock LLM.")
    return env


def require_venv():
    if not DRY_RUN and not os.path.exists(VENV_PYTHON):
        sys.exit("[run_local] .venv not found. Run: python run_local.py setup")


def master(args, env):
    require_venv()
    run([VENV_PYTHON, "master_run.py"] + args, env=env)


def step_setup(env):
    if sys.version_info < (3, 10):
        sys.exit(f"[run_local] Python 3.10+ is required, found {sys.version.split()[0]}")
    if not os.path.exists(VENV_PYTHON):
        run([sys.executable, "-m", "venv", VENV_DIR])
    run([VENV_PYTHON, "-m", "pip", "install", "--upgrade", "pip"])
    if IS_WINDOWS and detect_vram_gb() > 0:
        run([VENV_PYTHON, "-m", "pip", "install", "torch", "--index-url", TORCH_CUDA_INDEX])
    run([VENV_PYTHON, "-m", "pip", "install", "-r", "requirements.txt"])
    run([VENV_PYTHON, "-c", (
        "import torch; ok = torch.cuda.is_available(); "
        "print('[run_local] torch', torch.__version__, '| CUDA available:', ok, "
        "'|', torch.cuda.get_device_name(0) if ok else 'no GPU visible to torch')"
    )], env=env)


def step_build(args, env):
    cmd = ["--build-db", "--volumes", args.volumes, "--batch-size", env["LEXAGENT_BATCH_SIZE"]]
    if args.overwrite:
        cmd.append("--overwrite")
    master(cmd, env)


def step_smoke(env):
    master(["--test-phases"], env)


def step_eval(args, env):
    limit = str(args.limit)
    master(["--stress-test", "--stress-limit", limit], env)
    master(["--benchmark", "--benchmark-limit", limit], env)
    master(["--baseline", "--benchmark-limit", limit], env)
    master(["--ablation", "--ablation-limit", limit], env)


def main():
    global DRY_RUN
    parser = argparse.ArgumentParser(
        description="Local setup and run wrapper for LexAgent v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("stage", choices=["setup", "build", "smoke", "eval", "all"])
    parser.add_argument("--volumes", default="landmark", help="CAP volume preset or range (default: landmark)")
    parser.add_argument("--overwrite", action="store_true", help="Wipe and rebuild the database from scratch")
    parser.add_argument("--limit", type=int, default=5, help="Cases per evaluation run (default: 5, full suite: 100)")
    parser.add_argument("--mock", action="store_true", help="Use the mock LLM instead of Mistral-7B")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them")
    args = parser.parse_args()
    DRY_RUN = args.dry_run

    env = build_env(args.mock)

    if args.stage in ("setup", "all"):
        step_setup(env)
    if args.stage in ("build", "all"):
        step_build(args, env)
    if args.stage in ("smoke", "all"):
        step_smoke(env)
    if args.stage in ("eval", "all"):
        step_eval(args, env)

    print(f"\n[run_local] Stage '{args.stage}' finished.")


if __name__ == "__main__":
    main()
