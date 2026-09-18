#!/usr/bin/env python
"""Record the compute environment (run before any GPU job; stored with each run record).

    python scripts/env_report.py [--out env_report.json]

Reports OS, Python, CPU count, and -- only if available -- torch/CUDA/cuDNN versions, GPU
name(s) and VRAM, the ultralytics version, and nvidia-smi driver info. Never installs anything;
missing packages are reported as null.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _nvidia_smi() -> list | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return [dict(zip(("name", "memory_total", "driver_version"), (x.strip() for x in line.split(","))))
            for line in r.stdout.strip().splitlines()]


def collect() -> dict:
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "torch": None, "cuda_available": None, "cuda_version": None, "cudnn_version": None,
        "gpus": [], "ultralytics": None, "nvidia_smi": _nvidia_smi(),
    }
    try:
        import torch
    except ImportError:
        pass
    else:
        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_version"] = torch.version.cuda
        report["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                report["gpus"].append({"index": i, "name": p.name, "vram_gb": round(p.total_memory / 1024**3, 2)})
    try:
        import ultralytics
    except ImportError:
        pass
    else:
        report["ultralytics"] = ultralytics.__version__
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = collect()
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
