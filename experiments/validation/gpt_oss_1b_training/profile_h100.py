from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_PATH = RESULTS_DIR / "h100_profile.yaml"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "simulon",
        "profile",
        "gpu",
        "--name",
        "gpt-oss-1b-moe-h100",
        "--hidden-size",
        "1536",
        "--num-heads",
        "24",
        "--ffn-hidden-size",
        "6144",
        "--seq-len",
        "8192",
        "--batch-size",
        "1",
        "--vocab-size",
        "32000",
        "--dtype",
        "bf16",
        "--tp",
        "1",
        "--epoch-num",
        "20",
        "--output",
        str(OUTPUT_PATH),
    ]

    _ = subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)
    print(f"Profile complete: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
