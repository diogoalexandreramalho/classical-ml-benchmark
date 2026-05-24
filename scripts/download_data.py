"""Download raw datasets from UCI ML Repository into data/raw/.

Idempotent — skips files that already exist. Run with: `python scripts/download_data.py`.

Requires macOS or a Linux system with bsdtar (`apt install libarchive-tools`) for
unpacking Parkinson's, which UCI distributes as a .rar inside a .zip.

While UCI's TLS certificate is expired (May 23 2026), run with `--insecure`.
"""

import argparse
import sys

from data_science.datasets import DATASETS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Skip TLS verification (workaround for expired UCI cert).",
    )
    args = parser.parse_args()

    for ds in DATASETS.values():
        if ds.raw_path.exists():
            print(f"skip {ds.filename} (already exists)")
            continue
        print(f"fetching {ds.filename}")
        ds.download(insecure=args.insecure)
        print(f"  wrote {ds.raw_path} ({ds.raw_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
