import argparse
from pathlib import Path

from papers2code.graph import run_pipeline


def main():
    ap = argparse.ArgumentParser(description="Paper → Kaggle → Code scaffold → Wiki")
    ap.add_argument("--paper", required=True, help="Path to PDF or URL")
    ap.add_argument("--out", default="artifacts", help="Output directory")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    run_pipeline(paper_source=args.paper, out_dir=Path(args.out))

if __name__ == "__main__":
    main()
