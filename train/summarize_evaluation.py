"""Convert a full Co-DINO evaluation JSON to the compact format."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.evaluation_summary import compact_evaluation_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path,
                        help="Default: <input_stem>_simple.json next to input")
    args = parser.parse_args(argv)
    detailed = json.loads(args.input.read_text(encoding="utf-8"))
    simple = compact_evaluation_report(detailed)
    output = args.output or args.input.with_name(args.input.stem + "_simple.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(simple, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output.resolve())
    return simple


if __name__ == "__main__":
    main()
