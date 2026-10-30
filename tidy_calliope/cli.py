"""Command line: `tidy-calliope RESULTS... [-c config.yaml] [-o output_dir]`."""

import argparse
import json
import logging
import sys
from pathlib import Path

from tidy_calliope.config import DEFAULT_CONFIG


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="tidy-calliope",
        description="Post-process Calliope 0.7 result files (.nc) into tidy tables, one folder per scenario. "
        "Clustered (typical-day) runs are rebuilt to the full year.",
    )
    parser.add_argument("results", nargs="*", help=".nc files, folders with .nc files, or glob patterns")
    parser.add_argument("-c", "--config", help="YAML file with the settings to change (see --show-config)")
    parser.add_argument("-o", "--output-dir", default="tidy-results", help="output folder (default: %(default)s)")
    parser.add_argument("--no-checks", action="store_true", help="skip the consistency checks")
    parser.add_argument("--strict", action="store_true", help="exit with an error if a consistency check fails")
    parser.add_argument("--show-config", action="store_true", help="print the default configuration and exit")
    parser.add_argument("-q", "--quiet", action="store_true", help="only warnings and errors")
    args = parser.parse_args(argv)

    if args.show_config:
        print(Path(DEFAULT_CONFIG).read_text())
        return 0
    if not args.results:
        parser.error("give at least one result file or folder")
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    from tidy_calliope.core import process  # after logging setup

    summaries = process(args.results, args.output_dir, args.config, checks=not args.no_checks)
    failed = {s["scenario"]: s.get("checks_failed", []) for s in summaries if s.get("checks_failed")}
    for s in summaries:
        print(f"{s['scenario']}: {len(s['tables'])} tables in {s['runtime_seconds']} s -> {args.output_dir}/{s['scenario']}")
    if failed:
        print("Failed checks: " + json.dumps(failed), file=sys.stderr)
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
