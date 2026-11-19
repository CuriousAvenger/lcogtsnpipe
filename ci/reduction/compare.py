#!/usr/bin/env python3
"""Compare photlco dumps from master and branch reductions."""

import argparse
import json
import math
import sys

METRICS = ("mag", "psfmag", "zn", "fwhm", "wcs")


def measured(value):
    """Return value as a float, or None if the pipeline did not measure it."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number != 9999 else None


def load_images(path):
    with open(path) as dump:
        return {row["filename"]: row for row in json.load(dump) if row["filename"]}


def metric_problems(metric, reference, candidate, tolerance):
    problems = []
    measured_images = 0
    for filename, row in sorted(reference.items()):
        master, branch = measured(row[metric]), measured(candidate[filename][metric])
        if master is None and branch is None:
            continue
        measured_images += 1
        if master is None or branch is None or abs(branch - master) > tolerance:
            problems.append(f"{metric} {filename}: master={master}, branch={branch}")
    if not measured_images:
        problems.append(f"{metric}: no image has a measured value")
    return problems


def find_problems(reference, candidate, tolerance):
    if not reference:
        return ["master dump has no images"]
    if reference.keys() != candidate.keys():
        missing = sorted(reference.keys() - candidate.keys())
        extra = sorted(candidate.keys() - reference.keys())
        return [f"images differ: missing from branch={missing}, only in branch={extra}"]
    return [
        problem
        for metric in METRICS
        for problem in metric_problems(metric, reference, candidate, tolerance)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="master dump JSON")
    parser.add_argument("--candidate", required=True, help="branch dump JSON")
    parser.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args()

    reference = load_images(args.reference)
    candidate = load_images(args.candidate)
    problems = find_problems(reference, candidate, args.tolerance)
    if problems:
        print("\n".join(f"FAIL: {problem}" for problem in problems), file=sys.stderr)
        sys.exit(1)
    print(f"PASS: {len(reference)} images match master within {args.tolerance}")


if __name__ == "__main__":
    main()
