#!/usr/bin/env python3
"""Compare every image and column in the master and branch photlco dumps."""

import argparse
import json
import sys

# These record when the pipeline ran, so they differ on every run
RUN_TIME_COLUMNS = {"datecreated", "lastmodified", "lastunpacked"}
MISSING = "<missing>"


def load_images(path):
    with open(path) as dump:
        return {row["filename"]: row for row in json.load(dump)}


def is_same(master, branch):
    both_nan = master != master and branch != branch
    return master == branch or both_nan


def compare_image(filename, master, branch):
    differences = []
    for column in sorted((master.keys() | branch.keys()) - RUN_TIME_COLUMNS):
        master_value = master.get(column, MISSING)
        branch_value = branch.get(column, MISSING)
        if not is_same(master_value, branch_value):
            differences.append(f"{filename} {column}: master={master_value!r}, branch={branch_value!r}")
    return differences


def compare_dumps(master, branch):
    differences = [f"{filename}: missing from branch" for filename in sorted(master.keys() - branch.keys())]
    differences += [f"{filename}: only in branch" for filename in sorted(branch.keys() - master.keys())]
    for filename in sorted(master.keys() & branch.keys()):
        differences += compare_image(filename, master[filename], branch[filename])
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="master dump JSON")
    parser.add_argument("--candidate", required=True, help="branch dump JSON")
    args = parser.parse_args()

    master = load_images(args.reference)
    branch = load_images(args.candidate)
    if not master and not branch:
        sys.exit("FAIL: both dumps are empty")

    differences = compare_dumps(master, branch)
    if differences:
        print("\n".join(f"FAIL: {difference}" for difference in differences), file=sys.stderr)
        sys.exit(f"FAIL: {len(differences)} differences found")
    print(f"PASS: all {len(master)} images are identical")


if __name__ == "__main__":
    main()
