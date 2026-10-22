#!/usr/bin/env python3
"""Compare reduction results with the expected products for a CI case."""

import argparse
import math
import os
import sys
from pathlib import Path

import MySQLdb
from astropy.io import fits


def fail(message):
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def connect_database():
    return MySQLdb.connect(
        host=os.environ.get("LCOSNDBHOST", "db"),
        user=os.environ.get("LCOSNDBUSER", "supernova"),
        passwd=os.environ.get("LCOSNDBPASS", "supernova"),
        db=os.environ.get("LCOSNDBNAME", "supernova"),
    )


def finite_magnitude(value):
    return (
        isinstance(value, (int, float))
        and math.isfinite(value)
        and value != 9999
    )


def image_path(row):
    image = Path(row["filepath"] or "") / (row["filename"] or "")
    if not image.is_absolute():
        image = Path(os.environ.get("LCOSNDIR", "/supernova")) / image
    return image


def check_fits(path, require_object=False):
    if not path.is_file():
        fail(f"FITS product missing: {path}")
    try:
        with fits.open(path, memmap=True) as hdul:
            hdul.verify("exception")
            if hdul[0].data is None:
                fail(f"FITS data missing: {path.name}")
            if require_object and not hdul[0].header.get("OBJECT"):
                fail(f"OBJECT header missing: {path.name}")
    except Exception as exc:
        fail(f"cannot open FITS {path.name}: {exc}")


def compare_basic(cursor, args):
    cursor.execute(
        "SELECT filename, filepath, psf, psfmag, zcat, mag FROM photlco "
        "WHERE objname=%s AND dayobs=%s AND filetype=1",
        (args.target, args.epoch),
    )
    rows = cursor.fetchall()
    if len(rows) != args.expected_images:
        fail(f"expected {args.expected_images} photlco rows; found {len(rows)}")

    for column, expected in (
        ("psf", args.expected_images),
        ("zcat", args.expected_calibrated_images),
    ):
        completed = [row for row in rows if row[column] not in (None, "", "X")]
        if len(completed) != expected:
            fail(f"{column} completed for {len(completed)}/{expected} images")

    for column, expected in (
        ("psfmag", args.expected_images),
        ("mag", args.expected_calibrated_images),
    ):
        completed = [row for row in rows if finite_magnitude(row[column])]
        if len(completed) != expected:
            fail(f"{column} is finite for {len(completed)}/{expected} images")

    work_dir = Path(os.environ.get("LCOSNDIR", "/supernova"))
    for row in rows:
        image = image_path(row)
        check_fits(image, require_object=True)
        if not row["psf"]:
            fail(f"PSF filename missing for {image.name}")
        if not (image.parent / row["psf"]).is_file():
            fail(f"PSF product missing for {image.name}")

    catalog = work_dir / "standard" / "cat" / "apass" / f"{args.target}_apass.cat"
    if not catalog.is_file():
        fail(f"APASS catalog missing: {catalog}")
    print(f"PASS: {args.target} basic reduction products validated")


def compare_difference(cursor, args):
    cursor.execute(
        "SELECT filename, filepath, psf, psfmag, zcat, mag, filetype, difftype "
        "FROM photlco WHERE objname IN (%s, %s)",
        (args.target, "2018cow"),
    )
    rows = cursor.fetchall()
    templates = [
        row for row in rows
        if row["filetype"] == 4
        and args.template_epoch in (row["filename"] or "")
        and any(
            f"-{image_id:04d}-" in (row["filename"] or "")
            for image_id in (87, 89, 91)
        )
    ]
    if len(templates) != args.expected_templates:
        fail(f"expected {args.expected_templates} templates; found {len(templates)}")
    if any(row["psf"] in (None, "", "X") for row in templates):
        fail("one or more template PSFs are missing")

    differences = [
        row for row in rows
        if row["filetype"] == 3
        and row["difftype"] == 0
        and args.science_epoch in (row["filename"] or "")
    ]
    if len(differences) != args.expected_science:
        fail(f"expected {args.expected_science} HOTPANTS images; found {len(differences)}")

    for row in differences:
        for column in ("psfmag", "mag"):
            if not finite_magnitude(row[column]):
                fail(f"{column} missing for {row['filename']}")
        if row["zcat"] in (None, "", "X"):
            fail(f"zeropoint missing for {row['filename']}")
        check_fits(image_path(row))

    print(f"PASS: {len(templates)} templates and {len(differences)} HOTPANTS images validated")


def compare_live(cursor, args):
    cursor.execute(
        "SELECT psf, psfmag, zcat, mag, wcs FROM photlco "
        "WHERE objname=%s AND telescope LIKE %s AND dateobs >= %s AND dateobs < %s "
        "AND filetype=1",
        (args.target, f"%{args.telescope}%", args.start, args.end),
    )
    rows = cursor.fetchall()
    if not rows:
        fail(f"no science images found for {args.target} at {args.telescope}")

    checks = {
        "PSF": [row for row in rows if row["psf"] not in (None, "", "X")],
        "PSF photometry": [row for row in rows if finite_magnitude(row["psfmag"])],
        "zeropoint": [row for row in rows if row["zcat"] not in (None, "", "X")],
        "magnitude": [row for row in rows if finite_magnitude(row["mag"])],
    }
    for name, completed in checks.items():
        if not completed:
            fail(f"{name} stage produced no successful images")

    if args.check_wcs:
        solved = [row for row in rows if row["wcs"] is not None and row["wcs"] < 1]
        if not solved:
            fail("WCS stage produced no solved images")
    print(f"PASS: {len(rows)} science images found; requested stages produced results")


def compare_uband(cursor, args):
    work_dir = Path(os.environ.get("LCOSNDIR", "/supernova"))
    catalog_dir = work_dir / "standard" / "cat" / "landolt"
    for catalog in (args.standard_catalog, args.local_catalog):
        path = catalog_dir / catalog
        if not path.is_file():
            fail(f"Landolt catalog missing: {path}")

    cursor.execute(
        "SELECT classificationid, landolt_cat FROM targets WHERE id=%s",
        (args.standard_id,),
    )
    standard = cursor.fetchone()
    if not standard:
        fail(f"standard target {args.standard_id} is missing")
    if standard["classificationid"] != 1:
        fail(f"target {args.standard_id} is not classified as a standard")
    if standard["landolt_cat"] != args.local_catalog:
        fail(f"target {args.standard_id} does not use {args.local_catalog}")

    cursor.execute(
        "SELECT filename, filepath, psf, psfmag, zcat, mag FROM photlco "
        "WHERE objname=%s AND dayobs=%s AND filetype=1 AND filter='U'",
        (args.target, args.epoch),
    )
    rows = cursor.fetchall()
    if not rows:
        fail(f"no U-band science images found for {args.target} on {args.epoch}")
    for column in ("psf", "zcat"):
        completed = [row for row in rows if row[column] not in (None, "", "X")]
        if not completed:
            fail(f"U-band {column} stage produced no results")
    for column in ("psfmag", "mag"):
        completed = [row for row in rows if finite_magnitude(row[column])]
        if not completed:
            fail(f"U-band {column} stage produced no finite magnitudes")
    print(f"PASS: U-band Landolt calibration validated across {len(rows)} images")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)

    basic = modes.add_parser("basic", help="validate standard photometry")
    basic.add_argument("--target", required=True)
    basic.add_argument("--epoch", required=True)
    basic.add_argument("--expected-images", required=True, type=int)
    basic.add_argument("--expected-calibrated-images", required=True, type=int)

    difference = modes.add_parser("difference", help="validate HOTPANTS subtraction")
    difference.add_argument("--target", required=True)
    difference.add_argument("--science-epoch", required=True)
    difference.add_argument("--template-epoch", required=True)
    difference.add_argument("--expected-science", required=True, type=int)
    difference.add_argument("--expected-templates", required=True, type=int)

    live = modes.add_parser("live", help="validate live archive photometry")
    live.add_argument("--target", required=True)
    live.add_argument("--telescope", required=True)
    live.add_argument("--start", required=True)
    live.add_argument("--end", required=True)
    live.add_argument("--check-wcs", action="store_true")

    uband = modes.add_parser("uband", help="validate U-band Landolt calibration")
    uband.add_argument("--target", required=True)
    uband.add_argument("--epoch", required=True)
    uband.add_argument("--standard-id", required=True, type=int)
    uband.add_argument("--standard-catalog", required=True)
    uband.add_argument("--local-catalog", required=True)

    args = parser.parse_args()
    connection = connect_database()
    try:
        cursor = connection.cursor(MySQLdb.cursors.DictCursor)
        if args.mode == "basic":
            compare_basic(cursor, args)
        elif args.mode == "difference":
            compare_difference(cursor, args)
        elif args.mode == "live":
            compare_live(cursor, args)
        else:
            compare_uband(cursor, args)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
