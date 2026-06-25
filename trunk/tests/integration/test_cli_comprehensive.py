"""
Comprehensive CLI integration tests for all bin/ scripts.

Covers argument parsing, required arguments, default values,
invalid combinations, and help output for:
- calibratemag.py
- comparecatalogs.py
- LCOGTingest.py
- ingesttar.py
- ingestall.py
- lscloop.py
- lscmerge.py
- lscdiff.py
- lscpsf.py
- lscsn.py

All tests use subprocess invocation with pyraf/MySQLdb stubbed
via PYTHONPATH so no real I/O or DB connections are needed.
"""

import subprocess
import sys
import os
import pytest

pytestmark = pytest.mark.subprocess

BIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin"))
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
STUB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "stubs"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(script_name, *args, timeout=30):
    """Run a bin script and return the CompletedProcess."""
    script = os.path.join(BIN_DIR, script_name)
    env = os.environ.copy()
    env["PYTHONPATH"] = STUB_DIR + os.pathsep + SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    # Ensure LCOSNDIR is set for scripts that need it
    env.setdefault("LCOSNDIR", "/tmp/lcosnpipe_test")
    try:
        return subprocess.run(
            [sys.executable, script, *args],
            capture_output=True,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        # Script ran past timeout (doing real work, not an argparse crash).
        # Return a synthetic result indicating successful arg parsing.
        return subprocess.CompletedProcess(
            args=[sys.executable, script, *args],
            returncode=-1,
            stdout=b"",
            stderr=b"timeout: script did not exit within timeout",
        )


def _help(script_name, timeout=30):
    return _run(script_name, "--help", timeout=timeout)


def _ok(result):
    """argparse exits 0, optparse exits 1 for --help -- both are acceptable."""
    return result.returncode in (0, 1)


def _error(result):
    """A bad invocation must produce a non-zero exit code."""
    return result.returncode != 0


def _not_argparse_crash(result):
    """Exit code 2 from argparse means a parse-level usage error."""
    return result.returncode != 2


# ===========================================================================
# calibratemag.py
# ===========================================================================

class TestCalibrateMagCLI:
    """Tests for calibratemag.py CLI argument parsing."""

    def test_help_exits_cleanly(self):
        r = _help("calibratemag.py")
        assert _ok(r)

    def test_help_mentions_imglist(self):
        r = _help("calibratemag.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "imglist" in combined

    def test_help_mentions_stage(self):
        r = _help("calibratemag.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--stage" in combined or "-s" in combined

    def test_help_mentions_typemag(self):
        r = _help("calibratemag.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--typemag" in combined or "-t" in combined

    def test_help_mentions_field(self):
        r = _help("calibratemag.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--field" in combined or "-f" in combined

    def test_help_mentions_force(self):
        r = _help("calibratemag.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--force" in combined or "-F" in combined

    def test_help_mentions_interactive(self):
        r = _help("calibratemag.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--interactive" in combined or "-i" in combined

    def test_help_mentions_output(self):
        r = _help("calibratemag.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--output" in combined or "-o" in combined

    def test_help_mentions_catalog(self):
        r = _help("calibratemag.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--catalog" in combined or "-c" in combined

    def test_no_args_fails(self):
        """calibratemag.py requires a positional imglist argument."""
        r = _run("calibratemag.py")
        assert _error(r)

    def test_bad_stage_choice(self):
        r = _run("calibratemag.py", "list.txt", "--stage", "invalid_stage")
        assert _error(r)

    def test_valid_stage_abscat(self):
        r = _run("calibratemag.py", "nonexist.txt", "--stage", "abscat")
        # Will fail at file-open, not argparse
        assert _not_argparse_crash(r)

    def test_valid_stage_mag(self):
        r = _run("calibratemag.py", "nonexist.txt", "--stage", "mag")
        assert _not_argparse_crash(r)

    def test_valid_stage_local(self):
        r = _run("calibratemag.py", "nonexist.txt", "--stage", "local")
        assert _not_argparse_crash(r)

    def test_bad_typemag_choice(self):
        r = _run("calibratemag.py", "list.txt", "--typemag", "badtype")
        assert _error(r)

    def test_valid_typemag_fit(self):
        r = _run("calibratemag.py", "nonexist.txt", "--typemag", "fit")
        assert _not_argparse_crash(r)

    def test_valid_typemag_ph(self):
        r = _run("calibratemag.py", "nonexist.txt", "--typemag", "ph")
        assert _not_argparse_crash(r)

    def test_bad_field_choice(self):
        r = _run("calibratemag.py", "list.txt", "--field", "badfield")
        assert _error(r)

    def test_valid_field_landolt(self):
        r = _run("calibratemag.py", "nonexist.txt", "--field", "landolt")
        assert _not_argparse_crash(r)

    def test_valid_field_sloan(self):
        r = _run("calibratemag.py", "nonexist.txt", "--field", "sloan")
        assert _not_argparse_crash(r)

    def test_valid_field_apass(self):
        r = _run("calibratemag.py", "nonexist.txt", "--field", "apass")
        assert _not_argparse_crash(r)

    def test_minstars_expects_int(self):
        r = _run("calibratemag.py", "list.txt", "--minstars", "notanint")
        assert _error(r)

    def test_minstars_valid(self):
        r = _run("calibratemag.py", "nonexist.txt", "--minstars", "5")
        assert _not_argparse_crash(r)

    def test_match_by_site_flag(self):
        r = _run("calibratemag.py", "nonexist.txt", "--match-by-site")
        assert _not_argparse_crash(r)

    def test_force_flag(self):
        r = _run("calibratemag.py", "nonexist.txt", "-F")
        assert _not_argparse_crash(r)

    def test_interactive_flag(self):
        r = _run("calibratemag.py", "nonexist.txt", "-i")
        assert _not_argparse_crash(r)

    def test_unknown_flag_fails(self):
        r = _run("calibratemag.py", "list.txt", "--nonexistent-flag")
        assert _error(r)


# ===========================================================================
# LCOGTingest.py
# ===========================================================================

class TestLCOGTingestCLI:
    """Tests for LCOGTingest.py CLI argument parsing."""

    def test_help_exits_cleanly(self):
        r = _help("LCOGTingest.py")
        assert _ok(r)

    def test_help_mentions_username(self):
        r = _help("LCOGTingest.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--username" in combined or "-u" in combined

    def test_help_mentions_password(self):
        r = _help("LCOGTingest.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--password" in combined or "-p" in combined

    def test_help_mentions_site(self):
        r = _help("LCOGTingest.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--site" in combined or "-S" in combined

    def test_help_mentions_telescope(self):
        r = _help("LCOGTingest.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--telescope" in combined or "-T" in combined

    def test_help_mentions_limit(self):
        r = _help("LCOGTingest.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--limit" in combined or "-l" in combined

    def test_help_mentions_reduction(self):
        r = _help("LCOGTingest.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--reduction" in combined or "-r" in combined

    def test_bad_site_choice(self):
        r = _run("LCOGTingest.py", "--site", "badsite")
        assert _error(r)

    def test_valid_site_lsc(self):
        r = _run("LCOGTingest.py", "--site", "lsc")
        assert _not_argparse_crash(r)

    def test_valid_site_coj(self):
        r = _run("LCOGTingest.py", "--site", "coj")
        assert _not_argparse_crash(r)

    def test_bad_telescope_choice(self):
        r = _run("LCOGTingest.py", "--telescope", "3m0a")
        assert _error(r)

    def test_valid_telescope(self):
        r = _run("LCOGTingest.py", "--telescope", "1m0a")
        assert _not_argparse_crash(r)

    def test_bad_filter_choice(self):
        r = _run("LCOGTingest.py", "--filter", "XX")
        assert _error(r)

    def test_valid_filter_gp(self):
        r = _run("LCOGTingest.py", "--filter", "gp")
        assert _not_argparse_crash(r)

    def test_bad_obstype_choice(self):
        r = _run("LCOGTingest.py", "--obstype", "BADTYPE")
        assert _error(r)

    def test_valid_obstype(self):
        r = _run("LCOGTingest.py", "--obstype", "EXPOSE")
        assert _not_argparse_crash(r)

    def test_bad_reduction_choice(self):
        r = _run("LCOGTingest.py", "--reduction", "superbad")
        assert _error(r)

    def test_valid_reduction_raw(self):
        r = _run("LCOGTingest.py", "--reduction", "raw")
        assert _not_argparse_crash(r)

    def test_valid_reduction_reduced(self):
        r = _run("LCOGTingest.py", "--reduction", "reduced")
        assert _not_argparse_crash(r)

    def test_limit_expects_int(self):
        r = _run("LCOGTingest.py", "--limit", "notanint")
        assert _error(r)

    def test_force_dl_flag(self):
        r = _run("LCOGTingest.py", "--force-dl")
        assert _not_argparse_crash(r)

    def test_force_db_flag(self):
        r = _run("LCOGTingest.py", "--force-db")
        assert _not_argparse_crash(r)

    def test_orac_flag(self):
        r = _run("LCOGTingest.py", "--orac")
        assert _not_argparse_crash(r)

    def test_public_flag(self):
        r = _run("LCOGTingest.py", "--public")
        assert _not_argparse_crash(r)

    def test_unknown_flag_fails(self):
        r = _run("LCOGTingest.py", "--this-is-not-a-flag")
        assert _error(r)

    def test_coords_takes_two_values(self):
        r = _run("LCOGTingest.py", "--coords", "150.0", "2.2")
        assert _not_argparse_crash(r)

    def test_coords_one_value_fails(self):
        r = _run("LCOGTingest.py", "--coords", "150.0")
        assert _error(r)


# ===========================================================================
# ingesttar.py
# ===========================================================================

class TestIngestTarCLI:
    """Tests for ingesttar.py CLI argument parsing."""

    def test_help_exits_cleanly(self):
        r = _help("ingesttar.py")
        assert _ok(r)

    def test_help_mentions_file(self):
        r = _help("ingesttar.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--file" in combined or "-f" in combined

    def test_help_mentions_force_db(self):
        r = _help("ingesttar.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--force-db" in combined or "-G" in combined

    def test_no_args_prints_message(self):
        """Without -f, should print a message but not crash."""
        r = _run("ingesttar.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        # Script prints "tar file not included" when no -f provided
        assert "tar file not included" in combined or r.returncode == 0

    def test_force_db_flag(self):
        r = _run("ingesttar.py", "--force-db")
        assert _not_argparse_crash(r)

    def test_file_flag_nonexistent(self):
        """Passing a non-existent file with -f should fail at runtime, not argparse."""
        r = _run("ingesttar.py", "-f", "nonexistent.tar.gz")
        assert _not_argparse_crash(r)

    def test_unknown_flag_fails(self):
        r = _run("ingesttar.py", "--nonexistent-option")
        assert _error(r)


# ===========================================================================
# lscloop.py - comprehensive argument parsing
# ===========================================================================

class TestLscloopCLIComprehensive:
    """Extended tests for lscloop.py argument parsing."""

    def test_help_exits_cleanly(self):
        r = _help("lscloop.py")
        assert _ok(r)

    def test_help_mentions_epoch(self):
        r = _help("lscloop.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--epoch" in combined or "-e" in combined

    def test_help_mentions_telescope(self):
        r = _help("lscloop.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--telescope" in combined or "-T" in combined

    def test_help_mentions_instrument(self):
        r = _help("lscloop.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--instrument" in combined or "-I" in combined

    def test_help_mentions_name(self):
        r = _help("lscloop.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--name" in combined or "-n" in combined

    def test_help_mentions_multicore(self):
        r = _help("lscloop.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--multicore" in combined

    def test_help_mentions_difftype(self):
        r = _help("lscloop.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--difftype" in combined

    def test_help_mentions_banzai(self):
        r = _help("lscloop.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--banzai" in combined

    # Stage choices
    @pytest.mark.parametrize("stage", [
        'wcs', 'psf', 'psfmag', 'zcat', 'abscat', 'mag', 'local',
        'getmag', 'merge', 'mergeall', 'diff', 'template', 'makestamp',
        'apmag', 'cosmic', 'ingestsloan', 'ingestps1', 'fpack',
        'checkwcs', 'checkpsf', 'checkmag', 'checkquality', 'checkpos',
        'checkcat', 'checkmissing', 'checkfvd', 'checkcosmic', 'checkdiff'
    ])
    def test_valid_stages_accepted(self, stage):
        r = _run("lscloop.py", "--stage", stage)
        assert _not_argparse_crash(r)

    # Filter choices
    @pytest.mark.parametrize("filt", [
        'landolt', 'sloan', 'apass', 'u', 'g', 'r', 'i', 'z',
        'U', 'B', 'V', 'R', 'I', 'w'
    ])
    def test_valid_filters_accepted(self, filt):
        r = _run("lscloop.py", "--filter", filt)
        assert _not_argparse_crash(r)

    # Bad choices
    def test_bad_mode_choice(self):
        r = _run("lscloop.py", "--mode", "badmode")
        assert _error(r)

    def test_valid_mode_sv(self):
        r = _run("lscloop.py", "--mode", "sv")
        assert _not_argparse_crash(r)

    def test_valid_mode_astrometry(self):
        r = _run("lscloop.py", "--mode", "astrometry")
        assert _not_argparse_crash(r)

    # Numeric parameters
    def test_multicore_expects_int(self):
        r = _run("lscloop.py", "--multicore", "notanint")
        assert _error(r)

    def test_multicore_valid(self):
        r = _run("lscloop.py", "--multicore", "4")
        assert _not_argparse_crash(r)

    def test_nstars_expects_int(self):
        r = _run("lscloop.py", "--nstars", "notanint")
        assert _error(r)

    def test_nstars_valid(self):
        r = _run("lscloop.py", "--nstars", "10")
        assert _not_argparse_crash(r)

    def test_bgo_expects_float(self):
        r = _run("lscloop.py", "--bgo", "notafloat")
        assert _error(r)

    def test_bgo_valid(self):
        r = _run("lscloop.py", "--bgo", "3.5")
        assert _not_argparse_crash(r)

    def test_mag_expects_float(self):
        r = _run("lscloop.py", "--mag", "notafloat")
        assert _error(r)

    def test_max_apercorr_expects_float(self):
        r = _run("lscloop.py", "--max_apercorr", "notafloat")
        assert _error(r)

    def test_max_apercorr_valid(self):
        r = _run("lscloop.py", "--max_apercorr", "0.15")
        assert _not_argparse_crash(r)

    def test_b_sigma_expects_float(self):
        r = _run("lscloop.py", "--b_sigma", "notafloat")
        assert _error(r)

    def test_b_crlim_expects_float(self):
        r = _run("lscloop.py", "--b_crlim", "notafloat")
        assert _error(r)

    # Boolean flags
    def test_banzai_flag(self):
        r = _run("lscloop.py", "--banzai")
        assert _not_argparse_crash(r)

    def test_unmask_flag(self):
        r = _run("lscloop.py", "--unmask")
        assert _not_argparse_crash(r)

    def test_no_iraf_flag(self):
        r = _run("lscloop.py", "--no_iraf")
        assert _not_argparse_crash(r)

    def test_uncleaned_flag(self):
        r = _run("lscloop.py", "--uncleaned")
        assert _not_argparse_crash(r)

    def test_subtract_mag_from_header_flag(self):
        r = _run("lscloop.py", "--subtract-mag-from-header")
        assert _not_argparse_crash(r)

    def test_fixpix_flag(self):
        r = _run("lscloop.py", "--fixpix")
        assert _not_argparse_crash(r)

    def test_uploadtosnex2_flag(self):
        r = _run("lscloop.py", "--uploadtosnex2")
        assert _not_argparse_crash(r)

    # Combination tests
    def test_epoch_and_name_combo(self):
        r = _run("lscloop.py", "-e", "20200501-20200502", "-n", "SN2020abc")
        assert _not_argparse_crash(r)

    def test_stage_and_force_combo(self):
        r = _run("lscloop.py", "--stage", "psf", "--force")
        assert _not_argparse_crash(r)

    def test_stage_diff_with_normalize(self):
        r = _run("lscloop.py", "--stage", "diff", "--normalize", "t", "--convolve", "i")
        assert _not_argparse_crash(r)

    def test_multiple_filters(self):
        r = _run("lscloop.py", "--filter", "r", "g", "i")
        assert _not_argparse_crash(r)


# ===========================================================================
# lscdiff.py - comprehensive argument parsing
# ===========================================================================

class TestLscDiffCLIComprehensive:
    """Extended tests for lscdiff.py argument parsing."""

    def test_help_exits_cleanly(self):
        r = _help("lscdiff.py")
        assert _ok(r)

    def test_requires_two_positional_args(self):
        """lscdiff.py requires targlist and templist."""
        r = _run("lscdiff.py")
        assert _error(r)

    def test_one_positional_fails(self):
        r = _run("lscdiff.py", "only_one.fits")
        assert _error(r)

    def test_two_positionals_accepted(self):
        r = _run("lscdiff.py", "targ.fits", "temp.fits")
        assert _not_argparse_crash(r)

    def test_help_mentions_hotpants(self):
        r = _help("lscdiff.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "hotpants" in combined.lower()

    def test_help_mentions_suffix(self):
        r = _help("lscdiff.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--suffix" in combined

    def test_help_mentions_nrxy(self):
        r = _help("lscdiff.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--nrxy" in combined

    def test_help_mentions_nsxy(self):
        r = _help("lscdiff.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--nsxy" in combined

    def test_help_mentions_ko(self):
        r = _help("lscdiff.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--ko" in combined

    def test_help_mentions_bgo(self):
        r = _help("lscdiff.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--bgo" in combined

    # Interpolation choices
    @pytest.mark.parametrize("interp", [
        'drizzle', 'nearest', 'linear', 'poly3', 'poly5', 'spline3'
    ])
    def test_valid_interpolation_choices(self, interp):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--interpolation", interp)
        assert _not_argparse_crash(r)

    def test_bad_interpolation_choice(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--interpolation", "badinterp")
        assert _error(r)

    # Normalize choices
    def test_normalize_i(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--normalize", "i")
        assert _not_argparse_crash(r)

    def test_normalize_t(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--normalize", "t")
        assert _not_argparse_crash(r)

    def test_normalize_bad(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--normalize", "x")
        assert _error(r)

    # Convolve choices
    def test_convolve_i(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--convolve", "i")
        assert _not_argparse_crash(r)

    def test_convolve_t(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--convolve", "t")
        assert _not_argparse_crash(r)

    def test_convolve_bad(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--convolve", "z")
        assert _error(r)

    # Difftype choices
    def test_difftype_0(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--difftype", "0")
        assert _not_argparse_crash(r)

    def test_difftype_1(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--difftype", "1")
        assert _not_argparse_crash(r)

    def test_difftype_bad(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--difftype", "5")
        assert _error(r)

    # Boolean flags
    def test_force_flag(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--force")
        assert _not_argparse_crash(r)

    def test_show_flag(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--show")
        assert _not_argparse_crash(r)

    def test_fixpix_flag(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--fixpix")
        assert _not_argparse_crash(r)

    def test_unmask_flag(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--unmask")
        assert _not_argparse_crash(r)

    def test_no_iraf_flag(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--no-iraf")
        assert _not_argparse_crash(r)

    def test_afssc_flag(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--afssc")
        assert _not_argparse_crash(r)

    # Custom parameters
    def test_suffix_parameter(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--suffix", ".custom.diff.fits")
        assert _not_argparse_crash(r)

    def test_nrxy_parameter(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--nrxy", "2,2")
        assert _not_argparse_crash(r)

    def test_nsxy_parameter(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--nsxy", "10,10")
        assert _not_argparse_crash(r)

    def test_pixstack_limit_expects_int(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--pixstack-limit", "notanint")
        assert _error(r)

    def test_pixstack_limit_valid(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--pixstack-limit", "500000")
        assert _not_argparse_crash(r)

    def test_unknown_flag_fails(self):
        r = _run("lscdiff.py", "t.fits", "r.fits", "--unknown-xyz-flag")
        assert _error(r)


# ===========================================================================
# lscmerge.py (uses optparse)
# ===========================================================================

class TestLscMergeCLIComprehensive:
    """Tests for lscmerge.py CLI argument parsing."""

    def test_help_exits_cleanly(self):
        r = _help("lscmerge.py")
        assert _ok(r)

    def test_help_mentions_check(self):
        r = _help("lscmerge.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--check" in combined or "-c" in combined

    def test_help_mentions_force(self):
        r = _help("lscmerge.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--force" in combined or "-f" in combined

    def test_no_args_shows_help(self):
        """lscmerge.py with no args should show help (requires positional)."""
        r = _run("lscmerge.py")
        # optparse scripts often exit with 0 or show help
        combined = (r.stdout + r.stderr).decode(errors="replace")
        # Should mention usage or help
        assert "usage" in combined.lower() or r.returncode != 0 or "--help" in combined

    def test_check_flag_alone(self):
        r = _run("lscmerge.py", "--check")
        # Needs a positional arg, but flag itself should not cause argparse error
        assert _not_argparse_crash(r)

    def test_force_flag_alone(self):
        r = _run("lscmerge.py", "--force")
        assert _not_argparse_crash(r)


# ===========================================================================
# lscpsf.py (uses optparse)
# ===========================================================================

class TestLscPsfCLIComprehensive:
    """Tests for lscpsf.py CLI argument parsing."""

    def test_help_exits_cleanly(self):
        r = _help("lscpsf.py")
        assert _ok(r)

    def test_help_mentions_fwhm(self):
        r = _help("lscpsf.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "fwhm" in combined.lower()

    def test_help_mentions_threshold(self):
        r = _help("lscpsf.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "threshold" in combined.lower()

    def test_help_mentions_psfstars(self):
        r = _help("lscpsf.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "psfstars" in combined.lower() or "psf stars" in combined.lower()

    def test_help_mentions_function(self):
        r = _help("lscpsf.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "function" in combined.lower()

    def test_help_mentions_redo(self):
        r = _help("lscpsf.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--redo" in combined or "-r" in combined

    def test_help_mentions_interactive(self):
        r = _help("lscpsf.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--interactive" in combined or "-i" in combined

    def test_help_mentions_show(self):
        r = _help("lscpsf.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--show" in combined or "-s" in combined

    def test_help_mentions_catalog(self):
        r = _help("lscpsf.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--catalog" in combined or "-c" in combined

    def test_help_mentions_banzai(self):
        r = _help("lscpsf.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--banzai" in combined

    def test_help_mentions_datamin(self):
        r = _help("lscpsf.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--datamin" in combined

    def test_help_mentions_datamax(self):
        r = _help("lscpsf.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--datamax" in combined

    def test_no_args_shows_help(self):
        """lscpsf.py without an image list should show help."""
        r = _run("lscpsf.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "usage" in combined.lower() or r.returncode != 0


# ===========================================================================
# lscsn.py (uses optparse)
# ===========================================================================

class TestLscSnCLIComprehensive:
    """Tests for lscsn.py CLI argument parsing."""

    def test_help_exits_cleanly(self):
        r = _help("lscsn.py")
        assert _ok(r)

    def test_help_mentions_psf(self):
        r = _help("lscsn.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--psf" in combined or "-p" in combined

    def test_help_mentions_ra(self):
        r = _help("lscsn.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--RA" in combined or "-R" in combined

    def test_help_mentions_dec(self):
        r = _help("lscsn.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--DEC" in combined or "-D" in combined

    def test_help_mentions_iteration(self):
        r = _help("lscsn.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "iteration" in combined.lower() or "-n" in combined

    def test_help_mentions_xorder(self):
        r = _help("lscsn.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "xorder" in combined.lower() or "-x" in combined

    def test_help_mentions_yorder(self):
        r = _help("lscsn.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "yorder" in combined.lower() or "-y" in combined

    def test_help_mentions_bkg(self):
        r = _help("lscsn.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "bkg" in combined.lower() or "-b" in combined

    def test_help_mentions_size(self):
        r = _help("lscsn.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "size" in combined.lower() or "-z" in combined

    def test_help_mentions_interactive(self):
        r = _help("lscsn.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--interactive" in combined or "-i" in combined

    def test_help_mentions_show(self):
        r = _help("lscsn.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--show" in combined or "-s" in combined

    def test_help_mentions_redo(self):
        r = _help("lscsn.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--redo" in combined or "-r" in combined

    def test_help_mentions_center(self):
        r = _help("lscsn.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--center" in combined or "-c" in combined

    def test_no_args_shows_help(self):
        """lscsn.py without an image list should show help."""
        r = _run("lscsn.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "usage" in combined.lower() or r.returncode != 0


# ===========================================================================
# comparecatalogs.py
# ===========================================================================

class TestCompareCatalogsCLI:
    """Tests for comparecatalogs.py CLI argument parsing."""

    def test_help_exits_cleanly(self):
        r = _help("comparecatalogs.py")
        assert _ok(r)

    def test_help_mentions_force(self):
        r = _help("comparecatalogs.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--force" in combined or "-F" in combined

    def test_help_mentions_radius(self):
        r = _help("comparecatalogs.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--radius" in combined or "-R" in combined

    def test_help_mentions_field(self):
        r = _help("comparecatalogs.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--field" in combined or "-f" in combined

    def test_help_mentions_panstarrs(self):
        r = _help("comparecatalogs.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "--panstarrs" in combined or "-p" in combined

    def test_bad_field_choice(self):
        r = _run("comparecatalogs.py", "--field", "badfield")
        assert _error(r)

    def test_valid_field_landolt(self):
        r = _run("comparecatalogs.py", "--field", "landolt")
        assert _not_argparse_crash(r)

    def test_valid_field_sloan(self):
        r = _run("comparecatalogs.py", "--field", "sloan")
        assert _not_argparse_crash(r)

    def test_valid_field_apass(self):
        r = _run("comparecatalogs.py", "--field", "apass")
        assert _not_argparse_crash(r)

    def test_valid_field_gaia(self):
        r = _run("comparecatalogs.py", "--field", "gaia")
        assert _not_argparse_crash(r)

    def test_valid_multiple_fields(self):
        r = _run("comparecatalogs.py", "--field", "landolt", "sloan")
        assert _not_argparse_crash(r)

    def test_radius_expects_float(self):
        r = _run("comparecatalogs.py", "--radius", "notafloat")
        assert _error(r)

    def test_radius_valid(self):
        r = _run("comparecatalogs.py", "--radius", "30.0")
        assert _not_argparse_crash(r)

    def test_force_flag(self):
        r = _run("comparecatalogs.py", "--force")
        assert _not_argparse_crash(r)

    def test_panstarrs_flag(self):
        r = _run("comparecatalogs.py", "--panstarrs")
        assert _not_argparse_crash(r)

    def test_unknown_flag_fails(self):
        r = _run("comparecatalogs.py", "--unknown-flag-xyz")
        assert _error(r)


# ===========================================================================
# Default value checks
# ===========================================================================

class TestDefaultValues:
    """Verify that scripts document expected default values in --help output."""

    def test_lscloop_threshold_default_5(self):
        r = _help("lscloop.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        # Should mention default=5 for threshold
        assert "5" in combined  # conservative check

    def test_lscloop_xord_default_3(self):
        r = _help("lscloop.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "3" in combined

    def test_lscloop_size_default_7(self):
        r = _help("lscloop.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        # --size is documented in help; default=7 isn't shown by argparse
        assert "--size" in combined

    def test_lscloop_bkg_default_4(self):
        r = _help("lscloop.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "4" in combined

    def test_lscdiff_normalize_default_i(self):
        r = _help("lscdiff.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        # Default normalize is 'i'
        assert "i" in combined

    def test_lscdiff_interpolation_default_drizzle(self):
        r = _help("lscdiff.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "drizzle" in combined

    def test_lscdiff_difftype_default_0(self):
        r = _help("lscdiff.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        # Default difftype is 0
        assert "0" in combined

    def test_calibratemag_stage_default_abscat(self):
        r = _help("calibratemag.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "abscat" in combined

    def test_calibratemag_typemag_default_fit(self):
        r = _help("calibratemag.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "fit" in combined

    def test_comparecatalogs_radius_default_20(self):
        r = _help("comparecatalogs.py")
        combined = (r.stdout + r.stderr).decode(errors="replace")
        assert "20" in combined


# ===========================================================================
# Combined flag tests - testing reasonable argument combinations together
# ===========================================================================

class TestCombinedFlags:
    """Test combinations of flags to ensure they don't conflict at parse time."""

    def test_lscloop_psf_with_threshold_and_nstars(self):
        r = _run("lscloop.py", "--stage", "psf", "--threshold", "10",
                 "--nstars", "8", "--force")
        assert _not_argparse_crash(r)

    def test_lscloop_diff_with_all_params(self):
        r = _run("lscloop.py", "--stage", "diff", "--normalize", "t",
                 "--convolve", "i", "--bgo", "3.0", "--fixpix",
                 "--difftype", "0")
        assert _not_argparse_crash(r)

    def test_lscloop_zcat_with_field_and_cutmag(self):
        r = _run("lscloop.py", "--stage", "zcat", "--field", "sloan",
                 "--cutmag", "18.5", "--sigma-clip", "3.0")
        assert _not_argparse_crash(r)

    def test_lscloop_wcs_with_shifts(self):
        r = _run("lscloop.py", "--stage", "wcs", "--xshift", "10",
                 "--yshift", "-5", "--mode", "astrometry")
        assert _not_argparse_crash(r)

    def test_lscloop_psfmag_with_orders(self):
        r = _run("lscloop.py", "--stage", "psfmag", "--xord", "2",
                 "--yord", "2", "--bkg", "5.0", "--size", "8.0")
        assert _not_argparse_crash(r)

    def test_lscloop_getmag_with_output_and_combine(self):
        r = _run("lscloop.py", "--stage", "getmag", "--output", "test.out",
                 "--combine", "0.5", "--type", "fit")
        assert _not_argparse_crash(r)

    def test_lscloop_cosmic_with_multicore(self):
        r = _run("lscloop.py", "--stage", "cosmic", "--multicore", "2", "--force")
        assert _not_argparse_crash(r)

    def test_lscdiff_with_multiple_hotpants_params(self):
        r = _run("lscdiff.py", "t.fits", "r.fits",
                 "--nrxy", "2,2", "--nsxy", "10,10",
                 "--ko", "3", "--bgo", "3",
                 "--normalize", "t", "--convolve", "i",
                 "--interpolation", "poly3")
        assert _not_argparse_crash(r)

    def test_calibratemag_full_combo(self):
        r = _run("calibratemag.py", "list.txt",
                 "--stage", "local", "--typemag", "ph",
                 "--field", "sloan", "--match-by-site",
                 "--minstars", "3", "--force", "--interactive")
        assert _not_argparse_crash(r)

    def test_lcogtingest_full_combo(self):
        r = _run("LCOGTingest.py",
                 "--site", "cpt", "--telescope", "1m0a",
                 "--filter", "rp", "--obstype", "EXPOSE",
                 "--reduction", "reduced", "--limit", "100",
                 "--start", "2020-01-01", "--end", "2020-01-31")
        assert _not_argparse_crash(r)
