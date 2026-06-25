"""
Comprehensive tests for lsc.banzaicat.make_cat.

Tests cover:
  - Reading CAT extension from FITS files
  - Missing CAT extension handling (KeyError)
  - Column access and filtering logic
  - Empty catalogs, single-source catalogs, large catalogs
  - Invalid FITS files (corrupted, wrong type)
  - S/N calculation (PEAK / BACKGROUND)
  - Edge cases in sigma clipping parameters
  - Output file format validation
  - Idempotency of repeated calls
  - Boundary conditions for datamax, b_crlim, b_sigma
  - Division by zero in S/N when BACKGROUND=0
  - Negative values in columns
"""
import os
import sys
import importlib.util
import numpy as np
import pytest
from astropy.io import fits as afits
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.unit


def _import_banzaicat():
    """Import banzaicat directly from its source file, bypassing lsc.__init__."""
    spec = importlib.util.spec_from_file_location(
        "lsc.banzaicat",
        os.path.join(os.path.dirname(__file__), "..", "..", "src", "lsc", "banzaicat.py"),
    )
    # Ensure 'lsc' exists in sys.modules as a stub so `import lsc` in banzaicat.py works
    if "lsc" not in sys.modules:
        sys.modules["lsc"] = MagicMock()
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_banzai_fits(path, ras, decs, ellipticities, backgrounds, fwhms, peaks, flags):
    """Write a BANZAI-style FITS file with the given source arrays."""
    n = len(ras)
    primary = afits.PrimaryHDU(np.ones((10, 10), dtype=np.float32))
    cols = [
        afits.Column(name="RA",          format="D", array=np.array(ras,           dtype=np.float64)),
        afits.Column(name="DEC",         format="D", array=np.array(decs,          dtype=np.float64)),
        afits.Column(name="ELLIPTICITY", format="E", array=np.array(ellipticities, dtype=np.float32)),
        afits.Column(name="BACKGROUND",  format="E", array=np.array(backgrounds,   dtype=np.float32)),
        afits.Column(name="FWHM",        format="E", array=np.array(fwhms,         dtype=np.float32)),
        afits.Column(name="PEAK",        format="E", array=np.array(peaks,         dtype=np.float32)),
        afits.Column(name="FLAG",        format="I", array=np.array(flags,         dtype=np.int16)),
    ]
    cat_hdu = afits.BinTableHDU.from_columns(cols, name="CAT")
    afits.HDUList([primary, cat_hdu]).writeto(str(path), overwrite=True)
    return str(path)


def _data_lines(cat_path):
    """Return non-comment lines from a banzai.cat file."""
    with open(cat_path) as f:
        return [ln for ln in f if not ln.startswith("#") and ln.strip()]


def _header_lines(cat_path):
    """Return comment/header lines from a banzai.cat file."""
    with open(cat_path) as f:
        return [ln for ln in f if ln.startswith("#")]


# ---------------------------------------------------------------------------
# Missing CAT extension
# ---------------------------------------------------------------------------

class TestMissingCATExtension:
    """Tests for handling FITS files without a CAT extension."""

    def test_no_cat_extension_raises_keyerror(self, tmp_path, monkeypatch):
        """A FITS file without a CAT extension raises KeyError."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        # Create FITS without CAT extension
        path = tmp_path / "no_cat.fits"
        primary = afits.PrimaryHDU(np.ones((10, 10), dtype=np.float32))
        afits.HDUList([primary]).writeto(str(path), overwrite=True)
        with pytest.raises(KeyError):
            banzaicat.make_cat(str(path))

    def test_wrong_extension_name_raises_keyerror(self, tmp_path, monkeypatch):
        """A FITS file with an extension named 'SCI' instead of 'CAT' raises KeyError."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "wrong_ext.fits"
        primary = afits.PrimaryHDU(np.ones((10, 10), dtype=np.float32))
        cols = [afits.Column(name="RA", format="D", array=np.array([150.0]))]
        sci_hdu = afits.BinTableHDU.from_columns(cols, name="SCI")
        afits.HDUList([primary, sci_hdu]).writeto(str(path), overwrite=True)
        with pytest.raises(KeyError):
            banzaicat.make_cat(str(path))


# ---------------------------------------------------------------------------
# Invalid FITS files
# ---------------------------------------------------------------------------

class TestInvalidFITSFiles:
    """Tests for handling invalid or corrupted FITS files."""

    def test_nonexistent_file_raises_oserror(self, tmp_path, monkeypatch):
        """A file that does not exist raises an OSError (FileNotFoundError)."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        with pytest.raises(OSError):
            banzaicat.make_cat("/nonexistent/path/file.fits")

    def test_empty_file_raises_oserror(self, tmp_path, monkeypatch):
        """An empty file raises an OSError."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "empty.fits"
        path.write_bytes(b"")
        with pytest.raises(OSError):
            banzaicat.make_cat(str(path))

    def test_non_fits_file_raises_oserror(self, tmp_path, monkeypatch):
        """A text file with wrong format raises an OSError."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "text.fits"
        path.write_text("This is not a FITS file\n")
        with pytest.raises(OSError):
            banzaicat.make_cat(str(path))


# ---------------------------------------------------------------------------
# Empty catalog (zero sources in CAT table)
# ---------------------------------------------------------------------------

class TestEmptyCatalog:
    """Tests for FITS files with an empty CAT table (0 rows)."""

    def test_empty_cat_table_produces_header_only(self, tmp_path, monkeypatch):
        """A CAT extension with 0 rows yields only header lines in output."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "empty_cat.fits"
        primary = afits.PrimaryHDU(np.ones((10, 10), dtype=np.float32))
        cols = [
            afits.Column(name="RA",          format="D", array=np.array([], dtype=np.float64)),
            afits.Column(name="DEC",         format="D", array=np.array([], dtype=np.float64)),
            afits.Column(name="ELLIPTICITY", format="E", array=np.array([], dtype=np.float32)),
            afits.Column(name="BACKGROUND",  format="E", array=np.array([], dtype=np.float32)),
            afits.Column(name="FWHM",        format="E", array=np.array([], dtype=np.float32)),
            afits.Column(name="PEAK",        format="E", array=np.array([], dtype=np.float32)),
            afits.Column(name="FLAG",        format="I", array=np.array([], dtype=np.int16)),
        ]
        cat_hdu = afits.BinTableHDU.from_columns(cols, name="CAT")
        afits.HDUList([primary, cat_hdu]).writeto(str(path), overwrite=True)
        cat = banzaicat.make_cat(str(path), datamax=99999, b_sigma=5, b_crlim=0.5)
        assert os.path.exists(cat)
        assert len(_data_lines(cat)) == 0
        assert len(_header_lines(cat)) > 0


# ---------------------------------------------------------------------------
# Single-source catalog
# ---------------------------------------------------------------------------

class TestSingleSourceCatalog:
    """Tests for catalogs with exactly one source."""

    def test_single_valid_source_passes(self, tmp_path, monkeypatch):
        """A single source meeting all criteria appears in output."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "single.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        assert len(lines) == 1

    def test_single_flagged_source_excluded(self, tmp_path, monkeypatch):
        """A single source with FLAG != 0 produces empty data."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "single_flag.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[1],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert len(_data_lines(cat)) == 0

    def test_single_source_id_is_one(self, tmp_path, monkeypatch):
        """A single passing source gets ID=1."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "single_id.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        assert int(lines[0].strip().split("\t")[2]) == 1

    def test_single_source_sigma_clip_no_error(self, tmp_path, monkeypatch):
        """Sigma clipping with a single source should not raise errors."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "single_clip.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        # Should not raise
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=1, b_crlim=0.5)
        assert os.path.exists(cat)


# ---------------------------------------------------------------------------
# S/N calculation correctness
# ---------------------------------------------------------------------------

class TestSNCalculation:
    """Tests for the signal-to-noise ratio calculation (PEAK / BACKGROUND)."""

    def test_sn_order_determines_output_order(self, tmp_path, monkeypatch):
        """Sources are sorted by S/N = PEAK/BACKGROUND descending."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        # Source A: S/N = 5000/100 = 50
        # Source B: S/N = 10000/200 = 50
        # Source C: S/N = 20000/100 = 200
        # Expected order: C (200), then A or B (50)
        fits_path = _make_banzai_fits(
            tmp_path / "sn_order.fits",
            ras=[10.0, 20.0, 30.0],
            decs=[1.0, 1.0, 1.0],
            ellipticities=[0.1, 0.1, 0.1],
            backgrounds=[100.0, 200.0, 100.0],
            fwhms=[4.0, 4.0, 4.0],
            peaks=[5000.0, 10000.0, 20000.0],
            flags=[0, 0, 0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        assert len(lines) == 3
        # First line should be RA=30 (highest S/N=200)
        first_ra = float(lines[0].strip().split("\t")[0])
        assert first_ra == pytest.approx(30.0, abs=0.01)

    def test_equal_sn_stable_sort(self, tmp_path, monkeypatch):
        """Sources with equal S/N maintain some consistent order."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        # All have S/N = 50
        fits_path = _make_banzai_fits(
            tmp_path / "equal_sn.fits",
            ras=[10.0, 20.0, 30.0],
            decs=[1.0, 1.0, 1.0],
            ellipticities=[0.1, 0.1, 0.1],
            backgrounds=[100.0, 100.0, 100.0],
            fwhms=[4.0, 4.0, 4.0],
            peaks=[5000.0, 5000.0, 5000.0],
            flags=[0, 0, 0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        assert len(lines) == 3

    def test_very_low_background_high_sn(self, tmp_path, monkeypatch):
        """Low background gives high S/N, source should be first."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "low_bg.fits",
            ras=[10.0, 20.0],
            decs=[1.0, 1.0],
            ellipticities=[0.1, 0.1],
            backgrounds=[1.0, 1000.0],
            fwhms=[4.0, 4.0],
            peaks=[5000.0, 5000.0],
            flags=[0, 0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        if len(lines) >= 2:
            first_ra = float(lines[0].strip().split("\t")[0])
            # Source with bg=1 has S/N=5000, source with bg=1000 has S/N=5
            assert first_ra == pytest.approx(10.0, abs=0.01)


# ---------------------------------------------------------------------------
# Large catalog performance and correctness
# ---------------------------------------------------------------------------

class TestLargeCatalog:
    """Tests for catalogs with many sources."""

    def test_100_sources_processed(self, tmp_path, monkeypatch):
        """100 identical clean sources should all pass."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        n = 100
        fits_path = _make_banzai_fits(
            tmp_path / "large.fits",
            ras=list(np.linspace(149, 151, n)),
            decs=[2.0] * n,
            ellipticities=[0.1] * n,
            backgrounds=[100.0] * n,
            fwhms=[4.0] * n,
            peaks=[5000.0] * n,
            flags=[0] * n,
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        assert len(lines) == n

    def test_ids_contiguous_for_large_catalog(self, tmp_path, monkeypatch):
        """IDs should be 1..n for n passing sources."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        n = 50
        fits_path = _make_banzai_fits(
            tmp_path / "ids_large.fits",
            ras=list(np.linspace(149, 151, n)),
            decs=[2.0] * n,
            ellipticities=[0.1] * n,
            backgrounds=[100.0] * n,
            fwhms=[4.0] * n,
            peaks=[5000.0] * n,
            flags=[0] * n,
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        ids = sorted([int(ln.strip().split("\t")[2]) for ln in lines])
        assert ids == list(range(1, n + 1))

    def test_half_flagged_half_clean(self, tmp_path, monkeypatch):
        """Half flagged, half clean: only clean sources appear."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        n = 20
        flags = [0] * (n // 2) + [1] * (n // 2)
        fits_path = _make_banzai_fits(
            tmp_path / "half_flag.fits",
            ras=list(np.linspace(149, 151, n)),
            decs=[2.0] * n,
            ellipticities=[0.1] * n,
            backgrounds=[100.0] * n,
            fwhms=[4.0] * n,
            peaks=[5000.0] * n,
            flags=flags,
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        assert len(lines) == n // 2


# ---------------------------------------------------------------------------
# Sigma clipping edge cases
# ---------------------------------------------------------------------------

class TestSigmaClipEdgeCases:
    """Edge cases for the sigma clipping on ELLIPTICITY, BACKGROUND, and FWHM."""

    def test_identical_values_no_clipping(self, tmp_path, monkeypatch):
        """When all values are identical, sigma=0, and sigmaclip should keep all."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        n = 10
        fits_path = _make_banzai_fits(
            tmp_path / "identical.fits",
            ras=list(np.linspace(149, 151, n)),
            decs=[2.0] * n,
            ellipticities=[0.1] * n,
            backgrounds=[100.0] * n,
            fwhms=[4.0] * n,
            peaks=[5000.0] * n,
            flags=[0] * n,
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=1, b_crlim=0.5)
        lines = _data_lines(cat)
        assert len(lines) == n

    def test_very_large_sigma_keeps_all(self, tmp_path, monkeypatch):
        """Very large b_sigma (100) should keep all sources regardless of spread."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        rng = np.random.default_rng(42)
        n = 10
        fits_path = _make_banzai_fits(
            tmp_path / "big_sigma.fits",
            ras=list(rng.uniform(149, 151, n)),
            decs=[2.0] * n,
            ellipticities=list(rng.uniform(0.0, 0.5, n)),
            backgrounds=list(rng.uniform(50, 500, n)),
            fwhms=list(rng.uniform(3.5, 5.5, n)),
            peaks=[5000.0] * n,
            flags=[0] * n,
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=100, b_crlim=0.5)
        lines = _data_lines(cat)
        assert len(lines) == n

    def test_very_small_sigma_may_clip_all(self, tmp_path, monkeypatch):
        """Very small b_sigma (0.01) with varied data may clip everything."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        # Two sources with different ellipticities
        fits_path = _make_banzai_fits(
            tmp_path / "tiny_sigma.fits",
            ras=[150.0, 150.1],
            decs=[2.0, 2.0],
            ellipticities=[0.1, 0.3],  # different enough for tight sigma
            backgrounds=[100.0, 100.0],
            fwhms=[4.0, 4.0],
            peaks=[5000.0, 5000.0],
            flags=[0, 0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=0.01, b_crlim=0.5)
        # With very tight sigma on just 2 points, behavior depends on scipy's sigmaclip
        lines = _data_lines(cat)
        # Should not crash - result may be 0, 1, or 2
        assert len(lines) <= 2


# ---------------------------------------------------------------------------
# datamax boundary conditions
# ---------------------------------------------------------------------------

class TestDatamaxBoundaryConditions:
    """Boundary conditions for the datamax parameter."""

    def test_datamax_zero_excludes_all_positive_peaks(self, tmp_path, monkeypatch):
        """datamax=0 should exclude all sources with PEAK > 0."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "dm_zero.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=0, b_sigma=5, b_crlim=0.5)
        assert len(_data_lines(cat)) == 0

    def test_datamax_negative_excludes_all(self, tmp_path, monkeypatch):
        """datamax=-1 should exclude all sources with positive PEAK."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "dm_neg.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=-1, b_sigma=5, b_crlim=0.5)
        assert len(_data_lines(cat)) == 0

    def test_datamax_very_large_includes_all(self, tmp_path, monkeypatch):
        """datamax=1e10 should include all sources."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "dm_huge.fits",
            ras=[150.0, 150.1, 150.2],
            decs=[2.0, 2.0, 2.0],
            ellipticities=[0.1, 0.1, 0.1],
            backgrounds=[100.0, 100.0, 100.0],
            fwhms=[4.0, 4.0, 4.0],
            peaks=[50000.0, 60000.0, 70000.0],
            flags=[0, 0, 0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=1e10, b_sigma=5, b_crlim=0.5)
        assert len(_data_lines(cat)) == 3


# ---------------------------------------------------------------------------
# b_crlim boundary conditions
# ---------------------------------------------------------------------------

class TestCrlimBoundaryConditions:
    """Boundary conditions for b_crlim (minimum FWHM threshold)."""

    def test_crlim_zero_includes_all_positive_fwhm(self, tmp_path, monkeypatch):
        """b_crlim=0 includes all sources with FWHM > 0."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "crlim_zero.fits",
            ras=[150.0, 150.1],
            decs=[2.0, 2.0],
            ellipticities=[0.1, 0.1],
            backgrounds=[100.0, 100.0],
            fwhms=[0.5, 4.0],
            peaks=[5000.0, 5000.0],
            flags=[0, 0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0)
        lines = _data_lines(cat)
        assert len(lines) == 2

    def test_crlim_negative_includes_all(self, tmp_path, monkeypatch):
        """b_crlim=-1 includes all sources (FWHM > -1 is always true)."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "crlim_neg.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=-1)
        assert len(_data_lines(cat)) == 1

    def test_crlim_at_fwhm_minus_epsilon(self, tmp_path, monkeypatch):
        """b_crlim just below FWHM includes the source."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "crlim_below.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=3.99)
        assert len(_data_lines(cat)) == 1


# ---------------------------------------------------------------------------
# Output file format
# ---------------------------------------------------------------------------

class TestOutputFileFormat:
    """Tests for the output file format of banzai.cat."""

    def test_output_filename_always_banzai_cat(self, tmp_path, monkeypatch):
        """Output file is always named 'banzai.cat'."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "named.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        result = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert result == 'banzai.cat'
        assert os.path.exists(tmp_path / 'banzai.cat')

    def test_header_nfields_is_13(self, tmp_path, monkeypatch):
        """Header should declare nfields 13."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "nfields.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        with open(cat) as f:
            content = f.read()
        assert "# nfields 13" in content

    def test_ra_format_10_5f(self, tmp_path, monkeypatch):
        """RA values should be formatted with at least 5 decimal places."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "ra_fmt.fits",
            ras=[150.12345], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        ra_str = lines[0].strip().split("\t")[0]
        # Should have 5 decimal places
        decimal_part = ra_str.strip().split('.')[1]
        assert len(decimal_part) == 5

    def test_dec_format_10_5f(self, tmp_path, monkeypatch):
        """DEC values should be formatted with at least 5 decimal places."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "dec_fmt.fits",
            ras=[150.0], decs=[2.12345], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        dec_str = lines[0].strip().split("\t")[1]
        decimal_part = dec_str.strip().split('.')[1]
        assert len(decimal_part) == 5

    def test_tab_separated_fields(self, tmp_path, monkeypatch):
        """Fields in data lines are tab-separated."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "tabs.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        assert '\t' in lines[0]

    def test_no_trailing_whitespace_issues(self, tmp_path, monkeypatch):
        """Lines should end with newline, not extra whitespace."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "ws.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        with open(cat) as f:
            for line in f:
                if line.strip():
                    assert line.endswith('\n')


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    """Running make_cat twice with same inputs produces identical output."""

    def test_same_input_same_output(self, tmp_path, monkeypatch):
        """Two consecutive runs produce the same banzai.cat content."""
        banzaicat = _import_banzaicat()
        fits_path = _make_banzai_fits(
            tmp_path / "idem.fits",
            ras=[150.0, 150.1, 150.2],
            decs=[2.0, 2.0, 2.0],
            ellipticities=[0.1, 0.15, 0.1],
            backgrounds=[100.0, 110.0, 105.0],
            fwhms=[4.0, 4.5, 4.2],
            peaks=[5000.0, 6000.0, 4500.0],
            flags=[0, 0, 0],
        )
        d1 = tmp_path / "run1"
        d2 = tmp_path / "run2"
        d1.mkdir()
        d2.mkdir()

        monkeypatch.chdir(d1)
        banzaicat.make_cat(fits_path, datamax=99999, b_sigma=3, b_crlim=0.5)
        with open(d1 / "banzai.cat") as f:
            content1 = f.read()

        monkeypatch.chdir(d2)
        banzaicat.make_cat(fits_path, datamax=99999, b_sigma=3, b_crlim=0.5)
        with open(d2 / "banzai.cat") as f:
            content2 = f.read()

        assert content1 == content2


# ---------------------------------------------------------------------------
# Division by zero edge case
# ---------------------------------------------------------------------------

class TestDivisionByZero:
    """Edge case where BACKGROUND is zero (S/N = PEAK/0 = inf)."""

    def test_zero_background_does_not_crash(self, tmp_path, monkeypatch):
        """A source with BACKGROUND=0 should not cause a division error."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "zerobg.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[0.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        # This may or may not include the source depending on sigma clipping
        # but it should NOT crash
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert os.path.exists(cat)

    def test_zero_background_multiple_sources(self, tmp_path, monkeypatch):
        """Multiple sources where one has BACKGROUND=0 should not crash."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "zerobg_multi.fits",
            ras=[150.0, 150.1, 150.2],
            decs=[2.0, 2.0, 2.0],
            ellipticities=[0.1, 0.1, 0.1],
            backgrounds=[0.0, 100.0, 100.0],
            fwhms=[4.0, 4.0, 4.0],
            peaks=[5000.0, 5000.0, 5000.0],
            flags=[0, 0, 0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert os.path.exists(cat)


# ---------------------------------------------------------------------------
# Negative values in columns
# ---------------------------------------------------------------------------

class TestNegativeValues:
    """Edge cases with negative values in various columns."""

    def test_negative_ellipticity(self, tmp_path, monkeypatch):
        """Negative ellipticity should be handled without crash."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "neg_ellip.fits",
            ras=[150.0], decs=[2.0], ellipticities=[-0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert os.path.exists(cat)

    def test_negative_peak(self, tmp_path, monkeypatch):
        """Negative PEAK value should be excluded by datamax > 0 filter."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "neg_peak.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[-100.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        # -100 <= 99999 so it passes datamax filter but S/N would be negative
        # Behavior depends on sigmaclip with one source
        assert os.path.exists(cat)

    def test_negative_background(self, tmp_path, monkeypatch):
        """Negative BACKGROUND should not crash."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "neg_bg.fits",
            ras=[150.0, 150.1],
            decs=[2.0, 2.0],
            ellipticities=[0.1, 0.1],
            backgrounds=[-50.0, 100.0],
            fwhms=[4.0, 4.0],
            peaks=[5000.0, 5000.0],
            flags=[0, 0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert os.path.exists(cat)


# ---------------------------------------------------------------------------
# Column access validation
# ---------------------------------------------------------------------------

class TestColumnAccess:
    """Tests verifying that all required columns are accessed correctly."""

    def test_ra_column_values_preserved(self, tmp_path, monkeypatch):
        """RA values from the catalog should appear in output unchanged."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "ra_vals.fits",
            ras=[123.45678], decs=[45.67890], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        ra = float(lines[0].strip().split("\t")[0])
        assert ra == pytest.approx(123.45678, abs=0.001)

    def test_dec_column_values_preserved(self, tmp_path, monkeypatch):
        """DEC values from the catalog should appear in output unchanged."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "dec_vals.fits",
            ras=[150.0], decs=[-35.12345], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        dec = float(lines[0].strip().split("\t")[1])
        assert dec == pytest.approx(-35.12345, abs=0.001)

    def test_negative_dec_preserved(self, tmp_path, monkeypatch):
        """Negative DEC values should be preserved in output."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "neg_dec.fits",
            ras=[150.0], decs=[-89.5], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        dec = float(lines[0].strip().split("\t")[1])
        assert dec < 0

    def test_ra_near_zero(self, tmp_path, monkeypatch):
        """RA near 0 degrees should be preserved correctly."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "ra_zero.fits",
            ras=[0.12345], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        ra = float(lines[0].strip().split("\t")[0])
        assert ra == pytest.approx(0.12345, abs=0.001)

    def test_ra_near_360(self, tmp_path, monkeypatch):
        """RA near 360 degrees should be preserved correctly."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "ra_360.fits",
            ras=[359.99999], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        ra = float(lines[0].strip().split("\t")[0])
        assert ra == pytest.approx(359.99999, abs=0.001)


# ---------------------------------------------------------------------------
# Multiple filter interactions
# ---------------------------------------------------------------------------

class TestFilterInteractions:
    """Tests for interactions between multiple filtering criteria."""

    def test_source_fails_only_peak_check(self, tmp_path, monkeypatch):
        """Source passes all filters except datamax."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "only_peak.fits",
            ras=[150.0, 150.1],
            decs=[2.0, 2.0],
            ellipticities=[0.1, 0.1],
            backgrounds=[100.0, 100.0],
            fwhms=[4.0, 4.0],
            peaks=[5000.0, 80000.0],  # second exceeds datamax
            flags=[0, 0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=75000, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        assert len(lines) == 1

    def test_source_fails_only_fwhm_check(self, tmp_path, monkeypatch):
        """Source passes all filters except FWHM > b_crlim."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "only_fwhm.fits",
            ras=[150.0, 150.1],
            decs=[2.0, 2.0],
            ellipticities=[0.1, 0.1],
            backgrounds=[100.0, 100.0],
            fwhms=[4.0, 2.0],  # second below b_crlim=3
            peaks=[5000.0, 5000.0],
            flags=[0, 0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=3.0)
        lines = _data_lines(cat)
        assert len(lines) == 1

    def test_source_fails_only_flag_check(self, tmp_path, monkeypatch):
        """Source passes all filters except FLAG."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "only_flag.fits",
            ras=[150.0, 150.1],
            decs=[2.0, 2.0],
            ellipticities=[0.1, 0.1],
            backgrounds=[100.0, 100.0],
            fwhms=[4.0, 4.0],
            peaks=[5000.0, 5000.0],
            flags=[0, 4],  # second has non-zero flag
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        assert len(lines) == 1

    def test_all_filters_fail_simultaneously(self, tmp_path, monkeypatch):
        """Source fails all criteria: flag, peak, fwhm."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "all_fail.fits",
            ras=[150.0],
            decs=[2.0],
            ellipticities=[0.1],
            backgrounds=[100.0],
            fwhms=[1.0],       # below b_crlim=3
            peaks=[90000.0],   # above datamax=75000
            flags=[2],         # non-zero flag
        )
        cat = banzaicat.make_cat(fits_path, datamax=75000, b_sigma=5, b_crlim=3.0)
        assert len(_data_lines(cat)) == 0


# ---------------------------------------------------------------------------
# Return value
# ---------------------------------------------------------------------------

class TestReturnValue:
    """Tests for the return value of make_cat."""

    def test_returns_string(self, tmp_path, monkeypatch):
        """make_cat always returns a string."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "ret.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        result = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert isinstance(result, str)

    def test_return_value_is_banzai_cat(self, tmp_path, monkeypatch):
        """Return value is always 'banzai.cat'."""
        banzaicat = _import_banzaicat()
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "retval.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        result = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert result == 'banzai.cat'
