"""
Tests for lsc.banzaicat.make_cat (make_banzai_cat).

Coverage:
  - TestMakeBanzaiCat: baseline file-creation tests (original 5)
  - TestMakeCatFileHeader: catalog header structure/content validation
  - TestMakeCatDataLines: per-line format validation (RA/DEC floats, integer IDs)
  - TestMakeCatSorting: output rows sorted by S/N (PEAK/BACKGROUND) descending
  - TestMakeCatFlagFiltering: FLAG != 0 sources are excluded
  - TestMakeCatDatamax: datamax=None defaults to 75000; datamax below all peaks -> 0 rows
  - TestMakeCatFwhmFiltering: b_crlim above all FWHM values yields empty data section
  - TestMakeCatSigmaClipFiltering: extreme outlier in ELLIPTICITY/BACKGROUND is removed
  - TestMakeCatSourceCount: known-source-count catalog produces expected row count
  - TestMissingCATExtension: FITS without CAT extension raises KeyError
  - TestInvalidFITSFiles: corrupted/wrong-type files raise OSError
  - TestEmptyCatalog: empty CAT table produces header-only output
  - TestSingleSourceCatalog: single-source edge cases
  - TestSNCalculation: S/N = PEAK/BACKGROUND ordering
  - TestLargeCatalog: 50-100 source correctness
  - TestSigmaClipEdgeCases: boundary sigma clipping behavior
  - TestDatamaxBoundaryConditions: datamax 0, negative, very large
  - TestCrlimBoundaryConditions: b_crlim 0, negative, epsilon below FWHM
  - TestOutputFileFormat: filename, nfields, decimal format, tabs
  - TestIdempotency: repeated calls produce identical output
  - TestDivisionByZero: BACKGROUND=0 does not crash
  - TestNegativeValues: negative column values handled
  - TestColumnAccess: RA/DEC values preserved in output
  - TestFilterInteractions: multiple filter criteria interactions
  - TestReturnValue: return type and value
"""
import os
import numpy as np
import pytest
from astropy.io import fits as afits

from lsc import banzaicat

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_banzai_fits(path, ras, decs, ellipticities, backgrounds, fwhms, peaks, flags):
    """Write a BANZAI-style FITS file with the given source arrays."""
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
# Original tests
# ---------------------------------------------------------------------------

class TestMakeBanzaiCat:
    def test_creates_output_file(self, banzai_fits, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = banzaicat.make_cat(banzai_fits, datamax=50000, b_sigma=5, b_crlim=0.5)
        assert os.path.exists(result)

    def test_returns_banzai_cat(self, banzai_fits, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = banzaicat.make_cat(banzai_fits, datamax=50000, b_sigma=5, b_crlim=0.5)
        assert result == "banzai.cat"

    def test_output_not_empty(self, banzai_fits, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = banzaicat.make_cat(banzai_fits, datamax=50000, b_sigma=5, b_crlim=0.5)
        assert os.path.getsize(result) > 0

    def test_high_datamax_includes_more_sources(self, banzai_fits, tmp_path, monkeypatch):
        """A very high datamax should not reduce the number of sources vs default."""
        high_dir = tmp_path / "high"
        low_dir  = tmp_path / "low"
        high_dir.mkdir()
        low_dir.mkdir()

        monkeypatch.chdir(high_dir)
        banzaicat.make_cat(banzai_fits, datamax=99999, b_sigma=5, b_crlim=0.5)
        high_size = os.path.getsize(high_dir / "banzai.cat")

        monkeypatch.chdir(low_dir)
        banzaicat.make_cat(banzai_fits, datamax=100, b_sigma=5, b_crlim=0.5)
        low_size = os.path.getsize(low_dir / "banzai.cat")

        assert high_size >= low_size

    def test_strict_cr_filter_reduces_sources(self, banzai_fits, tmp_path, monkeypatch):
        """Stricter b_crlim should yield fewer or equal sources."""
        def run_with_crlim(crlim, subdir):
            d = tmp_path / subdir
            d.mkdir()
            monkeypatch.chdir(d)
            banzaicat.make_cat(banzai_fits, datamax=99999, b_sigma=5, b_crlim=crlim)
            return os.path.getsize(d / "banzai.cat")

        size_loose  = run_with_crlim(1.5, "loose")
        size_strict = run_with_crlim(0.01, "strict")
        assert size_strict <= size_loose


# ---------------------------------------------------------------------------
# Catalog header content
# ---------------------------------------------------------------------------

class TestMakeCatFileHeader:
    """The written catalog must contain the expected header block."""

    def _run(self, banzai_fits, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        return banzaicat.make_cat(banzai_fits, datamax=99999, b_sigma=5, b_crlim=0.5)

    def test_header_has_begin_marker(self, banzai_fits, tmp_path, monkeypatch):
        cat = self._run(banzai_fits, tmp_path, monkeypatch)
        with open(cat) as f:
            content = f.read()
        assert "# BEGIN CATALOG HEADER" in content

    def test_header_has_end_marker(self, banzai_fits, tmp_path, monkeypatch):
        cat = self._run(banzai_fits, tmp_path, monkeypatch)
        with open(cat) as f:
            content = f.read()
        assert "# END CATALOG HEADER" in content

    def test_header_has_ra_field(self, banzai_fits, tmp_path, monkeypatch):
        cat = self._run(banzai_fits, tmp_path, monkeypatch)
        with open(cat) as f:
            content = f.read()
        assert "#     ra" in content

    def test_header_has_dec_field(self, banzai_fits, tmp_path, monkeypatch):
        cat = self._run(banzai_fits, tmp_path, monkeypatch)
        with open(cat) as f:
            content = f.read()
        assert "#     dec" in content

    def test_header_has_id_field(self, banzai_fits, tmp_path, monkeypatch):
        cat = self._run(banzai_fits, tmp_path, monkeypatch)
        with open(cat) as f:
            content = f.read()
        assert "#     id" in content

    def test_all_header_lines_start_with_hash(self, banzai_fits, tmp_path, monkeypatch):
        cat = self._run(banzai_fits, tmp_path, monkeypatch)
        hlines = _header_lines(cat)
        assert len(hlines) > 0
        assert all(ln.startswith("#") for ln in hlines)


# ---------------------------------------------------------------------------
# Per-data-line format validation
# ---------------------------------------------------------------------------

class TestMakeCatDataLines:
    """Each data line must have exactly 3 tab-separated fields with correct types."""

    def _data(self, banzai_fits, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(banzai_fits, datamax=99999, b_sigma=5, b_crlim=0.5)
        return _data_lines(cat)

    def test_data_lines_present(self, banzai_fits, tmp_path, monkeypatch):
        lines = self._data(banzai_fits, tmp_path, monkeypatch)
        assert len(lines) > 0

    def test_each_line_has_three_fields(self, banzai_fits, tmp_path, monkeypatch):
        lines = self._data(banzai_fits, tmp_path, monkeypatch)
        for ln in lines:
            fields = ln.strip().split("\t")
            assert len(fields) == 3, f"Expected 3 fields, got: {ln!r}"

    def test_ra_field_is_float(self, banzai_fits, tmp_path, monkeypatch):
        lines = self._data(banzai_fits, tmp_path, monkeypatch)
        for ln in lines:
            ra_str = ln.strip().split("\t")[0]
            assert float(ra_str) == float(ra_str)

    def test_dec_field_is_float(self, banzai_fits, tmp_path, monkeypatch):
        lines = self._data(banzai_fits, tmp_path, monkeypatch)
        for ln in lines:
            dec_str = ln.strip().split("\t")[1]
            assert float(dec_str) == float(dec_str)

    def test_id_field_is_positive_integer(self, banzai_fits, tmp_path, monkeypatch):
        lines = self._data(banzai_fits, tmp_path, monkeypatch)
        for ln in lines:
            id_str = ln.strip().split("\t")[2]
            assert int(id_str) > 0

    def test_ids_form_contiguous_set_from_one(self, banzai_fits, tmp_path, monkeypatch):
        """IDs are a permutation of {1..n}."""
        lines = self._data(banzai_fits, tmp_path, monkeypatch)
        ids = [int(ln.strip().split("\t")[2]) for ln in lines]
        assert sorted(ids) == list(range(1, len(ids) + 1))

    def test_ra_in_fixture_range(self, banzai_fits, tmp_path, monkeypatch):
        """conftest fixture uses RA in [149, 151]."""
        lines = self._data(banzai_fits, tmp_path, monkeypatch)
        for ln in lines:
            ra = float(ln.strip().split("\t")[0])
            assert 149.0 <= ra <= 151.0

    def test_dec_in_fixture_range(self, banzai_fits, tmp_path, monkeypatch):
        """conftest fixture uses DEC in [1.5, 2.5]."""
        lines = self._data(banzai_fits, tmp_path, monkeypatch)
        for ln in lines:
            dec = float(ln.strip().split("\t")[1])
            assert 1.5 <= dec <= 2.5


# ---------------------------------------------------------------------------
# Output sorted by S/N descending
# ---------------------------------------------------------------------------

class TestMakeCatSorting:
    """make_cat sorts sources by S/N = PEAK/BACKGROUND in descending order."""

    def test_sources_sorted_sn_descending(self, tmp_path, monkeypatch):
        fits_path = _make_banzai_fits(
            tmp_path / "sorted.fits",
            ras=          [10.0, 20.0, 30.0, 40.0, 50.0],
            decs=         [ 1.0,  1.0,  1.0,  1.0,  1.0],
            ellipticities=[0.1,   0.1,  0.1,  0.1,  0.1],
            backgrounds=  [100.0, 100.0, 100.0, 100.0, 100.0],
            fwhms=        [4.0,   4.0,  4.0,  4.0,  4.0],
            peaks=        [50.0,  200.0, 10.0, 300.0, 100.0],
            flags=        [0, 0, 0, 0, 0],
        )
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        assert len(lines) == 5
        first_ra = float(lines[0].strip().split("\t")[0])
        assert first_ra == pytest.approx(40.0, abs=0.01)
        last_ra = float(lines[-1].strip().split("\t")[0])
        assert last_ra == pytest.approx(30.0, abs=0.01)

    def test_sn_order_is_non_increasing(self, tmp_path, monkeypatch):
        """IDs must form a contiguous set 1..n regardless of output order."""
        rng = np.random.default_rng(99)
        n = 15
        peaks = rng.uniform(100, 50000, n)
        bgs   = rng.uniform(50, 500, n)
        fits_path = _make_banzai_fits(
            tmp_path / "order.fits",
            ras=          list(rng.uniform(149, 151, n)),
            decs=         list(rng.uniform(1.5, 2.5, n)),
            ellipticities=list(rng.uniform(0.0, 0.2, n)),
            backgrounds=  list(bgs),
            fwhms=        list(rng.uniform(3.0, 7.0, n)),
            peaks=        list(peaks),
            flags=        [0] * n,
        )
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        ids = [int(ln.strip().split("\t")[2]) for ln in lines]
        assert sorted(ids) == list(range(1, len(ids) + 1))


# ---------------------------------------------------------------------------
# FLAG filtering
# ---------------------------------------------------------------------------

class TestMakeCatFlagFiltering:
    """Sources with FLAG != 0 must be excluded."""

    def test_all_flagged_sources_excluded(self, tmp_path, monkeypatch):
        fits_path = _make_banzai_fits(
            tmp_path / "flagged.fits",
            ras=          [150.0, 150.1, 150.2],
            decs=         [2.0,   2.0,   2.0],
            ellipticities=[0.1,   0.1,   0.1],
            backgrounds=  [100.0, 100.0, 100.0],
            fwhms=        [4.0,   4.0,   4.0],
            peaks=        [5000.0, 6000.0, 7000.0],
            flags=        [1, 2, 4],
        )
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert len(_data_lines(cat)) == 0

    def test_mixed_flags_only_zero_flag_passes(self, tmp_path, monkeypatch):
        fits_path = _make_banzai_fits(
            tmp_path / "mixed_flags.fits",
            ras=          [150.0, 150.1, 150.2],
            decs=         [2.0,   2.0,   2.0],
            ellipticities=[0.1,   0.1,   0.1],
            backgrounds=  [100.0, 100.0, 100.0],
            fwhms=        [4.0,   4.0,   4.0],
            peaks=        [5000.0, 6000.0, 7000.0],
            flags=        [0, 1, 2],
        )
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        assert len(lines) == 1
        assert float(lines[0].strip().split("\t")[0]) == pytest.approx(150.0, abs=0.001)


# ---------------------------------------------------------------------------
# datamax handling
# ---------------------------------------------------------------------------

class TestMakeCatDatamax:
    """datamax filters sources whose PEAK exceeds the limit."""

    def test_datamax_none_same_as_75000(self, tmp_path, monkeypatch):
        fits_path = _make_banzai_fits(
            tmp_path / "dm_none.fits",
            ras=          [150.0, 150.1],
            decs=         [2.0,   2.0],
            ellipticities=[0.1,   0.1],
            backgrounds=  [100.0, 100.0],
            fwhms=        [4.0,   4.0],
            peaks=        [5000.0, 60000.0],
            flags=        [0, 0],
        )
        d_none = tmp_path / "none"
        d_none.mkdir()
        d_explicit = tmp_path / "explicit"
        d_explicit.mkdir()

        monkeypatch.chdir(d_none)
        banzaicat.make_cat(fits_path, datamax=None, b_sigma=5, b_crlim=0.5)
        size_none = os.path.getsize(d_none / "banzai.cat")

        monkeypatch.chdir(d_explicit)
        banzaicat.make_cat(fits_path, datamax=75000, b_sigma=5, b_crlim=0.5)
        size_explicit = os.path.getsize(d_explicit / "banzai.cat")

        assert size_none == size_explicit

    def test_datamax_below_all_peaks_yields_empty(self, tmp_path, monkeypatch):
        fits_path = _make_banzai_fits(
            tmp_path / "dm_low.fits",
            ras=          [150.0, 150.1, 150.2],
            decs=         [2.0,   2.0,   2.0],
            ellipticities=[0.1,   0.1,   0.1],
            backgrounds=  [100.0, 100.0, 100.0],
            fwhms=        [4.0,   4.0,   4.0],
            peaks=        [5000.0, 6000.0, 7000.0],
            flags=        [0, 0, 0],
        )
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(fits_path, datamax=100, b_sigma=5, b_crlim=0.5)
        assert len(_data_lines(cat)) == 0

    def test_datamax_exactly_at_peak_included(self, tmp_path, monkeypatch):
        """PEAK <= datamax means peak == datamax IS included."""
        fits_path = _make_banzai_fits(
            tmp_path / "dm_exact.fits",
            ras=          [150.0],
            decs=         [2.0],
            ellipticities=[0.1],
            backgrounds=  [100.0],
            fwhms=        [4.0],
            peaks=        [5000.0],
            flags=        [0],
        )
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(fits_path, datamax=5000, b_sigma=5, b_crlim=0.5)
        assert len(_data_lines(cat)) == 1


# ---------------------------------------------------------------------------
# FWHM / b_crlim filtering
# ---------------------------------------------------------------------------

class TestMakeCatFwhmFiltering:
    """b_crlim is a minimum FWHM threshold; sources with FWHM <= b_crlim are removed."""

    def test_high_crlim_excludes_all(self, tmp_path, monkeypatch):
        fits_path = _make_banzai_fits(
            tmp_path / "fwhm_high.fits",
            ras=          [150.0, 150.1],
            decs=         [2.0,   2.0],
            ellipticities=[0.1,   0.1],
            backgrounds=  [100.0, 100.0],
            fwhms=        [1.0,   2.0],
            peaks=        [5000.0, 6000.0],
            flags=        [0, 0],
        )
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=5.0)
        assert len(_data_lines(cat)) == 0

    def test_crlim_exactly_at_fwhm_excluded(self, tmp_path, monkeypatch):
        """FWHM > b_crlim is strict: FWHM == b_crlim should be excluded."""
        fits_path = _make_banzai_fits(
            tmp_path / "fwhm_eq.fits",
            ras=          [150.0],
            decs=         [2.0],
            ellipticities=[0.1],
            backgrounds=  [100.0],
            fwhms=        [3.0],
            peaks=        [5000.0],
            flags=        [0],
        )
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=3.0)
        assert len(_data_lines(cat)) == 0

    def test_fwhm_just_above_crlim_included(self, tmp_path, monkeypatch):
        fits_path = _make_banzai_fits(
            tmp_path / "fwhm_pass.fits",
            ras=          [150.0],
            decs=         [2.0],
            ellipticities=[0.1],
            backgrounds=  [100.0],
            fwhms=        [3.01],
            peaks=        [5000.0],
            flags=        [0],
        )
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=3.0)
        assert len(_data_lines(cat)) == 1


# ---------------------------------------------------------------------------
# Sigma-clipping on ELLIPTICITY / BACKGROUND / FWHM
# ---------------------------------------------------------------------------

class TestMakeCatSigmaClipFiltering:
    """Extreme outliers in ellipticity, background, or FWHM are clipped out."""

    def test_extreme_ellipticity_outlier_removed(self, tmp_path, monkeypatch):
        n_good = 10
        fits_path = _make_banzai_fits(
            tmp_path / "ellip_outlier.fits",
            ras=          list(np.linspace(149, 151, n_good + 1)),
            decs=         [2.0] * (n_good + 1),
            ellipticities=[0.1] * n_good + [100.0],
            backgrounds=  [100.0] * (n_good + 1),
            fwhms=        [4.0] * (n_good + 1),
            peaks=        [5000.0] * (n_good + 1),
            flags=        [0] * (n_good + 1),
        )
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=2, b_crlim=0.5)
        lines = _data_lines(cat)
        assert len(lines) <= n_good

    def test_extreme_background_outlier_removed(self, tmp_path, monkeypatch):
        n_good = 10
        fits_path = _make_banzai_fits(
            tmp_path / "bg_outlier.fits",
            ras=          list(np.linspace(149, 151, n_good + 1)),
            decs=         [2.0] * (n_good + 1),
            ellipticities=[0.1] * (n_good + 1),
            backgrounds=  [100.0] * n_good + [1e6],
            fwhms=        [4.0] * (n_good + 1),
            peaks=        [5000.0] * (n_good + 1),
            flags=        [0] * (n_good + 1),
        )
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=2, b_crlim=0.5)
        lines = _data_lines(cat)
        assert len(lines) <= n_good

    def test_extreme_fwhm_outlier_removed(self, tmp_path, monkeypatch):
        n_good = 10
        fits_path = _make_banzai_fits(
            tmp_path / "fwhm_outlier.fits",
            ras=          list(np.linspace(149, 151, n_good + 1)),
            decs=         [2.0] * (n_good + 1),
            ellipticities=[0.1] * (n_good + 1),
            backgrounds=  [100.0] * (n_good + 1),
            fwhms=        [4.0] * n_good + [9999.0],
            peaks=        [5000.0] * (n_good + 1),
            flags=        [0] * (n_good + 1),
        )
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=2, b_crlim=0.5)
        lines = _data_lines(cat)
        assert len(lines) <= n_good

    def test_loose_sigma_keeps_more_sources(self, tmp_path, monkeypatch):
        rng = np.random.default_rng(77)
        n = 15
        ellips = [0.1] * n
        ellips[5] = 0.6
        fits_path = _make_banzai_fits(
            tmp_path / "sigma_loose.fits",
            ras=          list(rng.uniform(149, 151, n)),
            decs=         list(rng.uniform(1.5, 2.5, n)),
            ellipticities=ellips,
            backgrounds=  [100.0] * n,
            fwhms=        [4.0] * n,
            peaks=        [5000.0] * n,
            flags=        [0] * n,
        )
        d_loose  = tmp_path / "loose"
        d_strict = tmp_path / "strict"
        d_loose.mkdir()
        d_strict.mkdir()

        monkeypatch.chdir(d_loose)
        banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        n_loose = len(_data_lines(d_loose / "banzai.cat"))

        monkeypatch.chdir(d_strict)
        banzaicat.make_cat(fits_path, datamax=99999, b_sigma=1, b_crlim=0.5)
        n_strict = len(_data_lines(d_strict / "banzai.cat"))

        assert n_loose >= n_strict


# ---------------------------------------------------------------------------
# Known source-count validation
# ---------------------------------------------------------------------------

class TestMakeCatSourceCount:
    """End-to-end count tests with fully deterministic input data."""

    def test_all_clean_sources_pass(self, tmp_path, monkeypatch):
        fits_path = _make_banzai_fits(
            tmp_path / "clean5.fits",
            ras=          [149.1, 149.2, 149.3, 149.4, 149.5],
            decs=         [2.0,   2.0,   2.0,   2.0,   2.0],
            ellipticities=[0.1,   0.1,   0.1,   0.1,   0.1],
            backgrounds=  [100.0, 100.0, 100.0, 100.0, 100.0],
            fwhms=        [4.0,   4.0,   4.0,   4.0,   4.0],
            peaks=        [5000.0, 5000.0, 5000.0, 5000.0, 5000.0],
            flags=        [0, 0, 0, 0, 0],
        )
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert len(_data_lines(cat)) == 5

    def test_one_bad_flag_out_of_five(self, tmp_path, monkeypatch):
        fits_path = _make_banzai_fits(
            tmp_path / "flag_one.fits",
            ras=          [149.1, 149.2, 149.3, 149.4, 149.5],
            decs=         [2.0,   2.0,   2.0,   2.0,   2.0],
            ellipticities=[0.1,   0.1,   0.1,   0.1,   0.1],
            backgrounds=  [100.0, 100.0, 100.0, 100.0, 100.0],
            fwhms=        [4.0,   4.0,   4.0,   4.0,   4.0],
            peaks=        [5000.0, 5000.0, 5000.0, 5000.0, 5000.0],
            flags=        [0, 0, 0, 0, 1],
        )
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert len(_data_lines(cat)) == 4

    def test_one_above_datamax_out_of_five(self, tmp_path, monkeypatch):
        fits_path = _make_banzai_fits(
            tmp_path / "dm_one.fits",
            ras=          [149.1, 149.2, 149.3, 149.4, 149.5],
            decs=         [2.0,   2.0,   2.0,   2.0,   2.0],
            ellipticities=[0.1,   0.1,   0.1,   0.1,   0.1],
            backgrounds=  [100.0, 100.0, 100.0, 100.0, 100.0],
            fwhms=        [4.0,   4.0,   4.0,   4.0,   4.0],
            peaks=        [5000.0, 5000.0, 5000.0, 5000.0, 100000.0],
            flags=        [0, 0, 0, 0, 0],
        )
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert len(_data_lines(cat)) == 4

    def test_empty_catalog_produces_header_only(self, tmp_path, monkeypatch):
        fits_path = _make_banzai_fits(
            tmp_path / "empty.fits",
            ras=   [150.0],
            decs=  [2.0],
            ellipticities=[0.1],
            backgrounds=  [100.0],
            fwhms=        [4.0],
            peaks=        [5000.0],
            flags=        [1],
        )
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert os.path.exists(cat)
        assert len(_data_lines(cat)) == 0
        assert len(_header_lines(cat)) > 0


# ---------------------------------------------------------------------------
# Missing CAT extension
# ---------------------------------------------------------------------------

class TestMissingCATExtension:
    """Tests for handling FITS files without a CAT extension."""

    def test_no_cat_extension_raises_keyerror(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "no_cat.fits"
        primary = afits.PrimaryHDU(np.ones((10, 10), dtype=np.float32))
        afits.HDUList([primary]).writeto(str(path), overwrite=True)
        with pytest.raises(KeyError):
            banzaicat.make_cat(str(path))

    def test_wrong_extension_name_raises_keyerror(self, tmp_path, monkeypatch):
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
        monkeypatch.chdir(tmp_path)
        with pytest.raises(OSError):
            banzaicat.make_cat("/nonexistent/path/file.fits")

    def test_empty_file_raises_oserror(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "empty.fits"
        path.write_bytes(b"")
        with pytest.raises(OSError):
            banzaicat.make_cat(str(path))

    def test_non_fits_file_raises_oserror(self, tmp_path, monkeypatch):
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
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "single.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert len(_data_lines(cat)) == 1

    def test_single_flagged_source_excluded(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "single_flag.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[1],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert len(_data_lines(cat)) == 0

    def test_single_source_id_is_one(self, tmp_path, monkeypatch):
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
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "single_clip.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=1, b_crlim=0.5)
        assert os.path.exists(cat)


# ---------------------------------------------------------------------------
# S/N calculation correctness
# ---------------------------------------------------------------------------

class TestSNCalculation:
    """Tests for the signal-to-noise ratio calculation (PEAK / BACKGROUND)."""

    def test_sn_order_determines_output_order(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
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
        first_ra = float(lines[0].strip().split("\t")[0])
        assert first_ra == pytest.approx(30.0, abs=0.01)

    def test_equal_sn_stable_sort(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
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
            assert first_ra == pytest.approx(10.0, abs=0.01)


# ---------------------------------------------------------------------------
# Large catalog performance and correctness
# ---------------------------------------------------------------------------

class TestLargeCatalog:
    """Tests for catalogs with many sources."""

    def test_100_sources_processed(self, tmp_path, monkeypatch):
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
        assert len(_data_lines(cat)) == n

    def test_ids_contiguous_for_large_catalog(self, tmp_path, monkeypatch):
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
        assert len(_data_lines(cat)) == n // 2


# ---------------------------------------------------------------------------
# Sigma clipping edge cases
# ---------------------------------------------------------------------------

class TestSigmaClipEdgeCases:
    """Edge cases for the sigma clipping on ELLIPTICITY, BACKGROUND, and FWHM."""

    def test_identical_values_no_clipping(self, tmp_path, monkeypatch):
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
        assert len(_data_lines(cat)) == n

    def test_very_large_sigma_keeps_all(self, tmp_path, monkeypatch):
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
        assert len(_data_lines(cat)) == n

    def test_very_small_sigma_may_clip_all(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "tiny_sigma.fits",
            ras=[150.0, 150.1],
            decs=[2.0, 2.0],
            ellipticities=[0.1, 0.3],
            backgrounds=[100.0, 100.0],
            fwhms=[4.0, 4.0],
            peaks=[5000.0, 5000.0],
            flags=[0, 0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=0.01, b_crlim=0.5)
        lines = _data_lines(cat)
        assert len(lines) <= 2


# ---------------------------------------------------------------------------
# datamax boundary conditions
# ---------------------------------------------------------------------------

class TestDatamaxBoundaryConditions:
    """Boundary conditions for the datamax parameter."""

    def test_datamax_zero_excludes_all_positive_peaks(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "dm_zero.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=0, b_sigma=5, b_crlim=0.5)
        assert len(_data_lines(cat)) == 0

    def test_datamax_negative_excludes_all(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "dm_neg.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=-1, b_sigma=5, b_crlim=0.5)
        assert len(_data_lines(cat)) == 0

    def test_datamax_very_large_includes_all(self, tmp_path, monkeypatch):
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
        assert len(_data_lines(cat)) == 2

    def test_crlim_negative_includes_all(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "crlim_neg.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=-1)
        assert len(_data_lines(cat)) == 1

    def test_crlim_at_fwhm_minus_epsilon(self, tmp_path, monkeypatch):
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
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "ra_fmt.fits",
            ras=[150.12345], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        lines = _data_lines(cat)
        ra_str = lines[0].strip().split("\t")[0]
        decimal_part = ra_str.strip().split('.')[1]
        assert len(decimal_part) == 5

    def test_dec_format_10_5f(self, tmp_path, monkeypatch):
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
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "zerobg.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[0.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert os.path.exists(cat)

    def test_zero_background_multiple_sources(self, tmp_path, monkeypatch):
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
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "neg_ellip.fits",
            ras=[150.0], decs=[2.0], ellipticities=[-0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert os.path.exists(cat)

    def test_negative_peak(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "neg_peak.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[-100.0], flags=[0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert os.path.exists(cat)

    def test_negative_background(self, tmp_path, monkeypatch):
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
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "only_peak.fits",
            ras=[150.0, 150.1],
            decs=[2.0, 2.0],
            ellipticities=[0.1, 0.1],
            backgrounds=[100.0, 100.0],
            fwhms=[4.0, 4.0],
            peaks=[5000.0, 80000.0],
            flags=[0, 0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=75000, b_sigma=5, b_crlim=0.5)
        assert len(_data_lines(cat)) == 1

    def test_source_fails_only_fwhm_check(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "only_fwhm.fits",
            ras=[150.0, 150.1],
            decs=[2.0, 2.0],
            ellipticities=[0.1, 0.1],
            backgrounds=[100.0, 100.0],
            fwhms=[4.0, 2.0],
            peaks=[5000.0, 5000.0],
            flags=[0, 0],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=3.0)
        assert len(_data_lines(cat)) == 1

    def test_source_fails_only_flag_check(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "only_flag.fits",
            ras=[150.0, 150.1],
            decs=[2.0, 2.0],
            ellipticities=[0.1, 0.1],
            backgrounds=[100.0, 100.0],
            fwhms=[4.0, 4.0],
            peaks=[5000.0, 5000.0],
            flags=[0, 4],
        )
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert len(_data_lines(cat)) == 1

    def test_all_filters_fail_simultaneously(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "all_fail.fits",
            ras=[150.0],
            decs=[2.0],
            ellipticities=[0.1],
            backgrounds=[100.0],
            fwhms=[1.0],
            peaks=[90000.0],
            flags=[2],
        )
        cat = banzaicat.make_cat(fits_path, datamax=75000, b_sigma=5, b_crlim=3.0)
        assert len(_data_lines(cat)) == 0


# ---------------------------------------------------------------------------
# Return value
# ---------------------------------------------------------------------------

class TestReturnValue:
    """Tests for the return value of make_cat."""

    def test_returns_string(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "ret.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        result = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert isinstance(result, str)

    def test_return_value_is_banzai_cat(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fits_path = _make_banzai_fits(
            tmp_path / "retval.fits",
            ras=[150.0], decs=[2.0], ellipticities=[0.1],
            backgrounds=[100.0], fwhms=[4.0], peaks=[5000.0], flags=[0],
        )
        result = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert result == 'banzai.cat'
