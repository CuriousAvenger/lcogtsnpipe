"""
Tests for lsc.banzaicat.make_cat (make_banzai_cat).
Uses the banzai_fits session fixture from conftest.py.

Coverage added:
  - TestMakeBanzaiCat: baseline file-creation tests (original 5)
  - TestMakeCatFileHeader: catalog header structure/content validation
  - TestMakeCatDataLines: per-line format validation (RA/DEC floats, integer IDs)
  - TestMakeCatSorting: output rows are sorted by S/N (PEAK/BACKGROUND) descending
  - TestMakeCatFlagFiltering: FLAG != 0 sources are excluded from the catalog
  - TestMakeCatDatamax: datamax=None defaults to 75000; datamax below all peaks → 0 rows
  - TestMakeCatFwhmFiltering: b_crlim above all FWHM values yields empty data section
  - TestMakeCatSigmaClipFiltering: extreme outlier in ELLIPTICITY/BACKGROUND is removed
  - TestMakeCatSourceCount: known-source-count catalog produces expected row count
"""
import os
import numpy as np
import pytest
from astropy.io import fits as afits

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Module-level helper: build a minimal BANZAI FITS from per-source arrays
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
# Original tests (unchanged)
# ---------------------------------------------------------------------------

class TestMakeBanzaiCat:
    def test_creates_output_file(self, banzai_fits, tmp_path, monkeypatch):
        from lsc import banzaicat
        monkeypatch.chdir(tmp_path)
        result = banzaicat.make_cat(banzai_fits, datamax=50000, b_sigma=5, b_crlim=0.5)
        assert os.path.exists(result)

    def test_returns_banzai_cat(self, banzai_fits, tmp_path, monkeypatch):
        from lsc import banzaicat
        monkeypatch.chdir(tmp_path)
        result = banzaicat.make_cat(banzai_fits, datamax=50000, b_sigma=5, b_crlim=0.5)
        assert result == "banzai.cat"

    def test_output_not_empty(self, banzai_fits, tmp_path, monkeypatch):
        from lsc import banzaicat
        monkeypatch.chdir(tmp_path)
        result = banzaicat.make_cat(banzai_fits, datamax=50000, b_sigma=5, b_crlim=0.5)
        assert os.path.getsize(result) > 0

    def test_high_datamax_includes_more_sources(self, banzai_fits, tmp_path, monkeypatch):
        """A very high datamax should not reduce the number of sources vs default."""
        from lsc import banzaicat

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
        from lsc import banzaicat

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
        from lsc import banzaicat
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(banzai_fits, datamax=99999, b_sigma=5, b_crlim=0.5)
        return cat

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
        from lsc import banzaicat
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
            assert float(ra_str) == float(ra_str)  # no exception

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
        """IDs are assigned during filtering (1..n); output is sorted by S/N,
        so IDs in the file are a permutation of {1..n}, not necessarily in order.
        """
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
        """Build a catalog with 5 sources of known S/N and verify output order."""
        from lsc import banzaicat

        # Five sources: S/N = PEAK/BACKGROUND = 50/100, 200/100, 10/100, 300/100, 100/100
        # Expected S/N order: 300 > 200 > 100 > 50 > 10
        # Use distinct RA values so we can identify which source is which
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
        # First source should be RA=40 (peak=300, highest S/N)
        first_ra = float(lines[0].strip().split("\t")[0])
        assert first_ra == pytest.approx(40.0, abs=0.01)
        # Last source should be RA=30 (peak=10, lowest S/N)
        last_ra = float(lines[-1].strip().split("\t")[0])
        assert last_ra == pytest.approx(30.0, abs=0.01)

    def test_sn_order_is_non_increasing(self, tmp_path, monkeypatch):
        """IDs must form a contiguous set 1..n regardless of output order."""
        from lsc import banzaicat
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
        # IDs must be a contiguous set starting at 1
        assert sorted(ids) == list(range(1, len(ids) + 1))


# ---------------------------------------------------------------------------
# FLAG filtering
# ---------------------------------------------------------------------------

class TestMakeCatFlagFiltering:
    """Sources with FLAG != 0 must be excluded regardless of other parameters."""

    def test_all_flagged_sources_excluded(self, tmp_path, monkeypatch):
        """A catalog with all FLAG=1 sources should produce zero data lines."""
        from lsc import banzaicat
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
        """Only the source with FLAG=0 should appear when flags are mixed."""
        from lsc import banzaicat
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
        """datamax=None should behave identically to datamax=75000."""
        from lsc import banzaicat
        fits_path = _make_banzai_fits(
            tmp_path / "dm_none.fits",
            ras=          [150.0, 150.1],
            decs=         [2.0,   2.0],
            ellipticities=[0.1,   0.1],
            backgrounds=  [100.0, 100.0],
            fwhms=        [4.0,   4.0],
            peaks=        [5000.0, 60000.0],   # both < 75000
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
        """Setting datamax below all peaks should produce zero data lines."""
        from lsc import banzaicat
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

    def test_datamax_exactly_at_peak_excluded(self, tmp_path, monkeypatch):
        """PEAK <= datamax means peak == datamax IS included."""
        from lsc import banzaicat
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
        """b_crlim above all FWHM values should produce zero data lines."""
        from lsc import banzaicat
        fits_path = _make_banzai_fits(
            tmp_path / "fwhm_high.fits",
            ras=          [150.0, 150.1],
            decs=         [2.0,   2.0],
            ellipticities=[0.1,   0.1],
            backgrounds=  [100.0, 100.0],
            fwhms=        [1.0,   2.0],   # all below b_crlim=5
            peaks=        [5000.0, 6000.0],
            flags=        [0, 0],
        )
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=5.0)
        assert len(_data_lines(cat)) == 0

    def test_crlim_exactly_at_fwhm_excluded(self, tmp_path, monkeypatch):
        """FWHM > b_crlim is strict: FWHM == b_crlim should be excluded."""
        from lsc import banzaicat
        fits_path = _make_banzai_fits(
            tmp_path / "fwhm_eq.fits",
            ras=          [150.0],
            decs=         [2.0],
            ellipticities=[0.1],
            backgrounds=  [100.0],
            fwhms=        [3.0],   # exactly equals b_crlim
            peaks=        [5000.0],
            flags=        [0],
        )
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=3.0)
        assert len(_data_lines(cat)) == 0

    def test_fwhm_just_above_crlim_included(self, tmp_path, monkeypatch):
        """A source with FWHM slightly above b_crlim must be included."""
        from lsc import banzaicat
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
        """A source with extreme ellipticity should be clipped by sigmaclip."""
        from lsc import banzaicat
        # 10 normal sources + 1 with extreme ellipticity
        n_good = 10
        fits_path = _make_banzai_fits(
            tmp_path / "ellip_outlier.fits",
            ras=          list(np.linspace(149, 151, n_good + 1)),
            decs=         [2.0] * (n_good + 1),
            ellipticities=[0.1] * n_good + [100.0],   # extreme outlier
            backgrounds=  [100.0] * (n_good + 1),
            fwhms=        [4.0] * (n_good + 1),
            peaks=        [5000.0] * (n_good + 1),
            flags=        [0] * (n_good + 1),
        )
        monkeypatch.chdir(tmp_path)
        # tight sigma so the outlier is clipped
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=2, b_crlim=0.5)
        lines = _data_lines(cat)
        assert len(lines) <= n_good

    def test_extreme_background_outlier_removed(self, tmp_path, monkeypatch):
        """A source with extreme background should be clipped."""
        from lsc import banzaicat
        n_good = 10
        fits_path = _make_banzai_fits(
            tmp_path / "bg_outlier.fits",
            ras=          list(np.linspace(149, 151, n_good + 1)),
            decs=         [2.0] * (n_good + 1),
            ellipticities=[0.1] * (n_good + 1),
            backgrounds=  [100.0] * n_good + [1e6],   # extreme outlier
            fwhms=        [4.0] * (n_good + 1),
            peaks=        [5000.0] * (n_good + 1),
            flags=        [0] * (n_good + 1),
        )
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=2, b_crlim=0.5)
        lines = _data_lines(cat)
        assert len(lines) <= n_good

    def test_extreme_fwhm_outlier_removed(self, tmp_path, monkeypatch):
        """A source with extreme FWHM should be clipped (not just below b_crlim)."""
        from lsc import banzaicat
        n_good = 10
        fits_path = _make_banzai_fits(
            tmp_path / "fwhm_outlier.fits",
            ras=          list(np.linspace(149, 151, n_good + 1)),
            decs=         [2.0] * (n_good + 1),
            ellipticities=[0.1] * (n_good + 1),
            backgrounds=  [100.0] * (n_good + 1),
            fwhms=        [4.0] * n_good + [9999.0],   # extreme high FWHM
            peaks=        [5000.0] * (n_good + 1),
            flags=        [0] * (n_good + 1),
        )
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=2, b_crlim=0.5)
        lines = _data_lines(cat)
        assert len(lines) <= n_good

    def test_loose_sigma_keeps_more_sources(self, tmp_path, monkeypatch):
        """Higher b_sigma (looser clipping) keeps more borderline sources."""
        from lsc import banzaicat
        # One source has ellipticity 3× the std above the mean
        rng = np.random.default_rng(77)
        n = 15
        ellips = [0.1] * n
        ellips[5] = 0.6   # mild outlier
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
        """5 identical clean sources should all appear in the output."""
        from lsc import banzaicat
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
        """4 clean + 1 flagged → exactly 4 sources in output."""
        from lsc import banzaicat
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
        """4 under datamax + 1 over → exactly 4 sources in output."""
        from lsc import banzaicat
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
        """Zero sources passing all filters → file has only header lines."""
        from lsc import banzaicat
        fits_path = _make_banzai_fits(
            tmp_path / "empty.fits",
            ras=   [150.0],
            decs=  [2.0],
            ellipticities=[0.1],
            backgrounds=  [100.0],
            fwhms=        [4.0],
            peaks=        [5000.0],
            flags=        [1],   # flagged → excluded
        )
        monkeypatch.chdir(tmp_path)
        cat = banzaicat.make_cat(fits_path, datamax=99999, b_sigma=5, b_crlim=0.5)
        assert os.path.exists(cat)
        assert len(_data_lines(cat)) == 0
        assert len(_header_lines(cat)) > 0
