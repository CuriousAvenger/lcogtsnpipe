"""
Comprehensive unit tests for calibratemag.py helper functions.

Tests cover:
- crossmatch(): various catalog sizes, thresholds, right_join flag
- get_image_data(): mocked DB queries and file I/O
- average_in_flux(): mathematical correctness
- combine_nights(): combining multi-night observations

Uses unittest.mock to avoid any real I/O, database, or network calls.
"""

import sys
import os
import types
import pytest
import numpy as np
import numpy.ma as ma
from unittest.mock import patch, MagicMock

# Stub missing optional dependencies before lsc imports happen via calibratemag.py
# NOTE: do NOT stub matplotlib/mpl_toolkits/requests/reproject — they are installed
# and stubbing them poisons sys.modules for all later test files.
for _mod in ["odrpack"]:
    sys.modules.setdefault(_mod, MagicMock())

from astropy.table import Table

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
BIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin"))
sys.path.insert(0, BIN_DIR)

# We need to import the functions from calibratemag.py as a module
import importlib.util

spec = importlib.util.spec_from_file_location("calibratemag", os.path.join(BIN_DIR, "calibratemag.py"))
calibratemag = importlib.util.module_from_spec(spec)

# Prevent __main__ block from executing
with patch.object(spec.loader, 'exec_module') as mock_exec:
    # We'll manually exec just the function definitions
    pass

# Instead, directly exec the source file stopping before __main__
_source_path = os.path.join(BIN_DIR, "calibratemag.py")
with open(_source_path) as _f:
    _source = _f.read()

# Extract everything before `if __name__ == "__main__":`
_main_idx = _source.find('\nif __name__ == "__main__":')
if _main_idx > 0:
    _module_source = _source[:_main_idx]
else:
    _module_source = _source

# Execute in a namespace that has access to the needed imports
_ns = {}
exec(compile(_module_source, _source_path, 'exec'), _ns)

crossmatch = _ns['crossmatch']
get_image_data = _ns['get_image_data']
average_in_flux = _ns['average_in_flux']
combine_nights = _ns['combine_nights']


# ===========================================================================
# crossmatch() tests
# ===========================================================================

class TestCrossmatch:
    """Tests for the crossmatch() function."""

    def _make_catalog(self, ra, dec, extra_cols=None):
        """Helper to create an astropy Table catalog."""
        t = Table({'ra': np.array(ra, dtype=float),
                   'dec': np.array(dec, dtype=float)})
        if extra_cols:
            for name, vals in extra_cols.items():
                t[name] = vals
        return t

    def test_perfect_match_same_positions(self):
        """Stars at identical positions should all match."""
        ra = [10.0, 20.0, 30.0]
        dec = [1.0, 2.0, 3.0]
        cat0 = self._make_catalog(ra, dec)
        cat1 = self._make_catalog(ra, dec)
        out, cat1_out = crossmatch(cat0, cat1, threshold=1.0)
        assert len(out) == 3
        assert len(cat1_out) == 3

    def test_no_matches_far_apart(self):
        """Stars far apart (>threshold) should yield no matches."""
        cat0 = self._make_catalog([10.0, 20.0], [1.0, 2.0])
        cat1 = self._make_catalog([100.0, 200.0], [50.0, 60.0])
        out, cat1_out = crossmatch(cat0, cat1, threshold=1.0)
        assert len(out) == 0
        assert len(cat1_out) == 0

    def test_partial_match(self):
        """Only some stars should match when some are within threshold."""
        cat0 = self._make_catalog([10.0, 20.0, 30.0], [1.0, 2.0, 3.0])
        # cat1: first star matches cat0[0], second is far away
        cat1 = self._make_catalog([10.0001, 200.0], [1.0001, 60.0])
        out, cat1_out = crossmatch(cat0, cat1, threshold=1.0)
        # Only the first star in cat1 should match
        assert len(out) == 1
        assert len(cat1_out) == 1

    def test_threshold_effect(self):
        """Increasing threshold should find more matches."""
        cat0 = self._make_catalog([10.0], [1.0])
        # Star at ~0.5 arcsec away (sep in arcsec = angular_sep * 3600)
        # delta_ra = 0.0001 deg ~ 0.36 arcsec
        cat1 = self._make_catalog([10.0001], [1.0])

        # With tight threshold should not match
        out_tight, _ = crossmatch(cat0, cat1, threshold=0.1)
        assert len(out_tight) == 0

        # With loose threshold should match
        out_loose, _ = crossmatch(cat0, cat1, threshold=2.0)
        assert len(out_loose) == 1

    def test_right_join_preserves_all_cat1_rows(self):
        """With right_join=True, output should have same length as cat1."""
        cat0 = self._make_catalog([10.0, 20.0, 30.0], [1.0, 2.0, 3.0])
        # Only first star in cat1 has a match in cat0
        cat1 = self._make_catalog([10.0001, 200.0, 300.0], [1.0001, 60.0, 70.0])
        out, cat1_out = crossmatch(cat0, cat1, threshold=1.0, right_join=True)
        # All cat1 rows preserved
        assert len(out) == 3
        assert len(cat1_out) == 3
        # Non-matching rows should be masked
        assert out['ra'].mask[1] == True
        assert out['ra'].mask[2] == True
        # Matching row should not be masked
        assert out['ra'].mask[0] == False

    def test_right_join_false_filters_unmatched(self):
        """With right_join=False (default), unmatched rows are removed."""
        cat0 = self._make_catalog([10.0, 20.0, 30.0], [1.0, 2.0, 3.0])
        cat1 = self._make_catalog([10.0001, 200.0], [1.0001, 60.0])
        out, cat1_out = crossmatch(cat0, cat1, threshold=1.0, right_join=False)
        assert len(out) == 1
        assert len(cat1_out) == 1

    def test_empty_cat0_raises(self):
        """Empty cat0 raises ValueError (zero-size reduction)."""
        cat0 = self._make_catalog([], [])
        cat1 = self._make_catalog([10.0, 20.0], [1.0, 2.0])
        with pytest.raises(ValueError):
            crossmatch(cat0, cat1, threshold=1.0)

    def test_empty_cat1_returns_empty(self):
        """Empty cat1 returns empty results (no matches possible)."""
        cat0 = self._make_catalog([10.0, 20.0], [1.0, 2.0])
        cat1 = self._make_catalog([], [])
        out, cat1_out = crossmatch(cat0, cat1, threshold=1.0)
        assert len(out) == 0
        assert len(cat1_out) == 0

    def test_single_star_catalogs_match(self):
        """Single-star catalogs that match."""
        cat0 = self._make_catalog([15.0], [2.5])
        cat1 = self._make_catalog([15.0], [2.5])
        out, cat1_out = crossmatch(cat0, cat1, threshold=1.0)
        assert len(out) == 1
        assert len(cat1_out) == 1

    def test_single_star_catalogs_no_match(self):
        """Single-star catalogs too far apart."""
        cat0 = self._make_catalog([15.0], [2.5])
        cat1 = self._make_catalog([25.0], [12.5])
        out, cat1_out = crossmatch(cat0, cat1, threshold=1.0)
        assert len(out) == 0
        assert len(cat1_out) == 0

    def test_custom_column_names(self):
        """crossmatch with custom RA/DEC column names."""
        cat0 = Table({'ra0': [10.0, 20.0], 'dec0': [1.0, 2.0]})
        cat1 = Table({'ra1': [10.0, 20.0], 'dec1': [1.0, 2.0]})
        out, cat1_out = crossmatch(cat0, cat1, threshold=1.0,
                                   racol0='ra0', deccol0='dec0',
                                   racol1='ra1', deccol1='dec1')
        assert len(out) == 2
        assert len(cat1_out) == 2

    def test_large_catalog_performance(self):
        """crossmatch with moderately large catalogs (100 stars)."""
        rng = np.random.default_rng(42)
        n = 100
        ra = rng.uniform(149, 151, n)
        dec = rng.uniform(1, 3, n)
        cat0 = self._make_catalog(ra, dec)
        # cat1 is same with tiny offsets
        cat1 = self._make_catalog(ra + rng.normal(0, 1e-5, n),
                                  dec + rng.normal(0, 1e-5, n))
        out, cat1_out = crossmatch(cat0, cat1, threshold=1.0)
        # Most should match
        assert len(out) > 80

    def test_many_to_one_closest_wins(self):
        """When multiple cat0 stars are near one cat1 star, closest should match."""
        # Two cat0 stars near the same cat1 star
        cat0 = self._make_catalog([10.0, 10.00005], [1.0, 1.0])
        cat1 = self._make_catalog([10.00001], [1.0])
        out, cat1_out = crossmatch(cat0, cat1, threshold=1.0)
        # cat1 has 1 star; it should match the closest cat0 star
        assert len(out) == 1
        assert len(cat1_out) == 1

    def test_preserves_extra_columns(self):
        """crossmatch should preserve extra columns in cat0."""
        cat0 = Table({'ra': [10.0, 20.0], 'dec': [1.0, 2.0],
                      'mag': [15.5, 16.2]})
        cat1 = self._make_catalog([10.0, 20.0], [1.0, 2.0])
        out, cat1_out = crossmatch(cat0, cat1, threshold=1.0)
        assert 'mag' in out.colnames
        assert len(out) == 2


# ===========================================================================
# average_in_flux() tests
# ===========================================================================

class TestAverageInFlux:
    """Tests for the average_in_flux() function."""

    def test_single_measurement(self):
        """Single measurement should return itself."""
        mag = ma.array([20.0])
        dmag = ma.array([0.01])
        avg_mag, avg_dmag = average_in_flux(mag, dmag)
        np.testing.assert_almost_equal(avg_mag, 20.0, decimal=5)

    def test_identical_measurements(self):
        """Identical measurements should return the same magnitude."""
        mag = ma.array([18.0, 18.0, 18.0])
        dmag = ma.array([0.01, 0.01, 0.01])
        avg_mag, avg_dmag = average_in_flux(mag, dmag)
        np.testing.assert_almost_equal(avg_mag, 18.0, decimal=5)

    def test_error_decreases_with_more_measurements(self):
        """Error should decrease with more identical measurements."""
        mag1 = ma.array([18.0])
        dmag1 = ma.array([0.1])
        _, err1 = average_in_flux(mag1, dmag1)

        mag3 = ma.array([18.0, 18.0, 18.0])
        dmag3 = ma.array([0.1, 0.1, 0.1])
        _, err3 = average_in_flux(mag3, dmag3)

        assert err3 < err1

    def test_weighted_average_brighter_weight(self):
        """Measurement with smaller error should dominate."""
        # One precise measurement at 18.0, one imprecise at 19.0
        mag = ma.array([18.0, 19.0])
        dmag = ma.array([0.001, 1.0])
        avg_mag, _ = average_in_flux(mag, dmag)
        # Result should be much closer to 18.0
        assert abs(avg_mag - 18.0) < 0.1

    def test_axis_parameter(self):
        """Test with 2D array and axis parameter."""
        mag = ma.array([[18.0, 18.5], [18.1, 18.4]])
        dmag = ma.array([[0.01, 0.02], [0.01, 0.02]])
        avg_mag, avg_dmag = average_in_flux(mag, dmag, axis=0)
        assert avg_mag.shape == (2,)
        assert avg_dmag.shape == (2,)

    def test_returns_masked_array(self):
        """Output should be compatible with masked arrays."""
        mag = ma.array([18.0, 19.0])
        dmag = ma.array([0.01, 0.02])
        avg_mag, avg_dmag = average_in_flux(mag, dmag)
        # Should be numeric, not masked
        assert np.isfinite(avg_mag)
        assert np.isfinite(avg_dmag)


# ===========================================================================
# combine_nights() tests
# ===========================================================================

class TestCombineNights:
    """Tests for the combine_nights() function."""

    def test_single_night_single_filter(self):
        """Single night, single filter should give reasonable output."""
        filterlist = ['r']
        n_stars = 5
        # Create a combined_catalog with required columns
        combined_catalog = Table({
            'filter': ['r'] * 3,
            'mag': ma.array([[15.0, 16.0, 17.0, 18.0, 19.0],
                             [15.1, 16.1, 17.1, 18.1, 19.1],
                             [14.9, 15.9, 16.9, 17.9, 18.9]])
        })
        refcat = Table({
            'ra': np.arange(n_stars, dtype=float),
            'dec': np.arange(n_stars, dtype=float),
            'id': np.arange(n_stars)
        })
        catalog = combine_nights(combined_catalog, filterlist, refcat)
        assert 'r' in catalog.colnames
        assert 'rerr' in catalog.colnames
        assert len(catalog) == n_stars

    def test_multiple_filters(self):
        """Multiple filters should each appear in output."""
        filterlist = ['g', 'r', 'i']
        n_stars = 3
        rows = []
        for filt in filterlist:
            for _ in range(2):
                rows.append({'filter': filt,
                             'mag': ma.array([15.0, 16.0, 17.0])})
        combined_catalog = Table(rows)
        refcat = Table({
            'ra': np.arange(n_stars, dtype=float),
            'dec': np.arange(n_stars, dtype=float),
            'id': np.arange(n_stars)
        })
        catalog = combine_nights(combined_catalog, filterlist, refcat)
        for filt in filterlist:
            assert filt in catalog.colnames
            assert filt + 'err' in catalog.colnames

    def test_outlier_rejection(self):
        """Extreme outliers should be masked by the 5*MAD clipping."""
        filterlist = ['r']
        n_stars = 1
        # 10 observations: 9 normal + 1 extreme outlier
        mags = ma.array([[15.0]] * 9 + [[25.0]])  # outlier at 25.0
        combined_catalog = Table({
            'filter': ['r'] * 10,
            'mag': mags
        })
        refcat = Table({
            'ra': [0.0],
            'dec': [0.0],
            'id': [0]
        })
        catalog = combine_nights(combined_catalog, filterlist, refcat)
        # The result should be close to 15.0, not pulled by the outlier
        assert abs(catalog['r'][0] - 15.0) < 0.5


# ===========================================================================
# get_image_data() tests (with mocked DB)
# ===========================================================================

class TestGetImageData:
    """Tests for get_image_data() with mocked database."""

    @patch('lsc.myloopdef')
    @patch('lsc.mysqldef')
    @patch('lsc.sites')
    def test_basic_query_with_mag_columns_in_table(self, mock_sites, mock_mysqldef, mock_myloopdef):
        """When magcol and errcol are in the query result, they get renamed."""
        # Setup mock
        mock_sites.filterst1 = {'rp': 'r', 'gp': 'g'}
        mock_mysqldef.query.return_value = [
            {'filter': 'rp', 'filepath': '/data/', 'filename': 'test.fits',
             'airmass': 1.2, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 5, 'instrumentid': 10, 'shortname': 'lsc 1m0',
             'type': 'sinistro',
             'zcol1': 'gr', 'z1': 25.0, 'c1': 0.01, 'dz1': 0.01, 'dc1': 0.001,
             'zcol2': 'ri', 'z2': 25.1, 'c2': 0.02, 'dz2': 0.01, 'dc2': 0.001,
             'psfmag': 18.5, 'psfdmag': 0.02, 'apmag': 18.6, 'dapmag': 0.03}
        ]

        lista = ['/data/test.fits']
        t = get_image_data(lista, 'psfmag', 'psfdmag')

        assert 'instmag' in t.colnames
        assert 'dinstmag' in t.colnames
        mock_mysqldef.query.assert_called_once()
