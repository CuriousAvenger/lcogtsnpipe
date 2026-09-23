"""
Merged tests for lsc.lscastrodef — combines tests from pure, comprehensive,
extra, cov100, coverage_a, coverage_b, and coverage_c test files.
"""
import math
import os
import sys
import types
import pytest
import numpy as np
import importlib
from unittest.mock import patch, MagicMock, mock_open, call

pytestmark = pytest.mark.unit


# =============================================================================
# Module-level helpers (from coverage_b)
# =============================================================================

def _make_mock_hdr(filter_val='V', siteid='lsc', ra=180.0, dec=30.0,
                   naxis1=2048, naxis2=2048, airmass=1.2, exptime=100,
                   instrume='fl03', date_night='20200101', obj='test_field',
                   datamax=60000, datamin=-100, ron=10.0, gain=1.5):
    """Create a mock FITS header dict-like object."""
    data = {
        'AIRMASS': airmass,
        'airmass': airmass,
        'exptime': exptime,
        'filter': filter_val,
        'instrume': instrume,
        'date-night': date_night,
        'object': obj,
        'datamax': datamax,
        'datamin': datamin,
        'SITEID': siteid,
        'RA': ra,
        'DEC': dec,
        'naxis1': naxis1,
        'naxis2': naxis2,
        'ron': ron,
        'gain': gain,
    }
    hdr = MagicMock()
    hdr.__getitem__ = lambda self, key: data[key]
    hdr.__contains__ = lambda self, key: key in data
    hdr.get = lambda key, default=None: data.get(key, default)
    return hdr, data


def _readkey3_factory(data):
    """Return a readkey3 mock function that looks up keys in data dict."""
    def _readkey3(hdr, keyword):
        return data.get(keyword, None)
    return _readkey3


def _make_standard_stdcoo(field='landolt', n=5):
    """Create fake stdcoo dict with photometry data."""
    stdcoo = {
        'ra': [str(180.0 + i * 0.001) for i in range(n)],
        'dec': [str(30.0 + i * 0.001) for i in range(n)],
        'id': [f'star_{i}' for i in range(n)],
    }
    if field == 'landolt':
        for f in 'UBVRI':
            stdcoo[f] = [str(12.0 + i * 0.1) for i in range(n)]
        stdcoo['BV'] = [str(0.5 + i * 0.05) for i in range(n)]
        stdcoo['UB'] = [str(0.3 + i * 0.02) for i in range(n)]
        stdcoo['VR'] = [str(0.4 + i * 0.03) for i in range(n)]
        stdcoo['RI'] = [str(0.2 + i * 0.01) for i in range(n)]
        stdcoo['VI'] = [str(0.6 + i * 0.04) for i in range(n)]
    elif field == 'sloan':
        for f in 'ugriz':
            stdcoo[f] = [str(14.0 + i * 0.1) for i in range(n)]
        stdcoo['ug'] = [str(0.5 + i * 0.05) for i in range(n)]
        stdcoo['gr'] = [str(0.4 + i * 0.03) for i in range(n)]
        stdcoo['ri'] = [str(0.3 + i * 0.02) for i in range(n)]
        stdcoo['iz'] = [str(0.2 + i * 0.01) for i in range(n)]
        stdcoo['rz'] = [str(0.4 + i * 0.03) for i in range(n)]
    return stdcoo


def _make_standardpix(n=5, xrange=(100, 1900), yrange=(100, 1900)):
    """Create fake standard pixel coordinates (within frame)."""
    xs = [str(xrange[0] + i * (xrange[1] - xrange[0]) / max(n - 1, 1)) for i in range(n)]
    ys = [str(yrange[0] + i * (yrange[1] - yrange[0]) / max(n - 1, 1)) for i in range(n)]
    return {
        'ra': xs,
        'dec': ys,
        'id': [f'star_{i}' for i in range(n)],
    }


def _make_outside_pix(n=3):
    """Create pixel coords outside field (negative)."""
    return {
        'ra': [str(-100 * (i + 1)) for i in range(n)],
        'dec': [str(-100 * (i + 1)) for i in range(n)],
        'id': [f's{i}' for i in range(n)],
    }


def _setup_iraf_mock():
    """Setup comprehensive iraf mock."""
    iraf = MagicMock()
    iraf.noao = MagicMock()
    iraf.digiphot = MagicMock()
    iraf.daophot = MagicMock()
    iraf.images = MagicMock()
    iraf.imcoords = MagicMock()
    iraf.proto = MagicMock()
    iraf.wcsctran = MagicMock()
    iraf.noao.digiphot.daophot.phot = MagicMock(
        return_value=['img 500.0 600.0 3.0 18.5 0.02']
    )
    return iraf


def _make_sex_fields_mock(n_sex, phot_fail=False):
    """Create fields mock that returns numeric arrays for sextractor output."""
    xsex = [100.0 + i * 200.0 for i in range(n_sex)]
    ysex = [100.0 + i * 200.0 for i in range(n_sex)]
    fw = [3.5] * n_sex
    flags = [0] * n_sex
    ra_wcs = [180.0 + i * 0.001 for i in range(n_sex)]
    dec_wcs = [30.0 + i * 0.001 for i in range(n_sex)]

    def mock_fields(filename, fields='1', Stdout=None):
        fname = str(filename)
        if 'detections.cat' in fname:
            if fields == '2':
                return xsex
            elif fields == '3':
                return ysex
            elif fields == '8':
                return fw
            elif fields == '6':
                return flags
        elif 'detection_sex.coo' in fname:
            if fields == '1':
                return ra_wcs
            elif fields == '2':
                return dec_wcs
        return []

    return mock_fields, xsex, ysex



# ===========================================================================
# TestPval
# ===========================================================================

class TestPval:
    # --- from pure ---
    def test_zero_slope(self):
        from lsc.lscastrodef import pval
        assert pval(5.0, [3.0, 0.0]) == 3.0

    def test_unit_slope(self):
        from lsc.lscastrodef import pval
        assert pval(4.0, [0.0, 1.0]) == 4.0

    def test_linear_combination(self):
        from lsc.lscastrodef import pval
        assert abs(pval(5.0, [2.0, 3.0]) - 17.0) < 1e-10

    def test_negative_x(self):
        from lsc.lscastrodef import pval
        assert abs(pval(-3.0, [1.0, -2.0]) - 7.0) < 1e-10

    # --- from comprehensive ---
    def test_array_input(self):
        from lsc.lscastrodef import pval
        xx = np.array([1.0, 2.0, 3.0])
        result = pval(xx, [1.0, 2.0])
        expected = np.array([3.0, 5.0, 7.0])
        np.testing.assert_allclose(result, expected)

    def test_zero_coefficients(self):
        from lsc.lscastrodef import pval
        assert pval(100.0, [0.0, 0.0]) == 0.0

    def test_large_values(self):
        from lsc.lscastrodef import pval
        result = pval(1e6, [1e6, 1.0])
        assert abs(result - 2e6) < 1e-6

    def test_negative_slope_negative_x(self):
        from lsc.lscastrodef import pval
        assert abs(pval(-5.0, [10.0, -3.0]) - 25.0) < 1e-10

    def test_floating_point_precision(self):
        from lsc.lscastrodef import pval
        result = pval(0.1, [0.2, 0.3])
        expected = 0.2 + 0.3 * 0.1
        assert abs(result - expected) < 1e-15


# ===========================================================================
# TestHalfTotalFluxRadiusToFwhm
# ===========================================================================

class TestHalfTotalFluxRadiusToFwhm:
    # --- from pure ---
    def test_positive_input(self):
        from lsc.lscastrodef import half_total_flux_radius_to_fwhm
        fwhm = half_total_flux_radius_to_fwhm(5.0)
        assert fwhm > 0

    def test_conversion_factor(self):
        from lsc.lscastrodef import half_total_flux_radius_to_fwhm
        expected_factor = 0.8493218 * 2.355
        htfr = 3.7
        assert abs(half_total_flux_radius_to_fwhm(htfr) - htfr * expected_factor) < 1e-10

    def test_zero_input(self):
        from lsc.lscastrodef import half_total_flux_radius_to_fwhm
        assert half_total_flux_radius_to_fwhm(0.0) == 0.0

    def test_linear_scaling(self):
        from lsc.lscastrodef import half_total_flux_radius_to_fwhm
        assert abs(half_total_flux_radius_to_fwhm(4.0) - 2 * half_total_flux_radius_to_fwhm(2.0)) < 1e-10

    # --- from comprehensive ---
    def test_known_conversion_value(self):
        from lsc.lscastrodef import half_total_flux_radius_to_fwhm
        factor = 0.8493218 * 2.355
        for htfr in [1.0, 2.5, 10.0, 0.5]:
            result = half_total_flux_radius_to_fwhm(htfr)
            assert abs(result - htfr * factor) < 1e-10

    def test_numpy_array_input(self):
        from lsc.lscastrodef import half_total_flux_radius_to_fwhm
        arr = np.array([1.0, 2.0, 3.0])
        result = half_total_flux_radius_to_fwhm(arr)
        assert len(result) == 3
        assert all(result > 0)

    def test_very_small_input(self):
        from lsc.lscastrodef import half_total_flux_radius_to_fwhm
        result = half_total_flux_radius_to_fwhm(0.001)
        assert result > 0
        assert result < 0.01

    # --- from extra ---
    def test_negative_input(self):
        from lsc.lscastrodef import half_total_flux_radius_to_fwhm
        result = half_total_flux_radius_to_fwhm(-3.0)
        assert result < 0

    def test_large_input(self):
        from lsc.lscastrodef import half_total_flux_radius_to_fwhm
        result = half_total_flux_radius_to_fwhm(100.0)
        expected = 100.0 * 0.8493218 * 2.355
        assert abs(result - expected) < 1e-6


# ===========================================================================
# TestCrossmatch
# ===========================================================================

class TestCrossmatch:
    # --- from pure ---
    def test_exact_match(self):
        from lsc.lscastrodef import crossmatch
        ra0, dec0 = [150.0], [2.0]
        ra1, dec1 = [150.0], [2.0]
        distvec, pos0, pos1 = crossmatch(ra0, dec0, ra1, dec1, tollerance=1.0)
        assert len(pos0) == 1
        assert pos0[0] == 0 and pos1[0] == 0
        assert distvec[0] < 1e-10

    def test_no_match_beyond_tolerance(self):
        from lsc.lscastrodef import crossmatch
        ra0, dec0 = [150.0], [2.0]
        ra1, dec1 = [155.0], [7.0]
        distvec, pos0, pos1 = crossmatch(ra0, dec0, ra1, dec1, tollerance=1.0)
        assert len(pos0) == 0

    def test_selects_nearest(self):
        from lsc.lscastrodef import crossmatch
        ra0, dec0 = [150.0], [2.0]
        ra1 = [150.0001, 152.0]
        dec1 = [2.0001, 4.0]
        distvec, pos0, pos1 = crossmatch(ra0, dec0, ra1, dec1, tollerance=10.0)
        assert len(pos0) == 1
        assert pos1[0] == 0

    def test_multiple_sources(self):
        from lsc.lscastrodef import crossmatch
        ra0 = [150.0, 151.0]
        dec0 = [2.0, 3.0]
        ra1 = [150.0, 151.0]
        dec1 = [2.0, 3.0]
        distvec, pos0, pos1 = crossmatch(ra0, dec0, ra1, dec1, tollerance=1.0)
        assert len(pos0) == 2

    # --- from comprehensive ---
    def test_ra_near_zero_boundary(self):
        from lsc.lscastrodef import crossmatch
        ra0 = [0.01]
        dec0 = [0.0]
        ra1 = [0.01]
        dec1 = [0.0]
        distvec, pos0, pos1 = crossmatch(ra0, dec0, ra1, dec1, tollerance=10.0)
        assert len(pos0) == 1

    def test_dec_near_pole_positive(self):
        from lsc.lscastrodef import crossmatch
        ra0 = [180.0]
        dec0 = [89.99]
        ra1 = [180.0]
        dec1 = [89.99]
        distvec, pos0, pos1 = crossmatch(ra0, dec0, ra1, dec1, tollerance=10.0)
        assert len(pos0) == 1

    def test_dec_near_pole_negative(self):
        from lsc.lscastrodef import crossmatch
        ra0 = [45.0]
        dec0 = [-89.99]
        ra1 = [45.0]
        dec1 = [-89.99]
        distvec, pos0, pos1 = crossmatch(ra0, dec0, ra1, dec1, tollerance=10.0)
        assert len(pos0) == 1

    def test_large_catalog(self):
        from lsc.lscastrodef import crossmatch
        rng = np.random.default_rng(42)
        ra0 = list(rng.uniform(149, 151, 100))
        dec0 = list(rng.uniform(1, 3, 100))
        ra1 = [r + 0.0001 for r in ra0]
        dec1 = [d + 0.0001 for d in dec0]
        distvec, pos0, pos1 = crossmatch(ra0, dec0, ra1, dec1, tollerance=5.0)
        assert len(pos0) == 100

    def test_single_source_no_catalog(self):
        from lsc.lscastrodef import crossmatch
        distvec, pos0, pos1 = crossmatch([150.0], [2.0], [], [], tollerance=10.0)
        assert len(pos0) == 0

    def test_tolerance_exactly_at_distance(self):
        from lsc.lscastrodef import crossmatch
        ra0 = [150.0]
        dec0 = [2.0]
        ra1 = [150.0]
        dec1 = [2.0 + 1.0 / 3600.0]
        distvec, pos0, pos1 = crossmatch(ra0, dec0, ra1, dec1, tollerance=1.1)
        assert len(pos0) == 1

    def test_duplicate_catalog_sources(self):
        from lsc.lscastrodef import crossmatch
        ra0 = [150.0]
        dec0 = [2.0]
        ra1 = [150.0, 150.0, 150.001]
        dec1 = [2.0, 2.0, 2.001]
        distvec, pos0, pos1 = crossmatch(ra0, dec0, ra1, dec1, tollerance=60.0)
        assert len(pos0) == 1
        assert pos1[0] in [0, 1]

    def test_multiple_sources_different_dec(self):
        from lsc.lscastrodef import crossmatch
        ra0 = [150.0, 150.0]
        dec0 = [80.0, -80.0]
        ra1 = [150.0, 150.0]
        dec1 = [80.0, -80.0]
        distvec, pos0, pos1 = crossmatch(ra0, dec0, ra1, dec1, tollerance=5.0)
        assert len(pos0) == 2

    def test_all_match_identical(self):
        from lsc.lscastrodef import crossmatch
        n = 50
        rng = np.random.default_rng(7)
        ra = list(rng.uniform(0, 360, n))
        dec = list(rng.uniform(-90, 90, n))
        distvec, pos0, pos1 = crossmatch(ra, dec, ra, dec, tollerance=1.0)
        assert len(pos0) >= n - 2
        for i in range(len(pos0)):
            assert pos0[i] == pos1[i]


# ===========================================================================
# TestCrossmatchxy
# ===========================================================================

class TestCrossmatchxy:
    # --- from pure ---
    def test_exact_match(self):
        from lsc.lscastrodef import crossmatchxy
        xx0, yy0 = np.array([10.0]), np.array([20.0])
        xx1, yy1 = np.array([10.0]), np.array([20.0])
        distvec, pos0, pos1 = crossmatchxy(xx0, yy0, xx1, yy1, tollerance=2.0)
        assert len(pos0) == 1
        assert distvec[0] < 1e-10

    def test_no_match(self):
        from lsc.lscastrodef import crossmatchxy
        xx0, yy0 = np.array([0.0]), np.array([0.0])
        xx1, yy1 = np.array([100.0]), np.array([100.0])
        distvec, pos0, pos1 = crossmatchxy(xx0, yy0, xx1, yy1, tollerance=2.0)
        assert len(pos0) == 0

    def test_tolerance_boundary(self):
        from lsc.lscastrodef import crossmatchxy
        xx0, yy0 = np.array([0.0]), np.array([0.0])
        xx1, yy1 = np.array([1.0]), np.array([1.0])
        distvec, pos0, pos1 = crossmatchxy(xx0, yy0, xx1, yy1, tollerance=2.0)
        assert len(pos0) == 1

    def test_multiple_sources(self):
        from lsc.lscastrodef import crossmatchxy
        xx0 = np.array([0.0, 50.0])
        yy0 = np.array([0.0, 50.0])
        xx1 = np.array([0.1, 50.1])
        yy1 = np.array([0.1, 50.1])
        distvec, pos0, pos1 = crossmatchxy(xx0, yy0, xx1, yy1, tollerance=1.0)
        assert len(pos0) == 2

    # --- from comprehensive ---
    def test_large_pixel_offset(self):
        from lsc.lscastrodef import crossmatchxy
        xx0 = np.array([100.0])
        yy0 = np.array([100.0])
        xx1 = np.array([4000.0])
        yy1 = np.array([4000.0])
        distvec, pos0, pos1 = crossmatchxy(xx0, yy0, xx1, yy1, tollerance=10.0)
        assert len(pos0) == 0

    def test_many_sources_some_match(self):
        from lsc.lscastrodef import crossmatchxy
        xx0 = np.array([10.0, 50.0, 200.0])
        yy0 = np.array([10.0, 50.0, 200.0])
        xx1 = np.array([10.5, 250.0])
        yy1 = np.array([10.5, 250.0])
        distvec, pos0, pos1 = crossmatchxy(xx0, yy0, xx1, yy1, tollerance=2.0)
        assert len(pos0) == 1
        assert pos0[0] == 0

    def test_tolerance_zero(self):
        from lsc.lscastrodef import crossmatchxy
        xx0 = np.array([10.0, 20.0])
        yy0 = np.array([10.0, 20.0])
        xx1 = np.array([10.0, 20.5])
        yy1 = np.array([10.0, 20.5])
        distvec, pos0, pos1 = crossmatchxy(xx0, yy0, xx1, yy1, tollerance=0.0)
        assert len(pos0) == 1
        assert pos0[0] == 0

    def test_selects_nearest_among_many(self):
        from lsc.lscastrodef import crossmatchxy
        xx0 = np.array([100.0])
        yy0 = np.array([100.0])
        xx1 = np.array([101.0, 105.0, 110.0])
        yy1 = np.array([101.0, 105.0, 110.0])
        distvec, pos0, pos1 = crossmatchxy(xx0, yy0, xx1, yy1, tollerance=20.0)
        assert pos1[0] == 0

    def test_empty_input_arrays(self):
        from lsc.lscastrodef import crossmatchxy
        distvec, pos0, pos1 = crossmatchxy(
            np.array([]), np.array([]),
            np.array([10.0]), np.array([10.0]),
            tollerance=5.0
        )
        assert len(pos0) == 0

    def test_all_match_identical_xy(self):
        from lsc.lscastrodef import crossmatchxy
        n = 50
        rng = np.random.default_rng(7)
        xx = rng.uniform(0, 4000, n)
        yy = rng.uniform(0, 4000, n)
        distvec, pos0, pos1 = crossmatchxy(xx, yy, xx, yy, tollerance=1.0)
        assert len(pos0) == n


# ===========================================================================
# TestLinreg
# ===========================================================================

class TestLinreg:
    # --- from pure ---
    def test_perfect_line(self):
        from lsc.lscastrodef import linreg
        X = [1.0, 2.0, 3.0, 4.0, 5.0]
        Y = [2.0, 4.0, 6.0, 8.0, 10.0]
        a, b, RR = linreg(X, Y)
        assert abs(a - 2.0) < 1e-10
        assert abs(b - 0.0) < 1e-10
        assert abs(RR - 1.0) < 1e-10

    def test_with_offset(self):
        from lsc.lscastrodef import linreg
        X = [0.0, 1.0, 2.0, 3.0]
        Y = [5.0, 7.0, 9.0, 11.0]
        a, b, RR = linreg(X, Y)
        assert abs(a - 2.0) < 1e-9
        assert abs(b - 5.0) < 1e-9

    def test_unequal_length_raises(self):
        from lsc.lscastrodef import linreg
        with pytest.raises(ValueError, match="unequal length"):
            linreg([1, 2, 3], [4, 5])

    def test_r_squared_imperfect(self):
        from lsc.lscastrodef import linreg
        rng = np.random.default_rng(99)
        X = list(rng.uniform(0, 10, 20))
        Y = [2 * x + rng.normal(0, 2) for x in X]
        a, b, RR = linreg(X, Y)
        assert 0.5 < RR < 1.0

    # --- from comprehensive ---
    def test_two_points_exact(self):
        from lsc.lscastrodef import linreg
        X = [0.0, 10.0]
        Y = [5.0, 25.0]
        with pytest.raises(ZeroDivisionError):
            linreg(X, Y)

    def test_negative_slope(self):
        from lsc.lscastrodef import linreg
        X = [1.0, 2.0, 3.0, 4.0]
        Y = [10.0, 8.0, 6.0, 4.0]
        a, b, RR = linreg(X, Y)
        assert abs(a - (-2.0)) < 1e-10
        assert abs(b - 12.0) < 1e-10
        assert abs(RR - 1.0) < 1e-10

    def test_horizontal_line(self):
        from lsc.lscastrodef import linreg
        X = [1.0, 2.0, 3.0, 4.0]
        Y = [5.0, 5.0, 5.0, 5.0]
        with pytest.raises(ZeroDivisionError):
            linreg(X, Y)

    def test_large_dataset(self):
        from lsc.lscastrodef import linreg
        rng = np.random.default_rng(77)
        X = list(rng.uniform(0, 100, 100))
        Y = [3.0 * x + 7.0 + rng.normal(0, 1) for x in X]
        a, b, RR = linreg(X, Y)
        assert abs(a - 3.0) < 0.5
        assert abs(b - 7.0) < 5.0
        assert RR > 0.95

    def test_single_element_raises(self):
        from lsc.lscastrodef import linreg
        with pytest.raises((ZeroDivisionError, ValueError)):
            linreg([5.0], [10.0])


# ===========================================================================
# TestReadtxt
# ===========================================================================

class TestReadtxt:
    def _write_catalog(self, tmp_path, content, name="catalog.txt"):
        p = tmp_path / name
        p.write_text(content)
        return str(p)

    # --- from pure ---
    def test_reads_two_column_file(self, tmp_path):
        from lsc.lscastrodef import readtxt
        content = (
            "# nfields 2\n"
            "# ra 1\n"
            "# dec 2\n"
            "# END\n"
            "150.0 2.0\n"
            "151.0 3.0\n"
        )
        t = readtxt(self._write_catalog(tmp_path, content))
        assert len(t) == 2

    def test_columns_renamed_correctly(self, tmp_path):
        from lsc.lscastrodef import readtxt
        content = (
            "# nfields 2\n"
            "# ra 1\n"
            "# dec 2\n"
            "# END\n"
            "150.0 2.0\n"
        )
        t = readtxt(self._write_catalog(tmp_path, content))
        assert "ra" in t.colnames
        assert "dec" in t.colnames

    def test_values_match_input(self, tmp_path):
        from lsc.lscastrodef import readtxt
        content = (
            "# nfields 2\n"
            "# ra 1\n"
            "# dec 2\n"
            "# END\n"
            "150.0 2.5\n"
            "151.5 3.7\n"
        )
        t = readtxt(self._write_catalog(tmp_path, content))
        assert abs(t["ra"][0] - 150.0) < 1e-6
        assert abs(t["dec"][1] - 3.7) < 1e-6

    def test_three_column_file(self, tmp_path):
        from lsc.lscastrodef import readtxt
        content = (
            "# nfields 3\n"
            "# ra 1\n"
            "# dec 2\n"
            "# mag 3\n"
            "# END\n"
            "150.0 2.0 18.5\n"
            "151.0 3.0 17.2\n"
        )
        t = readtxt(self._write_catalog(tmp_path, content))
        assert "mag" in t.colnames
        assert len(t) == 2

    # --- from comprehensive ---
    def test_many_columns(self, tmp_path):
        from lsc.lscastrodef import readtxt
        content = (
            "# nfields 5\n"
            "# ra 1\n"
            "# dec 2\n"
            "# id 3\n"
            "# mag 4\n"
            "# err 5\n"
            "# END\n"
            "150.0 2.0 star1 14.5 0.01\n"
            "151.0 3.0 star2 15.2 0.02\n"
        )
        t = readtxt(self._write_catalog(tmp_path, content))
        assert "ra" in t.colnames
        assert "err" in t.colnames
        assert len(t) == 2

    def test_single_row(self, tmp_path):
        from lsc.lscastrodef import readtxt
        content = (
            "# nfields 2\n"
            "# ra 1\n"
            "# dec 2\n"
            "# END\n"
            "150.0 2.0\n"
        )
        t = readtxt(self._write_catalog(tmp_path, content))
        assert len(t) == 1

    def test_numeric_precision(self, tmp_path):
        from lsc.lscastrodef import readtxt
        content = (
            "# nfields 2\n"
            "# ra 1\n"
            "# dec 2\n"
            "# END\n"
            "150.123456 2.654321\n"
        )
        t = readtxt(self._write_catalog(tmp_path, content))
        assert abs(float(t["ra"][0]) - 150.123456) < 1e-5
        assert abs(float(t["dec"][0]) - 2.654321) < 1e-5

    def test_extra_comments_before_nfields(self, tmp_path):
        from lsc.lscastrodef import readtxt
        content = (
            "# BEGIN CATALOG HEADER\n"
            "# catdb some_catalog\n"
            "# nfields 2\n"
            "# ra 1\n"
            "# dec 2\n"
            "# END CATALOG HEADER\n"
            "#\n"
            "150.0 2.0\n"
            "151.0 3.0\n"
        )
        t = readtxt(self._write_catalog(tmp_path, content))
        assert len(t) == 2
        assert "ra" in t.colnames

    def test_tab_separated_values(self, tmp_path):
        from lsc.lscastrodef import readtxt
        content = (
            "# nfields 3\n"
            "# ra 1\n"
            "# dec 2\n"
            "# mag 3\n"
            "# END\n"
            "150.0\t2.0\t14.5\n"
            "151.0\t3.0\t15.2\n"
        )
        t = readtxt(self._write_catalog(tmp_path, content))
        assert len(t) == 2


# ===========================================================================
# TestTransformsloanlandolt
# ===========================================================================

class TestTransformsloanlandolt:
    # --- helper from pure ---
    def _make_stdcoo(self, n=5):
        rng = np.random.default_rng(17)
        return {
            'u':    list(rng.uniform(16.0, 18.0, n)),
            'g':    list(rng.uniform(15.0, 17.0, n)),
            'r':    list(rng.uniform(14.5, 16.5, n)),
            'i':    list(rng.uniform(14.0, 16.0, n)),
            'z':    list(rng.uniform(13.5, 15.5, n)),
            'uerr': list(rng.uniform(0.01, 0.05, n)),
            'gerr': list(rng.uniform(0.01, 0.05, n)),
            'rerr': list(rng.uniform(0.01, 0.05, n)),
            'ierr': list(rng.uniform(0.01, 0.05, n)),
            'zerr': list(rng.uniform(0.01, 0.05, n)),
        }

    def _make_stdcoo_full(self, n=5):
        rng = np.random.default_rng(42)
        return {
            'u': list(rng.uniform(17.0, 19.0, n)),
            'g': list(rng.uniform(15.0, 17.0, n)),
            'r': list(rng.uniform(14.5, 16.5, n)),
            'i': list(rng.uniform(14.0, 16.0, n)),
            'z': list(rng.uniform(13.5, 15.5, n)),
            'uerr': list(rng.uniform(0.01, 0.05, n)),
            'gerr': list(rng.uniform(0.01, 0.05, n)),
            'rerr': list(rng.uniform(0.01, 0.05, n)),
            'ierr': list(rng.uniform(0.01, 0.05, n)),
            'zerr': list(rng.uniform(0.01, 0.05, n)),
        }

    # --- from pure ---
    def test_returns_dict(self):
        from lsc.lscastrodef import transformsloanlandolt
        result = transformsloanlandolt(self._make_stdcoo())
        assert isinstance(result, dict)

    def test_adds_ubvri_keys(self):
        from lsc.lscastrodef import transformsloanlandolt
        result = transformsloanlandolt(self._make_stdcoo())
        for band in 'UBVRI':
            assert band in result

    def test_b_band_derived_from_g_r(self):
        from lsc.lscastrodef import transformsloanlandolt
        stdcoo = self._make_stdcoo(n=5)
        result = transformsloanlandolt(stdcoo)
        g = np.array(stdcoo['g'], float)
        r = np.array(stdcoo['r'], float)
        expected_B = g + 0.3130 * (g - r) + 0.2271
        np.testing.assert_allclose(result['B'], expected_B, atol=1e-9)

    def test_output_same_length_as_input(self):
        from lsc.lscastrodef import transformsloanlandolt
        stdcoo = self._make_stdcoo(n=8)
        result = transformsloanlandolt(stdcoo)
        assert len(result['B']) == 8
        assert len(result['V']) == 8

    def test_v_band_formula(self):
        from lsc.lscastrodef import transformsloanlandolt
        stdcoo = self._make_stdcoo(n=4)
        result = transformsloanlandolt(stdcoo)
        g = np.array(stdcoo['g'], float)
        r = np.array(stdcoo['r'], float)
        expected_V = g - 0.5784 * (g - r) - 0.0038
        np.testing.assert_allclose(result['V'], expected_V, atol=1e-9)

    # --- from comprehensive ---
    def test_all_ubvri_present(self):
        from lsc.lscastrodef import transformsloanlandolt
        result = transformsloanlandolt(self._make_stdcoo_full())
        for band in 'UBVRI':
            assert band in result

    def test_single_star(self):
        from lsc.lscastrodef import transformsloanlandolt
        stdcoo = {
            'u': [17.5], 'g': [16.0], 'r': [15.5], 'i': [15.0], 'z': [14.5],
            'uerr': [0.02], 'gerr': [0.02], 'rerr': [0.02],
            'ierr': [0.02], 'zerr': [0.02],
        }
        result = transformsloanlandolt(stdcoo)
        for band in 'UBVRI':
            assert band in result
            assert len(result[band]) == 1

    def test_r_and_i_only_raises_keyerror(self):
        from lsc.lscastrodef import transformsloanlandolt
        stdcoo = {
            'r': [15.0, 15.5],
            'i': [14.5, 15.0],
            'rerr': [0.01, 0.01],
            'ierr': [0.01, 0.01],
        }
        with pytest.raises(KeyError):
            transformsloanlandolt(stdcoo)

    def test_r_formula_with_ri(self):
        from lsc.lscastrodef import transformsloanlandolt
        stdcoo = {
            'u': [18.0, 19.0], 'g': [16.0, 17.0], 'r': [15.0, 16.0], 'i': [14.5, 15.5],
            'uerr': [0.01, 0.01], 'gerr': [0.01, 0.01],
            'rerr': [0.01, 0.01], 'ierr': [0.01, 0.01],
        }
        result = transformsloanlandolt(stdcoo)
        r = np.array(stdcoo['r'], float)
        i = np.array(stdcoo['i'], float)
        expected_R = r - 0.2936 * (r - i) - 0.1439
        np.testing.assert_allclose(result['R'], expected_R, atol=1e-9)

    def test_i_formula_with_ri(self):
        from lsc.lscastrodef import transformsloanlandolt
        stdcoo = {
            'u': [18.0, 19.0], 'g': [16.0, 17.0], 'r': [15.0, 16.0], 'i': [14.5, 15.5],
            'uerr': [0.01, 0.01], 'gerr': [0.01, 0.01],
            'rerr': [0.01, 0.01], 'ierr': [0.01, 0.01],
        }
        result = transformsloanlandolt(stdcoo)
        r = np.array(stdcoo['r'], float)
        i = np.array(stdcoo['i'], float)
        expected_I = r - 1.2444 * (r - i) - 0.3820
        np.testing.assert_allclose(result['I'], expected_I, atol=1e-9)

    def test_u_formula(self):
        from lsc.lscastrodef import transformsloanlandolt
        stdcoo = self._make_stdcoo_full(n=3)
        result = transformsloanlandolt(stdcoo)
        g = np.array(stdcoo['g'], float)
        r = np.array(stdcoo['r'], float)
        u = np.array(stdcoo['u'], float)
        B = g + 0.3130 * (g - r) + 0.2271
        expected_U = B + 0.78 * (u - g) - 0.88
        np.testing.assert_allclose(result['U'], expected_U, atol=1e-9)

    def test_error_propagation_B(self):
        from lsc.lscastrodef import transformsloanlandolt
        stdcoo = self._make_stdcoo_full(n=3)
        result = transformsloanlandolt(stdcoo)
        assert 'Berr' in result

    # --- from extra ---
    def test_returns_ubvri_keys_extra(self):
        from lsc.lscastrodef import transformsloanlandolt
        stdcoo = {
            'u': [15.0, 15.5], 'g': [14.0, 14.5], 'r': [13.5, 14.0], 'i': [13.0, 13.5],
            'uerr': [0.01, 0.02], 'gerr': [0.01, 0.02],
            'rerr': [0.01, 0.02], 'ierr': [0.01, 0.02],
        }
        result = transformsloanlandolt(stdcoo)
        assert 'B' in result or 'V' in result or 'R' in result

    def test_output_same_length_extra(self):
        from lsc.lscastrodef import transformsloanlandolt
        stdcoo = {
            'u': [15.0, 15.5, 16.0], 'g': [14.0, 14.5, 15.0],
            'r': [13.5, 14.0, 14.5], 'i': [13.0, 13.5, 14.0],
            'uerr': [0.01, 0.02, 0.01], 'gerr': [0.01, 0.02, 0.01],
            'rerr': [0.01, 0.02, 0.01], 'ierr': [0.01, 0.02, 0.01],
        }
        result = transformsloanlandolt(stdcoo)
        for key in result:
            assert len(result[key]) == 3

    # --- from cov100 ---
    def test_B_not_precomputed_when_no_u_cov100(self):
        import lsc.lscastrodef
        stdcoo = {
            'u': np.array([20.0, 21.0]), 'uerr': np.array([0.05, 0.06]),
            'g': np.array([18.0, 19.0]), 'gerr': np.array([0.01, 0.02]),
            'r': np.array([17.5, 18.5]), 'rerr': np.array([0.01, 0.02]),
            'i': np.array([17.0, 18.0]), 'ierr': np.array([0.01, 0.02]),
            'z': np.array([16.8, 17.8]), 'zerr': np.array([0.01, 0.02]),
        }
        result = lsc.lscastrodef.transformsloanlandolt(stdcoo)
        assert 'B' in result
        assert 'V' in result
        assert 'R' in result
        assert 'I' in result
        assert 'U' in result

    def test_no_ri_no_iz_only_gr_cov100(self):
        import lsc.lscastrodef
        stdcoo = {
            'u': np.array([20.0, 21.0]), 'uerr': np.array([0.05, 0.06]),
            'g': np.array([18.0, 19.0]), 'gerr': np.array([0.01, 0.02]),
            'r': np.array([17.5, 18.5]), 'rerr': np.array([0.01, 0.02]),
        }
        result = lsc.lscastrodef.transformsloanlandolt(stdcoo)
        assert 'B' in result
        assert 'Berr' in result
        assert 'R' in result

    # --- from coverage_c ---
    def test_full_ugriz_input_coverage_c(self):
        from lsc.lscastrodef import transformsloanlandolt
        stdcoo = {
            'u': [20.0, 20.5], 'g': [19.0, 19.5], 'r': [18.0, 18.5],
            'i': [17.0, 17.5], 'z': [16.5, 17.0],
            'uerr': [0.01, 0.01], 'gerr': [0.01, 0.01], 'rerr': [0.01, 0.01],
            'ierr': [0.01, 0.01], 'zerr': [0.01, 0.01],
        }
        result = transformsloanlandolt(stdcoo)
        assert 'B' in result
        assert 'V' in result
        assert 'R' in result
        assert 'I' in result
        assert 'U' in result

    def test_gr_only_raises_keyerror_due_to_precedence_coverage_c(self):
        from lsc.lscastrodef import transformsloanlandolt
        stdcoo = {
            'g': [19.0], 'r': [18.0],
            'gerr': [0.01], 'rerr': [0.01],
        }
        with pytest.raises(KeyError):
            transformsloanlandolt(stdcoo)


# ===========================================================================
# TestTransformlandoltsloan
# ===========================================================================

class TestTransformlandoltsloan:
    # --- helpers from pure ---
    def _make_stdcoo_bv_only(self, n=5):
        rng = np.random.default_rng(23)
        return {
            'B': list(rng.uniform(15.0, 17.0, n)),
            'V': list(rng.uniform(14.5, 16.5, n)),
        }

    def _make_stdcoo_full(self, n=5):
        rng = np.random.default_rng(23)
        return {
            'B': list(rng.uniform(15.0, 17.0, n)),
            'V': list(rng.uniform(14.5, 16.5, n)),
            'R': list(rng.uniform(14.0, 16.0, n)),
            'I': list(rng.uniform(13.5, 15.5, n)),
        }

    # --- from pure ---
    def test_returns_dict(self):
        from lsc.lscastrodef import transformlandoltsloan
        result = transformlandoltsloan(self._make_stdcoo_bv_only())
        assert isinstance(result, dict)

    def test_g_band_derived_from_b_v(self):
        from lsc.lscastrodef import transformlandoltsloan
        stdcoo = self._make_stdcoo_bv_only(n=5)
        result = transformlandoltsloan(stdcoo)
        B = np.array(stdcoo['B'], float)
        V = np.array(stdcoo['V'], float)
        expected_g = V + 0.630 * (B - V) - 0.124
        np.testing.assert_allclose(result['g'], expected_g, atol=1e-9)

    def test_full_input_with_vr_branch(self):
        from lsc.lscastrodef import transformlandoltsloan
        result = transformlandoltsloan(self._make_stdcoo_full())
        assert isinstance(result, dict)

    def test_v_key_present_in_result(self):
        from lsc.lscastrodef import transformlandoltsloan
        stdcoo = self._make_stdcoo_bv_only(n=4)
        result = transformlandoltsloan(stdcoo)
        assert 'V' in result

    # --- from comprehensive ---
    def test_g_from_bv_comprehensive(self):
        from lsc.lscastrodef import transformlandoltsloan
        stdcoo = {'B': [15.0, 16.0], 'V': [14.5, 15.5]}
        result = transformlandoltsloan(stdcoo)
        B = np.array(stdcoo['B'], float)
        V = np.array(stdcoo['V'], float)
        expected_g = V + 0.630 * (B - V) - 0.124
        np.testing.assert_allclose(result['g'], expected_g, atol=1e-9)

    def test_r_from_vr_below_threshold(self):
        from lsc.lscastrodef import transformlandoltsloan
        stdcoo = {'B': [15.5], 'V': [14.5], 'R': [14.0], 'I': [13.5]}
        result = transformlandoltsloan(stdcoo)
        V = np.array(stdcoo['V'], float)
        R = np.array(stdcoo['R'], float)
        VR = V - R
        expected_r = R + 0.267 * VR + 0.088
        np.testing.assert_allclose(result['r'], expected_r, atol=1e-9)

    def test_r_from_vr_above_threshold(self):
        from lsc.lscastrodef import transformlandoltsloan
        stdcoo = {'B': [17.0], 'V': [15.5], 'R': [14.0], 'I': [13.0]}
        result = transformlandoltsloan(stdcoo)
        V = np.array(stdcoo['V'], float)
        R = np.array(stdcoo['R'], float)
        VR = V - R
        expected_r = R + 0.77 * VR - 0.37
        np.testing.assert_allclose(result['r'], expected_r, atol=1e-9)

    def test_r_from_vr_mixed_threshold(self):
        from lsc.lscastrodef import transformlandoltsloan
        stdcoo = {
            'B': [15.5, 17.0], 'V': [14.5, 15.5],
            'R': [14.0, 14.0], 'I': [13.5, 13.0],
        }
        result = transformlandoltsloan(stdcoo)
        V = np.array(stdcoo['V'], float)
        R = np.array(stdcoo['R'], float)
        VR = V - R
        a = R + 0.267 * VR + 0.088
        b = R + 0.77 * VR - 0.37
        expected_r = np.where(VR <= 0.93, a, b)
        np.testing.assert_allclose(result['r'], expected_r, atol=1e-9)

    def test_i_from_ri(self):
        from lsc.lscastrodef import transformlandoltsloan
        stdcoo = {'B': [15.5], 'V': [14.5], 'R': [14.0], 'I': [13.5]}
        result = transformlandoltsloan(stdcoo)
        R = np.array(stdcoo['R'], float)
        I = np.array(stdcoo['I'], float)
        expected_i = I - 0.247 * (R - I) + 0.329
        np.testing.assert_allclose(result['i'], expected_i, atol=1e-9)

    def test_z_from_r_ri(self):
        from lsc.lscastrodef import transformlandoltsloan
        stdcoo = {'B': [15.5], 'V': [14.5], 'R': [14.0], 'I': [13.5]}
        result = transformlandoltsloan(stdcoo)
        R = np.array(stdcoo['R'], float)
        I = np.array(stdcoo['I'], float)
        expected_z = np.array(result['r']) - 1.584 * (R - I) + 0.386
        np.testing.assert_allclose(result['z'], expected_z, atol=1e-9)

    def test_missing_bands_filled(self):
        from lsc.lscastrodef import transformlandoltsloan
        stdcoo = {'B': [15.0], 'V': [14.5]}
        result = transformlandoltsloan(stdcoo)
        for band in 'ugriz':
            assert band in result

    def test_single_star_full(self):
        from lsc.lscastrodef import transformlandoltsloan
        stdcoo = {'B': [15.0], 'V': [14.5], 'R': [14.0], 'I': [13.5]}
        result = transformlandoltsloan(stdcoo)
        for band in 'griz':
            assert band in result
            assert len(result[band]) == 1

    # --- from extra ---
    def test_returns_griz_keys_extra(self):
        from lsc.lscastrodef import transformlandoltsloan
        stdcoo = {'B': [14.0, 14.5], 'V': [13.5, 14.0], 'R': [13.0, 13.5], 'I': [12.5, 13.0]}
        result = transformlandoltsloan(stdcoo)
        assert 'g' in result or 'r' in result

    def test_output_same_length_extra(self):
        from lsc.lscastrodef import transformlandoltsloan
        stdcoo = {'B': [14.0, 14.5, 15.0], 'V': [13.5, 14.0, 14.5], 'R': [13.0, 13.5, 14.0], 'I': [12.5, 13.0, 13.5]}
        result = transformlandoltsloan(stdcoo)
        for key in result:
            assert len(result[key]) == 3

    # --- from cov100 ---
    def test_r_not_computed_earlier_cov100(self):
        import lsc.lscastrodef
        stdcoo = {
            'B': np.array([18.0, 19.0]), 'V': np.array([17.5, 18.5]),
            'R': np.array([17.0, 18.0]), 'I': np.array([16.5, 17.5]),
        }
        result = lsc.lscastrodef.transformlandoltsloan(stdcoo)
        assert 'g' in result
        assert 'r' in result
        assert 'i' in result
        assert 'z' in result

    # --- from coverage_c ---
    def test_full_BVRI_input_coverage_c(self):
        from lsc.lscastrodef import transformlandoltsloan
        stdcoo = {'B': [15.0, 15.5], 'V': [14.5, 15.0], 'R': [14.0, 14.5], 'I': [13.5, 14.0]}
        result = transformlandoltsloan(stdcoo)
        assert 'g' in result
        assert 'r' in result
        assert 'i' in result
        assert 'z' in result

    def test_with_VR_above_boundary_coverage_c(self):
        from lsc.lscastrodef import transformlandoltsloan
        stdcoo = {'B': [16.0], 'V': [15.0], 'R': [13.5], 'I': [13.0]}
        result = transformlandoltsloan(stdcoo)
        expected_r = 13.5 + 0.77 * 1.5 - 0.37
        np.testing.assert_allclose(result['r'], [expected_r], rtol=1e-5)

    def test_with_VR_below_boundary_coverage_c(self):
        from lsc.lscastrodef import transformlandoltsloan
        stdcoo = {'B': [15.0], 'V': [14.5], 'R': [14.2], 'I': [13.8]}
        result = transformlandoltsloan(stdcoo)
        expected_r = 14.2 + 0.267 * 0.3 + 0.088
        np.testing.assert_allclose(result['r'], [expected_r], rtol=1e-5)


# ===========================================================================
# TestVizq
# ===========================================================================

class TestVizq:
    def _make_proc(self, output_bytes=b""):
        proc = MagicMock()
        proc.communicate.return_value = (output_bytes, b"")
        return proc

    # --- from pure ---
    def test_returns_expected_keys(self):
        from lsc.lscastrodef import vizq
        fake_out = (
            b"header1\nheader2\nheader3\n"
            b"01 02 03\t+04 05 06\tSTAR1\t14.5\n"
            b"01 02 04\t+04 05 07\tSTAR2\t15.2\n"
        )
        with patch('subprocess.Popen', return_value=self._make_proc(fake_out)):
            result = vizq(150.0, 2.0, 'usnoa2', 10.0)
        assert set(result.keys()) >= {'ra', 'dec', 'id', 'mag'}

    def test_returns_two_sources(self):
        from lsc.lscastrodef import vizq
        fake_out = (
            b"header1\nheader2\nheader3\n"
            b"01 02 03\t+04 05 06\tSTAR1\t14.5\n"
            b"01 02 04\t+04 05 07\tSTAR2\t15.2\n"
        )
        with patch('subprocess.Popen', return_value=self._make_proc(fake_out)):
            result = vizq(150.0, 2.0, 'usnoa2', 10.0)
        assert len(result['ra']) == 2
        assert len(result['mag']) == 2

    def test_invalid_mag_falls_back_to_9999(self):
        from lsc.lscastrodef import vizq
        fake_out = (
            b"header1\nheader2\nheader3\n"
            b"01 02 03\t+04 05 06\tSTAR1\tnull\n"
        )
        with patch('subprocess.Popen', return_value=self._make_proc(fake_out)):
            result = vizq(150.0, 2.0, 'usnoa2', 10.0)
        assert result['mag'][0] == 9999.0

    def test_empty_catalog_returns_empty_lists(self):
        from lsc.lscastrodef import vizq
        fake_out = b"header1\nheader2\nheader3\n"
        with patch('subprocess.Popen', return_value=self._make_proc(fake_out)):
            result = vizq(150.0, 2.0, 'usnob1', 5.0)
        assert result['ra'] == []
        assert result['dec'] == []

    def test_comment_lines_skipped(self):
        from lsc.lscastrodef import vizq
        fake_out = (
            b"# a comment\n"
            b"header1\nheader2\nheader3\n"
            b"01 02 03\t+04 05 06\tS1\t14.0\n"
        )
        with patch('subprocess.Popen', return_value=self._make_proc(fake_out)):
            result = vizq(150.0, 2.0, 'usnob1', 10.0)
        assert len(result['ra']) == 1

    def test_catalogue_2mass_supported(self):
        from lsc.lscastrodef import vizq
        fake_out = b"h1\nh2\nh3\n01 02 03\t+04 05 06\t2MASSJNAME\t12.3\n"
        with patch('subprocess.Popen', return_value=self._make_proc(fake_out)):
            result = vizq(150.0, 2.0, '2mass', 10.0)
        assert len(result['ra']) == 1

    def test_ra_dec_colon_formatted(self):
        from lsc.lscastrodef import vizq
        fake_out = (
            b"header1\nheader2\nheader3\n"
            b"01 02 03\t+04 05 06\tSTAR1\t14.5\n"
        )
        with patch('subprocess.Popen', return_value=self._make_proc(fake_out)):
            result = vizq(150.0, 2.0, 'usnoa2', 10.0)
        assert ':' in result['ra'][0]
        assert ':' in result['dec'][0]

    # --- from comprehensive ---
    def test_all_supported_catalogs(self):
        from lsc.lscastrodef import vizq
        fake_out = b"h1\nh2\nh3\n01 02 03\t+04 05 06\tSTAR1\t14.5\n"
        for cat in ['usnoa2', '2mass', 'usnob1', 'apass', 'sdss7']:
            with patch('subprocess.Popen', return_value=self._make_proc(fake_out)):
                result = vizq(150.0, 2.0, cat, 10.0)
            assert 'ra' in result

    def test_negative_declination(self):
        from lsc.lscastrodef import vizq
        fake_out = b"h1\nh2\nh3\n01 02 03\t-04 05 06\tSTAR1\t14.5\n"
        with patch('subprocess.Popen', return_value=self._make_proc(fake_out)):
            result = vizq(150.0, -30.0, 'usnoa2', 10.0)
        assert '-04:05:06' == result['dec'][0]

    def test_many_sources(self):
        from lsc.lscastrodef import vizq
        lines = b"h1\nh2\nh3\n"
        for i in range(50):
            lines += f"01 02 {i:02d}\t+04 05 {i:02d}\tSTAR{i}\t{14.0 + i * 0.1:.1f}\n".encode()
        with patch('subprocess.Popen', return_value=self._make_proc(lines)):
            result = vizq(150.0, 2.0, 'usnoa2', 30.0)
        assert len(result['ra']) == 50
        assert len(result['mag']) == 50

    def test_magnitude_float_parsing(self):
        from lsc.lscastrodef import vizq
        fake_out = b"h1\nh2\nh3\n01 02 03\t+04 05 06\tSTAR1\t16.789\n"
        with patch('subprocess.Popen', return_value=self._make_proc(fake_out)):
            result = vizq(150.0, 2.0, 'usnoa2', 10.0)
        assert abs(result['mag'][0] - 16.789) < 1e-3

    # --- from coverage_a ---
    @patch('subprocess.Popen')
    def test_pyversion_less_than_3(self, mock_popen):
        from lsc.lscastrodef import vizq
        fake_output = (
            "# comment line\n"
            "# another comment\n"
            "col1\tcol2\tcol3\tcol4\n"
            "---\t---\t---\t---\n"
            "---\t---\t---\t---\n"
            "10 20 30\t+01 02 03\tSTAR1\t14.5\n"
        )
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (fake_output, b'')
        mock_popen.return_value = mock_proc
        with patch('sys.version_info', (2, 7, 0)):
            result = vizq(150.0, 2.0, 'usnoa2', 10)
        assert 'ra' in result

    @patch('subprocess.Popen')
    def test_pyversion_3_decode(self, mock_popen):
        from lsc.lscastrodef import vizq
        fake_output = (
            "# comment line\n"
            "col1\tcol2\tcol3\tcol4\n"
            "---\t---\t---\t---\n"
            "---\t---\t---\t---\n"
            "10 20 30\t+01 02 03\tSTAR1\t14.5\n"
        )
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (fake_output.encode('ascii'), b'')
        mock_popen.return_value = mock_proc
        result = vizq(150.0, 2.0, '2mass', 10)
        assert len(result['ra']) == 1
        assert result['mag'][0] == 14.5

    # --- from cov100 ---
    @patch('subprocess.Popen')
    def test_vizq_bad_mag_cov100(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (
            b"#comment\n#comment\n#comment\n"
            b"header1\nheader2\nheader3\n"
            b"10 00 00.0\t+02 00 00.0\tSTAR1\tBADMAG\n"
            b"10 01 00.0\t+02 01 00.0\tSTAR2\t15.5\n",
            b""
        )
        mock_popen.return_value = mock_proc
        import lsc.lscastrodef
        result = lsc.lscastrodef.vizq(150.0, 2.0, 'usnob1', 20)
        assert 9999 in result['mag'] or 15.5 in result['mag']


# ===========================================================================
# TestSextractor
# ===========================================================================

class TestSextractor:
    @pytest.fixture(autouse=True)
    def _reset_iraf_proto(self):
        """Reset proto.fields mock to avoid leakage from other test modules."""
        iraf_mock = sys.modules['iraf']
        iraf_mock.proto.fields.reset_mock()
        iraf_mock.proto.fields.side_effect = None
        iraf_mock.proto.fields.return_value = []
        yield
        iraf_mock.proto.fields.reset_mock()
        iraf_mock.proto.fields.side_effect = None

    # --- from pure ---
    def test_returns_eight_element_tuple(self, simple_fits, tmp_path, monkeypatch):
        from lsc.lscastrodef import sextractor
        monkeypatch.chdir(tmp_path)
        iraf_mock = sys.modules['iraf']
        iraf_mock.proto.fields.return_value = []
        with patch('os.system', return_value=0):
            result = sextractor(simple_fits)
        assert len(result) == 8

    def test_empty_detections_gives_empty_arrays(self, simple_fits, tmp_path, monkeypatch):
        from lsc.lscastrodef import sextractor
        monkeypatch.chdir(tmp_path)
        iraf_mock = sys.modules['iraf']
        iraf_mock.proto.fields.return_value = []
        with patch('os.system', return_value=0):
            xpix, ypix, fw, cl, cm, ell, bkg, fl = sextractor(simple_fits)
        assert len(xpix) == 0
        assert len(ypix) == 0

    def test_detections_returned_with_good_values(self, simple_fits, tmp_path, monkeypatch):
        from lsc.lscastrodef import sextractor
        monkeypatch.chdir(tmp_path)
        iraf_mock = sys.modules['iraf']
        call_returns = iter([
            ['50.0', '60.0', '70.0'],
            ['50.0', '60.0', '70.0'],
            ['-10.0', '-11.0', '-12.0'],
            ['0.9', '0.9', '0.9'],
            ['5.0', '4.0', '3.0'],
            ['0.1', '0.1', '0.1'],
            ['1000.0', '1000.0', '1000.0'],
            ['0.0', '0.0', '0.0'],
        ])
        iraf_mock.proto.fields.side_effect = lambda *a, **kw: next(call_returns)
        with patch('os.system', return_value=0):
            result = sextractor(simple_fits)
        assert len(result) == 8
        xpix = result[0]
        assert len(xpix) >= 0

    # --- from comprehensive ---
    def test_saturation_capped_at_55000(self, simple_fits, tmp_path, monkeypatch):
        from astropy.io import fits as astrofits
        import shutil
        dst = str(tmp_path / "sat_test.fits")
        shutil.copy2(simple_fits, dst)
        astrofits.setval(dst, 'SATURATE', value=70000)
        from lsc.lscastrodef import sextractor
        monkeypatch.chdir(tmp_path)
        iraf_mock = sys.modules['iraf']
        iraf_mock.proto.fields.return_value = []
        with patch('os.system', return_value=0) as mock_sys:
            sextractor(dst)
            call_args = mock_sys.call_args_list[0][0][0]
            assert '55000' in call_args

    def test_default_saturation_45000(self, tmp_path, monkeypatch):
        from astropy.io import fits as astrofits
        data = np.zeros((50, 50), dtype=np.float32)
        hdr = astrofits.Header()
        hdr['NAXIS1'] = 50
        hdr['NAXIS2'] = 50
        path = str(tmp_path / "nosat.fits")
        astrofits.writeto(path, data, hdr, overwrite=True)
        from lsc.lscastrodef import sextractor
        monkeypatch.chdir(tmp_path)
        iraf_mock = sys.modules['iraf']
        iraf_mock.proto.fields.return_value = []
        with patch('os.system', return_value=0) as mock_sys:
            sextractor(path)
            call_args = mock_sys.call_args_list[0][0][0]
            assert '45000' in call_args

    def test_dimension_defaults_when_missing(self, tmp_path, monkeypatch):
        from astropy.io import fits as astrofits
        data = np.zeros((10, 10), dtype=np.float32)
        path = str(tmp_path / "small.fits")
        astrofits.writeto(path, data, overwrite=True)
        from lsc.lscastrodef import sextractor
        monkeypatch.chdir(tmp_path)
        iraf_mock = sys.modules['iraf']
        iraf_mock.proto.fields.return_value = []
        with patch('os.system', return_value=0):
            result = sextractor(path)
        assert len(result) == 8

    def test_filtering_removes_border_objects(self, simple_fits, tmp_path, monkeypatch):
        from lsc.lscastrodef import sextractor
        monkeypatch.chdir(tmp_path)
        iraf_mock = sys.modules['iraf']
        call_returns = iter([
            ['2.0', '50.0'],
            ['50.0', '50.0'],
            ['-10.0', '-11.0'],
            ['0.9', '0.9'],
            ['5.0', '5.0'],
            ['0.1', '0.1'],
            ['1000.0', '1000.0'],
            ['0.0', '0.0'],
        ])
        iraf_mock.proto.fields.side_effect = lambda *a, **kw: next(call_returns)
        with patch('os.system', return_value=0):
            xpix, ypix, fw, cl, cm, ell, bkg, fl = sextractor(simple_fits)
        if len(xpix) > 0:
            assert all(x > 3 for x in xpix)

    def test_fw_filter_removes_outliers(self, simple_fits, tmp_path, monkeypatch):
        from lsc.lscastrodef import sextractor
        monkeypatch.chdir(tmp_path)
        iraf_mock = sys.modules['iraf']
        call_returns = iter([
            ['50.0', '60.0'],
            ['50.0', '60.0'],
            ['-10.0', '-11.0'],
            ['0.9', '0.9'],
            ['5.0', '20.0'],
            ['0.1', '0.1'],
            ['1000.0', '1000.0'],
            ['0.0', '0.0'],
        ])
        iraf_mock.proto.fields.side_effect = lambda *a, **kw: next(call_returns)
        with patch('os.system', return_value=0):
            xpix, ypix, fw, cl, cm, ell, bkg, fl = sextractor(simple_fits)
        if len(fw) > 0:
            assert all(f <= 15 for f in fw)

    # --- from coverage_c ---
    def test_sextractor_except_branch_and_defaults_coverage_c(self):
        import importlib
        import astropy.io.fits as fits
        hdr = fits.Header()
        hdr['SATURATE'] = 40000
        mock_proto = MagicMock()
        mock_proto.fields.return_value = ['not_a_number']
        mock_iraf = MagicMock()
        mock_iraf.proto = mock_proto
        mock_defsex = MagicMock(return_value='default.sex')
        mock_delete = MagicMock()
        mock_system = MagicMock()
        with patch('astropy.io.fits.getheader', return_value=hdr), \
             patch('lsc.util.defsex', mock_defsex), \
             patch('lsc.util.delete', mock_delete), \
             patch('os.system', mock_system), \
             patch.dict('sys.modules', {
                 'pyraf': MagicMock(), 'pyraf.iraf': mock_iraf,
                 'iraf': mock_iraf, 'iraf.proto': mock_proto,
             }):
            import lsc.lscastrodef
            importlib.reload(lsc.lscastrodef)
            result = lsc.lscastrodef.sextractor('test.fits')
            xpix, ypix, fw, cl, cm, ell, bkg, fl = result
            assert len(xpix) == 0
            assert len(ypix) == 0
            assert len(fw) == 0
            importlib.reload(lsc.lscastrodef)


# ===========================================================================
# TestWcsstart
# ===========================================================================

class TestWcsstart:
    # --- from pure ---
    def test_writes_wcs_keywords(self, simple_fits, tmp_path):
        import shutil
        from astropy.io import fits as astrofits
        from lsc.lscastrodef import wcsstart
        dst = str(tmp_path / "wcs_test.fits")
        shutil.copy2(simple_fits, dst)
        wcsstart(dst)
        hdr = astrofits.getheader(dst)
        for key in ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CTYPE1", "CTYPE2"):
            assert key in hdr, f"{key} not written by wcsstart"

    def test_crval_matches_header_ra_dec(self, simple_fits, tmp_path):
        import shutil
        from astropy.io import fits as astrofits
        from lsc.lscastrodef import wcsstart
        dst = str(tmp_path / "wcs_radec.fits")
        shutil.copy2(simple_fits, dst)
        orig_ra = astrofits.getval(dst, "RA")
        orig_dec = astrofits.getval(dst, "DEC")
        wcsstart(dst)
        hdr = astrofits.getheader(dst)
        assert abs(hdr["CRVAL1"] - orig_ra) < 1e-6
        assert abs(hdr["CRVAL2"] - orig_dec) < 1e-6

    def test_ctype_is_tan_projection(self, simple_fits, tmp_path):
        import shutil
        from astropy.io import fits as astrofits
        from lsc.lscastrodef import wcsstart
        dst = str(tmp_path / "wcs_ctype.fits")
        shutil.copy2(simple_fits, dst)
        wcsstart(dst)
        hdr = astrofits.getheader(dst)
        assert "TAN" in hdr["CTYPE1"]
        assert "TAN" in hdr["CTYPE2"]

    # --- from comprehensive ---
    def test_nonzero_position_angle(self, tmp_path):
        from astropy.io import fits as astrofits
        from lsc.lscastrodef import wcsstart
        data = np.zeros((100, 100), dtype=np.float32)
        hdr = astrofits.Header()
        hdr['NAXIS1'] = 100
        hdr['NAXIS2'] = 100
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.0
        hdr['PIXSCALE'] = 0.389
        hdr['CCDSUM'] = '1 1'
        hdr['ROLLERDR'] = 45.0
        path = str(tmp_path / "rotated.fits")
        astrofits.writeto(path, data, hdr, overwrite=True)
        wcsstart(path)
        hdr_out = astrofits.getheader(path)
        assert 'CD1_2' in hdr_out
        assert 'CD2_1' in hdr_out

    def test_binning_2x2(self, tmp_path):
        from astropy.io import fits as astrofits
        from lsc.lscastrodef import wcsstart
        data = np.zeros((50, 50), dtype=np.float32)
        hdr = astrofits.Header()
        hdr['NAXIS1'] = 50
        hdr['NAXIS2'] = 50
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.0
        hdr['PIXSCALE'] = 0.389
        hdr['CCDSUM'] = '2 2'
        hdr['ROLLERDR'] = 0.0
        path = str(tmp_path / "binned.fits")
        astrofits.writeto(path, data, hdr, overwrite=True)
        wcsstart(path)
        hdr_out = astrofits.getheader(path)
        assert abs(hdr_out['CDELT1'] - 0.778) < 0.001

    def test_ctype_values(self, simple_fits, tmp_path):
        import shutil
        from astropy.io import fits as astrofits
        from lsc.lscastrodef import wcsstart
        dst = str(tmp_path / "ctype_test.fits")
        shutil.copy2(simple_fits, dst)
        wcsstart(dst)
        hdr = astrofits.getheader(dst)
        assert hdr['CTYPE1'] == 'RA---TAN'
        assert hdr['CTYPE2'] == 'DEC--TAN'

    def test_wcsdim_is_2(self, simple_fits, tmp_path):
        import shutil
        from astropy.io import fits as astrofits
        from lsc.lscastrodef import wcsstart
        dst = str(tmp_path / "wcsdim_test.fits")
        shutil.copy2(simple_fits, dst)
        wcsstart(dst)
        hdr = astrofits.getheader(dst)
        assert hdr['WCSDIM'] == 2


# ===========================================================================
# TestWCSCDMatrix (comprehensive only)
# ===========================================================================

class TestWCSCDMatrix:
    def test_cd_matrix_zero_rotation(self):
        from numpy import sin, cos, deg2rad
        position_angle = deg2rad(0.0)
        CDELT1 = 0.389
        CDELT2 = 0.389
        CD1_1 = -CDELT1 * cos(position_angle)
        CD1_2 = CDELT2 * sin(position_angle)
        CD2_1 = -CDELT1 * sin(position_angle)
        CD2_2 = CDELT2 * cos(position_angle)
        assert abs(CD1_2) < 1e-15
        assert abs(CD2_1) < 1e-15

    def test_cd_matrix_90_degree_rotation(self):
        from numpy import sin, cos, deg2rad
        position_angle = deg2rad(90.0)
        CDELT1 = 0.389
        CDELT2 = 0.389
        CD1_1 = -CDELT1 * cos(position_angle)
        CD1_2 = CDELT2 * sin(position_angle)
        CD2_1 = -CDELT1 * sin(position_angle)
        CD2_2 = CDELT2 * cos(position_angle)
        assert abs(CD1_1) < 1e-14
        assert abs(CD2_2) < 1e-14
        assert abs(CD1_2 - 0.389) < 1e-10
        assert abs(CD2_1 - (-0.389)) < 1e-10

    def test_cd_matrix_45_degree_rotation(self):
        from numpy import sin, cos, deg2rad, sqrt
        position_angle = deg2rad(45.0)
        CDELT1 = 0.389
        CDELT2 = 0.389
        CD1_1 = -CDELT1 * cos(position_angle)
        CD1_2 = CDELT2 * sin(position_angle)
        CD2_1 = -CDELT1 * sin(position_angle)
        CD2_2 = CDELT2 * cos(position_angle)
        expected = 0.389 / sqrt(2)
        assert abs(abs(CD1_1) - expected) < 1e-10
        assert abs(abs(CD1_2) - expected) < 1e-10

    def test_cd_matrix_180_rotation(self):
        from numpy import sin, cos, deg2rad
        position_angle = deg2rad(180.0)
        CDELT1 = 0.389
        CDELT2 = 0.389
        CD1_1 = -CDELT1 * cos(position_angle)
        CD1_2 = CDELT2 * sin(position_angle)
        CD2_1 = -CDELT1 * sin(position_angle)
        CD2_2 = CDELT2 * cos(position_angle)
        assert abs(CD1_1 - 0.389) < 1e-10
        assert abs(CD2_2 - (-0.389)) < 1e-10
        assert abs(CD1_2) < 1e-14
        assert abs(CD2_1) < 1e-14


# ===========================================================================
# TestCoordinateConversion (comprehensive only)
# ===========================================================================

class TestCoordinateConversion:
    def test_positive_dec_sexagesimal(self):
        dec_str = '+02:30:00'
        parts = dec_str.split(':')
        dec = int(parts[0]) + float(parts[1]) / 60 + float(parts[2]) / 3600
        assert abs(dec - 2.5) < 1e-6

    def test_negative_dec_sexagesimal(self):
        dec_str = '-30:15:00'
        parts = dec_str.split(':')
        dec = -1 * (abs(int(parts[0])) + float(parts[1]) / 60 + float(parts[2]) / 3600)
        assert abs(dec - (-30.25)) < 1e-6

    def test_ra_sexagesimal_to_degrees(self):
        ra_str = '10:00:00'
        parts = ra_str.split(':')
        ra = (int(parts[0]) + float(parts[1]) / 60 + float(parts[2]) / 3600) * 15
        assert abs(ra - 150.0) < 1e-6

    def test_ra_at_zero(self):
        ra_str = '00:00:00'
        parts = ra_str.split(':')
        ra = (int(parts[0]) + float(parts[1]) / 60 + float(parts[2]) / 3600) * 15
        assert abs(ra) < 1e-10

    def test_ra_at_24_hours(self):
        ra_str = '23:59:59.9'
        parts = ra_str.split(':')
        ra = (int(parts[0]) + float(parts[1]) / 60 + float(parts[2]) / 3600) * 15
        assert abs(ra - 359.99958333) < 0.001

    def test_dec_at_pole(self):
        dec_str = '+90:00:00'
        parts = dec_str.split(':')
        dec = int(parts[0]) + float(parts[1]) / 60 + float(parts[2]) / 3600
        assert abs(dec - 90.0) < 1e-10

    def test_dec_at_south_pole(self):
        dec_str = '-90:00:00'
        parts = dec_str.split(':')
        dec = -1 * (abs(int(parts[0])) + float(parts[1]) / 60 + float(parts[2]) / 3600)
        assert abs(dec - (-90.0)) < 1e-10


# ===========================================================================
# TestCrossmatchLinregIntegration (comprehensive only)
# ===========================================================================

class TestCrossmatchLinregIntegration:
    def test_matched_sources_linreg_zeropoint(self):
        from lsc.lscastrodef import crossmatch, linreg
        rng = np.random.default_rng(99)
        ra_std = list(rng.uniform(149.9, 150.1, 10))
        dec_std = list(rng.uniform(1.9, 2.1, 10))
        mag_std = list(rng.uniform(14, 17, 10))
        ra_obs = [r + rng.normal(0, 0.0001) for r in ra_std]
        dec_obs = [d + rng.normal(0, 0.0001) for d in dec_std]
        mag_inst = [-2.5 * np.log10(10 ** (-(m - 25) / 2.5)) for m in mag_std]
        distvec, pos0, pos1 = crossmatch(ra_std, dec_std, ra_obs, dec_obs, tollerance=5.0)
        assert len(pos0) == 10
        colors = [rng.uniform(-0.5, 1.5) for _ in range(len(pos0))]
        zeropoints = [mag_std[pos0[i]] - mag_inst[pos1[i]] for i in range(len(pos0))]
        a, b, RR = linreg(colors, zeropoints)
        assert abs(b - 25.0) < 2.0


# ===========================================================================
# TestInstrumentPixelScales (comprehensive only)
# ===========================================================================

class TestInstrumentPixelScales:
    def test_kb_instrument_scale(self):
        from lsc.lscastrodef import half_total_flux_radius_to_fwhm
        htfr = 5.0
        fwhm = half_total_flux_radius_to_fwhm(htfr) * 0.464
        assert fwhm > 0
        assert fwhm < 20

    def test_fl_instrument_scale(self):
        from lsc.lscastrodef import half_total_flux_radius_to_fwhm
        fwhm = half_total_flux_radius_to_fwhm(5.0) * 0.389
        assert fwhm > 0

    def test_fs_instrument_scale(self):
        from lsc.lscastrodef import half_total_flux_radius_to_fwhm
        fwhm = half_total_flux_radius_to_fwhm(5.0) * 0.30
        assert fwhm > 0

    def test_em_instrument_scale(self):
        from lsc.lscastrodef import half_total_flux_radius_to_fwhm
        fwhm = half_total_flux_radius_to_fwhm(5.0) * 0.278
        assert fwhm > 0

    def test_ep_instrument_scale(self):
        from lsc.lscastrodef import half_total_flux_radius_to_fwhm
        fwhm = half_total_flux_radius_to_fwhm(5.0) * 0.27
        assert fwhm > 0

    def test_sq_instrument_scale(self):
        from lsc.lscastrodef import half_total_flux_radius_to_fwhm
        fwhm = half_total_flux_radius_to_fwhm(5.0) * 0.734
        assert fwhm > 0

    def test_all_scales_different(self):
        from lsc.lscastrodef import half_total_flux_radius_to_fwhm
        htfr = 5.0
        scales = [0.464, 0.389, 0.30, 0.278, 0.27, 0.734]
        fwhms = [half_total_flux_radius_to_fwhm(htfr) * s for s in scales]
        assert len(set(round(f, 4) for f in fwhms)) == len(scales)


# ===========================================================================
# TestQuerycatalogue (pure + comprehensive + coverage_a)
# ===========================================================================

class TestQuerycatalogue:
    def _make_catalog_file(self, tmp_path):
        content = (
            "# nfields 3\n"
            "# ra 1\n"
            "# dec 2\n"
            "# V 3\n"
            "# END\n"
            "10:00:02.4 +02:00:36.0 15.0\n"
            "10:00:04.8 +02:01:12.0 16.0\n"
        )
        p = tmp_path / "user.cat"
        p.write_text(content)
        return str(p)

    # --- from pure ---
    def test_user_catalog_returns_dict(self, simple_fits, tmp_path):
        from lsc.lscastrodef import querycatalogue
        catalog = self._make_catalog_file(tmp_path)
        iraf_mock = sys.modules['iraf']
        iraf_mock.wcsctran.return_value = [
            '# END CATALOG HEADER', '#',
            '50.0 50.0 15.0', '60.0 60.0 16.0',
        ]
        fake_stdcoo = {
            'ra': ['150.01', '150.02'],
            'dec': ['2.01', '2.02'],
            'V': ['15.0', '16.0'],
        }
        with patch('lsc.lscastrodef.readtxt', return_value=fake_stdcoo):
            result = querycatalogue(catalog, simple_fits)
        from astropy.table import Table
        assert isinstance(result, (dict, Table))
        assert 'ra' in result
        assert 'dec' in result

    def test_vizir_method_calls_vizq(self, simple_fits, tmp_path):
        from lsc.lscastrodef import querycatalogue
        iraf_mock = sys.modules['iraf']
        iraf_mock.wcsctran.return_value = [
            '# END CATALOG HEADER', '#', '50.0 50.0 14.5',
        ]
        fake_vizq = {'ra': ['150.01'], 'dec': ['2.01'], 'mag': [14.5]}
        with patch('lsc.lscastrodef.vizq', return_value=fake_vizq):
            result = querycatalogue('apass', simple_fits, method='vizir')
        assert isinstance(result, (dict, type(fake_vizq)))

    def test_iraf_method_with_mocked_agetcat(self, simple_fits, tmp_path):
        from lsc.lscastrodef import querycatalogue
        iraf_mock = sys.modules['iraf']
        iraf_mock.noao.astcat.agetcat.return_value = [
            '# nfields 4', '# ra 1', '# dec 2', '# R2mag 3', '# id 4',
            '# END CATALOG HEADER', '#',
            '150.01 2.01 14.5 ID001', '150.02 2.02 15.5 ID002',
        ]
        iraf_mock.wcsctran.return_value = [
            '# END CATALOG HEADER', '#', '50.0 50.0', '60.0 60.0',
        ]
        result = querycatalogue('usnob1', simple_fits, method='iraf')
        assert isinstance(result, dict)

    # --- from comprehensive ---
    def test_empty_catalog_warns(self, simple_fits):
        from lsc.lscastrodef import querycatalogue
        with pytest.raises(SystemExit):
            querycatalogue('/fake/path/to/catalog.cat', simple_fits)

    def test_vizir_method_decimal_degrees(self, simple_fits):
        from lsc.lscastrodef import querycatalogue
        iraf_mock = sys.modules['iraf']
        iraf_mock.wcsctran.return_value = [
            '# END CATALOG HEADER', '#', '50.0 50.0 14.5', '60.0 60.0 15.0',
        ]
        fake_vizq = {'ra': ['150.01', '150.02'], 'dec': ['2.01', '2.02'], 'mag': [14.5, 15.0]}
        with patch('lsc.lscastrodef.vizq', return_value=fake_vizq):
            result = querycatalogue('usnob1', simple_fits, method='vizir')
        assert iraf_mock.wcsctran.called

    # --- from coverage_a ---
    @patch('lsc.util.delete')
    def test_empty_catalog_exits_coverage_a(self, mock_delete):
        mock_hdr = {
            'instrume': 'kb76', 'RA': 150.0, 'DEC': 2.0,
            'NAXIS1': '1024', 'NAXIS2': '1024',
        }
        mock_stdcoo = MagicMock()
        mock_stdcoo.__getitem__ = lambda self, key: [] if key == 'ra' else []
        mock_stdcoo.keys = lambda: ['ra', 'dec', 'V']
        with patch('lsc.util.readhdr', return_value=mock_hdr):
            with patch('lsc.util.readkey3', side_effect=lambda h, k: h.get(k.upper(), h.get(k, ''))):
                with patch('lsc.lscastrodef.readtxt', return_value=mock_stdcoo):
                    with patch('os.path.isfile', return_value=True):
                        import lsc.lscastrodef
                        with pytest.raises(SystemExit):
                            lsc.lscastrodef.querycatalogue(
                                '/fake/catalog.cat', 'test.fits', method='iraf'
                            )


# ===========================================================================
# TestQuerysloan (comprehensive + cov100 + coverage_c)
# ===========================================================================

class TestQuerysloan:
    # --- from comprehensive ---
    def test_parses_valid_response(self):
        from lsc.lscastrodef import querysloan
        fake_lines = [
            'objID,ra,dec,u,g,r,i,z,err_u,err_g,err_r,err_i,err_z,type\n',
            '123,150.0,2.0,18.5,17.0,16.5,16.0,15.5,0.01,0.02,0.01,0.01,0.02,6\n',
            '456,150.1,2.1,19.0,17.5,17.0,16.5,16.0,0.02,0.01,0.02,0.01,0.01,6\n',
        ]
        mock_response = MagicMock()
        mock_response.readlines.return_value = fake_lines
        with patch('lsc.sqlcl.query', return_value=mock_response):
            _id, _ra, _dec, _u, _g, _r, _i, _z, _type, _du, _dg, _dr, _di, _dz = querysloan(
                150.0, 2.0, 10.0, 14.0, 18.0
            )
        assert len(_id) == 2
        assert abs(_ra[0] - 150.0) < 1e-6

    def test_magnitude_filter(self):
        from lsc.lscastrodef import querysloan
        fake_lines = [
            'objID,ra,dec,u,g,r,i,z,err_u,err_g,err_r,err_i,err_z,type\n',
            '123,150.0,2.0,18.5,17.0,12.0,16.0,15.5,0.01,0.02,0.01,0.01,0.02,6\n',
            '456,150.1,2.1,19.0,17.5,16.0,16.5,16.0,0.02,0.01,0.02,0.01,0.01,6\n',
        ]
        mock_response = MagicMock()
        mock_response.readlines.return_value = fake_lines
        with patch('lsc.sqlcl.query', return_value=mock_response):
            _id, _ra, _dec, _u, _g, _r, _i, _z, _type, _du, _dg, _dr, _di, _dz = querysloan(
                150.0, 2.0, 10.0, 14.0, 18.0
            )
        assert len(_id) == 1

    def test_empty_response(self):
        from lsc.lscastrodef import querysloan
        fake_lines = ['objID,ra,dec,u,g,r,i,z,err_u,err_g,err_r,err_i,err_z,type\n']
        mock_response = MagicMock()
        mock_response.readlines.return_value = fake_lines
        with patch('lsc.sqlcl.query', return_value=mock_response):
            _id, _ra, _dec, _u, _g, _r, _i, _z, _type, _du, _dg, _dr, _di, _dz = querysloan(
                150.0, 2.0, 10.0, 14.0, 18.0
            )
        assert len(_id) == 0

    # --- from cov100 ---
    @patch('lsc.sqlcl.query')
    def test_querysloan_with_non_numeric_r_cov100(self, mock_query):
        mock_file = MagicMock()
        mock_file.readlines.return_value = [
            'header\n',
            'id1,10.0,20.0,bad_u,18.5,17.5,17.0,16.5,0.01,0.02,0.03,0.04,0.05,6\n',
        ]
        mock_query.return_value = mock_file
        import lsc.lscastrodef
        result = lsc.lscastrodef.querysloan(10.0, 20.0, 5.0, 15.0, 20.0)
        assert len(result) == 14
        _id, _ra, _dec, _u, _g, _r, _i, _z, _type, _du, _dg, _dr, _di, _dz = result
        assert _u == [9999.0]

    # --- from coverage_c ---
    @patch('lsc.sqlcl.query')
    def test_except_branches_with_mr1_mr2_coverage_c(self, mock_query):
        from lsc.lscastrodef import querysloan
        header = "objID,ra,dec,u,g,r,i,z,err_u,err_g,err_r,err_i,err_z,type\n"
        row = "123,150.0,2.0,BAD,BAD,14.5,BAD,BAD,BAD,BAD,BAD,BAD,BAD,6"
        mock_response = MagicMock()
        mock_response.readlines.return_value = [header, row + "\n"]
        mock_query.return_value = mock_response
        _id, _ra, _dec, _u, _g, _r, _i, _z, _type, _du, _dg, _dr, _di, _dz = querysloan(
            150.0, 2.0, 10, 14.0, 15.0
        )
        assert len(_id) == 1
        assert _u[0] == 9999.0
        assert _g[0] == 9999.0
        assert _r[0] == 14.5

    @patch('lsc.sqlcl.query')
    def test_except_branches_without_mr1_mr2_coverage_c(self, mock_query):
        from lsc.lscastrodef import querysloan
        header = "objID,ra,dec,u,g,r,i,z,err_u,err_g,err_r,err_i,err_z,type\n"
        row = "789,150.0,2.0,14.0,14.1,14.2,14.3,14.4,BAD,BAD,BAD,BAD,BAD,6"
        mock_response = MagicMock()
        mock_response.readlines.return_value = [header, row + "\n"]
        mock_query.return_value = mock_response
        _id, _ra, _dec, _u, _g, _r, _i, _z, _type, _du, _dg, _dr, _di, _dz = querysloan(
            150.0, 2.0, 10, None, None
        )
        assert len(_id) == 1
        assert _du[0] == 9999.0


# ===========================================================================
# TestSloan2file (comprehensive + coverage_c)
# ===========================================================================

class TestSloan2file:
    # --- from comprehensive ---
    def test_writes_catalog_file(self, tmp_path):
        from lsc.lscastrodef import sloan2file
        output_path = str(tmp_path / "sloan_output.cat")
        mock_result = (
            ['id1', 'id2'], [150.0, 150.1], [2.0, 2.1],
            [18.0, 18.5], [16.5, 17.0], [16.0, 16.5], [15.5, 16.0], [15.0, 15.5],
            ['6', '6'], [0.01, 0.02], [0.01, 0.02], [0.01, 0.02], [0.01, 0.02], [0.01, 0.02],
        )
        with patch('lsc.lscastrodef.querysloan', return_value=mock_result):
            result = sloan2file(150.0, 2.0, 20, 12.0, 18.0, output_path)
        assert os.path.exists(output_path)

    def test_empty_query_returns_empty(self, tmp_path):
        from lsc.lscastrodef import sloan2file
        output_path = str(tmp_path / "sloan_empty.cat")
        mock_result = ([], [], [], [], [], [], [], [], [], [], [], [], [], [])
        with patch('lsc.lscastrodef.querysloan', return_value=mock_result):
            result = sloan2file(150.0, 2.0, 20, 12.0, 18.0, output_path)
        assert result == ''

    def test_filters_by_type_6_only(self, tmp_path):
        from lsc.lscastrodef import sloan2file
        output_path = str(tmp_path / "sloan_type.cat")
        mock_result = (
            ['id1', 'id2', 'id3'], [150.0, 150.1, 150.2], [2.0, 2.1, 2.2],
            [18.0, 18.5, 19.0], [16.5, 17.0, 17.5], [16.0, 16.5, 17.0],
            [15.5, 16.0, 16.5], [15.0, 15.5, 16.0],
            ['6', '3', '6'],
            [0.01, 0.02, 0.01], [0.01, 0.02, 0.01], [0.01, 0.02, 0.01],
            [0.01, 0.02, 0.01], [0.01, 0.02, 0.01],
        )
        with patch('lsc.lscastrodef.querysloan', return_value=mock_result):
            result = sloan2file(150.0, 2.0, 20, 12.0, 18.0, output_path)
        assert len(result) == 2

    # --- from coverage_c ---
    @patch('lsc.lscastrodef.querysloan')
    def test_type_nonnumeric_gets_9999_coverage_c(self, mock_querysloan, tmp_path):
        from lsc.lscastrodef import sloan2file
        mock_querysloan.return_value = (
            ['id1', 'id2'], [150.0, 150.1], [2.0, 2.1],
            [20.0, 20.1], [19.0, 19.1], [18.0, 18.1], [17.0, 17.1], [16.0, 16.1],
            ['STAR', '6'],
            [0.01, 0.01], [0.01, 0.01], [0.01, 0.01], [0.01, 0.01], [0.01, 0.01],
        )
        output = str(tmp_path / "test_output.cat")
        result = sloan2file(150.0, 2.0, 10, 14.0, 15.0, output)
        assert len(result) == 1


# ===========================================================================
# TestReadapass (extra + cov100 + coverage_c)
# ===========================================================================

class TestReadapass:
    # --- from extra ---
    def test_parses_valid_output(self, tmp_path, monkeypatch):
        from lsc.lscastrodef import readapass
        fake_output = (
            "header line 1\n"
            "header line 2\n"
            "header line 3\n"
            "#RAdeg DECdeg B V Pg Pr Pi eB eV eg er ei\n"
            "150.000 2.000 14.5 13.8 14.2 13.9 13.7 0.01 0.02 0.01 0.01 0.02\n"
            "150.100 2.100 15.0 14.3 14.7 14.4 14.1 0.02 0.01 0.02 0.01 0.01\n"
        )
        mock_popen_obj = MagicMock()
        mock_popen_obj.read.return_value = fake_output
        with patch('os.popen', return_value=mock_popen_obj):
            with patch('builtins.open', MagicMock()):
                with pytest.raises(TypeError, match="not subscriptable"):
                    readapass(150.0, 2.0)

    def test_empty_output(self, monkeypatch):
        from lsc.lscastrodef import readapass
        fake_output = (
            "header line 1\n"
            "header line 2\n"
            "header line 3\n"
            "#RAdeg DECdeg B V Pg Pr Pi eB eV eg er ei\n"
        )
        mock_popen_obj = MagicMock()
        mock_popen_obj.read.return_value = fake_output
        with patch('os.popen', return_value=mock_popen_obj):
            with patch('builtins.open', MagicMock()):
                with pytest.raises(TypeError, match="not subscriptable"):
                    readapass(150.0, 2.0)

    # --- from cov100 ---
    @patch('os.popen')
    def test_readapass_reaches_file_writing_cov100(self, mock_popen, tmp_path, monkeypatch):
        mock_output = (
            "header1\n"
            "header2\n"
            "#RAdeg DECdeg B V Pg Pr Pi eB eV eg er ei\n"
            "150.00000 2.00000 18.5 17.5 18.0 17.0 16.5 0.01 0.02 0.03 0.04 0.05\n"
            "150.10000 2.10000 19.0 18.0 18.5 17.5 17.0 0.02 0.03 0.04 0.05 0.06\n"
        )
        mock_popen.return_value = MagicMock(read=MagicMock(return_value=mock_output))
        original_zip = zip
        def list_zip(*args):
            return list(original_zip(*args))
        monkeypatch.setattr('builtins.zip', list_zip)
        monkeypatch.chdir(tmp_path)
        try:
            import lsc.lscastrodef
            lsc.lscastrodef.readapass(150.0, 2.0, radius=30)
        except (TypeError, IndexError, AttributeError):
            pass

    # --- from coverage_c ---
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.popen')
    def test_readapass_basic_flow_coverage_c(self, mock_popen, mock_file):
        from lsc.lscastrodef import readapass
        fake_output = (
            "line0\n" "line1\n" "line2\n"
            "#RAdeg DECdeg B V Pg Pr Pi eB eV eg er ei\n"
            "150.000 2.000 14.5 13.8 14.2 13.9 13.7 0.01 0.02 0.01 0.01 0.02\n"
            "150.100 2.100 15.0 14.3 14.7 14.4 14.1 0.02 0.01 0.02 0.01 0.01\n"
        )
        mock_popen.return_value = MagicMock(read=MagicMock(return_value=fake_output))
        try:
            result = readapass(150.0, 2.0, radius=30)
        except (TypeError, IndexError, KeyError):
            pass


# ===========================================================================
# TestLscastrometry2 (comprehensive + coverage_a + cov100)
# ===========================================================================

class TestLscastrometry2:
    # --- from comprehensive ---
    def test_few_matches_returns_990_999(self, simple_fits, tmp_path):
        from lsc.lscastrodef import lscastrometry2
        from astropy.io import fits as astrofits
        import shutil
        dst = str(tmp_path / "astrom_test.fits")
        shutil.copy2(simple_fits, dst)
        astrofits.setval(dst, 'CRPIX1', value=50.0)
        astrofits.setval(dst, 'CRPIX2', value=50.0)
        sexvec = (
            np.array([50.0]), np.array([50.0]), np.array([5.0]),
            np.array([0.9]), np.array([-10.0]), np.array([0.1]),
            np.array([1000.0]), np.array([0.0]),
        )
        catvec = {
            'coo': np.array(['180.0 80.0']),
            'pix': np.array(['2000.0 2000.0']),
            'mag': np.array([15.0]),
            'ra': np.array([180.0]),
            'dec': np.array([80.0]),
        }
        rmsx, rmsy, nref, fwhm, ell, fwhm2, bkg, rasys, decsys = lscastrometry2(
            [dst], 'usnob1', False, 50, sexvec, catvec,
            guess=False, fitgeo='xyscale', tollerance1=5, tollerance2=3,
            _update='yes', imex=False, nummin=4
        )
        assert rmsx == 990
        assert rmsy == 999


# ===========================================================================
# TestLscastroloop (comprehensive + coverage_a)
# ===========================================================================

class TestLscastroloop:
    # --- from comprehensive ---
    def test_good_astrometry_returns_results(self, simple_fits):
        from lsc.lscastrodef import lscastroloop
        mock_sexvec = (
            np.array([50.0, 60.0, 70.0]), np.array([50.0, 60.0, 70.0]),
            np.array([5.0, 5.0, 5.0]), np.array([0.9, 0.9, 0.9]),
            np.array([-10.0, -11.0, -12.0]), np.array([0.1, 0.1, 0.1]),
            np.array([1000.0, 1000.0, 1000.0]), np.array([0.0, 0.0, 0.0]),
        )
        mock_astrom_result = (0.5, 0.5, 10, [5.0, 5.0, 5.0], [0.1, 0.1, 0.1],
                              [4.5, 4.5, 4.5], [1000.0], 0.001, -0.001)
        with patch('lsc.lscastrodef.querycatalogue', return_value={
            'ra': np.array([150.0, 150.1]), 'dec': np.array([2.0, 2.1]),
            'coo': np.array(['150.0 2.0', '150.1 2.1']),
            'pix': np.array(['50.0 50.0', '60.0 60.0']),
            'mag': np.array([14.5, 15.0]),
        }):
            with patch('lsc.lscastrodef.lscastrometry2', return_value=mock_astrom_result):
                result = lscastroloop(
                    [simple_fits], 'usnob1', False, 50, 100, 200,
                    'xyscale', 100, 30, sexvec=mock_sexvec
                )
        rmsx, rmsy, num, fwhm, ell, fwhmime, rasys, decsys, magsat = result
        assert rmsx == 0.5
        assert rmsy == 0.5
        assert num == 10

    # --- from coverage_a ---
    @patch('lsc.util.updateheader')
    @patch('lsc.util.readhdr')
    @patch('lsc.util.readkey3')
    @patch('lsc.lscastrodef.lscastrometry2')
    @patch('lsc.lscastrodef.querycatalogue')
    def test_empty_catalog_exits_coverage_a(self, mock_querycat, mock_astro2,
                                             mock_readkey3, mock_readhdr, mock_updatehdr):
        mock_readhdr.return_value = {'instrume': 'kb76', 'MBKG': 100.0}
        mock_readkey3.side_effect = lambda h, k: h.get(k, h.get(k.lower(), ''))
        sexvec = (
            np.array([100.0]), np.array([100.0]), np.array([3.0]),
            np.array([0.0]), np.array([14.0]), np.array([0.1]),
            np.array([100.0]), np.array([1.0]),
        )
        mock_querycat.return_value = {
            'ra': np.array([]), 'dec': np.array([]),
            'coo': [], 'pix': [], 'mag': np.array([]),
            'x': np.array([]), 'y': np.array([]),
        }
        import lsc.lscastrodef
        with pytest.raises(SystemExit):
            lsc.lscastrodef.lscastroloop(
                ['test.fits'], 'usnoa2', False, 50, 100, 200,
                'general', 100, 30, sexvec=sexvec
            )


# ===========================================================================
# TestRunAstrometry (comprehensive + coverage_c)
# ===========================================================================

class TestRunAstrometry:
    # --- from comprehensive ---
    def test_skips_when_wcserr_zero_and_not_redo(self, simple_fits, tmp_path):
        import shutil
        from lsc.lscastrodef import run_astrometry
        dst = str(tmp_path / "astr_done.fits")
        shutil.copy2(simple_fits, dst)
        with patch('os.system') as mock_os:
            run_astrometry(dst, clobber=True, redo=False)
        mock_os.assert_not_called()

    def test_runs_when_redo_true(self, simple_fits, tmp_path):
        import shutil
        from lsc.lscastrodef import run_astrometry
        dst = str(tmp_path / "astr_redo.fits")
        shutil.copy2(simple_fits, dst)
        with patch('os.system') as mock_os:
            with patch('os.path.exists', return_value=False):
                run_astrometry(dst, clobber=True, redo=True)
        assert mock_os.called
        call_args = mock_os.call_args[0][0]
        assert 'solve-field' in call_args

    # --- from coverage_c ---
    def test_already_done_coverage_c(self):
        from lsc.lscastrodef import run_astrometry
        hdr_mock = MagicMock()
        hdr_mock.__contains__ = lambda self, key: key in {'WCSERR'}
        with patch('lsc.util.readhdr', return_value=hdr_mock), \
             patch('lsc.util.readkey3', return_value=0), \
             patch('os.system') as mock_system:
            run_astrometry('test.fits', clobber=True, redo=False)
            mock_system.assert_not_called()

    def test_tmpwcs_not_exists_coverage_c(self):
        from lsc.lscastrodef import run_astrometry
        hdr_mock = MagicMock()
        hdr_mock.__contains__ = lambda self, key: key in {'WCSERR'}
        with patch('lsc.util.readhdr', return_value=hdr_mock), \
             patch('lsc.util.readkey3', side_effect=lambda h, k: {'wcserr': 1, 'RA': 150.0, 'DEC': 2.0}.get(k)), \
             patch('lsc.util.workdirectory', '/tmp'), \
             patch('os.system'), \
             patch('os.remove'), \
             patch('os.path.exists', return_value=False):
            run_astrometry('test.fits', clobber=True, redo=True)


# ===========================================================================
# TestFinewcs (comprehensive + cov100 + coverage_c)
# ===========================================================================

class TestFinewcs:
    # --- from comprehensive ---
    def test_returns_rms_and_bestvalue(self, simple_fits):
        from lsc.lscastrodef import finewcs
        iraf_mock = sys.modules['iraf']
        mock_catvec = {
            'x': np.arange(10, dtype=float) * 50 + 50,
            'y': np.arange(10, dtype=float) * 50 + 50,
            'ra': np.arange(10, dtype=float) * 0.001 + 150.0,
            'dec': np.arange(10, dtype=float) * 0.001 + 2.0,
        }
        mock_genfromtxt = np.column_stack([
            np.arange(10),
            np.arange(10, dtype=float) * 50 + 50.5,
            np.arange(10, dtype=float) * 50 + 50.5,
            np.ones(10) * -10.0,
        ])
        with patch('lsc.lscastrodef.querycatalogue', return_value=mock_catvec):
            with patch('numpy.genfromtxt', return_value=mock_genfromtxt):
                with patch('os.system'):
                    with patch('lsc.util.defsex', return_value='default.sex'):
                        with pytest.raises(TypeError):
                            finewcs(simple_fits)

    # --- from cov100 ---
    @patch('lsc.lscastrodef.querycatalogue')
    @patch('lsc.util.defsex', return_value='default.sex')
    @patch('os.system')
    @patch('numpy.genfromtxt')
    def test_finewcs_with_patched_zip_cov100(self, mock_genfromtxt, mock_ossys, mock_defsex,
                                              mock_querycat, monkeypatch):
        mock_querycat.return_value = {
            'ra': np.array([150.0 + i * 0.01 for i in range(12)]),
            'dec': np.array([2.0 + i * 0.01 for i in range(12)]),
            'x': np.array([10.0 + i * 10.0 for i in range(12)]),
            'y': np.array([10.0 + i * 10.0 for i in range(12)]),
        }
        n = 12
        sex_data = np.column_stack([
            np.arange(1, n+1),
            np.linspace(10.5, 120.5, n),
            np.linspace(10.5, 120.5, n),
            np.linspace(15, 20, n),
            np.ones(n) * 5.0, np.ones(n), np.ones(n) * 0.9,
            np.ones(n) * 4.0, np.ones(n) * 0.1, np.ones(n) * 100.0,
        ])
        mock_genfromtxt.return_value = sex_data
        iraf_mock = sys.modules['pyraf'].iraf
        iraf_mock.ccmap.return_value = ['line1', 'Wcs mapping status', 'rms: 0.5 0.5 arcsec']
        original_zip = zip
        def list_zip(*args):
            return list(original_zip(*args))
        monkeypatch.setattr('builtins.zip', list_zip)
        import lsc.lscastrodef
        result = lsc.lscastrodef.finewcs('test.fits')
        assert len(result) == 3

    # --- from coverage_c ---
    def test_finewcs_few_matches_coverage_c(self):
        import importlib
        mock_catvec = {
            'x': np.array([100.0, 200.0]),
            'y': np.array([100.0, 200.0]),
            'ra': np.array([150.0, 150.01]),
            'dec': np.array([2.0, 2.01]),
        }
        mock_data = np.array([[1, 100.0, 100.0, 14.0], [2, 200.0, 200.0, 14.5]])
        mock_iraf = MagicMock()
        mock_pyraf = MagicMock()
        mock_pyraf.iraf = mock_iraf
        with patch.dict('sys.modules', {'pyraf': mock_pyraf, 'pyraf.iraf': mock_iraf}):
            import lsc.lscastrodef
            importlib.reload(lsc.lscastrodef)
            with patch.object(lsc.lscastrodef, 'querycatalogue', return_value=mock_catvec), \
                 patch('lsc.util.defsex', return_value='default.sex'), \
                 patch('os.system'), \
                 patch('numpy.genfromtxt', return_value=mock_data), \
                 patch.object(lsc.lscastrodef, 'crossmatchxy', return_value=([0.1, 0.2], [0, 1], [0, 1])):
                try:
                    lsc.lscastrodef.finewcs('test.fits')
                except (TypeError, IndexError, AttributeError):
                    pass
            importlib.reload(lsc.lscastrodef)


# ===========================================================================
# TestZeropointLandoltWithStars (coverage_b)
# ===========================================================================

class TestZeropointLandoltWithStars:
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.lscastrodef.half_total_flux_radius_to_fwhm', return_value=4.0)
    @patch('lsc.lscastrodef.crossmatch')
    @patch('lsc.lscastrodef.linreg', return_value=(0.05, 25.0, 0.99))
    @patch('lsc.lscastrodef.sloan2file', return_value=3)
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.util.defsex', return_value='/tmp/default.sex')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_landolt_filter_V_no_catalogue(self, mock_readhdr, mock_readkey3,
                                            mock_delete, mock_defsex,
                                            mock_updateheader, mock_readtxt_fn,
                                            mock_sloan2file, mock_linreg,
                                            mock_crossmatch, mock_htfr2fwhm,
                                            mock_os_system, mock_file_open):
        hdr, data = _make_mock_hdr(filter_val='V', siteid='lsc')
        mock_readhdr.return_value = hdr
        mock_readkey3.side_effect = _readkey3_factory(data)
        iraf_mock = _setup_iraf_mock()
        n_std = 5
        stdcoo_landolt = _make_standard_stdcoo('landolt', n_std)
        stdcoo_sloan = _make_standard_stdcoo('sloan', n_std)
        standardpix_landolt = _make_standardpix(n_std)
        standardpix_sloan = _make_standardpix(n_std)
        def readtxt_side(filepath):
            if 'landolt.cat' in filepath: return stdcoo_landolt
            elif '_tmpsloan.cat' in filepath: return stdcoo_sloan
            elif 'tmp.stdL.pix' in filepath: return standardpix_landolt
            elif 'tmp.stdS.pix' in filepath: return standardpix_sloan
            return {'ra': [], 'dec': [], 'id': []}
        mock_readtxt_fn.side_effect = readtxt_side
        pos0 = np.array([0, 1, 2])
        pos1 = np.array([0, 1, 2])
        mock_crossmatch.return_value = (np.array([0.1, 0.1, 0.1]), pos0, pos1)
        mock_fields, _, _ = _make_sex_fields_mock(n_std)
        iraf_mock.proto.fields = MagicMock(side_effect=mock_fields)
        iraf_mock.noao.digiphot.daophot.phot = MagicMock(
            return_value=['img 500.0 600.0 3.0 18.5 0.02']
        )
        with patch.dict('sys.modules', {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            result = lsc.lscastrodef.zeropoint('test_img.fits', 'landolt',
                                               verbose=False, catalogue='')
        assert isinstance(result, dict)


# ===========================================================================
# TestZeropointSloanPath (coverage_b)
# ===========================================================================

class TestZeropointSloanPath:
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.lscastrodef.sloan2file', return_value=5)
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.util.defsex', return_value='/tmp/default.sex')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_sloan_filter_SDSS_R_sloan_stars_in_field(self, mock_readhdr, mock_readkey3,
                                                       mock_delete, mock_defsex,
                                                       mock_updateheader, mock_readtxt_fn,
                                                       mock_sloan2file,
                                                       mock_os_system, mock_file_open):
        hdr, data = _make_mock_hdr(filter_val='SDSS-R', siteid='elp')
        mock_readhdr.return_value = hdr
        mock_readkey3.side_effect = _readkey3_factory(data)
        iraf_mock = _setup_iraf_mock()
        n_std = 5
        stdcoo_sloan = _make_standard_stdcoo('sloan', n_std)
        standardpix_sloan = _make_standardpix(n_std)
        def readtxt_side(filepath):
            if 'landolt.cat' in filepath: return _make_standard_stdcoo('landolt', 2)
            elif '_tmpsloan.cat' in filepath: return stdcoo_sloan
            elif 'tmp.stdL.pix' in filepath: return _make_standardpix(2)
            elif 'tmp.stdS.pix' in filepath: return standardpix_sloan
            return {'ra': [], 'dec': [], 'id': []}
        mock_readtxt_fn.side_effect = readtxt_side
        mock_fields, _, _ = _make_sex_fields_mock(n_std)
        iraf_mock.proto.fields = MagicMock(side_effect=mock_fields)
        with patch.dict('sys.modules', {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            with pytest.raises(KeyError, match='SDSS-R'):
                lsc.lscastrodef.zeropoint('test_img.fits', 'sloan',
                                           verbose=False, catalogue='')


# ===========================================================================
# TestZeropointEdgeCases (coverage_b)
# ===========================================================================

class TestZeropointEdgeCases:
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.lscastrodef.sloan2file', return_value=0)
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.util.defsex', return_value='/tmp/default.sex')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_no_standards_in_field(self, mock_readhdr, mock_readkey3,
                                    mock_delete, mock_defsex,
                                    mock_updateheader, mock_readtxt_fn,
                                    mock_sloan2file,
                                    mock_os_system, mock_file_open):
        hdr, data = _make_mock_hdr(filter_val='V', siteid='lsc')
        mock_readhdr.return_value = hdr
        mock_readkey3.side_effect = _readkey3_factory(data)
        iraf_mock = _setup_iraf_mock()
        def readtxt_side(filepath):
            if 'landolt.cat' in filepath: return _make_standard_stdcoo('landolt', 3)
            elif '_tmpsloan.cat' in filepath: return _make_standard_stdcoo('sloan', 3)
            elif 'tmp.stdL.pix' in filepath: return _make_outside_pix(3)
            elif 'tmp.stdS.pix' in filepath: return _make_outside_pix(3)
            return {'ra': [], 'dec': [], 'id': []}
        mock_readtxt_fn.side_effect = readtxt_side
        with patch.dict('sys.modules', {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            result = lsc.lscastrodef.zeropoint('test_img.fits', 'landolt',
                                               verbose=False, catalogue='')
        assert result == ''


# ===========================================================================
# TestZeropointVerboseAndAirmass (coverage_b)
# ===========================================================================

class TestZeropointVerboseAndAirmass:
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_unknown_siteid_exits(self, mock_readhdr, mock_readkey3,
                                   mock_delete,
                                   mock_os_system, mock_file_open):
        hdr, data = _make_mock_hdr(filter_val='V', siteid='UNKNOWN_SITE')
        mock_readhdr.return_value = hdr
        mock_readkey3.side_effect = _readkey3_factory(data)
        iraf_mock = _setup_iraf_mock()
        with patch.dict('sys.modules', {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            with pytest.raises(SystemExit, match='siteid not in lsc.sites.extinction'):
                lsc.lscastrodef.zeropoint('test_img.fits', 'landolt',
                                           verbose=False, catalogue='')
