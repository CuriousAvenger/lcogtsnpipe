"""
Tests for pure functions in lsc.externaldata.
Tested: jd2date, MJDnow, SDSS_gain_dark, northupeastleft.
No network or subprocess access required.
"""
import datetime
import os
import pytest
import numpy as np
from astropy.io import fits

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# jd2date
# ---------------------------------------------------------------------------

class TestJd2date:
    def test_j2000_epoch(self):
        from lsc.externaldata import jd2date
        # JD 2451544.5 = Jan 1, 2000 00:00:00 UTC
        result = jd2date(2451544.5)
        assert result == datetime.datetime(2000, 1, 1, 0, 0, 0)

    def test_known_date(self):
        from lsc.externaldata import jd2date
        # JD 2451545.0 = Jan 1.5, 2000 = Jan 1, 2000 12:00 UTC
        result = jd2date(2451545.0)
        assert result.year == 2000
        assert result.month == 1
        assert result.day == 1
        assert result.hour == 12

    def test_returns_datetime(self):
        from lsc.externaldata import jd2date
        assert isinstance(jd2date(2451544.5), datetime.datetime)

    def test_monotonic(self):
        from lsc.externaldata import jd2date
        d1 = jd2date(2451544.5)
        d2 = jd2date(2451574.5)  # 30 days later
        assert d2 > d1

    def test_half_day_offset(self):
        from lsc.externaldata import jd2date
        # adding 0.5 JD should add 12 hours
        d1 = jd2date(2451544.5)
        d2 = jd2date(2451545.0)
        delta = d2 - d1
        assert abs(delta.total_seconds() - 12 * 3600) < 1


# ---------------------------------------------------------------------------
# MJDnow
# ---------------------------------------------------------------------------

class TestMJDnow:
    def test_fixed_date(self):
        from lsc.externaldata import MJDnow
        # MJD epoch: Jan 1, 2012 = MJD 55927
        fixed = datetime.datetime(2012, 1, 1, 0, 0, 0)
        result = MJDnow(datenow=fixed)
        assert abs(result - 55927.0) < 1e-6

    def test_one_day_later(self):
        from lsc.externaldata import MJDnow
        d0 = datetime.datetime(2012, 1, 1, 0, 0, 0)
        d1 = datetime.datetime(2012, 1, 2, 0, 0, 0)
        assert abs(MJDnow(datenow=d1) - MJDnow(datenow=d0) - 1.0) < 1e-6

    def test_no_arg_returns_float(self):
        from lsc.externaldata import MJDnow
        result = MJDnow()
        assert isinstance(result, float)
        # Should be a plausible MJD (>2000 = MJD 51545)
        assert result > 51545

    def test_verbose_branch(self, capsys):
        from lsc.externaldata import MJDnow
        fixed = datetime.datetime(2012, 1, 1, 0, 0, 0)
        MJDnow(datenow=fixed, verbose=True)
        out = capsys.readouterr().out
        assert "JD=" in out or "55927" in out

    def test_one_year_difference(self):
        from lsc.externaldata import MJDnow
        d0 = datetime.datetime(2012, 1, 1, 0, 0, 0)
        d1 = datetime.datetime(2013, 1, 1, 0, 0, 0)
        diff = MJDnow(datenow=d1) - MJDnow(datenow=d0)
        assert 365 <= diff <= 366


# ---------------------------------------------------------------------------
# SDSS_gain_dark — lookup table for SDSS CCD parameters
# ---------------------------------------------------------------------------

class TestSDSSGainDark:
    # --- camcol 1 ---
    def test_camcol1_u(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=1, ugriz='u', run=500)
        assert gain == 1.62
        assert dark == 9.61

    def test_camcol1_g(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=1, ugriz='g', run=500)
        assert gain == 3.32
        assert dark == 15.6025

    def test_camcol1_r(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=1, ugriz='r', run=500)
        assert gain == 4.71

    def test_camcol1_i(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=1, ugriz='i', run=500)
        assert gain == 5.165

    def test_camcol1_z(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=1, ugriz='z', run=500)
        assert gain == 4.745

    def test_camcol1_invalid_band_prints(self, capsys):
        from lsc.externaldata import SDSS_gain_dark
        try:
            SDSS_gain_dark(camcol=1, ugriz='X', run=500)
        except Exception:
            pass
        out = capsys.readouterr().out
        assert "ERROR" in out

    # --- camcol 2 ---
    def test_camcol2_u_low_run(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=2, ugriz='u', run=500)
        assert gain == 1.595

    def test_camcol2_u_high_run(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=2, ugriz='u', run=1500)
        assert gain == 1.825

    def test_camcol2_g(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=2, ugriz='g', run=500)
        assert gain == 3.855

    def test_camcol2_r(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=2, ugriz='r', run=500)
        assert gain == 4.6

    def test_camcol2_i_low_run(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=2, ugriz='i', run=500)
        assert dark == 5.76

    def test_camcol2_i_high_run(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=2, ugriz='i', run=2000)
        assert dark == 6.25

    def test_camcol2_z(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=2, ugriz='z', run=500)
        assert gain == 5.155

    # --- camcol 3 ---
    def test_camcol3_u(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=3, ugriz='u', run=500)
        assert gain == 1.59

    def test_camcol3_g(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=3, ugriz='g', run=500)
        assert gain == 3.845

    def test_camcol3_r(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=3, ugriz='r', run=500)
        assert gain == 4.72

    def test_camcol3_i(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=3, ugriz='i', run=500)
        assert gain == 4.86

    def test_camcol3_z(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=3, ugriz='z', run=500)
        assert gain == 4.885

    # --- camcol 4 ---
    def test_camcol4_u(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=4, ugriz='u', run=500)
        assert gain == 1.6

    def test_camcol4_g(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=4, ugriz='g', run=500)
        assert gain == 3.995

    def test_camcol4_r(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=4, ugriz='r', run=500)
        assert gain == 4.76

    def test_camcol4_i_low_run(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=4, ugriz='i', run=500)
        assert dark == 6.25

    def test_camcol4_i_high_run(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=4, ugriz='i', run=2000)
        assert dark == 7.5625

    def test_camcol4_z_low_run(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=4, ugriz='z', run=500)
        assert dark == 9.61

    def test_camcol4_z_high_run(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=4, ugriz='z', run=2000)
        assert dark == 12.6025

    # --- camcol 5 ---
    def test_camcol5_u(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=5, ugriz='u', run=500)
        assert gain == 1.47

    def test_camcol5_g(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=5, ugriz='g', run=500)
        assert gain == 4.05

    def test_camcol5_r(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=5, ugriz='r', run=500)
        assert gain == 4.725

    def test_camcol5_i(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=5, ugriz='i', run=500)
        assert gain == 4.64

    def test_camcol5_z_low_run(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=5, ugriz='z', run=500)
        assert dark == 1.8225

    def test_camcol5_z_high_run(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=5, ugriz='z', run=2000)
        assert dark == 2.1025

    # --- camcol 6 ---
    def test_camcol6_u(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=6, ugriz='u', run=500)
        assert gain == 2.17

    def test_camcol6_g(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=6, ugriz='g', run=500)
        assert gain == 4.035

    def test_camcol6_r(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=6, ugriz='r', run=500)
        assert gain == 4.895

    def test_camcol6_i(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=6, ugriz='i', run=500)
        assert gain == 4.76

    def test_camcol6_z(self):
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=6, ugriz='z', run=500)
        assert gain == 4.69

    def test_camcol6_invalid_band_prints(self, capsys):
        from lsc.externaldata import SDSS_gain_dark
        try:
            SDSS_gain_dark(camcol=6, ugriz='X', run=500)
        except Exception:
            pass
        out = capsys.readouterr().out
        assert "ERROR" in out

    def test_invalid_camcol_prints(self, capsys):
        from lsc.externaldata import SDSS_gain_dark
        try:
            SDSS_gain_dark(camcol=99, ugriz='u', run=500)
        except Exception:
            pass
        out = capsys.readouterr().out
        assert "ERROR" in out

    def test_returns_two_values(self):
        from lsc.externaldata import SDSS_gain_dark
        result = SDSS_gain_dark(camcol=1, ugriz='i', run=500)
        assert len(result) == 2

    def test_all_camcol1_bands(self):
        from lsc.externaldata import SDSS_gain_dark
        for band in 'ugriz':
            gain, dark = SDSS_gain_dark(camcol=1, ugriz=band, run=500)
            assert gain > 0
            assert dark >= 0

    def test_all_camcols_r_band(self):
        from lsc.externaldata import SDSS_gain_dark
        for camcol in range(1, 7):
            gain, dark = SDSS_gain_dark(camcol=camcol, ugriz='r', run=500)
            assert 3.0 < gain < 6.0

    def test_camcol2_u_run_equals_1500_prints_error(self, capsys):
        """camcol=2, u-band, run==1100 → 'ERROR in SDSS_dark_gain: RUN not set!'"""
        from lsc.externaldata import SDSS_gain_dark
        try:
            SDSS_gain_dark(camcol=2, ugriz='u', run=1100)
        except Exception:
            pass
        out = capsys.readouterr().out
        assert 'ERROR' in out

    def test_camcol2_i_run_equals_1500_prints_error(self, capsys):
        """camcol=2, i-band, run==1500 → 'ERROR in SDSS_dark_gain: RUN not set!'"""
        from lsc.externaldata import SDSS_gain_dark
        try:
            SDSS_gain_dark(camcol=2, ugriz='i', run=1500)
        except Exception:
            pass
        out = capsys.readouterr().out
        assert 'ERROR' in out

    def test_camcol2_invalid_band_prints_error(self, capsys):
        """camcol=2, invalid band → 'ERROR in SDSS_dark_gain: UGRIZ not set!'"""
        from lsc.externaldata import SDSS_gain_dark
        try:
            SDSS_gain_dark(camcol=2, ugriz='X', run=500)
        except Exception:
            pass
        out = capsys.readouterr().out
        assert 'ERROR' in out

    def test_camcol3_invalid_band_prints_error(self, capsys):
        """camcol=3, invalid band → error printed."""
        from lsc.externaldata import SDSS_gain_dark
        try:
            SDSS_gain_dark(camcol=3, ugriz='X', run=500)
        except Exception:
            pass
        out = capsys.readouterr().out
        assert 'ERROR' in out

    def test_camcol4_i_run_equals_1500_prints_error(self, capsys):
        """camcol=4, i-band, run==1500 → error printed."""
        from lsc.externaldata import SDSS_gain_dark
        try:
            SDSS_gain_dark(camcol=4, ugriz='i', run=1500)
        except Exception:
            pass
        out = capsys.readouterr().out
        assert 'ERROR' in out

    def test_camcol4_z_run_equals_1500_prints_error(self, capsys):
        """camcol=4, z-band, run==1500 → error printed."""
        from lsc.externaldata import SDSS_gain_dark
        try:
            SDSS_gain_dark(camcol=4, ugriz='z', run=1500)
        except Exception:
            pass
        out = capsys.readouterr().out
        assert 'ERROR' in out

    def test_camcol4_invalid_band_prints_error(self, capsys):
        """camcol=4, invalid band → error printed."""
        from lsc.externaldata import SDSS_gain_dark
        try:
            SDSS_gain_dark(camcol=4, ugriz='X', run=500)
        except Exception:
            pass
        out = capsys.readouterr().out
        assert 'ERROR' in out

    def test_camcol5_z_run_equals_1500_prints_error(self, capsys):
        """camcol=5, z-band, run==1500 → error printed."""
        from lsc.externaldata import SDSS_gain_dark
        try:
            SDSS_gain_dark(camcol=5, ugriz='z', run=1500)
        except Exception:
            pass
        out = capsys.readouterr().out
        assert 'ERROR' in out

    def test_camcol5_invalid_band_prints_error(self, capsys):
        """camcol=5, invalid band → error printed."""
        from lsc.externaldata import SDSS_gain_dark
        try:
            SDSS_gain_dark(camcol=5, ugriz='X', run=500)
        except Exception:
            pass
        out = capsys.readouterr().out
        assert 'ERROR' in out

def _make_wcs_header(cd1_1, cd1_2, cd2_1, cd2_2, naxis1=50, naxis2=50):
    """Create a minimal WCS header for northupeastleft tests."""
    hdr = fits.Header()
    hdr['NAXIS1'] = naxis1
    hdr['NAXIS2'] = naxis2
    hdr['CD1_1'] = cd1_1
    hdr['CD1_2'] = cd1_2
    hdr['CD2_1'] = cd2_1
    hdr['CD2_2'] = cd2_2
    hdr['CRPIX1'] = naxis1 / 2.0
    hdr['CRPIX2'] = naxis2 / 2.0
    hdr['CRVAL1'] = 150.0
    hdr['CRVAL2'] = 2.0
    hdr['DATASEC'] = '[1:50,1:50]'
    return hdr


class TestNorthupeastleft:
    def test_no_transform_needed_returns_same_data(self):
        """cd1_1 < 0, cd2_2 > 0, |cd1_1| > |cd1_2| → no transformation needed."""
        from lsc.externaldata import northupeastleft
        data = np.ones((50, 50), dtype=float)
        hdr = _make_wcs_header(cd1_1=-0.5, cd1_2=0.0, cd2_1=0.0, cd2_2=0.5)
        result_data, result_hdr = northupeastleft(data=data, header=hdr)
        assert result_data.shape == (50, 50)
        assert result_hdr['CD1_1'] == -0.5
        assert result_hdr['CD2_2'] == 0.5

    def test_flips_x_when_cd1_1_positive(self, capsys):
        """cd1_1 > 0 → should flip x and negate cd1_1."""
        from lsc.externaldata import northupeastleft
        data = np.arange(50 * 50, dtype=float).reshape(50, 50)
        hdr = _make_wcs_header(cd1_1=0.5, cd1_2=0.0, cd2_1=0.0, cd2_2=0.5)
        result_data, result_hdr = northupeastleft(data=data, header=hdr)
        assert result_hdr['CD1_1'] == -0.5
        assert "flipping around x" in capsys.readouterr().out

    def test_flips_y_when_cd2_2_negative(self, capsys):
        """cd2_2 < 0 → should flip y and negate cd2_2."""
        from lsc.externaldata import northupeastleft
        data = np.arange(50 * 50, dtype=float).reshape(50, 50)
        hdr = _make_wcs_header(cd1_1=-0.5, cd1_2=0.0, cd2_1=0.0, cd2_2=-0.5)
        result_data, result_hdr = northupeastleft(data=data, header=hdr)
        assert result_hdr['CD2_2'] == 0.5
        assert "flipping around y" in capsys.readouterr().out

    def test_transposes_when_cd1_2_dominant(self, capsys):
        """|cd1_2| > |cd1_1| → should transpose and swap axes."""
        from lsc.externaldata import northupeastleft
        data = np.ones((40, 50), dtype=float)
        hdr = _make_wcs_header(cd1_1=0.1, cd1_2=0.5, cd2_1=0.5, cd2_2=0.1,
                                naxis1=50, naxis2=40)
        result_data, result_hdr = northupeastleft(data=data, header=hdr)
        out = capsys.readouterr().out
        assert "swapping" in out or result_data.shape == (50, 40)

    def test_returns_tuple(self):
        """northupeastleft without filename always returns (data, header)."""
        from lsc.externaldata import northupeastleft
        data = np.ones((30, 30), dtype=float)
        hdr = _make_wcs_header(cd1_1=-0.3, cd1_2=0.0, cd2_1=0.0, cd2_2=0.3,
                                naxis1=30, naxis2=30)
        result = northupeastleft(data=data, header=hdr)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_filename_writes_back_to_file(self, tmp_path):
        """When filename is given, function modifies the file in-place (no return value)."""
        from lsc.externaldata import northupeastleft
        data = np.arange(30 * 30, dtype=float).reshape(30, 30)
        hdr = _make_wcs_header(cd1_1=-0.3, cd1_2=0.0, cd2_1=0.0, cd2_2=0.3,
                                naxis1=30, naxis2=30)
        fname = str(tmp_path / 'nuel.fits')
        fits.writeto(fname, data, hdr, overwrite=True)
        result = northupeastleft(filename=fname)
        # Function returns None when filename is used (writes in-place)
        assert result is None
        assert os.path.exists(fname)

    def test_filename_no_flip_needed_preserves_data(self, tmp_path):
        """When no flip/transpose is needed, data should remain unchanged."""
        from lsc.externaldata import northupeastleft
        data = np.arange(100, dtype=float).reshape(10, 10)
        hdr = _make_wcs_header(cd1_1=-0.3, cd1_2=0.0, cd2_1=0.0, cd2_2=0.3,
                                naxis1=10, naxis2=10)
        fname = str(tmp_path / 'nuel_noflip.fits')
        fits.writeto(fname, data, hdr, overwrite=True)
        northupeastleft(filename=fname)
        result_data = fits.getdata(fname)
        np.testing.assert_array_equal(result_data, data)

    def test_filename_flip_x_modifies_file(self, tmp_path, capsys):
        """When cd1_1>0, flip is applied and file is written with new content."""
        from lsc.externaldata import northupeastleft
        data = np.arange(100, dtype=float).reshape(10, 10)
        hdr = _make_wcs_header(cd1_1=0.3, cd1_2=0.0, cd2_1=0.0, cd2_2=0.3,
                                naxis1=10, naxis2=10)
        fname = str(tmp_path / 'nuel_flip.fits')
        fits.writeto(fname, data, hdr, overwrite=True)
        northupeastleft(filename=fname)
        out = capsys.readouterr().out
        assert 'flipping' in out
        result_hdr = fits.getheader(fname)
        assert result_hdr['CD1_1'] < 0


# ---------------------------------------------------------------------------
# downloadsdss — download SDSS images using astroquery
# ---------------------------------------------------------------------------

class TestDownloadsdss:
    def test_empty_query_result_returns_empty_string(self, tmp_path, monkeypatch):
        """When SDSS.query_region returns None, downloadsdss returns ''."""
        monkeypatch.chdir(tmp_path)
        from unittest.mock import patch, MagicMock
        from lsc.externaldata import downloadsdss

        with patch('astroquery.sdss.SDSS.query_region', return_value=None):
            result = downloadsdss(150.0, 2.0, 'r', _radius=20)
        assert result == ''

    def test_empty_table_returns_empty_string(self, tmp_path, monkeypatch):
        """Empty astropy table (no rows) from query_region returns ''."""
        monkeypatch.chdir(tmp_path)
        from unittest.mock import patch
        from astropy.table import Table
        from lsc.externaldata import downloadsdss

        empty_table = Table({'run': [], 'camcol': [], 'field': []})
        with patch('astroquery.sdss.SDSS.query_region', return_value=empty_table):
            result = downloadsdss(150.0, 2.0, 'r', _radius=20)
        assert result == ''

    def test_already_downloaded_skips_and_appends(self, tmp_path, monkeypatch):
        """When file already exists and force=False, it appends to filevec without downloading."""
        monkeypatch.chdir(tmp_path)
        from unittest.mock import patch
        from astropy.table import Table
        from lsc.externaldata import downloadsdss

        # Create a fake "already downloaded" file
        band = 'r'
        run, camcol, field = 756, 1, 51
        existing = tmp_path / f'{band}_SDSS_{run}_{camcol}_{field}c.fits'
        existing.write_text('placeholder')
        output1_name = str(tmp_path / f'{band}_SDSS_{run}_{camcol}_{field}.fits')
        open(output1_name, 'w').write('placeholder')

        xid = Table({
            'run': [run],
            'camcol': [camcol],
            'field': [field],
        })
        with patch('astroquery.sdss.SDSS.query_region', return_value=xid):
            # Force=False, file exists → should skip download
            result = downloadsdss(150.0, 2.0, band, _radius=20, force=False)
        # Should return a list (filevec has the c.fits and weight.fits names)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# getimages — query PS1 images (mocked network)
# ---------------------------------------------------------------------------

class TestGetimages:
    def test_returns_astropy_table(self):
        """getimages returns an astropy Table (network mocked)."""
        from unittest.mock import patch
        from astropy.table import Table
        from lsc.externaldata import getimages

        fake_table = Table({
            'filename': ['rings.v3.skycell.0000.0000.stk.g.unconv.fits'],
            'filter': ['g'],
            'ra': [150.0],
            'dec': [2.0],
        })
        with patch('astropy.table.Table.read', return_value=fake_table):
            result = getimages(150.0, 2.0, size=500, filters='g')
        assert isinstance(result, Table)


# ---------------------------------------------------------------------------
# geturl — construct PS1 image URL (mocked network)
# ---------------------------------------------------------------------------

class TestGeturl:
    def test_returns_string_url(self):
        """geturl returns a non-empty string containing the expected format."""
        from unittest.mock import patch
        from astropy.table import Table
        from lsc.externaldata import geturl

        fake_table = Table({
            'filename': ['rings.v3.skycell.0000.stk.g.unconv.fits'],
            'filter': ['g'],
        })
        with patch('lsc.externaldata.getimages', return_value=fake_table):
            result = geturl(150.0, 2.0, size=500, filters='g', format='fits')
        assert isinstance(result, (str, list))

    def test_color_with_fits_raises(self):
        """geturl raises ValueError when color=True and format='fits'."""
        from lsc.externaldata import geturl
        with pytest.raises(ValueError, match="color"):
            geturl(150.0, 2.0, format='fits', color=True)

    def test_invalid_format_raises(self):
        """geturl raises ValueError for an unsupported format."""
        from lsc.externaldata import geturl
        with pytest.raises(ValueError, match="format"):
            geturl(150.0, 2.0, format='tiff')
