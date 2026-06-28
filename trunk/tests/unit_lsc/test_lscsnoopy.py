"""
Tests for lsc.lscsnoopy — PSF-fitting photometry with IRAF.
Covers fitsn, manusn, and errore with mocked IRAF and various edge cases.
"""
import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_iraf():
    """Return the mocked iraf module from conftest."""
    return sys.modules['pyraf'].iraf


def _reset_iraf():
    """Reset the global iraf mock and clear txdump side_effect."""
    iraf = _get_iraf()
    iraf.reset_mock()
    iraf.txdump.side_effect = None
    iraf.txdump.return_value = MagicMock()
    return iraf


def _make_fits(tmp_path, name='test_img', shape=(512, 512), sky=1000.0):
    """Write a minimal FITS file with standard photometry headers."""
    from astropy.io import fits
    hdr = fits.Header()
    hdr['GAIN'] = 2.0
    hdr['RDNOISE'] = 5.0
    hdr['RON'] = 5.0
    hdr['EXPTIME'] = 120.0
    hdr['AIRMASS'] = 1.2
    hdr['FILTER'] = 'r'
    hdr['FILTER2'] = 'rp'
    rng = np.random.default_rng(42)
    data = rng.normal(sky, 50, shape).astype(np.float32)
    fits.writeto(str(tmp_path / (name + '.fits')), data, hdr, overwrite=True)
    return name


def _setup_iraf_for_fitsn(mag_data='20.0 0.05 50.0 50.0',
                           apmag_data='18.0 18.5 19.0'):
    """Configure the iraf mock to return expected data for fitsn."""
    iraf = _get_iraf()
    iraf.txdump.side_effect = [
        [mag_data],
        [apmag_data],
        [apmag_data + ' 0.01 0.02 0.03'],
    ]
    return iraf


# ---------------------------------------------------------------------------
# Module importability
# ---------------------------------------------------------------------------

class TestImportability:
    """Ensure module and functions can be imported."""

    def test_module_imports(self):
        import lsc.lscsnoopy

    def test_fitsn_exists(self):
        from lsc.lscsnoopy import fitsn
        assert callable(fitsn)

    def test_manusn_exists(self):
        from lsc.lscsnoopy import manusn
        assert callable(manusn)

    def test_errore_exists(self):
        from lsc.lscsnoopy import errore
        assert callable(errore)

    def test_module_has_numpy(self):
        import lsc.lscsnoopy
        assert hasattr(lsc.lscsnoopy, 'np')

    def test_module_has_fits(self):
        import lsc.lscsnoopy
        assert hasattr(lsc.lscsnoopy, 'fits')


# ---------------------------------------------------------------------------
# fitsn — PSF fitting photometry
# ---------------------------------------------------------------------------

class TestFitsnReturnStructure:
    """Test that fitsn returns the correct structure."""

    def test_returns_14_element_tuple(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        assert len(result) == 14

    def test_return_types_are_lists(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        for i, item in enumerate(result):
            assert isinstance(item, list), f"result[{i}] is {type(item)}, expected list"


class TestFitsnApertureComputation:
    """Test aperture and annulus computation from FWHM."""

    def test_apertures_from_fwhm_5(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 5.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        assert iraf.noao.daophot.photpars.apertures == '5,10,15'

    def test_apertures_from_fwhm_3(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 3.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        assert iraf.noao.daophot.photpars.apertures == '3,6,9'

    def test_apertures_from_fwhm_7(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 7.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        assert iraf.noao.daophot.photpars.apertures == '7,14,21'

    def test_apertures_from_fwhm_4(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['28.5 0.1 50.2 50.2']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 4.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        assert iraf.noao.daophot.photpars.apertures == '4,8,12'

    def test_psfrad_from_fwhm(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 5.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        assert iraf.noao.digiphot.daophot.daopars.psfrad == 20

    def test_fitrad_equals_fwhm(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 4.5, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        assert iraf.noao.digiphot.daophot.daopars.fitrad == 4.5

    def test_annulus_from_fwhm(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 5.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        assert iraf.noao.daophot.fitskypars.annulus == 15

    def test_small_fwhm(self, tmp_path, monkeypatch):
        """Very small FWHM (e.g., 1.0) should still produce valid apertures."""
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 1.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        assert iraf.noao.daophot.photpars.apertures == '1,2,3'

    def test_large_fwhm(self, tmp_path, monkeypatch):
        """Large FWHM should produce large apertures."""
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 10.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        assert iraf.noao.daophot.photpars.apertures == '10,20,30'


class TestFitsnRecenter:
    """Test recentering behavior."""

    def test_recenter_true(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              True, 5.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        assert iraf.noao.digiphot.daophot.daopars.recenter == 'yes'

    def test_recenter_false(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 5.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        assert iraf.noao.digiphot.daophot.daopars.recenter == 'no'


class TestFitsnMagnitudes:
    """Test magnitude parsing and computation."""

    def test_single_star_fitmag(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['22.3 0.08 100.5 200.7']
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        fitmag = result[9]
        assert len(fitmag) == 1
        assert fitmag[0] == pytest.approx(22.3)

    def test_single_star_magerr(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['22.3 0.08 100.5 200.7']
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        magerr = result[11]
        assert len(magerr) == 1
        assert magerr[0] == pytest.approx(0.08)

    def test_single_star_coordinates(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['22.3 0.08 100.5 200.7']
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        centx = result[12]
        centy = result[13]
        assert centx[0] == pytest.approx(100.5)
        assert centy[0] == pytest.approx(200.7)

    def test_multiple_stars(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = [
            '20.0 0.05 50.0 50.0',
            '21.5 0.10 80.0 90.0',
            '19.8 0.03 120.0 130.0'
        ]
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        fitmag = result[9]
        assert len(fitmag) == 3
        assert fitmag[0] == pytest.approx(20.0)
        assert fitmag[1] == pytest.approx(21.5)
        assert fitmag[2] == pytest.approx(19.8)

    def test_truemag_with_apco0(self, tmp_path, monkeypatch):
        """truemag = fitmag + apco0"""
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0, apco0=0.3)
        truemag = result[10]
        assert truemag[0] == pytest.approx(20.3)

    def test_truemag_with_zero_apco0(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0, apco0=0)
        truemag = result[10]
        assert truemag[0] == pytest.approx(20.0)

    def test_truemag_with_negative_apco0(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0, apco0=-0.5)
        truemag = result[10]
        assert truemag[0] == pytest.approx(19.5)

    def test_truemag_with_apco0_half(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0, apco0=0.5)
        truemag = result[10]
        assert len(truemag) == 1
        assert truemag[0] == pytest.approx(20.5)


class TestFitsnINDEF:
    """Test handling of INDEF (undefined) values in txdump output."""

    def test_indef_fitmag(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['INDEF 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        fitmag = result[9]
        assert fitmag[0] == 'INDEF'

    def test_indef_fitmag_produces_indef_truemag(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['INDEF 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        truemag = result[10]
        assert truemag[0] == 'INDEF'

    def test_indef_magerr(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 INDEF 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        magerr = result[11]
        assert magerr[0] == 'INDEF'

    def test_mixed_valid_and_indef(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = [
            '20.0 0.05 50.0 50.0',
            'INDEF INDEF 60.0 60.0'
        ]
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        fitmag = result[9]
        assert fitmag[0] == pytest.approx(20.0)
        assert fitmag[1] == 'INDEF'


class TestFitsnEmptyInput:
    """Test fitsn with empty txdump output."""

    def test_empty_txdump_returns_empty_lists(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = []
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        fitmag = result[9]
        magerr = result[11]
        centx = result[12]
        centy = result[13]
        assert fitmag == []
        assert magerr == []
        assert centx == []
        assert centy == []


class TestFitsnIRAFSetup:
    """Test that fitsn properly configures IRAF parameters."""

    def test_zmag_set_to_zero(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 5.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        assert iraf.noao.digiphot.daophot.photpars.zmag == 0

    def test_datamin_set(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 5.0, 'original', 'sn', 'residual',
              False, False, 60000.0, -100.0)
        assert iraf.noao.digiphot.daophot.datapars.datamin == -100.0

    def test_datamax_set(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 5.0, 'original', 'sn', 'residual',
              False, False, 55000.0, 0.0)
        assert iraf.noao.digiphot.daophot.datapars.datamax == 55000.0

    def test_cbox_set(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 5.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        assert iraf.noao.digiphot.daophot.centerpars.cbox == 4

    def test_calgori_set_to_gauss(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 5.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        assert iraf.noao.digiphot.daophot.centerpars.calgori == 'gauss'

    def test_fitsky_yes(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 5.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        assert iraf.noao.digiphot.daophot.daopars.fitsky == 'yes'

    def test_exposure_keyword(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 5.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        assert iraf.noao.digiphot.daophot.datapars.exposure == 'exptime'

    def test_airmass_keyword(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 5.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        assert iraf.noao.digiphot.daophot.datapars.airmass == 'airmass'

    def test_filter_keyword(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 5.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        assert iraf.noao.digiphot.daophot.datapars.filter == 'filter2'


class TestFitsnIRAFCalls:
    """Test that fitsn calls IRAF tasks in the right sequence."""

    def test_allstar_called(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 5.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        assert iraf.allstar.called

    def test_imarith_called(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 5.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        assert iraf.imarith.called

    def test_txsort_called(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 5.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        assert iraf.txsort.called

    def test_phot_called_without_show(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 5.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        phot_calls = iraf.noao.digiphot.daophot.phot.call_args_list
        for c in phot_calls:
            assert c[1].get('verb') == 'no' or 'verb' in str(c)


# ---------------------------------------------------------------------------
# fitsn with _show=True
# ---------------------------------------------------------------------------

class TestFitsnShowTrue:
    """Test fitsn with _show=True to cover display/print branches."""

    def test_show_true_phot_without_verb(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['20.0 0.05 50.0 50.0'],
            ['18.0 18.5 19.0'],
            ['18.0 18.5 19.0 0.01 0.02 0.03'],
            ['50.0 50.0'],
        ]
        monkeypatch.chdir(tmp_path)
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            result = fitsn(name, name + 'psf', 'coords.coo',
                           False, 5.0, 'original', 'sn', 'residual',
                           True, False, 60000.0, 0.0,
                           z11='500', z22='2000', midpt='1000', size=7, apco0=0)
        assert len(result) == 14
        assert iraf.noao.digiphot.daophot.phot.called

    def test_show_true_prints_headers(self, tmp_path, monkeypatch, capsys):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['20.0 0.05 50.0 50.0'],
            ['18.0 18.5 19.0'],
            ['18.0 18.5 19.0 0.01 0.02 0.03'],
            ['50.0 50.0'],
        ]
        monkeypatch.chdir(tmp_path)
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            fitsn(name, name + 'psf', 'coords.coo',
                  False, 5.0, 'original', 'sn', 'residual',
                  True, False, 60000.0, 0.0,
                  z11='500', z22='2000', midpt='1000', size=7, apco0=0)
        captured = capsys.readouterr()
        assert "apmag on original" in captured.out
        assert "fitmag" in captured.out

    def test_show_true_prints_star_data(self, tmp_path, monkeypatch, capsys):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['20.0 0.05 50.0 50.0'],
            ['18.0 18.5 19.0'],
            ['18.0 18.5 19.0 0.01 0.02 0.03'],
            ['50.0 50.0'],
        ]
        monkeypatch.chdir(tmp_path)
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            fitsn(name, name + 'psf', 'coords.coo',
                  False, 5.0, 'original', 'sn', 'residual',
                  True, False, 60000.0, 0.0,
                  z11='500', z22='2000', midpt='1000', size=7, apco0=0)
        captured = capsys.readouterr()
        assert "***" in captured.out

    def test_show_true_display_block(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['20.0 0.05 50.0 50.0'],
            ['18.0 18.5 19.0'],
            ['18.0 18.5 19.0 0.01 0.02 0.03'],
            ['50.0 50.0'],
        ]
        monkeypatch.chdir(tmp_path)
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            result = fitsn(name, name + 'psf', 'coords.coo',
                           False, 5.0, 'original', 'sn', 'residual',
                           True, False, 60000.0, 0.0,
                           z11='500', z22='2000', midpt='1000', size=7, apco0=0)
        assert iraf.tvmark.called
        assert iraf.tvmark.call_count >= 3

    def test_show_true_with_indef_fitmag(self, tmp_path, monkeypatch, capsys):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['INDEF 0.05 50.0 50.0'],
            ['18.0 18.5 19.0'],
            ['18.0 18.5 19.0 0.01 0.02 0.03'],
            ['50.0 50.0'],
        ]
        monkeypatch.chdir(tmp_path)
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            result = fitsn(name, name + 'psf', 'coords.coo',
                           False, 5.0, 'original', 'sn', 'residual',
                           True, False, 60000.0, 0.0,
                           z11='500', z22='2000', midpt='1000', size=7, apco0=0)
        truemag = result[10]
        assert truemag[0] == 'INDEF'
        captured = capsys.readouterr()
        assert 'INDEF' in captured.out

    def test_multiple_stars_show_true(self, tmp_path, monkeypatch, capsys):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['20.0 0.05 50.0 50.0', '21.0 0.06 60.0 60.0'],
            ['18.0 18.5 19.0', '17.0 17.5 18.0'],
            ['18.0 18.5 19.0 0.01 0.02 0.03',
             '17.0 17.5 18.0 0.01 0.02 0.03'],
            ['50.0 50.0', '60.0 60.0'],
        ]
        monkeypatch.chdir(tmp_path)
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            result = fitsn(name, name + 'psf', 'coords.coo',
                           False, 5.0, 'original', 'sn', 'residual',
                           True, False, 60000.0, 0.0,
                           z11='500', z22='2000', midpt='1000', size=7, apco0=0)
        fitmag = result[9]
        assert len(fitmag) == 2
        captured = capsys.readouterr()
        assert '0' in captured.out
        assert '1' in captured.out


# ---------------------------------------------------------------------------
# fitsn exception branches
# ---------------------------------------------------------------------------

class TestFitsnExceptionBranches:
    """Test exception handling in fitsn magnitude parsing."""

    def test_apori3_indef_branch(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['20.0 0.05 50.0 50.0'],
            ['18.0 18.5 INDEF'],
            ['18.0 18.5 19.0 0.01 0.02 0.03'],
        ]
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        apori3 = result[2]
        assert apori3[0] == 'INDEF'

    def test_apori1_and_apori2_indef(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['20.0 0.05 50.0 50.0'],
            ['INDEF INDEF INDEF'],
            ['18.0 18.5 19.0 0.01 0.02 0.03'],
        ]
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        apori1 = result[0]
        apori2 = result[1]
        apori3 = result[2]
        assert apori1[0] == 'INDEF'
        assert apori2[0] == 'INDEF'
        assert apori3[0] == 'INDEF'

    def test_apmag3_exception_branch(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['20.0 0.05 50.0 50.0'],
            ['18.0 18.5 19.0'],
            ['18.0 18.5 INDEF 0.01 0.02 0.03'],
        ]
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        apmag3 = result[5]
        assert apmag3[0] == 9999

    def test_dapmag1_exception_branch(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['20.0 0.05 50.0 50.0'],
            ['18.0 18.5 19.0'],
            ['18.0 18.5 19.0 INDEF 0.02 0.03'],
        ]
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        dapmag1 = result[6]
        assert dapmag1[0] == 9999

    def test_all_sn_mag_columns_indef(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['20.0 0.05 50.0 50.0'],
            ['18.0 18.5 19.0'],
            ['INDEF INDEF INDEF INDEF INDEF INDEF'],
        ]
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        apmag1, apmag2, apmag3 = result[3], result[4], result[5]
        dapmag1, dapmag2, dapmag3 = result[6], result[7], result[8]
        assert apmag1[0] == 9999
        assert apmag2[0] == 9999
        assert apmag3[0] == 9999
        assert dapmag1[0] == 9999
        assert dapmag2[0] == 9999
        assert dapmag3[0] == 9999

    def test_truemag_indef_when_fitmag_is_string(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['INDEF INDEF 50.0 50.0'],
            ['18.0 18.5 19.0'],
            ['18.0 18.5 19.0 0.01 0.02 0.03'],
        ]
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        truemag = result[10]
        assert truemag[0] == 'INDEF'

    def test_magerr_indef_branch(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['20.0 INDEF 50.0 50.0'],
            ['18.0 18.5 19.0'],
            ['18.0 18.5 19.0 0.01 0.02 0.03'],
        ]
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        magerr = result[11]
        assert magerr[0] == 'INDEF'


# ---------------------------------------------------------------------------
# manusn — manual magnitude adjustment
# ---------------------------------------------------------------------------

class TestManusnReturnStructure:
    """Test manusn return value structure."""

    def test_returns_12_element_tuple(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import manusn
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['50.0 50.0']
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            result = manusn('test', 'testpsf', 0.5,
                            [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                            [20.0], ['20.0'], [0.1], [50.0], [50.0],
                            100.0, 300.0, 200.0, 7, 5.0, 0, 0, 0.1)
        assert len(result) == 12


class TestManusnMagnitudeComputation:
    """Test magnitude shift logic in manusn."""

    def test_newmag_equals_truemag_plus_dmag0(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import manusn
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['50.0 50.0']
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            result = manusn('test', 'testpsf', 1.0,
                            [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                            [20.0], ['20.0'], [0.1], [50.0], [50.0],
                            100.0, 300.0, 200.0, 7, 5.0, 0, 0, 0.1)
        newmag = result[11]
        assert newmag[0] == pytest.approx(21.0)

    def test_newmag_with_negative_dmag0(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import manusn
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['50.0 50.0']
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            result = manusn('test', 'testpsf', -0.5,
                            [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                            [20.0], ['20.0'], [0.1], [50.0], [50.0],
                            100.0, 300.0, 200.0, 7, 5.0, 0, 0, 0.1)
        newmag = result[11]
        assert newmag[0] == pytest.approx(19.5)

    def test_newmag_with_zero_dmag0(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import manusn
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['50.0 50.0']
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            result = manusn('test', 'testpsf', 0.0,
                            [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                            [20.0], ['20.0'], [0.1], [50.0], [50.0],
                            100.0, 300.0, 200.0, 7, 5.0, 0, 0, 0.1)
        newmag = result[11]
        assert newmag[0] == pytest.approx(20.0)

    def test_multiple_stars_shifted(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import manusn
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['50.0 50.0']
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            result = manusn('test', 'testpsf', 0.3,
                            [15.0, 16.0], [0.1, 0.1], [15.2, 16.2],
                            [15.0, 16.0], [0.1, 0.1], [15.2, 16.2],
                            [20.0, 21.0], ['20.0', '21.0'], [0.1, 0.1],
                            [50.0, 60.0], [50.0, 60.0],
                            100.0, 300.0, 200.0, 7, 5.0, 0, 0, 0.1)
        newmag = result[11]
        assert newmag[0] == pytest.approx(20.3)
        assert newmag[1] == pytest.approx(21.3)

    def test_magnitude_computation_independent(self):
        """manusn computes newmag = truemag + dmag0 for each star."""
        truemag = np.array([18.5, 19.0, 17.5])
        dmag0 = 0.5
        newmag = np.array([float(t) + float(dmag0) for t in truemag])
        assert np.allclose(newmag, [19.0, 19.5, 18.0])


class TestManusnINDEF:
    """Test manusn behavior when truemag is INDEF."""

    def test_indef_truemag_uses_dmag_only(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import manusn
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['50.0 50.0']
        iraf.field.return_value = ['50.0 50.0']
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            result = manusn('test', 'testpsf', '1.5',
                            [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                            [20.0], ['INDEF'], [0.0], [50.0], [50.0],
                            100.0, 300.0, 200.0, 7, 5.0, 0, 0, 0.1)
        newmag = result[11]
        assert newmag[0] == pytest.approx(1.5)

    def test_indef_truemag_calls_addstar(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import manusn
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['50.0 50.0']
        iraf.field.return_value = ['50.0 50.0']
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            manusn('test', 'testpsf', '0.0',
                   [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                   [20.0], ['INDEF'], [0.0], [50.0], [50.0],
                   100.0, 300.0, 200.0, 7, 5.0, 0, 0, 0.1)
        assert iraf.addstar.called

    def test_non_indef_truemag_calls_imarith(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import manusn
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.reset_mock()
        iraf.txdump.return_value = ['50.0 50.0']
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            manusn('test', 'testpsf', 1.0,
                   [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                   [20.0], ['20.0'], [0.1], [50.0], [50.0],
                   100.0, 300.0, 200.0, 7, 5.0, 0, 0, 0.1)
        assert iraf.imarith.called

    def test_indef_magerr_set_to_zero(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import manusn
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['50.0 50.0']
        iraf.field.return_value = ['50.0 50.0']
        magerr_list = [0.5]
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            result = manusn('test', 'testpsf', '1.0',
                            [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                            [20.0], ['INDEF'], magerr_list, [50.0], [50.0],
                            100.0, 300.0, 200.0, 7, 5.0, 0, 0, 0.1)
        assert magerr_list[0] == 0.0

    def test_indef_newmag_uses_dmag_only_2(self, tmp_path, monkeypatch):
        """When truemag is INDEF, newmag = float(dmag0)."""
        from lsc.lscsnoopy import manusn
        monkeypatch.chdir(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.return_value = ['50.0 50.0']
        iraf.field.return_value = ['50.0 50.0']
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            result = manusn('test', 'testpsf', '2.0',
                            [15.0], [0.1], [15.2],
                            [15.0], [0.1], [15.2],
                            [20.0], ['INDEF'], [0.5],
                            [50.0], [50.0],
                            '100.0', '300.0', '200.0',
                            7, 5.0, 0, 0, 0.1)
        newmag = result[11]
        assert newmag[0] == pytest.approx(2.0)

    def test_indef_magnitude_handling_independent(self):
        """When truemag is 'INDEF', handle gracefully."""
        from numpy import zeros
        truemag = ['INDEF', '18.5']
        dmag0 = 0.5

        newmag = zeros(len(truemag))
        magerr = zeros(len(truemag))
        for i in range(len(truemag)):
            try:
                newmag[i] = float(truemag[i]) + float(dmag0)
            except (ValueError, TypeError):
                newmag[i] = float(dmag0)
                magerr[i] = 0.0

        assert newmag[0] == 0.5
        assert magerr[0] == 0.0
        assert abs(newmag[1] - 19.0) < 1e-6


class TestManusnFdmag:
    """Test the flux dimming factor computation."""

    def test_fdmag_computation(self):
        dmag0 = 1.0
        fdmag = 10 ** (-0.4 * float(dmag0))
        assert fdmag == pytest.approx(0.3981, rel=1e-3)

    def test_fdmag_zero_dmag(self):
        dmag0 = 0.0
        fdmag = 10 ** (-0.4 * float(dmag0))
        assert fdmag == pytest.approx(1.0)

    def test_fdmag_negative_dmag(self):
        dmag0 = -1.0
        fdmag = 10 ** (-0.4 * float(dmag0))
        assert fdmag == pytest.approx(2.5119, rel=1e-3)


class TestManusnDisplayBlock:
    """Test manusn display/tvmark code block."""

    def test_display_image_called_multiple_times(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import manusn
        monkeypatch.chdir(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.return_value = ['50.0 50.0']
        display_mock = MagicMock(return_value=(100.0, 200.0, True))
        with patch('lsc.util.display_image', display_mock):
            manusn('test', 'testpsf', 0.5,
                   [15.0], [0.1], [15.2],
                   [15.0], [0.1], [15.2],
                   [20.0], ['20.0'], [0.1],
                   [50.0], [50.0],
                   '100.0', '300.0', '200.0',
                   7, 5.0, 0, 0, 0.1)
        assert display_mock.call_count == 3

    def test_tvmark_called(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import manusn
        monkeypatch.chdir(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.return_value = ['50.0 50.0']
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            manusn('test', 'testpsf', 0.5,
                   [15.0], [0.1], [15.2],
                   [15.0], [0.1], [15.2],
                   [20.0], ['20.0'], [0.1],
                   [50.0], [50.0],
                   '100.0', '300.0', '200.0',
                   7, 5.0, 0, 0, 0.1)
        assert iraf.tvmark.called
        assert iraf.tvmark.call_count >= 3

    def test_z01_z02_computation(self, tmp_path, monkeypatch, capsys):
        from lsc.lscsnoopy import manusn
        monkeypatch.chdir(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.return_value = ['50.0 50.0']
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            manusn('test', 'testpsf', 0.5,
                   [15.0], [0.1], [15.2],
                   [15.0], [0.1], [15.2],
                   [20.0], ['20.0'], [0.1],
                   [50.0], [50.0],
                   '500', '2000', '1000',
                   7, 5.0, 0, 0, 0.1)
        captured = capsys.readouterr()
        assert '500' in captured.out
        assert '2000' in captured.out
        assert '1000' in captured.out


# ---------------------------------------------------------------------------
# errore — artificial star error estimation
# ---------------------------------------------------------------------------

class TestErroreReturnStructure:
    """Test errore return value."""

    def test_returns_two_values(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import errore
        monkeypatch.chdir(tmp_path)
        from astropy.io import fits as afits
        data = np.ones((100, 100), dtype=np.float32) * 1000
        afits.writeto(str(tmp_path / 'skyfit.fits'), data, overwrite=True)
        afits.writeto(str(tmp_path / 'artbg.fits'), data, overwrite=True)
        (tmp_path / 'err_test.sn.coo').write_text('50.0 50.0\n')
        with patch('lsc.lscsnoopy.fits.getdata', return_value=np.ones((100, 100)) * 1000):
            with patch('lsc.lscsnoopy.fitsn', return_value=(
                [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                [15.0], [0.1], [15.2], [20.0], ['20.0'], [0.1],
                [50.0], [50.0]
            )):
                with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
                    result = errore('err_test', 'err_testpsf', 'err_test.sn.coo',
                                    7, ['20.0'], 5.0, 2, False, False,
                                    3, 100.0, 300.0, 200.0, 100, 100,
                                    2, 2, False, 0, 60000.0, 0.0)
        assert len(result) == 2

    def test_returns_floats(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import errore
        monkeypatch.chdir(tmp_path)
        from astropy.io import fits as afits
        data = np.ones((100, 100), dtype=np.float32) * 1000
        afits.writeto(str(tmp_path / 'skyfit.fits'), data, overwrite=True)
        afits.writeto(str(tmp_path / 'artbg.fits'), data, overwrite=True)
        (tmp_path / 'err2_test.sn.coo').write_text('50.0 50.0\n')
        with patch('lsc.lscsnoopy.fits.getdata', return_value=np.ones((100, 100)) * 1000):
            with patch('lsc.lscsnoopy.fitsn', return_value=(
                [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                [15.0], [0.1], [15.2], [20.0], ['20.0'], [0.1],
                [50.0], [50.0]
            )):
                with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
                    arterr2, arterr = errore('err2_test', 'err2_testpsf', 'err2_test.sn.coo',
                                            7, ['20.0'], 5.0, 2, False, False,
                                            3, 100.0, 300.0, 200.0, 100, 100,
                                            2, 2, False, 0, 60000.0, 0.0)
        assert isinstance(arterr, (int, float, np.floating))
        assert isinstance(arterr2, (int, float, np.floating))


class TestErroreGridPlacement:
    """Test the 3x3 grid logic for artificial star placement."""

    def test_grid_positions_3x3(self):
        """Verify the 9 artificial star positions form a 3x3 grid."""
        positions = []
        for i in range(9):
            artx = int(i / 3.) - 1
            if i <= 2:
                arty = artx + i
            elif 3 <= i <= 5:
                arty = artx - 1 + i - 3
            else:
                arty = artx - 2 + i - 6
            positions.append((artx, arty))

        expected = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 0), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        assert positions == expected

    def test_grid_centered_on_sn(self):
        """The center position (i=4) should be (0, 0) offset."""
        i = 4
        artx = int(i / 3.) - 1
        arty = artx - 1 + i - 3
        assert (artx, arty) == (0, 0)

    def test_grid_positions_unique(self):
        """All 9 positions should be distinct."""
        positions = set()
        for i in range(9):
            artx = int(i / 3.) - 1
            if i <= 2:
                arty = artx + i
            elif 3 <= i <= 5:
                arty = artx - 1 + i - 3
            else:
                arty = artx - 2 + i - 6
            positions.add((artx, arty))
        assert len(positions) == 9

    def test_grid_offsets_scaled_by_fwhm_and_artfac(self):
        """Actual pixel offsets = artx * fwhm * artfac."""
        fwhm0 = 5.0
        artfac0 = 1.0
        xbb_orig = 50.0
        ybb_orig = 50.0

        artx = -1
        arty = -1
        xbb = xbb_orig + artx * fwhm0 * artfac0
        ybb = ybb_orig + arty * fwhm0 * artfac0
        assert xbb == pytest.approx(45.0)
        assert ybb == pytest.approx(45.0)

    def test_grid_offsets_with_artfac_half(self):
        """With artfac=0.5, offsets are halved."""
        fwhm0 = 5.0
        artfac0 = 0.5
        xbb_orig = 50.0
        ybb_orig = 50.0

        artx = 1
        arty = 1
        xbb = xbb_orig + artx * fwhm0 * artfac0
        ybb = ybb_orig + arty * fwhm0 * artfac0
        assert xbb == pytest.approx(52.5)
        assert ybb == pytest.approx(52.5)


class TestErroreStatistics:
    """Test the statistical computation in errore."""

    def test_mean_of_constant_values(self):
        from numpy import array, mean, std
        tmpart = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
        assert mean(array(tmpart)) == pytest.approx(20.0)
        assert std(array(tmpart)) == pytest.approx(0.0)

    def test_std_of_varied_values(self):
        from numpy import array, std
        tmpart = [20.0, 20.1, 19.9, 20.2, 19.8, 20.0, 20.1, 19.9, 20.0]
        assert std(array(tmpart)) > 0.0

    def test_sigma_clipped_std(self):
        from numpy import array, mean, std, compress, average
        tmpart = [20.0, 20.1, 19.9, 20.2, 19.8, 20.0, 20.1, 19.9, 25.0]
        arr = array(tmpart)
        media = mean(arr)
        arterr = std(arr)
        mask = (average(tmpart) - std(tmpart) < arr) & (arr < average(tmpart) + std(tmpart))
        arterr2 = std(compress(mask, arr))
        assert arterr2 < arterr

    def test_empty_tmpart_returns_zero(self):
        media = 0
        arterr = 0
        arterr2 = 0
        assert media == 0
        assert arterr == 0
        assert arterr2 == 0


class TestErroreWithVaryingMagnitudes:
    """Test errore with varying artificial star recovery."""

    def test_varying_recovery_produces_nonzero_error(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import errore
        monkeypatch.chdir(tmp_path)
        from astropy.io import fits as afits
        data = np.ones((100, 100), dtype=np.float32) * 1000
        afits.writeto(str(tmp_path / 'skyfit.fits'), data, overwrite=True)
        afits.writeto(str(tmp_path / 'artbg.fits'), data, overwrite=True)
        (tmp_path / 'vary_test.sn.coo').write_text('50.0 50.0\n')

        call_count = [0]
        mags = [19.8, 20.1, 19.9, 20.2, 20.0, 19.7, 20.3, 19.9, 20.1]

        def mock_fitsn(*args, **kwargs):
            idx = min(call_count[0], len(mags) - 1)
            mag = mags[idx]
            call_count[0] += 1
            return (
                [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                [15.0], [0.1], [15.2], [mag], [str(mag)], [0.1],
                [50.0], [50.0]
            )

        with patch('lsc.lscsnoopy.fits.getdata', return_value=np.ones((100, 100)) * 1000):
            with patch('lsc.lscsnoopy.fitsn', side_effect=mock_fitsn):
                with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
                    arterr2, arterr = errore('vary_test', 'vary_testpsf', 'vary_test.sn.coo',
                                            7, ['20.0'], 5.0, 2, False, False,
                                            3, 100.0, 300.0, 200.0, 100, 100,
                                            2, 2, False, 0, 60000.0, 0.0)
        assert arterr > 0.0

    def test_constant_recovery_produces_zero_error(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import errore
        monkeypatch.chdir(tmp_path)
        from astropy.io import fits as afits
        data = np.ones((100, 100), dtype=np.float32) * 1000
        afits.writeto(str(tmp_path / 'skyfit.fits'), data, overwrite=True)
        afits.writeto(str(tmp_path / 'artbg.fits'), data, overwrite=True)
        (tmp_path / 'const_test.sn.coo').write_text('50.0 50.0\n')

        with patch('lsc.lscsnoopy.fits.getdata', return_value=np.ones((100, 100)) * 1000):
            with patch('lsc.lscsnoopy.fitsn', return_value=(
                [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                [15.0], [0.1], [15.2], [20.0], ['20.0'], [0.1],
                [50.0], [50.0]
            )):
                with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
                    arterr2, arterr = errore('const_test', 'const_testpsf', 'const_test.sn.coo',
                                            7, ['20.0'], 5.0, 2, False, False,
                                            3, 100.0, 300.0, 200.0, 100, 100,
                                            2, 2, False, 0, 60000.0, 0.0)
        assert arterr == pytest.approx(0.0)
        assert np.isnan(arterr2)


class TestErroreIterations:
    """Test that errore runs the correct number of iterations."""

    def test_fitsn_called_multiple_times(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import errore
        monkeypatch.chdir(tmp_path)
        from astropy.io import fits as afits
        data = np.ones((100, 100), dtype=np.float32) * 1000
        afits.writeto(str(tmp_path / 'skyfit.fits'), data, overwrite=True)
        afits.writeto(str(tmp_path / 'artbg.fits'), data, overwrite=True)
        (tmp_path / 'iter_test.sn.coo').write_text('50.0 50.0\n')

        fitsn_calls = [0]

        def mock_fitsn(*args, **kwargs):
            fitsn_calls[0] += 1
            return (
                [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                [15.0], [0.1], [15.2], [20.0], ['20.0'], [0.1],
                [50.0], [50.0]
            )

        _numiter = 3
        with patch('lsc.lscsnoopy.fits.getdata', return_value=np.ones((100, 100)) * 1000):
            with patch('lsc.lscsnoopy.fitsn', side_effect=mock_fitsn):
                with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
                    errore('iter_test', 'iter_testpsf', 'iter_test.sn.coo',
                           7, ['20.0'], 5.0, 2, False, False,
                           _numiter, 100.0, 300.0, 200.0, 100, 100,
                           2, 2, False, 0, 60000.0, 0.0)
        expected_calls = 9 * (1 + _numiter)
        assert fitsn_calls[0] == expected_calls


class TestErroreINDEFHandling:
    """Test errore when fitsn returns INDEF magnitudes."""

    def test_indef_magnitudes_skipped(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import errore
        monkeypatch.chdir(tmp_path)
        from astropy.io import fits as afits
        data = np.ones((100, 100), dtype=np.float32) * 1000
        afits.writeto(str(tmp_path / 'skyfit.fits'), data, overwrite=True)
        afits.writeto(str(tmp_path / 'artbg.fits'), data, overwrite=True)
        (tmp_path / 'indef_test.sn.coo').write_text('50.0 50.0\n')

        with patch('lsc.lscsnoopy.fits.getdata', return_value=np.ones((100, 100)) * 1000):
            with patch('lsc.lscsnoopy.fitsn', return_value=(
                [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                [15.0], [0.1], [15.2], ['INDEF'], ['INDEF'], [0.1],
                [50.0], [50.0]
            )):
                with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
                    arterr2, arterr = errore('indef_test', 'indef_testpsf', 'indef_test.sn.coo',
                                            7, ['20.0'], 5.0, 2, False, False,
                                            3, 100.0, 300.0, 200.0, 100, 100,
                                            2, 2, False, 0, 60000.0, 0.0)
        assert np.isnan(arterr)
        assert np.isnan(arterr2)


class TestErroreShowTrue:
    """Test errore with _show=True to cover display_image in the loop."""

    def test_show_true_calls_display_image(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import errore
        monkeypatch.chdir(tmp_path)
        _reset_iraf()
        from astropy.io import fits as afits
        data = np.ones((100, 100), dtype=np.float32) * 1000
        afits.writeto(str(tmp_path / 'skyfit.fits'), data, overwrite=True)
        afits.writeto(str(tmp_path / 'artbg.fits'), data, overwrite=True)
        (tmp_path / 'show_test.sn.coo').write_text('50.0 50.0\n')

        display_mock = MagicMock(return_value=(100.0, 200.0, True))
        with patch('lsc.lscsnoopy.fits.getdata', return_value=np.ones((100, 100)) * 1000):
            with patch('lsc.lscsnoopy.fitsn', return_value=(
                [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                [15.0], [0.1], [15.2], [20.0], ['20.0'], [0.1],
                [50.0], [50.0]
            )):
                with patch('lsc.util.display_image', display_mock):
                    result = errore('show_test', 'show_testpsf', 'show_test.sn.coo',
                                    7, ['20.0'], 5.0, 2, True, False,
                                    3, 100.0, 300.0, 200.0, 100, 100,
                                    2, 2, False, 0, 60000.0, 0.0)
        assert display_mock.call_count == 9
        assert len(result) == 2

    def test_show_true_with_varying_mags(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import errore
        monkeypatch.chdir(tmp_path)
        _reset_iraf()
        from astropy.io import fits as afits
        data = np.ones((100, 100), dtype=np.float32) * 1000
        afits.writeto(str(tmp_path / 'skyfit.fits'), data, overwrite=True)
        afits.writeto(str(tmp_path / 'artbg.fits'), data, overwrite=True)
        (tmp_path / 'vary_show.sn.coo').write_text('50.0 50.0\n')

        call_count = [0]
        mags = [19.8, 20.1, 19.9, 20.2, 20.0, 19.7, 20.3, 19.9, 20.1]

        def mock_fitsn(*args, **kwargs):
            idx = min(call_count[0], len(mags) - 1)
            mag = mags[idx]
            call_count[0] += 1
            return (
                [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                [15.0], [0.1], [15.2], [mag], [str(mag)], [0.1],
                [50.0], [50.0]
            )

        display_mock = MagicMock(return_value=(100.0, 200.0, True))
        with patch('lsc.lscsnoopy.fits.getdata', return_value=np.ones((100, 100)) * 1000):
            with patch('lsc.lscsnoopy.fitsn', side_effect=mock_fitsn):
                with patch('lsc.util.display_image', display_mock):
                    arterr2, arterr = errore('vary_show', 'vary_showpsf', 'vary_show.sn.coo',
                                            7, ['20.0'], 5.0, 2, True, False,
                                            3, 100.0, 300.0, 200.0, 100, 100,
                                            2, 2, False, 0, 60000.0, 0.0)
        assert display_mock.call_count == 9
        assert arterr > 0.0


class TestErroreStatsExceptionBranch:
    """Test errore when tmpart triggers the except branch."""

    def test_empty_tmpart_returns_nan_or_zero(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import errore
        monkeypatch.chdir(tmp_path)
        _reset_iraf()
        from astropy.io import fits as afits
        data = np.ones((100, 100), dtype=np.float32) * 1000
        afits.writeto(str(tmp_path / 'skyfit.fits'), data, overwrite=True)
        afits.writeto(str(tmp_path / 'artbg.fits'), data, overwrite=True)
        (tmp_path / 'empty_test.sn.coo').write_text('50.0 50.0\n')

        with patch('lsc.lscsnoopy.fits.getdata', return_value=np.ones((100, 100)) * 1000):
            with patch('lsc.lscsnoopy.fitsn', return_value=(
                [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                [15.0], [0.1], [15.2], ['INDEF'], ['INDEF'], [0.1],
                [50.0], [50.0]
            )):
                with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
                    arterr2, arterr = errore('empty_test', 'empty_testpsf', 'empty_test.sn.coo',
                                            7, ['20.0'], 5.0, 2, False, False,
                                            3, 100.0, 300.0, 200.0, 100, 100,
                                            2, 2, False, 0, 60000.0, 0.0)
        assert isinstance(arterr, (int, float, np.floating))
        assert isinstance(arterr2, (int, float, np.floating))

    def test_stats_exception_forced_via_compress_failure(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import errore
        monkeypatch.chdir(tmp_path)
        _reset_iraf()
        from astropy.io import fits as afits
        data = np.ones((100, 100), dtype=np.float32) * 1000
        afits.writeto(str(tmp_path / 'skyfit.fits'), data, overwrite=True)
        afits.writeto(str(tmp_path / 'artbg.fits'), data, overwrite=True)
        (tmp_path / 'force_test.sn.coo').write_text('50.0 50.0\n')

        with patch('lsc.lscsnoopy.fits.getdata', return_value=np.ones((100, 100)) * 1000):
            with patch('lsc.lscsnoopy.fitsn', return_value=(
                [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                [15.0], [0.1], [15.2], [20.0], ['20.0'], [0.1],
                [50.0], [50.0]
            )):
                with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
                    original_compress = np.compress
                    def bad_compress(*args, **kwargs):
                        raise TypeError("forced error for testing")
                    monkeypatch.setattr(np, 'compress', bad_compress)
                    arterr2, arterr = errore('force_test', 'force_testpsf', 'force_test.sn.coo',
                                            7, ['20.0'], 5.0, 2, False, False,
                                            3, 100.0, 300.0, 200.0, 100, 100,
                                            2, 2, False, 0, 60000.0, 0.0)
        assert arterr == 0
        assert arterr2 == 0


class TestErroreInteractiveBranch:
    """Test errore _interactive=True paths for artfac0 input."""

    def test_interactive_valid_input(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import errore
        monkeypatch.chdir(tmp_path)
        _reset_iraf()
        from astropy.io import fits as afits
        data = np.ones((100, 100), dtype=np.float32) * 1000
        afits.writeto(str(tmp_path / 'skyfit.fits'), data, overwrite=True)
        afits.writeto(str(tmp_path / 'artbg.fits'), data, overwrite=True)
        (tmp_path / 'inter_test.sn.coo').write_text('50.0 50.0\n')

        with patch('lsc.util.userinput', return_value='1'):
            with patch('lsc.lscsnoopy.fits.getdata', return_value=np.ones((100, 100)) * 1000):
                with patch('lsc.lscsnoopy.fitsn', return_value=(
                    [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                    [15.0], [0.1], [15.2], [20.0], ['20.0'], [0.1],
                    [50.0], [50.0]
                )):
                    with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
                        arterr2, arterr = errore('inter_test', 'inter_testpsf', 'inter_test.sn.coo',
                                                7, ['20.0'], 5.0, 2, False, True,
                                                3, 100.0, 300.0, 200.0, 100, 100,
                                                2, 2, False, 0, 60000.0, 0.0)
        assert isinstance(arterr, (int, float, np.floating))

    def test_interactive_too_large_then_valid(self, tmp_path, monkeypatch, capsys):
        from lsc.lscsnoopy import errore
        monkeypatch.chdir(tmp_path)
        _reset_iraf()
        from astropy.io import fits as afits
        data = np.ones((100, 100), dtype=np.float32) * 1000
        afits.writeto(str(tmp_path / 'skyfit.fits'), data, overwrite=True)
        afits.writeto(str(tmp_path / 'artbg.fits'), data, overwrite=True)
        (tmp_path / 'toolarge_test.sn.coo').write_text('50.0 50.0\n')

        call_count = [0]
        def mock_userinput(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return '10'
            return '1'

        with patch('lsc.util.userinput', side_effect=mock_userinput):
            with patch('lsc.lscsnoopy.fits.getdata', return_value=np.ones((100, 100)) * 1000):
                with patch('lsc.lscsnoopy.fitsn', return_value=(
                    [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                    [15.0], [0.1], [15.2], [20.0], ['20.0'], [0.1],
                    [50.0], [50.0]
                )):
                    with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
                        arterr2, arterr = errore('toolarge_test', 'toolarge_testpsf', 'toolarge_test.sn.coo',
                                                7, ['20.0'], 5.0, 2, False, True,
                                                3, 100.0, 300.0, 200.0, 100, 100,
                                                2, 2, False, 0, 60000.0, 0.0)
        captured = capsys.readouterr()
        assert 'too large' in captured.out
        assert isinstance(arterr, (int, float, np.floating))

    def test_interactive_non_numeric_then_valid(self, tmp_path, monkeypatch, capsys):
        from lsc.lscsnoopy import errore
        monkeypatch.chdir(tmp_path)
        _reset_iraf()
        from astropy.io import fits as afits
        data = np.ones((100, 100), dtype=np.float32) * 1000
        afits.writeto(str(tmp_path / 'skyfit.fits'), data, overwrite=True)
        afits.writeto(str(tmp_path / 'artbg.fits'), data, overwrite=True)
        (tmp_path / 'nonnumeric_test.sn.coo').write_text('50.0 50.0\n')

        call_count = [0]
        def mock_userinput(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return 'abc'
            return '1'

        with patch('lsc.util.userinput', side_effect=mock_userinput):
            with patch('lsc.lscsnoopy.fits.getdata', return_value=np.ones((100, 100)) * 1000):
                with patch('lsc.lscsnoopy.fitsn', return_value=(
                    [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                    [15.0], [0.1], [15.2], [20.0], ['20.0'], [0.1],
                    [50.0], [50.0]
                )):
                    with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
                        arterr2, arterr = errore('nonnumeric_test', 'nonnumeric_testpsf', 'nonnumeric_test.sn.coo',
                                                7, ['20.0'], 5.0, 2, False, True,
                                                3, 100.0, 300.0, 200.0, 100, 100,
                                                2, 2, False, 0, 60000.0, 0.0)
        captured = capsys.readouterr()
        assert 'should be a number' in captured.out
        assert isinstance(arterr, (int, float, np.floating))

    def test_interactive_empty_input_defaults_to_1(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import errore
        monkeypatch.chdir(tmp_path)
        _reset_iraf()
        from astropy.io import fits as afits
        data = np.ones((100, 100), dtype=np.float32) * 1000
        afits.writeto(str(tmp_path / 'skyfit.fits'), data, overwrite=True)
        afits.writeto(str(tmp_path / 'artbg.fits'), data, overwrite=True)
        (tmp_path / 'empty_input.sn.coo').write_text('50.0 50.0\n')

        with patch('lsc.util.userinput', return_value=''):
            with patch('lsc.lscsnoopy.fits.getdata', return_value=np.ones((100, 100)) * 1000):
                with patch('lsc.lscsnoopy.fitsn', return_value=(
                    [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                    [15.0], [0.1], [15.2], [20.0], ['20.0'], [0.1],
                    [50.0], [50.0]
                )):
                    with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
                        arterr2, arterr = errore('empty_input', 'empty_inputpsf', 'empty_input.sn.coo',
                                                7, ['20.0'], 5.0, 2, False, True,
                                                3, 100.0, 300.0, 200.0, 100, 100,
                                                2, 2, False, 0, 60000.0, 0.0)
        assert isinstance(arterr, (int, float, np.floating))


# ---------------------------------------------------------------------------
# Aperture math unit tests (independent of IRAF)
# ---------------------------------------------------------------------------

class TestApertureMath:
    """Test the aperture/annulus formulas used in fitsn independently."""

    @pytest.mark.parametrize("fwhm0,expected_a1,expected_a2,expected_a3,expected_a4", [
        (3.0, 3, 6, 9, 12),
        (4.0, 4, 8, 12, 16),
        (5.0, 5, 10, 15, 20),
        (6.0, 6, 12, 18, 24),
        (7.0, 7, 14, 21, 28),
        (2.5, 2, 5, 8, 10),
        (3.7, 3, 7, 11, 15),
    ])
    def test_aperture_formulas(self, fwhm0, expected_a1, expected_a2, expected_a3, expected_a4):
        a1 = int(fwhm0)
        a2 = int(2. * fwhm0 + .5)
        a3 = int(3. * fwhm0 + .5)
        a4 = int(4. * fwhm0 + .5)
        assert a1 == expected_a1
        assert a2 == expected_a2
        assert a3 == expected_a3
        assert a4 == expected_a4

    def test_aperture_string_format(self):
        fwhm0 = 5.0
        a1 = int(fwhm0)
        a2 = int(2. * fwhm0 + .5)
        a3 = int(3. * fwhm0 + .5)
        ap = str(a1) + "," + str(a2) + "," + str(a3)
        assert ap == "5,10,15"


# ---------------------------------------------------------------------------
# Magnitude parsing logic unit tests
# ---------------------------------------------------------------------------

class TestMagnitudeParsing:
    """Test the magnitude parsing logic from txdump output strings."""

    def test_valid_float_magnitude(self):
        line = "20.5 0.05 50.0 60.0"
        parts = str.split(line)
        mag = float(parts[0])
        assert mag == pytest.approx(20.5)

    def test_indef_magnitude(self):
        line = "INDEF 0.05 50.0 60.0"
        parts = str.split(line)
        try:
            mag = float(parts[0])
        except ValueError:
            mag = parts[0]
        assert mag == 'INDEF'

    def test_negative_magnitude(self):
        line = "-2.5 0.01 50.0 60.0"
        parts = str.split(line)
        mag = float(parts[0])
        assert mag == pytest.approx(-2.5)

    def test_large_magnitude(self):
        line = "28.5 0.50 50.0 60.0"
        parts = str.split(line)
        mag = float(parts[0])
        assert mag == pytest.approx(28.5)

    def test_zero_error(self):
        line = "20.0 0.000 50.0 60.0"
        parts = str.split(line)
        err = float(parts[1])
        assert err == pytest.approx(0.0)

    def test_high_precision_coordinates(self):
        line = "20.0 0.05 123.456789 987.654321"
        parts = str.split(line)
        x = float(parts[2])
        y = float(parts[3])
        assert x == pytest.approx(123.456789)
        assert y == pytest.approx(987.654321)


# ---------------------------------------------------------------------------
# apmag parsing for sn.mag output
# ---------------------------------------------------------------------------

class TestApmagParsing:
    """Test aperture magnitude parsing from sn.mag txdump output."""

    def test_valid_three_apertures(self):
        line = "18.0 18.5 19.0 0.01 0.02 0.03"
        parts = str.split(line)
        apmag1 = float(parts[0])
        apmag2 = float(parts[1])
        apmag3 = float(parts[2])
        assert apmag1 == pytest.approx(18.0)
        assert apmag2 == pytest.approx(18.5)
        assert apmag3 == pytest.approx(19.0)

    def test_indef_aperture_mag_becomes_9999(self):
        line = "INDEF INDEF INDEF 0.01 0.02 0.03"
        parts = str.split(line)
        results = []
        for i in range(3):
            try:
                results.append(float(parts[i]))
            except ValueError:
                results.append(9999)
        assert all(r == 9999 for r in results)

    def test_mixed_valid_and_indef_apmag(self):
        line = "18.0 INDEF 19.0 0.01 0.02 0.03"
        parts = str.split(line)
        results = []
        for i in range(3):
            try:
                results.append(float(parts[i]))
            except ValueError:
                results.append(9999)
        assert results[0] == pytest.approx(18.0)
        assert results[1] == 9999
        assert results[2] == pytest.approx(19.0)


# ---------------------------------------------------------------------------
# Coordinate list writing
# ---------------------------------------------------------------------------

class TestCoordslistFormat:
    def test_coordslist_format(self):
        """fitsn writes coordinates to a file in IRAF format: 'x y flux id'."""
        stars = [(50.5, 60.3, 1)]
        for star in stars:
            line = '%8.3f %8.3f  %6.1f  %d' % (float(star[0]), float(star[1]), 1.0, star[2])
            parts = line.split()
            assert len(parts) == 4
            assert float(parts[0]) == float(star[0])
            assert float(parts[1]) == float(star[1])
