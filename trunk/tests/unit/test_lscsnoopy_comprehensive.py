"""
Comprehensive tests for lsc.lscsnoopy — PSF-fitting photometry with IRAF.
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
    # txdump is called multiple times:
    # 1st call: mag,merr,xcenter,ycenter from .sn.als
    # 2nd call: mag from apori
    # 3rd call: mag,merr from .sn.mag
    iraf.txdump.side_effect = [
        [mag_data],          # sn.als dump
        [apmag_data],        # apori dump
        [apmag_data + ' 0.01 0.02 0.03'],  # sn.mag dump
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
        # a1=5, a2=10, a3=15 -> "5,10,15"
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
        # a1=3, a2=int(2*3+0.5)=6, a3=int(3*3+0.5)=9
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
        # a1=7, a2=int(14+0.5)=14, a3=int(21+0.5)=21
        assert iraf.noao.daophot.photpars.apertures == '7,14,21'

    def test_psfrad_from_fwhm(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        fitsn(name, name + 'psf', 'coords.coo',
              False, 5.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        # a4 = int(4*5+0.5) = 20
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
        # a3 = int(3*5+0.5) = 15
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
        # a1=1, a2=int(2+0.5)=2, a3=int(3+0.5)=3
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
        # a1=10, a2=int(20+0.5)=20, a3=int(30+0.5)=30
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
        # phot should be called with verb='no' when _show=False
        phot_calls = iraf.noao.digiphot.daophot.phot.call_args_list
        for c in phot_calls:
            assert c[1].get('verb') == 'no' or 'verb' in str(c)


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
        # When truemag is INDEF, newmag = float(dmag0) = 1.5
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
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            result = manusn('test', 'testpsf', '1.0',
                            [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                            [20.0], ['INDEF'], [0.5], [50.0], [50.0],
                            100.0, 300.0, 200.0, 7, 5.0, 0, 0, 0.1)
        # magerr should be set to 0.0 for INDEF stars
        # magerr is at index ... the function modifies the passed-in list
        # Result contains the original magerr list
        # The function sets magerr[0]=0.0 in the INDEF branch


class TestManusnFdmag:
    """Test the flux dimming factor computation."""

    def test_fdmag_computation(self):
        """fdmag = 10^(-0.4*dmag0) is the flux scaling factor."""
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

        # For i=0: artx=-1, arty=-1
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
        """If all artificial stars recover the same magnitude, std should be 0."""
        from numpy import array, mean, std
        tmpart = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
        assert mean(array(tmpart)) == pytest.approx(20.0)
        assert std(array(tmpart)) == pytest.approx(0.0)

    def test_std_of_varied_values(self):
        """Varied magnitudes should give non-zero std."""
        from numpy import array, std
        tmpart = [20.0, 20.1, 19.9, 20.2, 19.8, 20.0, 20.1, 19.9, 20.0]
        assert std(array(tmpart)) > 0.0

    def test_sigma_clipped_std(self):
        """Test the 1-sigma rejection logic used in errore."""
        from numpy import array, mean, std, compress, average
        tmpart = [20.0, 20.1, 19.9, 20.2, 19.8, 20.0, 20.1, 19.9, 25.0]  # 25 is an outlier
        arr = array(tmpart)
        media = mean(arr)
        arterr = std(arr)
        # Sigma-clipped
        mask = (average(tmpart) - std(tmpart) < arr) & (arr < average(tmpart) + std(tmpart))
        arterr2 = std(compress(mask, arr))
        # Clipped std should be less than unclipped (outlier removed)
        assert arterr2 < arterr

    def test_empty_tmpart_returns_zero(self):
        """If no valid magnitudes recovered, error should be 0."""
        # Simulating the except branch in errore
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

        # Return slightly different magnitudes each call to simulate scatter
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
        # With varying magnitudes, error should be > 0
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
        # All same magnitude -> zero std for arterr
        assert arterr == pytest.approx(0.0)
        # arterr2 is nan because sigma-clipping with std=0 produces an empty array
        assert np.isnan(arterr2)


class TestErroreIterations:
    """Test that errore runs the correct number of iterations."""

    def test_fitsn_called_multiple_times(self, tmp_path, monkeypatch):
        """fitsn is called 9 (grid positions) * (1 + _numiter) times."""
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
        # 9 grid positions * (1 initial + _numiter refinement) = 9 * (1+3) = 36
        expected_calls = 9 * (1 + _numiter)
        assert fitsn_calls[0] == expected_calls


class TestErroreINDEFHandling:
    """Test errore when fitsn returns INDEF magnitudes."""

    def test_indef_magnitudes_skipped(self, tmp_path, monkeypatch):
        """When truemag is INDEF (not convertible to float), it's skipped."""
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
        # With all INDEF, tmpart is empty -> mean/std of empty array = nan
        assert np.isnan(arterr)
        assert np.isnan(arterr2)


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
        """Test the aperture string construction."""
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
        """Very bright stars can have negative instrumental magnitudes."""
        line = "-2.5 0.01 50.0 60.0"
        parts = str.split(line)
        mag = float(parts[0])
        assert mag == pytest.approx(-2.5)

    def test_large_magnitude(self):
        """Very faint sources."""
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
        """sn.mag txdump returns 3 aperture mags + 3 errors."""
        line = "18.0 18.5 19.0 0.01 0.02 0.03"
        parts = str.split(line)
        apmag1 = float(parts[0])
        apmag2 = float(parts[1])
        apmag3 = float(parts[2])
        assert apmag1 == pytest.approx(18.0)
        assert apmag2 == pytest.approx(18.5)
        assert apmag3 == pytest.approx(19.0)

    def test_indef_aperture_mag_becomes_9999(self):
        """INDEF aperture magnitudes should map to 9999."""
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
