"""
Tests for lsc.lscsnoopy.
With iraf fully mocked via conftest.py we can run fitsn through its body
(minus the _show=True display branches) by supplying a real FITS file.
"""
import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_iraf():
    return sys.modules['pyraf'].iraf


def _make_snoopy_fits(tmp_path, name='sn_test'):
    """Write a FITS file with the keys fitsn reads."""
    from astropy.io import fits
    hdr = fits.Header()
    hdr['GAIN'] = 2.0
    hdr['RDNOISE'] = 5.0
    hdr['RON'] = 5.0
    hdr['EXPTIME'] = 120.0
    data = np.random.default_rng(99).normal(1000, 50, (512, 512)).astype(np.float32)
    fits.writeto(str(tmp_path / (name + '.fits')), data, hdr, overwrite=True)
    return name


# ── importability ─────────────────────────────────────────────────────────────

class TestLscsnoopyImportable:
    def test_module_imports(self):
        import lsc.lscsnoopy  # noqa: F401

    def test_fitsn_is_callable(self):
        from lsc.lscsnoopy import fitsn
        assert callable(fitsn)

    def test_manusn_is_callable(self):
        from lsc.lscsnoopy import manusn
        assert callable(manusn)

    def test_errore_is_callable(self):
        from lsc.lscsnoopy import errore
        assert callable(errore)


# ── fitsn ─────────────────────────────────────────────────────────────────────

class TestFitsn:
    """fitsn main body executes with _show=False; iraf mocked; real FITS file."""

    def _configure_iraf_txdump(self, data='28.5 0.1 50.2 50.2'):
        iraf = _get_iraf()
        iraf.txdump.return_value = [data]
        return iraf

    def test_returns_14_lists(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_snoopy_fits(tmp_path)
        self._configure_iraf_txdump()
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        assert len(result) == 14

    def test_fitmag_populated_from_txdump(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_snoopy_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['28.5 0.1 50.2 50.2']
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        fitmag = result[9]
        assert len(fitmag) == 1
        assert fitmag[0] == pytest.approx(28.5)

    def test_recenter_true_sets_answ_yes(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_snoopy_fits(tmp_path)
        self._configure_iraf_txdump()
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        result = fitsn(name, name + 'psf', 'coords.coo',
                       True, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        # daopars.recenter should have been called with 'yes'
        assert iraf.noao.digiphot.daophot.daopars.recenter == 'yes'

    def test_recenter_false_sets_answ_no(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_snoopy_fits(tmp_path)
        self._configure_iraf_txdump()
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        assert iraf.noao.digiphot.daophot.daopars.recenter == 'no'

    def test_truemag_computed_with_apco0(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_snoopy_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0, apco0=0.5)
        truemag = result[10]
        assert len(truemag) == 1
        assert truemag[0] == pytest.approx(20.5)

    def test_indef_fitmag_handled(self, tmp_path, monkeypatch):
        """INDEF string in txdump output → appended as-is, truemag='INDEF'."""
        from lsc.lscsnoopy import fitsn
        name = _make_snoopy_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['INDEF 0.5 50.0 50.0']
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        fitmag = result[9]
        assert fitmag[0] == 'INDEF'

    def test_apertures_computed_from_fwhm(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_snoopy_fits(tmp_path)
        self._configure_iraf_txdump()
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        fitsn(name, name + 'psf', 'coords.coo',
              False, 4.0, 'original', 'sn', 'residual',
              False, False, 60000.0, 0.0)
        # a1=4, a2=8, a3=12 → apertures="4,8,12"
        assert iraf.noao.daophot.photpars.apertures == '4,8,12'

    def test_empty_txdump_returns_empty_lists(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_snoopy_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = []
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        fitmag = result[9]
        assert fitmag == []

    def test_multiple_stars(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import fitsn
        name = _make_snoopy_fits(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['20.0 0.05 50.0 50.0', '21.0 0.06 60.0 60.0']
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        fitmag = result[9]
        assert len(fitmag) == 2


# ── manusn ───────────────────────────────────────────────────────────────────

class TestManusn:
    """manusn uses display_image; mock it to avoid ds9 dependency."""

    def test_basic_run_returns_values(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import manusn
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['50.0 50.0']
        # Mock display_image which requires ds9
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            result = manusn('test', 'testpsf', 0.0,
                            [15.0], [0.0], [15.0], [15.0], [0.0], [15.0],
                            [20.0], ['20.5'], [0.1], [50.0], [50.0],
                            100.0, 300.0, 200.0, 7, 5.0, 0, 0, 0.1)
        assert len(result) == 12

    def test_truemag_indef_branch(self, tmp_path, monkeypatch):
        """When truemag[0]=='INDEF', addstar is called instead of imarith."""
        from lsc.lscsnoopy import manusn
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['50.0 50.0']
        iraf.field.return_value = ['50.0 50.0']
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            result = manusn('test', 'testpsf', '0.0',  # dmag0 must be str (source bug: ' '+dmag0)
                            [15.0], [0.0], [15.0], [15.0], [0.0], [15.0],
                            [20.0], ['INDEF'], [0.0], [50.0], [50.0],
                            100.0, 300.0, 200.0, 7, 5.0, 0, 0, 0.1)
        assert iraf.addstar.called

    def test_newmag_shifted_by_dmag0(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import manusn
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['50.0 50.0']
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            result = manusn('test', 'testpsf', 1.0,
                            [15.0], [0.0], [15.0], [15.0], [0.0], [15.0],
                            [20.0], ['20.0'], [0.1], [50.0], [50.0],
                            100.0, 300.0, 200.0, 7, 5.0, 0, 0, 0.1)
        newmag = result[11]
        assert newmag[0] == pytest.approx(21.0)


# ── errore ───────────────────────────────────────────────────────────────────

class TestErrore:
    """errore does 9 iterations of addstar + fitsn; mock fitsn to control output."""

    def test_errore_returns_two_values(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import errore
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.txdump.return_value = ['50.0 50.0']
        # Create skyfit.fits so os.system('cp skyfit.fits artskyfit.fits') has a source
        from astropy.io import fits as afits
        data = np.ones((50, 50), dtype=np.float32)
        afits.writeto(str(tmp_path / 'skyfit.fits'), data, overwrite=True)
        afits.writeto(str(tmp_path / 'artbg.fits'), data, overwrite=True)
        # Create img.sn.coo with a coordinate
        (tmp_path / 'sn_test.sn.coo').write_text('50.0 50.0\n')
        # Mock fits.getdata to avoid reading artbg.fits
        with patch('lsc.lscsnoopy.fits.getdata', return_value=np.ones((50, 50)) * 1000):
            with patch('lsc.lscsnoopy.fitsn', return_value=(
                [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                [15.0], [0.1], [15.2], [20.0], ['20.5'], [0.1],
                [50.0], [50.0]
            )):
                with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
                    result = errore('sn_test', 'sn_testpsf', 'sn_test.sn.coo',
                                    7, ['20.5'], 5.0, 2, False, False,
                                    3, 100.0, 300.0, 200.0, 512, 512,
                                    2, 2, False, 0, 60000.0, 0.0)
        assert len(result) == 2

    def test_errore_returns_float_errors(self, tmp_path, monkeypatch):
        from lsc.lscsnoopy import errore
        monkeypatch.chdir(tmp_path)
        from astropy.io import fits as afits
        data = np.ones((50, 50), dtype=np.float32)
        afits.writeto(str(tmp_path / 'skyfit.fits'), data, overwrite=True)
        afits.writeto(str(tmp_path / 'artbg.fits'), data, overwrite=True)
        (tmp_path / 'sn2_test.sn.coo').write_text('50.0 50.0\n')
        with patch('lsc.lscsnoopy.fits.getdata', return_value=np.ones((50, 50)) * 1000):
            with patch('lsc.lscsnoopy.fitsn', return_value=(
                [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                [15.0], [0.1], [15.2], [20.0], ['20.0'], [0.1],
                [50.0], [50.0]
            )):
                with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
                    arterr2, arterr = errore('sn2_test', 'sn2_testpsf', 'sn2_test.sn.coo',
                                            7, ['20.0'], 5.0, 2, False, False,
                                            3, 100.0, 300.0, 200.0, 512, 512,
                                            2, 2, False, 0, 60000.0, 0.0)
        # Both should be float values (0 or actual std)
        assert isinstance(arterr, (int, float))
        assert isinstance(arterr2, (int, float))
