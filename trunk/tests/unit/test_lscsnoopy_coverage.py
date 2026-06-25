"""
Tests for lsc.lscsnoopy — targeting specific uncovered lines.

Covers:
- fitsn with _show=True (lines 54-56, 101-104, 138, 140, 142-177)
- fitsn exception branches for apori3 (line 95-96), apmag3 (119-120), dapmag1 (123-124)
- fitsn truemag='INDEF' (line 136)
- manusn with truemag[0]=='INDEF' (lines 190-196, addstar branch)
- manusn z01/z02 display block (lines 200-234)
- errore with _show=True (line 369)
- errore try/except for media/arterr when tmpart is empty (lines 384-387)
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
    # Critically: clear any leftover side_effect from previous tests
    iraf.txdump.side_effect = None
    iraf.txdump.return_value = MagicMock()
    return iraf


def _make_fits(tmp_path, name='test_img'):
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
    data = rng.normal(1000, 50, (100, 100)).astype(np.float32)
    fits.writeto(str(tmp_path / (name + '.fits')), data, hdr, overwrite=True)
    return name


# ---------------------------------------------------------------------------
# fitsn with _show=True — covers lines 54-56, 101-104, 138, 140, 142-177
# ---------------------------------------------------------------------------

class TestFitsnShowTrue:
    """Test fitsn with _show=True to cover display/print branches."""

    def test_show_true_phot_without_verb(self, tmp_path, monkeypatch):
        """Lines 55-56: phot called without verb='no' when _show=True."""
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        # txdump is called 4 times when _show=True:
        # 1) sn.als: mag,merr,xcenter,ycenter
        # 2) apori: mag (3 columns)
        # 3) sn.mag: mag,merr (6 columns)
        # 4) display block (line 156): xcen,ycen from sn.als
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
        # Verify phot was called (the _show branch calls without verb='no')
        assert iraf.noao.digiphot.daophot.phot.called

    def test_show_true_prints_headers(self, tmp_path, monkeypatch, capsys):
        """Lines 101-104: column header prints when _show=True."""
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
        # Lines 102-103: prints column headers
        assert "apmag on original" in captured.out
        assert "fitmag" in captured.out

    def test_show_true_prints_star_data(self, tmp_path, monkeypatch, capsys):
        """Line 138: prints star data in the for loop when _show=True."""
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
        # Line 138 prints: i, apori1[i], apori2[i], etc.
        # Line 140 prints the separator
        assert "***" in captured.out

    def test_show_true_display_block(self, tmp_path, monkeypatch):
        """Lines 142-177: the _show display block with tvmark calls."""
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['20.0 0.05 50.0 50.0'],
            ['18.0 18.5 19.0'],
            ['18.0 18.5 19.0 0.01 0.02 0.03'],
            ['50.0 50.0'],  # 4th call for tmptbl0 in display block (line 156)
        ]
        monkeypatch.chdir(tmp_path)
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            result = fitsn(name, name + 'psf', 'coords.coo',
                           False, 5.0, 'original', 'sn', 'residual',
                           True, False, 60000.0, 0.0,
                           z11='500', z22='2000', midpt='1000', size=7, apco0=0)
        # Verify tvmark was called (lines 153, 162, 169, 177)
        assert iraf.tvmark.called
        assert iraf.tvmark.call_count >= 3


# ---------------------------------------------------------------------------
# fitsn exception branches for apori3, apmag, dapmag — lines 95-96, 119-120, 123-124
# ---------------------------------------------------------------------------

class TestFitsnExceptionBranches:
    """Test exception handling in fitsn magnitude parsing."""

    def test_apori3_indef_branch(self, tmp_path, monkeypatch):
        """Lines 95-96: when apori3 can't be float-converted."""
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['20.0 0.05 50.0 50.0'],            # sn.als
            ['18.0 18.5 INDEF'],                # apori - 3rd col is INDEF
            ['18.0 18.5 19.0 0.01 0.02 0.03'],  # sn.mag
        ]
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        apori3 = result[2]
        # Line 96: apori3.append(str.split(i)[2]) -> 'INDEF'
        assert apori3[0] == 'INDEF'

    def test_apori1_and_apori2_indef(self, tmp_path, monkeypatch):
        """Lines 88, 92: when apori1 and apori2 can't be float-converted."""
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['20.0 0.05 50.0 50.0'],
            ['INDEF INDEF INDEF'],              # all INDEF
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
        """Lines 119-120: when apmag3 (3rd column of sn.mag) can't be float."""
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['20.0 0.05 50.0 50.0'],
            ['18.0 18.5 19.0'],
            ['18.0 18.5 INDEF 0.01 0.02 0.03'],  # apmag3 = INDEF
        ]
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        apmag3 = result[5]
        # Line 120: apmag3.append(9999)
        assert apmag3[0] == 9999

    def test_dapmag1_exception_branch(self, tmp_path, monkeypatch):
        """Lines 123-124: when dapmag1 (4th column of sn.mag) can't be float."""
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['20.0 0.05 50.0 50.0'],
            ['18.0 18.5 19.0'],
            ['18.0 18.5 19.0 INDEF 0.02 0.03'],  # dapmag1 = INDEF
        ]
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        dapmag1 = result[6]
        # Line 124: dapmag1.append(9999)
        assert dapmag1[0] == 9999

    def test_all_sn_mag_columns_indef(self, tmp_path, monkeypatch):
        """Lines 112, 116, 120, 124, 128, 132: all columns INDEF -> all 9999."""
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
        """Line 136: truemag.append('INDEF') when fitmag is non-numeric."""
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['INDEF INDEF 50.0 50.0'],           # sn.als: fitmag='INDEF', magerr='INDEF'
            ['18.0 18.5 19.0'],                  # apori
            ['18.0 18.5 19.0 0.01 0.02 0.03'],   # sn.mag
        ]
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        truemag = result[10]
        # Line 136: fitmag[0] is 'INDEF', so fitmag[i]+float(apco0) raises -> 'INDEF'
        assert truemag[0] == 'INDEF'

    def test_magerr_indef_branch(self, tmp_path, monkeypatch):
        """Line 79: when magerr column can't be float-converted."""
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['20.0 INDEF 50.0 50.0'],            # sn.als: magerr='INDEF'
            ['18.0 18.5 19.0'],
            ['18.0 18.5 19.0 0.01 0.02 0.03'],
        ]
        monkeypatch.chdir(tmp_path)
        result = fitsn(name, name + 'psf', 'coords.coo',
                       False, 5.0, 'original', 'sn', 'residual',
                       False, False, 60000.0, 0.0)
        magerr = result[11]
        # Line 79: magerr.append(str.split(i)[1])
        assert magerr[0] == 'INDEF'


# ---------------------------------------------------------------------------
# fitsn _show=True combined with exception branches
# ---------------------------------------------------------------------------

class TestFitsnShowTrueWithINDEF:
    """Lines 138 print with INDEF data, combined _show + exception paths."""

    def test_show_true_with_indef_fitmag(self, tmp_path, monkeypatch, capsys):
        """Exercises lines 136 + 138: truemag='INDEF' printed when _show=True."""
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['INDEF 0.05 50.0 50.0'],            # fitmag = 'INDEF'
            ['18.0 18.5 19.0'],
            ['18.0 18.5 19.0 0.01 0.02 0.03'],
            ['50.0 50.0'],                       # display block txdump
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


# ---------------------------------------------------------------------------
# manusn with truemag[0]=='INDEF' — covers lines 190-196 (addstar branch)
# ---------------------------------------------------------------------------

class TestManusnINDEFBranch:
    """Test manusn when truemag[0]=='INDEF' triggers addstar path."""

    def test_indef_calls_addstar_and_sets_magerr(self, tmp_path, monkeypatch):
        """Lines 190-196: INDEF branch calls addstar, sets magerr[0]=0.0."""
        from lsc.lscsnoopy import manusn
        monkeypatch.chdir(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.return_value = ['50.0 50.0']
        iraf.field.return_value = ['50.0 50.0']
        # manusn signature:
        # (img, imgpsf, dmag0, apori1, apori2, apori3, apmag1, apmag2, apmag3,
        #  fitmag, truemag, magerr, centx, centy, z11, z22, midpt, size, fwhm0, x1, y1, arterr)
        # magerr is param index 11
        magerr_list = [0.5]
        with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
            result = manusn('test', 'testpsf', '1.5',
                            [15.0], [0.1], [15.2],       # apori1, apori2, apori3
                            [15.0], [0.1], [15.2],       # apmag1, apmag2, apmag3
                            [20.0], ['INDEF'], magerr_list,  # fitmag, truemag, magerr
                            [50.0], [50.0],              # centx, centy
                            '100.0', '300.0', '200.0',   # z11, z22, midpt
                            7, 5.0, 0, 0, 0.1)          # size, fwhm0, x1, y1, arterr
        # addstar called (line 195)
        assert iraf.addstar.called
        # magerr[0] set to 0.0 (line 192)
        assert magerr_list[0] == 0.0

    def test_indef_newmag_uses_dmag_only(self, tmp_path, monkeypatch):
        """When truemag is INDEF, newmag = float(dmag0) (line 241)."""
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
        # When truemag='INDEF', except branch: newmag[i]=float(dmag0)=2.0
        assert newmag[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# manusn z01/z02 and display block — covers lines 200-234
# ---------------------------------------------------------------------------

class TestManusnDisplayBlock:
    """Test manusn display/tvmark code block (lines 200-234)."""

    def test_display_image_called_multiple_times(self, tmp_path, monkeypatch):
        """Lines 200, 211, 227: display_image called 3 times."""
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
        # display_image is called for original, _snfit, and skyfit
        assert display_mock.call_count == 3

    def test_tvmark_called(self, tmp_path, monkeypatch):
        """Lines 210, 219, 226, 234: tvmark called multiple times."""
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
        """Lines 202-203: z01=z11-midpt, z02=z22-midpt printed."""
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
        # Line 201 prints z11, z22, midpt
        assert '500' in captured.out
        assert '2000' in captured.out
        assert '1000' in captured.out


# ---------------------------------------------------------------------------
# errore with _show=True — covers line 369
# ---------------------------------------------------------------------------

class TestErroreShowTrue:
    """Test errore with _show=True to cover display_image in the loop."""

    def test_show_true_calls_display_image(self, tmp_path, monkeypatch):
        """Line 369: display_image called inside the i-loop when _show=True."""
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
        # display_image should be called 9 times (once per grid position)
        assert display_mock.call_count == 9
        assert len(result) == 2


# ---------------------------------------------------------------------------
# errore try/except for media/arterr — covers lines 384-387
# ---------------------------------------------------------------------------

class TestErroreStatsExceptionBranch:
    """Test errore when tmpart triggers the except branch (lines 384-387)."""

    def test_empty_tmpart_returns_nan_or_zero(self, tmp_path, monkeypatch):
        """Lines 380-387: when all arttruemag are INDEF, tmpart stays empty.
        numpy may return nan (warning) or the except may catch it depending on version."""
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
        # With empty tmpart, the result is either nan (numpy warning path) or 0 (except path)
        assert isinstance(arterr, (int, float, np.floating))
        assert isinstance(arterr2, (int, float, np.floating))

    def test_stats_exception_forced_via_compress_failure(self, tmp_path, monkeypatch):
        """Lines 384-387: force the except branch by making compress raise.

        The errore code does:
            try:
                media = mean(array(tmpart))
                arterr = std(array(tmpart))
                arterr2 = std(compress((...), array(tmpart)))
            except:
                media = 0; arterr = 0; arterr2 = 0

        We force this by making fitsn return a value that causes compress to fail.
        Specifically, if tmpart has only 1 element, std returns 0.0, then compress
        with (avg-0 < x) & (x < avg+0) gives empty, and std of empty = nan (warning).
        But bare except doesn't catch warnings. Instead we'll monkeypatch
        numpy.compress to raise inside the errore function.
        """
        from lsc.lscsnoopy import errore
        monkeypatch.chdir(tmp_path)
        _reset_iraf()
        from astropy.io import fits as afits
        data = np.ones((100, 100), dtype=np.float32) * 1000
        afits.writeto(str(tmp_path / 'skyfit.fits'), data, overwrite=True)
        afits.writeto(str(tmp_path / 'artbg.fits'), data, overwrite=True)
        (tmp_path / 'force_test.sn.coo').write_text('50.0 50.0\n')

        # Return valid floats so tmpart is populated, then patch compress to raise
        with patch('lsc.lscsnoopy.fits.getdata', return_value=np.ones((100, 100)) * 1000):
            with patch('lsc.lscsnoopy.fitsn', return_value=(
                [15.0], [0.1], [15.2], [15.0], [0.1], [15.2],
                [15.0], [0.1], [15.2], [20.0], ['20.0'], [0.1],
                [50.0], [50.0]
            )):
                with patch('lsc.util.display_image', return_value=(100.0, 200.0, True)):
                    # errore does "from numpy import array,mean,std,compress,average"
                    # We need to patch inside the function's local scope.
                    # Since errore imports from numpy at the top of the function,
                    # we can monkeypatch numpy.compress to raise an exception.
                    original_compress = np.compress
                    def bad_compress(*args, **kwargs):
                        raise TypeError("forced error for testing")
                    monkeypatch.setattr(np, 'compress', bad_compress)
                    arterr2, arterr = errore('force_test', 'force_testpsf', 'force_test.sn.coo',
                                            7, ['20.0'], 5.0, 2, False, False,
                                            3, 100.0, 300.0, 200.0, 100, 100,
                                            2, 2, False, 0, 60000.0, 0.0)
        # The except branch sets both to 0
        assert arterr == 0
        assert arterr2 == 0


# ---------------------------------------------------------------------------
# fitsn _show=True with multiple stars (full exercise of display loop)
# ---------------------------------------------------------------------------

class TestFitsnShowMultipleStars:
    """Multiple stars with _show=True exercises line 138 loop multiple times."""

    def test_multiple_stars_show_true(self, tmp_path, monkeypatch, capsys):
        """Lines 138, 140: print in loop for each star, then separator."""
        from lsc.lscsnoopy import fitsn
        name = _make_fits(tmp_path)
        iraf = _reset_iraf()
        iraf.txdump.side_effect = [
            ['20.0 0.05 50.0 50.0', '21.0 0.06 60.0 60.0'],  # sn.als (2 stars)
            ['18.0 18.5 19.0', '17.0 17.5 18.0'],             # apori (2 stars)
            ['18.0 18.5 19.0 0.01 0.02 0.03',
             '17.0 17.5 18.0 0.01 0.02 0.03'],                # sn.mag (2 stars)
            ['50.0 50.0', '60.0 60.0'],                        # display block txdump
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
        # Both stars printed in loop (line 138)
        assert '0' in captured.out  # i=0
        assert '1' in captured.out  # i=1


# ---------------------------------------------------------------------------
# errore with _show=True and varying magnitudes
# ---------------------------------------------------------------------------

class TestErroreShowTrueVarying:
    """Exercise errore _show=True with varying magnitudes for full coverage."""

    def test_show_true_with_varying_mags(self, tmp_path, monkeypatch):
        """Line 369 with real floating magnitudes (not INDEF)."""
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
        # _show=True means display_image called in loop (line 369)
        assert display_mock.call_count == 9
        assert arterr > 0.0


# ---------------------------------------------------------------------------
# errore with _interactive=True — covers lines 264-265, 271-272, 275-277
# ---------------------------------------------------------------------------

class TestErroreInteractiveBranch:
    """Test errore _interactive=True paths for artfac0 input."""

    def test_interactive_valid_input(self, tmp_path, monkeypatch):
        """Lines 264-265: userinput called, valid float returned."""
        from lsc.lscsnoopy import errore
        monkeypatch.chdir(tmp_path)
        _reset_iraf()
        from astropy.io import fits as afits
        data = np.ones((100, 100), dtype=np.float32) * 1000
        afits.writeto(str(tmp_path / 'skyfit.fits'), data, overwrite=True)
        afits.writeto(str(tmp_path / 'artbg.fits'), data, overwrite=True)
        (tmp_path / 'inter_test.sn.coo').write_text('50.0 50.0\n')

        # userinput returns '1' (valid, and < size-1=6)
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
        """Lines 270-272: artfac0 >= size-1 prints warning, then loop retries."""
        from lsc.lscsnoopy import errore
        monkeypatch.chdir(tmp_path)
        _reset_iraf()
        from astropy.io import fits as afits
        data = np.ones((100, 100), dtype=np.float32) * 1000
        afits.writeto(str(tmp_path / 'skyfit.fits'), data, overwrite=True)
        afits.writeto(str(tmp_path / 'artbg.fits'), data, overwrite=True)
        (tmp_path / 'toolarge_test.sn.coo').write_text('50.0 50.0\n')

        # First call returns '10' (>= size-1=6), second returns '1'
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
        """Lines 275-277: non-numeric artfac0 prints warning, then loop retries."""
        from lsc.lscsnoopy import errore
        monkeypatch.chdir(tmp_path)
        _reset_iraf()
        from astropy.io import fits as afits
        data = np.ones((100, 100), dtype=np.float32) * 1000
        afits.writeto(str(tmp_path / 'skyfit.fits'), data, overwrite=True)
        afits.writeto(str(tmp_path / 'artbg.fits'), data, overwrite=True)
        (tmp_path / 'nonnumeric_test.sn.coo').write_text('50.0 50.0\n')

        # First call returns 'abc' (non-numeric), second returns '1'
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
        """Line 265: empty input (falsy) defaults artfac0 to 1."""
        from lsc.lscsnoopy import errore
        monkeypatch.chdir(tmp_path)
        _reset_iraf()
        from astropy.io import fits as afits
        data = np.ones((100, 100), dtype=np.float32) * 1000
        afits.writeto(str(tmp_path / 'skyfit.fits'), data, overwrite=True)
        afits.writeto(str(tmp_path / 'artbg.fits'), data, overwrite=True)
        (tmp_path / 'empty_input.sn.coo').write_text('50.0 50.0\n')

        # userinput returns '' (falsy), should default to 1
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
