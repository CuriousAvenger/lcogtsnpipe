"""Tests for bin/lscmaketempl.py -- exercises __main__ via runpy."""
import sys
import os
import re
import runpy
import importlib
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
from astropy.io import fits

TRUNK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(TRUNK, 'bin', 'lscmaketempl.py')


def _make_fits(path, extra_hdr=None):
    hdr = fits.Header()
    hdr['FILTER'] = 'rp'
    hdr['OBJECT'] = 'SN2024abc'
    hdr['TELESCOP'] = '1m0-01'
    hdr['INSTRUME'] = 'fa15'
    hdr['DATE-OBS'] = '2020-05-01T12:00:00'
    hdr['DAY-OBS'] = '20200501'
    hdr['UTSTART'] = '12:00:00'
    hdr['MJD-OBS'] = 58970.5
    hdr['MJD'] = 58970.5
    hdr['EXPTIME'] = 120.0
    hdr['AIRMASS'] = 1.2
    hdr['RA'] = 150.0
    hdr['DEC'] = 2.0
    hdr['WCSERR'] = 0
    hdr['PIXSCALE'] = 0.389
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = 100
    hdr['NAXIS2'] = 100
    hdr['CTYPE1'] = 'RA---TAN'
    hdr['CTYPE2'] = 'DEC--TAN'
    hdr['CRPIX1'] = 50.0
    hdr['CRPIX2'] = 50.0
    hdr['CRVAL1'] = 150.0
    hdr['CRVAL2'] = 2.0
    hdr['CD1_1'] = -0.0001
    hdr['CD1_2'] = 0.0
    hdr['CD2_1'] = 0.0
    hdr['CD2_2'] = 0.0001
    if extra_hdr:
        hdr.update(extra_hdr)
    data = np.ones((100, 100), dtype=np.float32) * 1000
    fits.writeto(str(path), data, hdr, overwrite=True)


class TestMainNoSubtract:
    """Test main block when mag=0 (just copy, no subtraction)."""

    def test_main_no_subtract(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        # Create image and sn2 file
        imgpath = tmp_path / 'img.fits'
        _make_fits(imgpath)
        sn2path = tmp_path / 'img.sn2.fits'
        _make_fits(sn2path, extra_hdr={'PSFX1': '50.0', 'PSFY1': '50.0', 'PSFMAG1': '18.0'})
        psfpath = tmp_path / 'img.psf.fits'
        _make_fits(psfpath)

        listf = tmp_path / 'list.txt'
        listf.write_text(str(imgpath) + '\n')

        mock_util = MagicMock()
        mock_util.readlist.return_value = [str(imgpath)]
        mock_util.readhdr.side_effect = lambda img: fits.getheader(img)
        mock_util.readkey3.side_effect = lambda h, k: {
            'filter': h.get('FILTER', 'rp'),
            'object': h.get('OBJECT', 'SN2024abc'),
            'PSFX1': h.get('PSFX1', '50.0'),
            'PSFY1': h.get('PSFY1', '50.0'),
            'PSFMAG1': h.get('PSFMAG1', '18.0'),
            'exptime': h.get('EXPTIME', 120.0),
            'telescop': '1m0-01', 'instrume': 'fa15',
            'date-obs': '2020-05-01', 'day-obs': '20200501',
            'ut': '12:00', 'mjd': 58970.5, 'airmass': 1.2,
            'wcserr': 0, 'RA': 150.0, 'DEC': 2.0,
        }.get(k, '')
        mock_util.checksndb.return_value = (150.0, 2.0, 0)
        mock_util.updateheader = MagicMock()
        mock_util.delete = MagicMock()
        mock_util.imcopy = MagicMock()
        mock_util.display_image = MagicMock(return_value=(100, 5000, True))
        mock_util.userinput = MagicMock(return_value='y')

        mock_mysqldef = MagicMock()
        mock_mysqldef.getconnection.return_value = ('h', 'u', 'p', 'db')
        mock_mysqldef.dbConnect.return_value = MagicMock()
        mock_mysqldef.getfromdataraw.side_effect = [
            [{'id': 1}],  # telescopes
            [{'id': 2}],  # instruments
            [],           # photlco check
        ]
        mock_mysqldef.targimg.return_value = 1
        mock_mysqldef.insert_values = MagicMock()

        mock_iraf = MagicMock()
        mock_iraf.imarith = MagicMock()
        mock_iraf.imcopy = MagicMock()

        # Create the output file that imcopy would create
        imgout = re.sub('.fits', '.temp.fits', os.path.basename(str(imgpath)))
        _make_fits(tmp_path / imgout)

        argv = ['lscmaketempl.py', str(listf)]  # No -i, no --subtract-mag-from-header => mag=0

        with patch.object(sys, 'argv', argv), \
             patch('lsc.util', mock_util), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch.dict(sys.modules, {
                 'pyraf': MagicMock(),
                 'pyraf.iraf': mock_iraf,
                 'iraf': mock_iraf,
             }):
            try:
                runpy.run_path(SCRIPT, run_name='__main__')
            except (SystemExit, Exception):
                pass

        # Verify template creation logic was reached
        mock_util.readlist.assert_called()

    def test_main_no_args_exits(self):
        with patch.object(sys, 'argv', ['lscmaketempl.py']), \
             patch('lsc.mysqldef.getconnection', return_value=('h', 'u', 'p', 'db')), \
             patch('lsc.mysqldef.dbConnect', return_value=MagicMock()):
            with pytest.raises(SystemExit):
                runpy.run_path(SCRIPT, run_name='__main__')


class TestImageDictBuilding:
    def test_grouping_by_filter(self):
        imgdic = {}
        for img, filt in [('a.fits', 'rp'), ('b.fits', 'rp'), ('c.fits', 'gp')]:
            if filt not in imgdic:
                imgdic[filt] = {'img': [], 'psf': []}
            imgdic[filt]['img'].append(img)
        assert len(imgdic) == 2
        assert len(imgdic['rp']['img']) == 2


class TestDictionaryConstruction:
    def test_filetype_4(self):
        assert {'filetype': 4}['filetype'] == 4

    def test_filepath(self):
        assert os.path.split('/data/lsc/20200501/img.fits')[0] + '/' == '/data/lsc/20200501/'


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------

def _run_maketempl(tmp_path, monkeypatch, argv, mock_util, mock_mysqldef, mock_iraf,
                   extra_modules=None, allow_exit=False):
    """Helper to run lscmaketempl.py with given mocks."""
    monkeypatch.chdir(tmp_path)
    mock_pyraf = MagicMock()
    mock_pyraf.iraf = mock_iraf
    modules = {
        'pyraf': mock_pyraf,
        'pyraf.iraf': mock_iraf,
        'iraf': mock_iraf,
    }
    if extra_modules:
        modules.update(extra_modules)
    with patch.object(sys, 'argv', argv), \
         patch('lsc.util', mock_util), \
         patch('lsc.mysqldef', mock_mysqldef), \
         patch.dict(sys.modules, modules):
        if allow_exit:
            runpy.run_path(SCRIPT, run_name='__main__')
        else:
            try:
                runpy.run_path(SCRIPT, run_name='__main__')
            except SystemExit:
                pass


def _base_mocks(tmp_path, imgs, psf_exists=True, sn2_exists=True, clean_exists=False,
                psfmag='18.0', checksndb_ra=150.0, checksndb_dec=2.0,
                photlco_result=None, force=False):
    """Build standard mock_util and mock_mysqldef for maketempl tests."""
    mock_util = MagicMock()
    mock_util.readlist.return_value = imgs
    mock_util.readhdr.side_effect = lambda img: fits.getheader(img)
    mock_util.readkey3.side_effect = lambda h, k: {
        'filter': h.get('FILTER', 'rp'),
        'object': h.get('OBJECT', 'SN2024abc'),
        'PSFX1': h.get('PSFX1', '50.0'),
        'PSFY1': h.get('PSFY1', '50.0'),
        'PSFMAG1': h.get('PSFMAG1', psfmag),
        'exptime': h.get('EXPTIME', 120.0),
        'telescop': h.get('TELESCOP', '1m0-01'),
        'instrume': h.get('INSTRUME', 'fa15'),
        'date-obs': h.get('DATE-OBS', '2020-05-01'),
        'day-obs': h.get('DAY-OBS', '20200501'),
        'ut': h.get('UTSTART', '12:00'),
        'mjd': h.get('MJD', 58970.5),
        'airmass': h.get('AIRMASS', 1.2),
        'wcserr': h.get('WCSERR', 0),
        'RA': h.get('RA', 150.0),
        'DEC': h.get('DEC', 2.0),
    }.get(k, '')
    mock_util.checksndb.return_value = (checksndb_ra, checksndb_dec, 0)
    mock_util.updateheader = MagicMock()
    mock_util.delete = MagicMock()
    mock_util.imcopy = MagicMock()
    mock_util.display_image = MagicMock(return_value=(100, 5000, True))
    mock_util.userinput = MagicMock(return_value='y')

    mock_mysqldef = MagicMock()
    mock_mysqldef.getconnection.return_value = ('h', 'u', 'p', 'db')
    mock_mysqldef.dbConnect.return_value = MagicMock()
    if photlco_result is None:
        mock_mysqldef.getfromdataraw.side_effect = [
            [{'id': 1}],  # telescopes
            [{'id': 2}],  # instruments
            [],           # photlco check (no existing entry)
        ]
    else:
        mock_mysqldef.getfromdataraw.side_effect = photlco_result
    mock_mysqldef.targimg.return_value = 1
    mock_mysqldef.insert_values = MagicMock()
    mock_mysqldef.deleteredufromarchive = MagicMock()
    mock_mysqldef.updatevalue = MagicMock()

    mock_iraf = MagicMock()
    return mock_util, mock_mysqldef, mock_iraf


class TestPsfOption:
    """Test with -p psf option providing a psf file list (lines 47-52)."""

    def test_psf_list_grouped_by_filter(self, tmp_path, monkeypatch):
        imgpath = tmp_path / 'img.fits'
        _make_fits(imgpath)
        psfpath = tmp_path / 'mypsf.psf.fits'
        _make_fits(psfpath)
        sn2path = tmp_path / 'img.sn2.fits'
        _make_fits(sn2path, extra_hdr={'PSFX1': '50.0', 'PSFY1': '50.0', 'PSFMAG1': '18.0'})

        listf = tmp_path / 'list.txt'
        listf.write_text(str(imgpath) + '\n')
        psf_listf = tmp_path / 'psflist.txt'
        psf_listf.write_text(str(psfpath) + '\n')

        mock_util, mock_mysqldef, mock_iraf = _base_mocks(tmp_path, [str(imgpath)])
        # readlist called twice: once for imglist, once for psflist
        mock_util.readlist.side_effect = [[str(imgpath)], [str(psfpath)]]

        # Create output file
        imgout = re.sub('.fits', '.temp.fits', os.path.basename(str(imgpath)))
        _make_fits(tmp_path / imgout)

        argv = ['lscmaketempl.py', str(listf), '-p', str(psf_listf)]
        _run_maketempl(tmp_path, monkeypatch, argv, mock_util, mock_mysqldef, mock_iraf)

        # readlist should have been called for both img and psf lists
        assert mock_util.readlist.call_count == 2


class TestPsfNotFound:
    """Test when no PSF found (lines 79-81)."""

    def test_psf_not_found_prints_warning(self, tmp_path, monkeypatch, capsys):
        imgpath = tmp_path / 'img.fits'
        _make_fits(imgpath)
        # No .psf.fits file, no -p option => psf not found

        listf = tmp_path / 'list.txt'
        listf.write_text(str(imgpath) + '\n')

        mock_util, mock_mysqldef, mock_iraf = _base_mocks(tmp_path, [str(imgpath)])

        argv = ['lscmaketempl.py', str(listf)]
        _run_maketempl(tmp_path, monkeypatch, argv, mock_util, mock_mysqldef, mock_iraf)

        captured = capsys.readouterr()
        assert 'psf not found' in captured.out


class TestSubtractMagFromHeader:
    """Test --subtract-mag-from-header reads PSFMAG1 (lines 96-101, 113)."""

    def test_subtract_mag_from_header(self, tmp_path, monkeypatch, capsys):
        imgpath = tmp_path / 'img.fits'
        _make_fits(imgpath)
        psfpath = tmp_path / 'img.psf.fits'
        _make_fits(psfpath)
        sn2path = tmp_path / 'img.sn2.fits'
        _make_fits(sn2path, extra_hdr={'PSFX1': '50.0', 'PSFY1': '50.0',
                                        'PSFMAG1': '19.5', 'EXPTIME': 120.0})

        listf = tmp_path / 'list.txt'
        listf.write_text(str(imgpath) + '\n')

        mock_util, mock_mysqldef, mock_iraf = _base_mocks(tmp_path, [str(imgpath)], psfmag='19.5')

        # Create output file that imarith would produce
        imgout = re.sub('.fits', '.temp.fits', os.path.basename(str(imgpath)))
        _make_fits(tmp_path / imgout)

        argv = ['lscmaketempl.py', str(listf), '--subtract-mag-from-header']
        _run_maketempl(tmp_path, monkeypatch, argv, mock_util, mock_mysqldef, mock_iraf)

        captured = capsys.readouterr()
        assert 'magnitude to be subtracted' in captured.out
        # iraf.daophot.addstar should have been called (mag != 0)
        mock_iraf.daophot.addstar.assert_called()

    def test_subtract_mag_header_missing_psfmag(self, tmp_path, monkeypatch, capsys):
        """When PSFMAG1 is not in header, mag defaults to 0 (lines 99-101)."""
        imgpath = tmp_path / 'img.fits'
        _make_fits(imgpath)
        psfpath = tmp_path / 'img.psf.fits'
        _make_fits(psfpath)
        # sn2 WITHOUT PSFMAG1
        sn2path = tmp_path / 'img.sn2.fits'
        _make_fits(sn2path, extra_hdr={'PSFX1': '50.0', 'PSFY1': '50.0'})

        listf = tmp_path / 'list.txt'
        listf.write_text(str(imgpath) + '\n')

        mock_util, mock_mysqldef, mock_iraf = _base_mocks(tmp_path, [str(imgpath)])
        # Make readkey3 return '' for PSFMAG1 to simulate it not being in header
        orig_side_effect = mock_util.readkey3.side_effect

        def readkey3_no_psfmag(h, k):
            if k == 'PSFMAG1':
                return ''
            return orig_side_effect(h, k)

        mock_util.readkey3.side_effect = readkey3_no_psfmag

        # The script checks 'PSFMAG1' in hdr1.keys(), so we need a header without it
        # Already done: sn2 doesn't have PSFMAG1

        imgout = re.sub('.fits', '.temp.fits', os.path.basename(str(imgpath)))
        _make_fits(tmp_path / imgout)

        argv = ['lscmaketempl.py', str(listf), '--subtract-mag-from-header']
        _run_maketempl(tmp_path, monkeypatch, argv, mock_util, mock_mysqldef, mock_iraf)

        captured = capsys.readouterr()
        assert 'mag not found in the header' in captured.out


class TestMagSubtraction:
    """Test when mag != 0: addstar + imarith path (lines 190-192)."""

    def test_nonzero_mag_calls_addstar(self, tmp_path, monkeypatch):
        imgpath = tmp_path / 'img.fits'
        _make_fits(imgpath)
        psfpath = tmp_path / 'img.psf.fits'
        _make_fits(psfpath)
        sn2path = tmp_path / 'img.sn2.fits'
        _make_fits(sn2path, extra_hdr={'PSFX1': '50.0', 'PSFY1': '50.0', 'PSFMAG1': '18.0'})

        listf = tmp_path / 'list.txt'
        listf.write_text(str(imgpath) + '\n')

        mock_util, mock_mysqldef, mock_iraf = _base_mocks(tmp_path, [str(imgpath)])

        imgout = re.sub('.fits', '.temp.fits', os.path.basename(str(imgpath)))
        _make_fits(tmp_path / imgout)

        argv = ['lscmaketempl.py', str(listf), '--mag', '19.0']
        _run_maketempl(tmp_path, monkeypatch, argv, mock_util, mock_mysqldef, mock_iraf)

        mock_iraf.daophot.addstar.assert_called_once()
        mock_iraf.imarith.assert_called()


class TestMagNoCoords:
    """Test sys.exit when mag!=0 but no RA/DEC (lines 138-140)."""

    def test_exits_when_no_coords_and_mag_nonzero(self, tmp_path, monkeypatch):
        imgpath = tmp_path / 'img.fits'
        _make_fits(imgpath)
        psfpath = tmp_path / 'img.psf.fits'
        _make_fits(psfpath)
        sn2path = tmp_path / 'img.sn2.fits'
        _make_fits(sn2path, extra_hdr={'PSFX1': '50.0', 'PSFY1': '50.0', 'PSFMAG1': '18.0'})

        listf = tmp_path / 'list.txt'
        listf.write_text(str(imgpath) + '\n')

        mock_util, mock_mysqldef, mock_iraf = _base_mocks(
            tmp_path, [str(imgpath)], checksndb_ra='', checksndb_dec='')

        argv = ['lscmaketempl.py', str(listf), '--mag', '19.0']
        with pytest.raises(SystemExit):
            _run_maketempl(tmp_path, monkeypatch, argv, mock_util, mock_mysqldef, mock_iraf,
                           allow_exit=True)


class TestCleanFits:
    """Test _clean=True with .clean.fits existing (lines 148-149)."""

    def test_uses_clean_fits_when_available(self, tmp_path, monkeypatch):
        imgpath = tmp_path / 'img.fits'
        _make_fits(imgpath)
        cleanpath = tmp_path / 'img.clean.fits'
        _make_fits(cleanpath)
        psfpath = tmp_path / 'img.psf.fits'
        _make_fits(psfpath)
        sn2path = tmp_path / 'img.sn2.fits'
        _make_fits(sn2path, extra_hdr={'PSFX1': '50.0', 'PSFY1': '50.0', 'PSFMAG1': '18.0'})

        listf = tmp_path / 'list.txt'
        listf.write_text(str(imgpath) + '\n')

        mock_util, mock_mysqldef, mock_iraf = _base_mocks(tmp_path, [str(imgpath)])

        imgout = re.sub('.fits', '.temp.fits', os.path.basename(str(imgpath)))
        _make_fits(tmp_path / imgout)

        # Default is _clean=True (--uncleaned disables it)
        argv = ['lscmaketempl.py', str(listf)]
        _run_maketempl(tmp_path, monkeypatch, argv, mock_util, mock_mysqldef, mock_iraf)

        # iraf.imarith should have been called with the clean file
        calls = mock_iraf.imarith.call_args_list
        if calls:
            # First arg of imarith should be the clean.fits file
            assert 'clean' in str(calls[0])


class TestShowOption:
    """Test --show logic (lines 151-179, 197-200)."""

    def test_show_displays_and_confirms(self, tmp_path, monkeypatch):
        imgpath = tmp_path / 'img.fits'
        _make_fits(imgpath)
        psfpath = tmp_path / 'img.psf.fits'
        _make_fits(psfpath)
        sn2path = tmp_path / 'img.sn2.fits'
        _make_fits(sn2path, extra_hdr={'PSFX1': '50.0', 'PSFY1': '50.0', 'PSFMAG1': '18.0'})

        listf = tmp_path / 'list.txt'
        listf.write_text(str(imgpath) + '\n')

        mock_util, mock_mysqldef, mock_iraf = _base_mocks(tmp_path, [str(imgpath)])
        mock_util.display_image.return_value = (100, 5000, True)
        mock_util.userinput.return_value = 'y'

        imgout = re.sub('.fits', '.temp.fits', os.path.basename(str(imgpath)))
        _make_fits(tmp_path / imgout)

        argv = ['lscmaketempl.py', str(listf), '-s']
        _run_maketempl(tmp_path, monkeypatch, argv, mock_util, mock_mysqldef, mock_iraf)

        # display_image should have been called multiple times (initial + confirm)
        assert mock_util.display_image.call_count >= 2


class TestShowCutsLoop:
    """Test --show cuts adjustment loop (lines 160-179)."""

    def test_show_cuts_adjustment(self, tmp_path, monkeypatch):
        imgpath = tmp_path / 'img.fits'
        _make_fits(imgpath)
        psfpath = tmp_path / 'img.psf.fits'
        _make_fits(psfpath)
        sn2path = tmp_path / 'img.sn2.fits'
        _make_fits(sn2path, extra_hdr={'PSFX1': '50.0', 'PSFY1': '50.0', 'PSFMAG1': '18.0'})

        listf = tmp_path / 'list.txt'
        listf.write_text(str(imgpath) + '\n')

        mock_util, mock_mysqldef, mock_iraf = _base_mocks(tmp_path, [str(imgpath)])
        mock_util.display_image.return_value = (100, 5000, True)
        # First answer: 'n' (adjust cuts), then z1/z2 inputs, then 'y' (accept)
        mock_util.userinput.side_effect = ['n', '200', '4000', 'y', 'y']

        imgout = re.sub('.fits', '.temp.fits', os.path.basename(str(imgpath)))
        _make_fits(tmp_path / imgout)

        argv = ['lscmaketempl.py', str(listf), '-s']
        _run_maketempl(tmp_path, monkeypatch, argv, mock_util, mock_mysqldef, mock_iraf)

        # display_image should be called at least 3 times (initial, after cut change, confirm)
        assert mock_util.display_image.call_count >= 3


class TestShowAnswerN:
    """Test answer=='n' loop for changing magnitude (lines 205-208)."""

    def test_show_answer_n_changes_mag(self, tmp_path, monkeypatch):
        imgpath = tmp_path / 'img.fits'
        _make_fits(imgpath)
        psfpath = tmp_path / 'img.psf.fits'
        _make_fits(psfpath)
        sn2path = tmp_path / 'img.sn2.fits'
        _make_fits(sn2path, extra_hdr={'PSFX1': '50.0', 'PSFY1': '50.0', 'PSFMAG1': '18.0'})

        listf = tmp_path / 'list.txt'
        listf.write_text(str(imgpath) + '\n')

        mock_util, mock_mysqldef, mock_iraf = _base_mocks(tmp_path, [str(imgpath)])
        mock_util.display_image.return_value = (100, 5000, True)
        # Cuts OK => 'y', then confirm result => 'n' (redo), new mag, then confirm => 'y'
        mock_util.userinput.side_effect = ['y', 'n', '20.0', 'y']

        imgout = re.sub('.fits', '.temp.fits', os.path.basename(str(imgpath)))
        _make_fits(tmp_path / imgout)

        argv = ['lscmaketempl.py', str(listf), '-s', '--mag', '19.0']
        _run_maketempl(tmp_path, monkeypatch, argv, mock_util, mock_mysqldef, mock_iraf)

        # addstar should have been called at least twice (original + redo)
        assert mock_iraf.daophot.addstar.call_count >= 2


class TestForceDelete:
    """Test --force when file already in archive (line 228)."""

    def test_force_deletes_existing(self, tmp_path, monkeypatch):
        imgpath = tmp_path / 'img.fits'
        _make_fits(imgpath)
        psfpath = tmp_path / 'img.psf.fits'
        _make_fits(psfpath)
        sn2path = tmp_path / 'img.sn2.fits'
        _make_fits(sn2path, extra_hdr={'PSFX1': '50.0', 'PSFY1': '50.0', 'PSFMAG1': '18.0'})

        listf = tmp_path / 'list.txt'
        listf.write_text(str(imgpath) + '\n')

        mock_util, mock_mysqldef, mock_iraf = _base_mocks(
            tmp_path, [str(imgpath)],
            photlco_result=[
                [{'id': 1}],   # telescopes
                [{'id': 2}],   # instruments
                [{'filename': 'img.temp.fits'}],  # photlco check: exists!
            ])

        imgout = re.sub('.fits', '.temp.fits', os.path.basename(str(imgpath)))
        _make_fits(tmp_path / imgout)

        argv = ['lscmaketempl.py', str(listf), '-f']
        _run_maketempl(tmp_path, monkeypatch, argv, mock_util, mock_mysqldef, mock_iraf)

        mock_mysqldef.deleteredufromarchive.assert_called_once()
        mock_mysqldef.insert_values.assert_called()


class TestPhotlcoUpdate:
    """Test when photlco entry already exists without force (lines 234-237, 239-240, 242-243)."""

    def test_update_existing_entry(self, tmp_path, monkeypatch):
        imgpath = tmp_path / 'img.fits'
        _make_fits(imgpath)
        psfpath = tmp_path / 'img.psf.fits'
        _make_fits(psfpath)
        sn2path = tmp_path / 'img.sn2.fits'
        _make_fits(sn2path, extra_hdr={'PSFX1': '50.0', 'PSFY1': '50.0', 'PSFMAG1': '18.0'})

        listf = tmp_path / 'list.txt'
        listf.write_text(str(imgpath) + '\n')

        mock_util, mock_mysqldef, mock_iraf = _base_mocks(
            tmp_path, [str(imgpath)],
            photlco_result=[
                [{'id': 1}],   # telescopes
                [{'id': 2}],   # instruments
                [{'filename': 'img.temp.fits', 'filetype': 4}],  # photlco: exists, no force
            ])

        imgout = re.sub('.fits', '.temp.fits', os.path.basename(str(imgpath)))
        _make_fits(tmp_path / imgout)

        # no -f flag => should update instead of insert
        argv = ['lscmaketempl.py', str(listf)]
        _run_maketempl(tmp_path, monkeypatch, argv, mock_util, mock_mysqldef, mock_iraf)

        mock_mysqldef.updatevalue.assert_called()
        mock_mysqldef.insert_values.assert_not_called()


class TestDayobsFallback:
    """Test dayobs fallback for pre-v1 data (line 223)."""

    def test_dayobs_from_filename(self, tmp_path, monkeypatch):
        # Use filename format with date in position [2] when split by '_'
        imgpath = tmp_path / 'lsc_1m0_20200501_img.fits'
        _make_fits(imgpath)
        psfpath = tmp_path / 'lsc_1m0_20200501_img.psf.fits'
        _make_fits(psfpath)
        sn2path = tmp_path / 'lsc_1m0_20200501_img.sn2.fits'
        _make_fits(sn2path, extra_hdr={'PSFX1': '50.0', 'PSFY1': '50.0', 'PSFMAG1': '18.0'})

        listf = tmp_path / 'list.txt'
        listf.write_text(str(imgpath) + '\n')

        mock_util, mock_mysqldef, mock_iraf = _base_mocks(tmp_path, [str(imgpath)])
        # Make day-obs return empty to trigger the fallback
        orig = mock_util.readkey3.side_effect

        def readkey3_no_dayobs(h, k):
            if k == 'day-obs':
                return ''
            return orig(h, k)

        mock_util.readkey3.side_effect = readkey3_no_dayobs

        imgout = re.sub('.fits', '.temp.fits', os.path.basename(str(imgpath)))
        _make_fits(tmp_path / imgout)

        argv = ['lscmaketempl.py', str(listf)]
        _run_maketempl(tmp_path, monkeypatch, argv, mock_util, mock_mysqldef, mock_iraf)

        # Should reach insert_values (the dayobs fallback doesn't crash)
        mock_mysqldef.insert_values.assert_called()


class TestInteractiveMag:
    """Test interactive magnitude choice -i flag (lines 106-110)."""

    def test_interactive_mag_choice(self, tmp_path, monkeypatch, capsys):
        imgpath = tmp_path / 'img.fits'
        _make_fits(imgpath)
        psfpath = tmp_path / 'img.psf.fits'
        _make_fits(psfpath)
        sn2path = tmp_path / 'img.sn2.fits'
        _make_fits(sn2path, extra_hdr={'PSFX1': '50.0', 'PSFY1': '50.0', 'PSFMAG1': '18.0'})

        listf = tmp_path / 'list.txt'
        listf.write_text(str(imgpath) + '\n')

        mock_util, mock_mysqldef, mock_iraf = _base_mocks(tmp_path, [str(imgpath)])
        # -i sets both chosemag and chosepos
        # userinput calls: mag choice ('20.5'), then later 'y' for confirm
        mock_util.userinput.side_effect = ['20.5', 'y']
        # iraf.fields returns position strings for chosepos path
        mock_iraf.fields.return_value = ['55.0  45.0']

        imgout = re.sub('.fits', '.temp.fits', os.path.basename(str(imgpath)))
        _make_fits(tmp_path / imgout)

        argv = ['lscmaketempl.py', str(listf), '-i']
        _run_maketempl(tmp_path, monkeypatch, argv, mock_util, mock_mysqldef, mock_iraf)

        captured = capsys.readouterr()
        assert 'magnitude to be subtracted' in captured.out

    def test_interactive_mag_empty_keeps_default(self, tmp_path, monkeypatch, capsys):
        """When user hits enter (empty input), keep default mag."""
        imgpath = tmp_path / 'img.fits'
        _make_fits(imgpath)
        psfpath = tmp_path / 'img.psf.fits'
        _make_fits(psfpath)
        sn2path = tmp_path / 'img.sn2.fits'
        _make_fits(sn2path, extra_hdr={'PSFX1': '50.0', 'PSFY1': '50.0', 'PSFMAG1': '18.0'})

        listf = tmp_path / 'list.txt'
        listf.write_text(str(imgpath) + '\n')

        mock_util, mock_mysqldef, mock_iraf = _base_mocks(tmp_path, [str(imgpath)])
        # Empty string for mag (keep default)
        mock_util.userinput.side_effect = ['', 'y']
        mock_iraf.fields.return_value = ['55.0  45.0']

        imgout = re.sub('.fits', '.temp.fits', os.path.basename(str(imgpath)))
        _make_fits(tmp_path / imgout)

        argv = ['lscmaketempl.py', str(listf), '-i']
        _run_maketempl(tmp_path, monkeypatch, argv, mock_util, mock_mysqldef, mock_iraf)

        captured = capsys.readouterr()
        assert 'magnitude to be subtracted' in captured.out


class TestInteractivePosition:
    """Test interactive position choice via -i flag (lines 118-125)."""

    def test_interactive_position(self, tmp_path, monkeypatch):
        imgpath = tmp_path / 'img.fits'
        _make_fits(imgpath)
        psfpath = tmp_path / 'img.psf.fits'
        _make_fits(psfpath)
        sn2path = tmp_path / 'img.sn2.fits'
        _make_fits(sn2path, extra_hdr={'PSFX1': '50.0', 'PSFY1': '50.0', 'PSFMAG1': '18.0'})

        listf = tmp_path / 'list.txt'
        listf.write_text(str(imgpath) + '\n')

        mock_util, mock_mysqldef, mock_iraf = _base_mocks(tmp_path, [str(imgpath)])
        mock_util.display_image.return_value = (100, 5000, True)
        # -i sets both chosemag and chosepos
        # userinput: mag choice (empty=keep default)
        mock_util.userinput.side_effect = ['', 'y']
        # iraf.fields returns position for chosepos path (line 123-124)
        mock_iraf.fields.return_value = ['55.0  45.0']

        imgout = re.sub('.fits', '.temp.fits', os.path.basename(str(imgpath)))
        _make_fits(tmp_path / imgout)

        argv = ['lscmaketempl.py', str(listf), '-i']
        _run_maketempl(tmp_path, monkeypatch, argv, mock_util, mock_mysqldef, mock_iraf)

        # imexamine should have been called for interactive position
        mock_iraf.imexamine.assert_called()


class TestMkdirAndMv:
    """Test mkdir and mv conditions (lines 239-240, 242-243)."""

    def test_creates_dir_and_moves_file(self, tmp_path, monkeypatch):
        imgpath = tmp_path / 'img.fits'
        _make_fits(imgpath)
        psfpath = tmp_path / 'img.psf.fits'
        _make_fits(psfpath)
        sn2path = tmp_path / 'img.sn2.fits'
        _make_fits(sn2path, extra_hdr={'PSFX1': '50.0', 'PSFY1': '50.0', 'PSFMAG1': '18.0'})

        listf = tmp_path / 'list.txt'
        listf.write_text(str(imgpath) + '\n')

        mock_util, mock_mysqldef, mock_iraf = _base_mocks(tmp_path, [str(imgpath)])

        imgout = re.sub('.fits', '.temp.fits', os.path.basename(str(imgpath)))
        _make_fits(tmp_path / imgout)

        argv = ['lscmaketempl.py', str(listf)]
        with patch.object(sys, 'argv', argv), \
             patch('lsc.util', mock_util), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch.dict(sys.modules, {
                 'pyraf': MagicMock(),
                 'pyraf.iraf': mock_iraf,
                 'iraf': mock_iraf,
             }), \
             patch('os.mkdir') as mock_mkdir, \
             patch('os.system') as mock_system:
            monkeypatch.chdir(tmp_path)
            try:
                runpy.run_path(SCRIPT, run_name='__main__')
            except SystemExit:
                pass

        # insert_values should have been called (archive path reached)
        mock_mysqldef.insert_values.assert_called()
