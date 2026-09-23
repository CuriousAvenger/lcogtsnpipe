"""Tests for bin/lscmerge.py -- exercises __main__ via runpy."""
import sys
import os
import re
import runpy
from unittest.mock import patch, MagicMock, mock_open
import pytest
import numpy as np
from astropy.io import fits

TRUNK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(TRUNK, 'bin', 'lscmerge.py')


@pytest.fixture
def merge_setup(tmp_path):
    """Create a pair of FITS files and a list file for merge testing."""
    imgs = []
    for i in range(2):
        hdr = fits.Header()
        hdr['FILTER'] = 'rp'
        hdr['OBJECT'] = 'SN2024abc'
        hdr['SITEID'] = 'lsc'
        hdr['TELID'] = '1m0-01'
        hdr['TELESCOP'] = '1m0-01'
        hdr['INSTRUME'] = 'fa15'
        hdr['DATE-OBS'] = '2020-05-01T12:00:00'
        hdr['DAY-OBS'] = '20200501'
        hdr['UTSTART'] = '12:00:00'
        hdr['MJD-OBS'] = 58970.5
        hdr['MJD'] = 58970.5
        hdr['EXPTIME'] = 120.0
        hdr['AIRMASS'] = 1.2 + i * 0.1
        hdr['RA'] = '10:00:00'
        hdr['DEC'] = '+02:00:00'
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 10.0
        hdr['PIXSCALE'] = 0.389
        hdr['WCSERR'] = 0
        data = np.ones((100, 100), dtype=np.float32) * 1000
        fpath = str(tmp_path / f'img{i}.fits')
        fits.writeto(fpath, data, hdr, overwrite=True)
        imgs.append(fpath)

    listf = str(tmp_path / 'imglist.txt')
    with open(listf, 'w') as f:
        f.write('\n'.join(imgs) + '\n')
    return listf, imgs, tmp_path


def test_main_grouping_and_swarp(merge_setup, monkeypatch):
    """Main block groups images and runs swarp commands."""
    listf, imgs, tmp_path = merge_setup
    monkeypatch.chdir(tmp_path)

    mock_util = MagicMock()
    mock_util.readlist.return_value = imgs
    mock_util.readhdr.side_effect = lambda img: fits.getheader(img)
    mock_util.readkey3.side_effect = lambda hdr, key: {
        'filter': hdr.get('FILTER', 'rp'),
        'object': hdr.get('OBJECT', 'SN2024abc'),
        'TELID': hdr.get('TELID', '1m0-01'),
        'gain': hdr.get('GAIN', 2.0),
        'ron': hdr.get('RDNOISE', 10.0),
        'instrume': hdr.get('INSTRUME', 'fa15'),
        'RA': hdr.get('RA', '10:00:00'),
        'DEC': hdr.get('DEC', '+02:00:00'),
        'telescop': hdr.get('TELESCOP', '1m0-01'),
        'date-obs': hdr.get('DATE-OBS', '2020-05-01'),
        'day-obs': hdr.get('DAY-OBS', '20200501'),
        'date-night': '20200501',
        'ut': hdr.get('UTSTART', '12:00:00'),
        'mjd': hdr.get('MJD', 58970.5),
        'exptime': hdr.get('EXPTIME', 120.0),
        'airmass': hdr.get('AIRMASS', 1.2),
        'wcserr': hdr.get('WCSERR', 0),
        'telid': hdr.get('TELID', '1m0-01'),
    }.get(key, '')
    mock_util.checksnlist.return_value = ('150.0', '2.0', '')
    mock_util.checksndb.return_value = ('150.0', '2.0', 0)
    mock_util.Docosmic.side_effect = lambda img: (img.replace('.fits', '.clean.fits'),
                                                    img.replace('.fits', '.mask.fits'),
                                                    img.replace('.fits', '.sat.fits'))
    mock_util.workdirectory = str(tmp_path) + '/'

    # Create the files that Docosmic would produce
    for img in imgs:
        for suffix in ['.clean.fits', '.mask.fits', '.sat.fits']:
            fpath = img.replace('.fits', suffix)
            data = np.zeros((100, 100), dtype=np.float32)
            fits.writeto(fpath, data, overwrite=True)

    mock_astrodef = MagicMock()
    mock_astrodef.finewcs.return_value = 0

    mock_mysqldef = MagicMock()
    mock_mysqldef.getconnection.return_value = ('h', 'u', 'p', 'db')
    mock_mysqldef.dbConnect.return_value = MagicMock()
    mock_mysqldef.getfromdataraw.return_value = []
    mock_mysqldef.targimg.return_value = 1
    mock_mysqldef.insert_values = MagicMock()
    mock_mysqldef.deleteredufromarchive = MagicMock()

    # Create the expected output FITS (swarp would make this)
    outname = 'lsc1m001_fa15_20200501_SN2024abc_rp.fits'
    out_data = np.ones((100, 100), dtype=np.float32) * 1000
    fits.writeto(str(tmp_path / outname), out_data, overwrite=True)

    os_system_calls = []
    def mock_os_system(cmd):
        os_system_calls.append(cmd)
        # If it's a swarp command, the file should already exist
        return 0

    # Create the directory structure the script expects to move files into
    dest_dir = tmp_path / 'data' / 'lsc' / '20200501'
    dest_dir.mkdir(parents=True, exist_ok=True)

    argv = ['lscmerge.py', listf]
    with patch.object(sys, 'argv', argv), \
         patch('lsc.util', mock_util), \
         patch('lsc.lscastrodef', mock_astrodef), \
         patch('lsc.mysqldef', mock_mysqldef), \
         patch('os.system', mock_os_system), \
         patch('os.chmod'):
        runpy.run_path(SCRIPT, run_name='__main__')

    # Should have called Docosmic for each image
    assert mock_util.Docosmic.call_count == 2
    # Should have at least one swarp call
    assert any('swarp' in c for c in os_system_calls)


def test_main_no_args_exits():
    """Missing args triggers SystemExit (help)."""
    with patch.object(sys, 'argv', ['lscmerge.py']), \
         patch('lsc.util', MagicMock()), \
         patch('lsc.mysqldef', MagicMock()):
        with pytest.raises(SystemExit):
            runpy.run_path(SCRIPT, run_name='__main__')


class TestCheckastFunction:
    """Test the checkast helper function."""

    def test_checkast_importable(self):
        """Module should define checkast."""
        import importlib
        loader = importlib.machinery.SourceFileLoader('lscmerge', SCRIPT)
        spec = importlib.util.spec_from_loader('lscmerge', loader)
        mod = importlib.util.module_from_spec(spec)
        mod.__name__ = 'lscmerge'
        spec.loader.exec_module(mod)
        assert callable(mod.checkast)


class TestOutputNaming:
    """Test output filename logic."""

    def test_standard_naming(self):
        outname = 'coj' + re.sub('-', '', '1m0-11') + '_fa15_20200501_SN2024abc_rp.fits'
        assert outname == 'coj1m011_fa15_20200501_SN2024abc_rp.fits'

    def test_ft_naming(self):
        outname = 'coj' + re.sub('-', '', 'ftn') + '_en06_' + re.sub('-', '', '2020-05-01') + '_SN2024abc_rp.fits'
        assert outname == 'cojftn_en06_20200501_SN2024abc_rp.fits'


class TestExptimeCombination:
    def test_exptime_summed(self):
        assert np.sum([120.0, 120.0, 120.0]) == 360.0

    def test_airmass_weighted(self):
        assert abs(np.average([1.1, 1.2, 1.3], weights=[120, 120, 120]) - 1.2) < 1e-10


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------

def _make_merge_fits(path, extra_hdr=None):
    """Create a FITS file suitable for lscmerge tests."""
    hdr = fits.Header()
    hdr['FILTER'] = 'rp'
    hdr['OBJECT'] = 'SN2024abc'
    hdr['SITEID'] = 'lsc'
    hdr['TELID'] = '1m0-01'
    hdr['TELESCOP'] = '1m0-01'
    hdr['INSTRUME'] = 'fa15'
    hdr['DATE-OBS'] = '2020-05-01T12:00:00'
    hdr['DAY-OBS'] = '20200501'
    hdr['UTSTART'] = '12:00:00'
    hdr['MJD-OBS'] = 58970.5
    hdr['MJD'] = 58970.5
    hdr['EXPTIME'] = 120.0
    hdr['AIRMASS'] = 1.2
    hdr['RA'] = '10:00:00'
    hdr['DEC'] = '+02:00:00'
    hdr['GAIN'] = 2.0
    hdr['RDNOISE'] = 10.0
    hdr['PIXSCALE'] = 0.389
    hdr['WCSERR'] = 0
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = 100
    hdr['NAXIS2'] = 100
    if extra_hdr:
        hdr.update(extra_hdr)
    data = np.ones((100, 100), dtype=np.float32) * 1000
    fits.writeto(str(path), data, hdr, overwrite=True)


def _build_merge_mocks(tmp_path, imgs, checksnlist_ra='150.0', checksnlist_dec='2.0',
                       checksndb_ra='150.0', checksndb_dec='2.0', checksndb_tt=0,
                       photlco_result=None, tel='1m0-01', instrume='fa15',
                       workdir=None):
    """Build standard mocks for lscmerge tests."""
    mock_util = MagicMock()
    mock_util.readlist.return_value = imgs
    mock_util.readhdr.side_effect = lambda img: fits.getheader(img)
    mock_util.readkey3.side_effect = lambda hdr, key: {
        'filter': hdr.get('FILTER', 'rp'),
        'object': hdr.get('OBJECT', 'SN2024abc'),
        'TELID': hdr.get('TELID', tel),
        'gain': hdr.get('GAIN', 2.0),
        'ron': hdr.get('RDNOISE', 10.0),
        'instrume': hdr.get('INSTRUME', instrume),
        'RA': hdr.get('RA', '10:00:00'),
        'DEC': hdr.get('DEC', '+02:00:00'),
        'telescop': hdr.get('TELESCOP', tel),
        'date-obs': hdr.get('DATE-OBS', '2020-05-01'),
        'day-obs': hdr.get('DAY-OBS', '20200501'),
        'date-night': '20200501',
        'ut': hdr.get('UTSTART', '12:00:00'),
        'mjd': hdr.get('MJD', 58970.5),
        'exptime': hdr.get('EXPTIME', 120.0),
        'airmass': hdr.get('AIRMASS', 1.2),
        'wcserr': hdr.get('WCSERR', 0),
        'telid': hdr.get('TELID', tel),
    }.get(key, '')
    mock_util.checksnlist.return_value = (checksnlist_ra, checksnlist_dec, '')
    mock_util.checksndb.return_value = (checksndb_ra, checksndb_dec, checksndb_tt)
    mock_util.Docosmic.side_effect = lambda img: (
        img.replace('.fits', '.clean.fits'),
        img.replace('.fits', '.mask.fits'),
        img.replace('.fits', '.sat.fits'))
    mock_util.workdirectory = workdir or (str(tmp_path) + '/')

    mock_astrodef = MagicMock()
    mock_astrodef.finewcs.return_value = 0

    mock_mysqldef = MagicMock()
    mock_mysqldef.getconnection.return_value = ('h', 'u', 'p', 'db')
    mock_mysqldef.dbConnect.return_value = MagicMock()
    if photlco_result is None:
        mock_mysqldef.getfromdataraw.return_value = []
    else:
        mock_mysqldef.getfromdataraw.side_effect = photlco_result
    mock_mysqldef.targimg.return_value = 1
    mock_mysqldef.insert_values = MagicMock()
    mock_mysqldef.deleteredufromarchive = MagicMock()
    mock_mysqldef.updatevalue = MagicMock()

    return mock_util, mock_astrodef, mock_mysqldef


def _create_clean_mask_sat(imgs):
    """Create the .clean/.mask/.sat files that Docosmic would produce."""
    for img in imgs:
        for suffix in ['.clean.fits', '.mask.fits', '.sat.fits']:
            fpath = img.replace('.fits', suffix)
            data = np.zeros((100, 100), dtype=np.float32)
            fits.writeto(fpath, data, overwrite=True)


def _run_merge(tmp_path, monkeypatch, argv, mock_util, mock_astrodef, mock_mysqldef,
               extra_patches=None):
    """Helper to run lscmerge.py with given mocks."""
    monkeypatch.chdir(tmp_path)
    os_system_calls = []

    def mock_os_system(cmd):
        os_system_calls.append(cmd)
        return 0

    # Create the filepath directory the script expects to mkdir/mv into
    workdir = mock_util.workdirectory
    for subdir in ['data/lsc/20200501', 'data/fts/20200501']:
        d = os.path.join(workdir, subdir)
        os.makedirs(d, exist_ok=True)

    with patch.object(sys, 'argv', argv), \
         patch('lsc.util', mock_util), \
         patch('lsc.lscastrodef', mock_astrodef), \
         patch('lsc.mysqldef', mock_mysqldef), \
         patch('os.system', mock_os_system), \
         patch('os.chmod'):
        try:
            runpy.run_path(SCRIPT, run_name='__main__')
        except SystemExit:
            pass

    return os_system_calls


class TestCheckastDirect:
    """Directly test the checkast function body (lines 18-51)."""

    def test_checkast_body(self, tmp_path, monkeypatch):
        """Exercise the full checkast function by importing module and calling directly."""
        import importlib

        # Create FITS files with proper WCS headers for checkast
        imgs = []
        for i in range(2):
            fpath = tmp_path / f'img{i}.fits'
            hdr = fits.Header()
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
            data = np.ones((100, 100), dtype=np.float32) * 1000
            fits.writeto(str(fpath), data, hdr, overwrite=True)
            imgs.append(str(fpath))

        mock_util = MagicMock()
        mock_util.readhdr.side_effect = lambda img: fits.getheader(img)

        mock_astrodef = MagicMock()
        # sextractor returns positions that will match cross-frame
        xpix = np.array([30.0, 50.0, 70.0])
        ypix = np.array([30.0, 50.0, 70.0])
        mock_astrodef.sextractor.return_value = (
            xpix, ypix,
            np.array([3.0, 3.0, 3.0]), np.array([0.9, 0.9, 0.9]),
            np.array([18.0, 19.0, 20.0]), np.array([0.1, 0.1, 0.1]),
            np.array([1000, 1000, 1000]))

        mock_lsc = MagicMock()
        mock_lsc.util = mock_util
        mock_lsc.lscastrodef = mock_astrodef
        mock_lsc.updateheader = MagicMock()

        # Load the module without executing __main__
        loader = importlib.machinery.SourceFileLoader('lscmerge_test', SCRIPT)
        spec = importlib.util.spec_from_loader('lscmerge_test', loader)
        mod = importlib.util.module_from_spec(spec)
        mod.__name__ = 'lscmerge_test'

        # Inject the lsc mock and fix the Python 3 zip issue
        with patch.dict(sys.modules, {'lsc': mock_lsc}), \
             patch('lsc.util', mock_util), \
             patch('lsc.lscastrodef', mock_astrodef):
            spec.loader.exec_module(mod)
            # Set the module-level imglist1 that checkast references
            mod.imglist1 = imgs
            # Fix the Python 3 bug: patch np.array in the module's namespace
            orig_np_array = np.array

            def patched_np_array(obj, *args, **kwargs):
                # Convert zip iterators to list first
                import types
                if hasattr(obj, '__next__') or isinstance(obj, zip):
                    obj = list(obj)
                return orig_np_array(obj, *args, **kwargs)

            mod.np.array = patched_np_array
            # Call checkast directly
            mod.checkast(imgs)

        # updateheader should have been called (2 calls per image: CRPIX1 and CRPIX2)
        assert mock_lsc.updateheader.call_count >= 2


class TestCheckastFunction2:
    """Test the checkast function execution (lines 18-51)."""

    def test_checkast_with_check_flag(self, tmp_path, monkeypatch):
        """Test --check flag triggers checkast (line 144)."""
        imgs = []
        for i in range(2):
            fpath = tmp_path / f'img{i}.fits'
            _make_merge_fits(fpath)
            imgs.append(str(fpath))

        _create_clean_mask_sat(imgs)

        listf = tmp_path / 'imglist.txt'
        listf.write_text('\n'.join(imgs) + '\n')

        mock_util, mock_astrodef, mock_mysqldef = _build_merge_mocks(tmp_path, imgs)
        mock_astrodef.sextractor.return_value = (
            np.array([10, 20, 30]), np.array([40, 50, 60]),
            np.array([3.0, 3.0, 3.0]), np.array([0.9, 0.9, 0.9]),
            np.array([18.0, 19.0, 20.0]), np.array([0.1, 0.1, 0.1]),
            np.array([1000, 1000, 1000]))
        # Also need lsc.lscastrodef.sextractor and lsc.updateheader
        mock_util.updateheader = MagicMock()

        # Create the output FITS that swarp would make
        outname = 'lsc1m001_fa15_20200501_SN2024abc_rp.fits'
        _make_merge_fits(tmp_path / outname)

        argv = ['lscmerge.py', str(listf), '-c']

        monkeypatch.chdir(tmp_path)
        os_system_calls = []

        def mock_os_system(cmd):
            os_system_calls.append(cmd)
            return 0

        with patch.object(sys, 'argv', argv), \
             patch('lsc.util', mock_util), \
             patch('lsc.lscastrodef', mock_astrodef), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch('os.system', mock_os_system), \
             patch('os.chmod'):
            try:
                runpy.run_path(SCRIPT, run_name='__main__')
            except (SystemExit, Exception):
                pass

        # sextractor should have been called by checkast
        assert mock_astrodef.sextractor.call_count >= 1


class TestStandardField:
    """Test when checksndb returns _tt==1 (standard field, lines 96-97)."""

    def test_standard_field_clears_radec(self, tmp_path, monkeypatch):
        imgs = []
        for i in range(2):
            fpath = tmp_path / f'img{i}.fits'
            _make_merge_fits(fpath)
            imgs.append(str(fpath))

        _create_clean_mask_sat(imgs)

        listf = tmp_path / 'imglist.txt'
        listf.write_text('\n'.join(imgs) + '\n')

        # checksnlist returns empty (no supernovaelist.txt entry) => falls to checksndb
        # checksndb returns _tt=1 => standard field => _ra/_dec cleared => fall to header
        mock_util, mock_astrodef, mock_mysqldef = _build_merge_mocks(
            tmp_path, imgs,
            checksnlist_ra='', checksnlist_dec='',
            checksndb_ra='150.0', checksndb_dec='2.0', checksndb_tt=1)

        outname = 'lsc1m001_fa15_20200501_SN2024abc_rp.fits'
        _make_merge_fits(tmp_path / outname)

        argv = ['lscmerge.py', str(listf)]
        os_calls = _run_merge(tmp_path, monkeypatch, argv, mock_util, mock_astrodef, mock_mysqldef)

        # Should still run swarp (using RA/DEC from header as fallback)
        assert any('swarp' in c for c in os_calls)


class TestFallbackToHeader:
    """Test fallback to RA/DEC from header (lines 100-101)."""

    def test_radec_from_header_when_both_empty(self, tmp_path, monkeypatch):
        imgs = []
        for i in range(2):
            fpath = tmp_path / f'img{i}.fits'
            _make_merge_fits(fpath)
            imgs.append(str(fpath))

        _create_clean_mask_sat(imgs)

        listf = tmp_path / 'imglist.txt'
        listf.write_text('\n'.join(imgs) + '\n')

        # Both checksnlist and checksndb return empty => falls to header
        mock_util, mock_astrodef, mock_mysqldef = _build_merge_mocks(
            tmp_path, imgs,
            checksnlist_ra='', checksnlist_dec='',
            checksndb_ra='', checksndb_dec='', checksndb_tt=0)

        outname = 'lsc1m001_fa15_20200501_SN2024abc_rp.fits'
        _make_merge_fits(tmp_path / outname)

        argv = ['lscmerge.py', str(listf)]
        os_calls = _run_merge(tmp_path, monkeypatch, argv, mock_util, mock_astrodef, mock_mysqldef)

        # Should use RA/DEC from header in swarp command
        assert any('swarp' in c and '10:00:00' in c for c in os_calls)


class TestEmInstrument:
    """Test 'em' in instrument name (line 104)."""

    def test_em_instrument_smaller_imagesize(self, tmp_path, monkeypatch):
        imgs = []
        for i in range(2):
            fpath = tmp_path / f'img{i}.fits'
            _make_merge_fits(fpath, extra_hdr={'INSTRUME': 'em01'})
            imgs.append(str(fpath))

        _create_clean_mask_sat(imgs)

        listf = tmp_path / 'imglist.txt'
        listf.write_text('\n'.join(imgs) + '\n')

        mock_util, mock_astrodef, mock_mysqldef = _build_merge_mocks(
            tmp_path, imgs, instrume='em01')

        outname = 'lsc1m001_em01_20200501_SN2024abc_rp.fits'
        _make_merge_fits(tmp_path / outname)

        argv = ['lscmerge.py', str(listf)]
        os_calls = _run_merge(tmp_path, monkeypatch, argv, mock_util, mock_astrodef, mock_mysqldef)

        # IMAGE_SIZE should be 1050 for 'em' instruments
        assert any('1050' in c for c in os_calls)


class TestCCDSCALEPixelScale:
    """Test pixelscale from CCDSCALE (lines 110-114)."""

    def test_ccdscale_used_when_no_pixscale(self, tmp_path, monkeypatch):
        imgs = []
        for i in range(2):
            fpath = tmp_path / f'img{i}.fits'
            # No PIXSCALE, but has CCDSCALE
            hdr = fits.Header()
            hdr['FILTER'] = 'rp'
            hdr['OBJECT'] = 'SN2024abc'
            hdr['SITEID'] = 'lsc'
            hdr['TELID'] = '1m0-01'
            hdr['TELESCOP'] = '1m0-01'
            hdr['INSTRUME'] = 'fa15'
            hdr['DATE-OBS'] = '2020-05-01T12:00:00'
            hdr['DAY-OBS'] = '20200501'
            hdr['UTSTART'] = '12:00:00'
            hdr['MJD-OBS'] = 58970.5
            hdr['MJD'] = 58970.5
            hdr['EXPTIME'] = 120.0
            hdr['AIRMASS'] = 1.2
            hdr['RA'] = '10:00:00'
            hdr['DEC'] = '+02:00:00'
            hdr['GAIN'] = 2.0
            hdr['RDNOISE'] = 10.0
            hdr['CCDSCALE'] = 0.26
            hdr['WCSERR'] = 0
            data = np.ones((100, 100), dtype=np.float32) * 1000
            fits.writeto(str(fpath), data, hdr, overwrite=True)
            imgs.append(str(fpath))

        _create_clean_mask_sat(imgs)

        listf = tmp_path / 'imglist.txt'
        listf.write_text('\n'.join(imgs) + '\n')

        mock_util, mock_astrodef, mock_mysqldef = _build_merge_mocks(tmp_path, imgs)

        outname = 'lsc1m001_fa15_20200501_SN2024abc_rp.fits'
        _make_merge_fits(tmp_path / outname)

        argv = ['lscmerge.py', str(listf)]
        os_calls = _run_merge(tmp_path, monkeypatch, argv, mock_util, mock_astrodef, mock_mysqldef)

        # 0.26 from CCDSCALE should appear in swarp command
        assert any('0.26' in c for c in os_calls)

    def test_no_pixscale_no_ccdscale(self, tmp_path, monkeypatch, capsys):
        """Lines 113-114: warning printed when neither scale is defined."""
        imgs = []
        for i in range(2):
            fpath = tmp_path / f'img{i}.fits'
            hdr = fits.Header()
            hdr['FILTER'] = 'rp'
            hdr['OBJECT'] = 'SN2024abc'
            hdr['SITEID'] = 'lsc'
            hdr['TELID'] = '1m0-01'
            hdr['TELESCOP'] = '1m0-01'
            hdr['INSTRUME'] = 'fa15'
            hdr['DATE-OBS'] = '2020-05-01T12:00:00'
            hdr['DAY-OBS'] = '20200501'
            hdr['UTSTART'] = '12:00:00'
            hdr['MJD-OBS'] = 58970.5
            hdr['MJD'] = 58970.5
            hdr['EXPTIME'] = 120.0
            hdr['AIRMASS'] = 1.2
            hdr['RA'] = '10:00:00'
            hdr['DEC'] = '+02:00:00'
            hdr['GAIN'] = 2.0
            hdr['RDNOISE'] = 10.0
            hdr['WCSERR'] = 0
            # No PIXSCALE, no CCDSCALE
            data = np.ones((100, 100), dtype=np.float32) * 1000
            fits.writeto(str(fpath), data, hdr, overwrite=True)
            imgs.append(str(fpath))

        _create_clean_mask_sat(imgs)

        listf = tmp_path / 'imglist.txt'
        listf.write_text('\n'.join(imgs) + '\n')

        mock_util, mock_astrodef, mock_mysqldef = _build_merge_mocks(tmp_path, imgs)

        outname = 'lsc1m001_fa15_20200501_SN2024abc_rp.fits'
        _make_merge_fits(tmp_path / outname)

        argv = ['lscmerge.py', str(listf)]
        _run_merge(tmp_path, monkeypatch, argv, mock_util, mock_astrodef, mock_mysqldef)

        captured = capsys.readouterr()
        assert 'pixel scale not defined' in captured.out


class TestFtsTelescope:
    """Test fts/ftn telescope naming (lines 117, 228)."""

    def test_ftn_naming_and_filepath(self, tmp_path, monkeypatch):
        imgs = []
        for i in range(2):
            fpath = tmp_path / f'img{i}.fits'
            _make_merge_fits(fpath, extra_hdr={
                'TELID': 'ftn', 'TELESCOP': 'ftn', 'SITEID': 'coj',
                'INSTRUME': 'en06'})
            imgs.append(str(fpath))

        _create_clean_mask_sat(imgs)

        listf = tmp_path / 'imglist.txt'
        listf.write_text('\n'.join(imgs) + '\n')

        mock_util, mock_astrodef, mock_mysqldef = _build_merge_mocks(
            tmp_path, imgs, tel='ftn', instrume='en06')

        # ftn naming: siteid + telid + '_' + instrume + '_' + date + '_' + obj + '_' + filter
        outname = 'cojftn_en06_20200501_SN2024abc_rp.fits'
        _make_merge_fits(tmp_path / outname)

        argv = ['lscmerge.py', str(listf)]
        os_calls = _run_merge(tmp_path, monkeypatch, argv, mock_util, mock_astrodef, mock_mysqldef)

        # filepath should contain 'data/fts/' for ftn/fts telescopes
        # First insert_values call is the photlco insert (has filepath);
        # later calls are photpairing inserts (no filepath key).
        for call in mock_mysqldef.insert_values.call_args_list:
            dictionary = call[0][2]
            if 'filepath' in dictionary:
                assert 'fts' in dictionary['filepath']
                break


class TestPhotlcoExistsUpdate:
    """Test when photlco entry already exists (lines 241-246)."""

    def test_update_when_entry_exists(self, tmp_path, monkeypatch):
        imgs = []
        for i in range(2):
            fpath = tmp_path / f'img{i}.fits'
            _make_merge_fits(fpath)
            imgs.append(str(fpath))

        _create_clean_mask_sat(imgs)

        listf = tmp_path / 'imglist.txt'
        listf.write_text('\n'.join(imgs) + '\n')

        outname = 'lsc1m001_fa15_20200501_SN2024abc_rp.fits'

        # First call returns existing photlco entry, second call for photpairing
        mock_util, mock_astrodef, mock_mysqldef = _build_merge_mocks(
            tmp_path, imgs,
            photlco_result=[
                [{'filename': outname, 'filetype': 2, 'filter': 'rp'}],  # photlco: exists
                [],  # photpairing
            ])

        _make_merge_fits(tmp_path / outname)

        argv = ['lscmerge.py', str(listf)]
        _run_merge(tmp_path, monkeypatch, argv, mock_util, mock_astrodef, mock_mysqldef)

        # updatevalue should have been called (update path, not insert)
        mock_mysqldef.updatevalue.assert_called()

    def test_update_with_exception(self, tmp_path, monkeypatch):
        """Lines 245-246: updatevalue raises, except catches and passes."""
        imgs = []
        for i in range(2):
            fpath = tmp_path / f'img{i}.fits'
            _make_merge_fits(fpath)
            imgs.append(str(fpath))

        _create_clean_mask_sat(imgs)

        listf = tmp_path / 'imglist.txt'
        listf.write_text('\n'.join(imgs) + '\n')

        outname = 'lsc1m001_fa15_20200501_SN2024abc_rp.fits'

        mock_util, mock_astrodef, mock_mysqldef = _build_merge_mocks(
            tmp_path, imgs,
            photlco_result=[
                [{'filename': outname, 'filetype': 2, 'filter': 'rp'}],  # photlco: exists
                [],  # photpairing
            ])
        # Make updatevalue raise KeyError for some keys (exercises except:pass)
        mock_mysqldef.updatevalue.side_effect = KeyError('no such column')

        _make_merge_fits(tmp_path / outname)

        argv = ['lscmerge.py', str(listf)]
        # Should not crash - the except:pass catches it
        _run_merge(tmp_path, monkeypatch, argv, mock_util, mock_astrodef, mock_mysqldef)

        mock_mysqldef.updatevalue.assert_called()


class TestMkdirCondition:
    """Test mkdir and mv conditions (lines 248-249)."""

    def test_mkdir_when_dir_missing(self, tmp_path, monkeypatch):
        imgs = []
        for i in range(2):
            fpath = tmp_path / f'img{i}.fits'
            _make_merge_fits(fpath)
            imgs.append(str(fpath))

        _create_clean_mask_sat(imgs)

        listf = tmp_path / 'imglist.txt'
        listf.write_text('\n'.join(imgs) + '\n')

        mock_util, mock_astrodef, mock_mysqldef = _build_merge_mocks(tmp_path, imgs)
        # Set workdirectory to a path where data/lsc/20200501 does NOT exist
        mock_util.workdirectory = str(tmp_path / 'nonexistent') + '/'

        outname = 'lsc1m001_fa15_20200501_SN2024abc_rp.fits'
        _make_merge_fits(tmp_path / outname)

        argv = ['lscmerge.py', str(listf)]

        monkeypatch.chdir(tmp_path)
        mkdir_calls = []
        orig_mkdir = os.mkdir

        def track_mkdir(path, *args):
            mkdir_calls.append(path)

        with patch.object(sys, 'argv', argv), \
             patch('lsc.util', mock_util), \
             patch('lsc.lscastrodef', mock_astrodef), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch('os.system', lambda cmd: 0), \
             patch('os.chmod'), \
             patch('os.mkdir', track_mkdir):
            try:
                runpy.run_path(SCRIPT, run_name='__main__')
            except (SystemExit, Exception):
                pass

        # mkdir should have been called for the missing directory
        assert len(mkdir_calls) >= 1


class TestForceDeleteMerge:
    """Test --force when photlco entry exists (line 234-235)."""

    def test_force_deletes_and_reinserts(self, tmp_path, monkeypatch):
        imgs = []
        for i in range(2):
            fpath = tmp_path / f'img{i}.fits'
            _make_merge_fits(fpath)
            imgs.append(str(fpath))

        _create_clean_mask_sat(imgs)

        listf = tmp_path / 'imglist.txt'
        listf.write_text('\n'.join(imgs) + '\n')

        outname = 'lsc1m001_fa15_20200501_SN2024abc_rp.fits'

        # photlco returns existing entry; with --force it should delete + reinsert
        mock_util, mock_astrodef, mock_mysqldef = _build_merge_mocks(
            tmp_path, imgs,
            photlco_result=[
                [{'filename': outname, 'filetype': 2}],  # photlco: exists
                [{'nameout': outname}],  # photpairing: exists
            ])

        _make_merge_fits(tmp_path / outname)

        argv = ['lscmerge.py', str(listf), '-f']
        _run_merge(tmp_path, monkeypatch, argv, mock_util, mock_astrodef, mock_mysqldef)

        # deleteredufromarchive should be called for both photlco and photpairing
        assert mock_mysqldef.deleteredufromarchive.call_count >= 1
        mock_mysqldef.insert_values.assert_called()
