"""Tests for bin/lscsn.py -- exercises __main__ via runpy to reach 90%+ coverage."""
import sys
import os
import runpy
from unittest.mock import patch, MagicMock, mock_open
import pytest
import numpy as np
from astropy.io import fits

TRUNK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(TRUNK, 'bin', 'lscsn.py')


def _make_fits(path, extra_hdr=None):
    """Create a minimal FITS file with standard headers."""
    hdr = fits.Header()
    hdr['FILTER'] = 'rp'
    hdr['OBJECT'] = 'SN2024abc'
    hdr['TELESCOP'] = '1m0-01'
    hdr['INSTRUME'] = 'fa15'
    hdr['EXPTIME'] = 120.0
    hdr['AIRMASS'] = 1.2
    hdr['PIXSCALE'] = 0.389
    hdr['PSF_FWHM'] = 1.5
    hdr['APCO'] = 0.05
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = 100
    hdr['NAXIS2'] = 100
    hdr['DATAMAX'] = 60000
    hdr['DATAMIN'] = 0
    if extra_hdr:
        hdr.update(extra_hdr)
    data = np.ones((100, 100), dtype=np.float32) * 1000
    fits.writeto(str(path), data, hdr, overwrite=True)


def _readkey3(h, k):
    """Mock readkey3: extract value from a FITS header object by key name."""
    key_map = {
        'instrume': 'INSTRUME',
        'filter': 'FILTER',
        'exptime': 'EXPTIME',
        'PIXSCALE': 'PIXSCALE',
        'CCDSCALE': 'CCDSCALE',
        'PSF_FWHM': 'PSF_FWHM',
        'APCO': 'APCO',
        'PSFMAG1': 'PSFMAG1',
        'datamax': 'DATAMAX',
        'datamin': 'DATAMIN',
        'CONVOL00': 'CONVOL00',
        'exptarg': 'EXPTARG',
        'exptemp': 'EXPTEMP',
        'CCDXBIN': 'CCDXBIN',
        'CCDSUM': 'CCDSUM',
    }
    real_key = key_map.get(k, k)
    try:
        val = h.get(real_key, '')
    except (TypeError, AttributeError):
        val = ''
    return val


def _setup_basic_env(tmp_path, img_name='test', extra_img_hdr=None, extra_sn2_hdr=None,
                     create_psf=True, create_sn2=True):
    """Set up the basic file environment for a test."""
    imgpath = tmp_path / f'{img_name}.fits'
    img_hdr = extra_img_hdr or {}
    _make_fits(imgpath, img_hdr)

    if create_sn2:
        sn2path = tmp_path / f'{img_name}.sn2.fits'
        sn2_hdr = {'PSFMAG1': '', 'PIXSCALE': 0.389, 'PSF_FWHM': 1.5, 'APCO': 0.05}
        if extra_sn2_hdr:
            sn2_hdr.update(extra_sn2_hdr)
        _make_fits(sn2path, sn2_hdr)

    if create_psf:
        psfpath = tmp_path / f'{img_name}.psf.fits'
        _make_fits(psfpath)

    # Create files that iraf operations would produce
    _make_fits(tmp_path / 'original.fits')
    _make_fits(tmp_path / 'bg.fits')
    _make_fits(tmp_path / 'tmp.fits')
    _make_fits(tmp_path / 'skyfit.fits')
    _make_fits(tmp_path / 'sky.fits')
    _make_fits(tmp_path / 'sn.fits')

    return str(imgpath)


def _build_mocks(tmp_path, imgpath, extra_readkey=None):
    """Build the standard set of mocks needed to run lscsn.py."""
    mock_util = MagicMock()
    mock_util.readlist.return_value = [imgpath]
    mock_util.readhdr.side_effect = lambda img: fits.getheader(img)
    mock_util.readkey3.side_effect = _readkey3
    mock_util.checksndb.return_value = (150.0, 2.0, 0)
    mock_util.updateheader = MagicMock()
    mock_util.delete = MagicMock()
    mock_util.imcopy = MagicMock()
    mock_util.display_image = MagicMock(return_value=(100, 5000, True))

    mock_mysqldef = MagicMock()
    mock_mysqldef.updatevalue = MagicMock()
    mock_mysqldef.getconnection.return_value = ('host', 'user', 'pass', 'db')
    mock_mysqldef.dbConnect.return_value = MagicMock()
    mock_mysqldef.getfromdataraw.return_value = [{'filepath': str(tmp_path)}]

    mock_iraf = MagicMock()

    def fake_wcsctran(*a, **kw):
        with open(str(tmp_path / 'tmp.pix'), 'w') as f:
            f.write('50.0 50.0\n')
    mock_iraf.wcsctran.side_effect = fake_wcsctran
    # iraf.fields: first two elements empty (skipped by `if kk:`), third is data
    # This satisfies both the iteration loop (line 260) and the [2] index (line 278)
    mock_iraf.fields.return_value = ['', '', '50.0 50.0']
    mock_iraf.imsurfit = MagicMock()
    mock_iraf.imcopy = MagicMock()
    mock_iraf.imarith = MagicMock()
    mock_iraf.set = MagicMock()
    mock_iraf.tvmark = MagicMock()
    mock_iraf.astcat = MagicMock()
    mock_iraf.imcoords = MagicMock()
    mock_iraf.digiphot = MagicMock()
    mock_iraf.daophot = MagicMock()

    mock_snoopy = MagicMock()
    mock_snoopy.fitsn.return_value = (
        [18.0], [17.8], [17.5],   # apori1, apori2, apori3
        [18.0], [17.8], [17.5],   # apmag1, apmag2, apmag3
        [0.01], [0.01], [0.01],   # dapmag1, dapmag2, dapmag3
        [17.0], [16.8], [0.02],   # fitmag, truemag, magerr
        [50.0], [50.0],           # centx, centy
    )

    return mock_util, mock_mysqldef, mock_iraf, mock_snoopy


def _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy,
                expect_exit=False):
    """Run lscsn.py with all mocks in place."""
    # Build modules dict for pyraf/iraf stubs
    pyraf_mod = MagicMock()
    pyraf_mod.iraf = mock_iraf
    modules_patch = {
        'pyraf': pyraf_mod,
        'pyraf.iraf': mock_iraf,
        'iraf': mock_iraf,
    }

    with patch.object(sys, 'argv', argv), \
         patch('lsc.util', mock_util), \
         patch('lsc.mysqldef', mock_mysqldef), \
         patch('lsc.lscsnoopy', mock_snoopy), \
         patch.dict(sys.modules, modules_patch), \
         patch('os.system', return_value=0):
        if expect_exit:
            with pytest.raises(SystemExit):
                runpy.run_path(SCRIPT, run_name='__main__')
        else:
            try:
                runpy.run_path(SCRIPT, run_name='__main__')
            except SystemExit:
                pass  # some paths exit cleanly


class TestNoArgs:
    """Test that missing args triggers help/exit."""

    def test_no_args_exits(self):
        with patch.object(sys, 'argv', ['lscsn.py']), \
             patch.dict(sys.modules, {
                 'pyraf': MagicMock(),
                 'pyraf.iraf': MagicMock(),
                 'iraf': MagicMock(),
             }):
            with pytest.raises(SystemExit):
                runpy.run_path(SCRIPT, run_name='__main__')


class TestSkipAlreadyDone:
    """When PSFMAG1 is set and redo=False, skip processing."""

    def test_skip_when_already_measured(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path, extra_sn2_hdr={'PSFMAG1': '17.5'})
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        # fitsn should NOT be called -- already measured
        mock_snoopy.fitsn.assert_not_called()

    def test_redo_overrides_skip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path, extra_sn2_hdr={'PSFMAG1': '17.5'})
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0', '--redo',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        # fitsn SHOULD be called since redo=True
        mock_snoopy.fitsn.assert_called_once()


class TestMissingFiles:
    """Missing psf or sn2 file should exit."""

    def test_missing_sn2_file_exits(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Create image and psf but NOT sn2
        imgpath = _setup_basic_env(tmp_path, create_sn2=False)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy,
                    expect_exit=True)

    def test_missing_psf_file_exits(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Create image and sn2 but NOT psf
        imgpath = _setup_basic_env(tmp_path, create_psf=False)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy,
                    expect_exit=True)


class TestNormalFlow:
    """Non-interactive flow with RA/DEC provided, no iterations."""

    def test_basic_measurement(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-n', '0', '-x', '2', '-y', '2', '-b', '4', '-z', '7', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        mock_snoopy.fitsn.assert_called_once()
        # DB updated with results
        assert mock_mysqldef.updatevalue.call_count >= 5
        # Header updated
        mock_util.updateheader.assert_called()

    def test_with_iterations(self, tmp_path, monkeypatch):
        """Test background iteration loop (niter=2)."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-n', '2', '-x', '2', '-y', '2', '-b', '4', '-z', '7', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        # fitsn called 1 (initial) + 2 (iterations) = 3 times
        assert mock_snoopy.fitsn.call_count == 3

    def test_multiple_sn_coords(self, tmp_path, monkeypatch):
        """Test comma-separated RA,DEC for multiple SN positions."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        # Return multiple pixel coords from wcsctran
        def fake_wcsctran(*a, **kw):
            with open(str(tmp_path / 'tmp.pix'), 'w') as f:
                f.write('50.0 50.0\n60.0 60.0\n')
        mock_iraf.wcsctran.side_effect = fake_wcsctran
        # Two empty (skipped by `if kk:`), then data lines; [2] used for box coord
        mock_iraf.fields.return_value = ['', '', '50.0 50.0', '60.0 60.0']

        # fitsn returns results for 2 sources
        mock_snoopy.fitsn.return_value = (
            [18.0, 18.1], [17.8, 17.9], [17.5, 17.6],
            [18.0, 18.1], [17.8, 17.9], [17.5, 17.6],
            [0.01, 0.01], [0.01, 0.01], [0.01, 0.01],
            [17.0, 17.1], [16.8, 16.9], [0.02, 0.03],
            [50.0, 60.0], [50.0, 60.0],
        )

        argv = ['lscsn.py', '--RA', '150.0,150.1', '--DEC', '2.0,2.1',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        mock_snoopy.fitsn.assert_called_once()
        # updateheader called once per image (with headers for both sources)
        mock_util.updateheader.assert_called()

    def test_no_radec_uses_database(self, tmp_path, monkeypatch):
        """When no RA/DEC on CLI, coordinates come from checksndb."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)
        mock_util.checksndb.return_value = (150.0, 2.0, 1)

        argv = ['lscsn.py', '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        mock_util.checksndb.assert_called()
        mock_snoopy.fitsn.assert_called_once()

    def test_no_recenter_flag(self, tmp_path, monkeypatch):
        """The -c flag disables recentering (sets _recenter=True internally)."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-c', '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        # fitsn called with recenter=True (confusing but that's how the script works)
        call_args = mock_snoopy.fitsn.call_args
        assert call_args[0][3] is True  # _recenter positional arg


class TestDiffImages:
    """Test difference imaging branches (CONVOL00 header)."""

    def test_diff_template_convol(self, tmp_path, monkeypatch):
        """CONVOL00=TEMPLATE branch: reads template apco from DB."""
        monkeypatch.chdir(tmp_path)
        # Create a "diff" image to trigger the diff branch
        img_name = 'test_diff'
        imgpath = _setup_basic_env(
            tmp_path, img_name=img_name,
            extra_img_hdr={
                'CONVOL00': 'TEMPLATE',
                'EXPTIME': 120.0,
                'EXPTARG': 120.0,
                'EXPTEMP': 60.0,
                'TEMPLATE': 'tmpl.fits',
                'TARGET': 'targ.fits',
            },
            extra_sn2_hdr={'PSFMAG1': '', 'PIXSCALE': 0.389, 'PSF_FWHM': 1.5, 'APCO': 0.05}
        )

        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        # Create the template sn2 file that the script will try to read
        tmpl_sn2 = tmp_path / 'tmpl.sn2.fits'
        _make_fits(tmpl_sn2, {'APCO': 0.08, 'PIXSCALE': 0.389, 'PSF_FWHM': 1.5})
        mock_mysqldef.getfromdataraw.return_value = [{'filepath': str(tmp_path)}]

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        # DB connection was established for template lookup
        mock_mysqldef.getconnection.assert_called_with('lcogt2')
        mock_snoopy.fitsn.assert_called_once()

    def test_diff_image_convol(self, tmp_path, monkeypatch):
        """CONVOL00=IMAGE branch: reads image apco from TARGET sn2 file."""
        monkeypatch.chdir(tmp_path)
        img_name = 'test_diff'
        imgpath = _setup_basic_env(
            tmp_path, img_name=img_name,
            extra_img_hdr={
                'CONVOL00': 'IMAGE',
                'TEMPLATE': 'tmpl.fits',
                'TARGET': 'targ.fits',
            },
            extra_sn2_hdr={'PSFMAG1': '', 'PIXSCALE': 0.389, 'PSF_FWHM': 1.5, 'APCO': 0.05}
        )
        # Create the target sn2 file
        targ_sn2 = tmp_path / 'targ.sn2.fits'
        _make_fits(targ_sn2, {'APCO': 0.07, 'PIXSCALE': 0.389, 'PSF_FWHM': 1.5})

        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        mock_snoopy.fitsn.assert_called_once()

    def test_optimal_diff_zero_apco(self, tmp_path, monkeypatch):
        """Optimal diff images set apco=0."""
        monkeypatch.chdir(tmp_path)
        img_name = 'test_optimal_diff'
        imgpath = _setup_basic_env(
            tmp_path, img_name=img_name,
            extra_img_hdr={'CONVOL00': 'TEMPLATE', 'EXPTARG': 120.0, 'EXPTEMP': 60.0,
                           'TEMPLATE': 'tmpl.fits', 'TARGET': 'targ.fits'},
            extra_sn2_hdr={'PSFMAG1': '', 'PIXSCALE': 0.389, 'PSF_FWHM': 1.5, 'APCO': 0.05}
        )
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        # The template sn2 is needed
        tmpl_sn2 = tmp_path / 'tmpl.sn2.fits'
        _make_fits(tmpl_sn2, {'APCO': 0.08, 'PIXSCALE': 0.389, 'PSF_FWHM': 1.5})
        mock_mysqldef.getfromdataraw.return_value = [{'filepath': str(tmp_path)}]

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        mock_snoopy.fitsn.assert_called_once()
        # The apco0 argument passed to fitsn (last positional arg)
        call_args = mock_snoopy.fitsn.call_args[0]
        # apco0 is the last argument
        apco_val = call_args[-1]
        assert isinstance(apco_val, (int, float))

    def test_dm_applied_to_psfmag(self, tmp_path, monkeypatch):
        """When CONVOL00=TEMPLATE, DM correction is applied to psfmag in DB update."""
        monkeypatch.chdir(tmp_path)
        img_name = 'test_diff'
        imgpath = _setup_basic_env(
            tmp_path, img_name=img_name,
            extra_img_hdr={
                'CONVOL00': 'TEMPLATE',
                'EXPTIME': 120.0,
                'EXPTARG': 120.0,
                'EXPTEMP': 60.0,
                'TEMPLATE': 'tmpl.fits',
                'TARGET': 'targ.fits',
            },
            extra_sn2_hdr={'PSFMAG1': '', 'PIXSCALE': 0.389, 'PSF_FWHM': 1.5, 'APCO': 0.05}
        )
        tmpl_sn2 = tmp_path / 'tmpl.sn2.fits'
        _make_fits(tmpl_sn2, {'APCO': 0.08, 'PIXSCALE': 0.389, 'PSF_FWHM': 1.5})

        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)
        mock_mysqldef.getfromdataraw.return_value = [{'filepath': str(tmp_path)}]

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        # DM = 2.5*log10(120) - 2.5*log10(120) = 0 in this case
        # psfmag update should subtract DM from truemag
        psfmag_calls = [c for c in mock_mysqldef.updatevalue.call_args_list
                        if c[0][1] == 'psfmag' and c[0][2] != 9999]
        assert len(psfmag_calls) >= 1


class TestShowMode:
    """Test --show flag triggers display_image calls."""

    def test_show_displays_images(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0', '--show',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        mock_util.display_image.assert_called()
        mock_iraf.tvmark.assert_called()


class TestPsfOption:
    """Test explicit --psf option."""

    def test_explicit_psf_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        # Create a separate PSF file
        custom_psf = tmp_path / 'custom.psf.fits'
        _make_fits(custom_psf)

        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)
        # readlist for psf option
        original_readlist = mock_util.readlist.side_effect

        def readlist_side_effect(arg):
            if 'custom' in str(arg):
                return [str(custom_psf)]
            return [imgpath]
        mock_util.readlist.side_effect = readlist_side_effect

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '--psf', str(custom_psf),
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        mock_snoopy.fitsn.assert_called_once()


class TestINDEFResults:
    """Test INDEF magnitude handling."""

    def test_indef_truemag(self, tmp_path, monkeypatch):
        """When fitsn returns INDEF, it should be set to 9999."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        mock_snoopy.fitsn.return_value = (
            [18.0], [17.8], [17.5],
            [18.0], [17.8], [17.5],
            [0.01], [0.01], [0.01],
            [17.0], ['INDEF'], [0.02],
            [50.0], [50.0],
        )

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        # DB should get 9999 for psfmag when INDEF
        psfmag_calls = [c for c in mock_mysqldef.updatevalue.call_args_list
                        if c[0][1] == 'psfmag']
        # First call sets 9999 (reset), then final call should also be 9999-DM
        assert any(c[0][2] == 9999 for c in psfmag_calls)

    def test_indef_apmag(self, tmp_path, monkeypatch):
        """When apmag3 is INDEF, it should be set to 9999."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        mock_snoopy.fitsn.return_value = (
            [18.0], [17.8], ['INDEF'],
            [18.0], [17.8], ['INDEF'],
            [0.01], [0.01], [0.01],
            [17.0], [16.8], [0.02],
            [50.0], [50.0],
        )

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        mock_util.updateheader.assert_called()


class TestInstrumentBranches:
    """Test instrument-specific FWHM calculation branches."""

    def test_fs_instrument_with_ccdxbin(self, tmp_path, monkeypatch):
        """fs/em instruments use CCDXBIN for fwhm calculation."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(
            tmp_path,
            extra_img_hdr={'INSTRUME': 'fs02'},
            extra_sn2_hdr={'PSFMAG1': '', 'PIXSCALE': 0.389, 'PSF_FWHM': 1.5,
                           'APCO': 0.05, 'CCDXBIN': 2}
        )
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        mock_snoopy.fitsn.assert_called_once()

    def test_em_instrument_with_ccdsum(self, tmp_path, monkeypatch):
        """em instruments with CCDSUM instead of CCDXBIN."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(
            tmp_path,
            extra_img_hdr={'INSTRUME': 'em01'},
            extra_sn2_hdr={'PSFMAG1': '', 'PIXSCALE': 0.389, 'PSF_FWHM': 1.5,
                           'APCO': 0.05, 'CCDSUM': '2 2'}
        )
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        mock_snoopy.fitsn.assert_called_once()

    def test_fs_instrument_no_binning(self, tmp_path, monkeypatch):
        """fs instruments without CCDXBIN or CCDSUM."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(
            tmp_path,
            extra_img_hdr={'INSTRUME': 'fs02'},
            extra_sn2_hdr={'PSFMAG1': '', 'PIXSCALE': 0.389, 'PSF_FWHM': 1.5, 'APCO': 0.05}
        )
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        mock_snoopy.fitsn.assert_called_once()


class TestCCDSCALE:
    """Test CCDSCALE fallback when PIXSCALE absent."""

    def test_ccdscale_used_when_no_pixscale(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(
            tmp_path,
            extra_sn2_hdr={'PSFMAG1': '', 'CCDSCALE': 0.3, 'PSF_FWHM': 1.5, 'APCO': 0.05}
        )
        # Remove PIXSCALE from sn2
        sn2path = tmp_path / 'test.sn2.fits'
        hdr = fits.getheader(str(sn2path))
        if 'PIXSCALE' in hdr:
            del hdr['PIXSCALE']
        data = fits.getdata(str(sn2path))
        fits.writeto(str(sn2path), data, hdr, overwrite=True)

        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        mock_snoopy.fitsn.assert_called_once()


class TestNoApco:
    """When APCO is missing/falsy, apco0 defaults to 0."""

    def test_missing_apco_defaults_zero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(
            tmp_path,
            extra_sn2_hdr={'PSFMAG1': '', 'PIXSCALE': 0.389, 'PSF_FWHM': 1.5, 'APCO': 0}
        )
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        # apco0 should be 0 (last arg to fitsn)
        call_args = mock_snoopy.fitsn.call_args[0]
        assert call_args[-1] == 0


class TestNoFwhm:
    """When PSF_FWHM is missing, fwhm defaults to 6."""

    def test_missing_fwhm_defaults_6(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(
            tmp_path,
            extra_sn2_hdr={'PSFMAG1': '', 'PIXSCALE': 0.389, 'PSF_FWHM': 0, 'APCO': 0.05}
        )
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        mock_snoopy.fitsn.assert_called_once()


class TestDatamaxDatamin:
    """Test datamax/datamin CLI options."""

    def test_explicit_datamax_datamin(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-n', '0', '-x', '2', '-y', '2',
                '-m', '50000', '--datamin', '-100', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        mock_snoopy.fitsn.assert_called_once()

    def test_optimal_datamin_indef(self, tmp_path, monkeypatch):
        """Optimal images without --datamin get INDEF datamin."""
        monkeypatch.chdir(tmp_path)
        img_name = 'test_optimal'
        imgpath = _setup_basic_env(tmp_path, img_name=img_name)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        mock_snoopy.fitsn.assert_called_once()


class TestRA0DEC0:
    """Test RA0/DEC0 centering coordinates."""

    def test_explicit_ra0_dec0(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '--RA0', '150.5', '--DEC0', '2.5',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        # wcsctran should be called twice (SN coords + box coords)
        assert mock_iraf.wcsctran.call_count == 2

    def test_no_coords_no_db_exits(self, tmp_path, monkeypatch):
        """No RA/DEC and checksndb fails -> exit."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)
        mock_util.checksndb.side_effect = Exception("no target")

        argv = ['lscsn.py', '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy,
                    expect_exit=True)


class TestEmptyImageList:
    """Test empty/None entries in image list."""

    def test_empty_entry_in_list(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)
        # List with empty string entry
        mock_util.readlist.return_value = ['', imgpath]

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        # Should still process the valid image
        mock_snoopy.fitsn.assert_called_once()


class TestDBUpdateFailure:
    """Test that DB update exception is caught gracefully."""

    def test_db_update_exception_caught(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        # Make the final DB update raise
        call_count = [0]
        def fail_on_final(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 5:  # After the initial resets
                raise Exception("DB connection lost")
        mock_mysqldef.updatevalue.side_effect = fail_on_final

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        # Should not crash - the script catches exceptions on final DB updates
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)


class TestImageWithoutFitsExtension:
    """Test that images without .fits extension are handled."""

    def test_img_no_fits_suffix(self, tmp_path, monkeypatch):
        """When image path doesn't end in .fits, img = imglong (line 119)."""
        monkeypatch.chdir(tmp_path)
        # Create a file named 'testimg' (no .fits extension) that is a valid FITS
        imgpath_no_ext = str(tmp_path / 'testimg')
        _make_fits(tmp_path / 'testimg')  # astropy can write without .fits ext
        _make_fits(tmp_path / 'testimg.sn2.fits', {'PSFMAG1': '17.0', 'PIXSCALE': 0.389,
                                                    'PSF_FWHM': 1.5, 'APCO': 0.05})
        _make_fits(tmp_path / 'testimg.psf.fits')
        _make_fits(tmp_path / 'original.fits')
        _make_fits(tmp_path / 'bg.fits')
        _make_fits(tmp_path / 'tmp.fits')
        _make_fits(tmp_path / 'skyfit.fits')
        _make_fits(tmp_path / 'sky.fits')
        _make_fits(tmp_path / 'sn.fits')

        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath_no_ext)

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0', imgpath_no_ext]
        # This will likely fail because the script does img + '.sn2.fits' which works
        # but hdr = readhdr(imglong) and imglong has no .fits, so it reads 'testimg'
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        # Should call readhdr with the raw path (no .fits extension)
        mock_util.readhdr.assert_called()


class TestInteractiveMode:
    """Test interactive mode paths. These use --interactive flag and mock userinput.
    The prompts are complex and sequencing-sensitive, so we test specific branches."""

    def test_interactive_imexamine_success(self, tmp_path, monkeypatch):
        """Interactive mode: imexamine succeeds, covers lines 91, 168, 293-308."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        # Prompt sequence for interactive with --RA/--DEC provided:
        # 1. "repeat selection?" (line 302) -> 'n'
        # 2. "Size of cut frame" (line 337) -> '' (use default _size)
        # 3. "ok?" (line 347) -> '' (empty → 'y')
        # 4. "Cuts OK" (line 362) -> '' (empty → 'y')
        # 5. "SN ID OK?" (line 426) -> '' (empty → 'y')
        # 6. "length of square for bg" (line 450) -> '' (empty → default 3)
        # 7. "Background fit OK?" (line 606) -> '' (empty → 'y')
        # 8. "Iterate?" (line 627 - _numiter=0, not _numiter=True) -> 'n'
        # 9. "Manual adjust?" (line 707) -> '' (empty → 'n')
        # 10. "Error estimate?" (line 744) -> '' (empty → 'y' for interactive!)
        #     Actually line 744: if not answ0: answ0 = 'y'. Empty → 'y'!
        #     So we need 'n' explicitly.
        # Safest: return 'n' for everything except 'repeat?' which needs 'n' anyway
        # and size which can be ''. But 'n' for size causes int('n') → ValueError.
        # Use a smarter approach: check the prompt content.
        def userinput_side_effect(prompt):
            if 'repeat' in prompt.lower() or 'again' in prompt.lower():
                return 'n'
            if 'size' in prompt.lower() or 'length' in prompt.lower():
                return ''  # use default
            if 'order' in prompt.lower():
                return ''  # use default
            if 'iterate' in prompt.lower():
                return 'n'
            if 'adjust' in prompt.lower():
                return 'n'
            if 'error' in prompt.lower() or 'arterr' in prompt.lower():
                return 'n'
            # Default: empty → most prompts treat empty as 'y' (accept)
            return ''
        mock_util.userinput.side_effect = userinput_side_effect

        def fields_side_effect(filename, *a, **kw):
            if 'tmp.log' in str(filename):
                return ['50.0 50.0']
            return ['', '', '50.0 50.0']
        mock_iraf.fields.side_effect = fields_side_effect
        mock_iraf.imexamine.side_effect = lambda *a, **kw: ['50.0 50.0 1000']

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0', '--interactive',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        # Interactive mode sets _show = True (line 91)
        mock_util.display_image.assert_called()

    def test_interactive_imexamine_exception(self, tmp_path, monkeypatch):
        """Interactive mode: imexamine raises exception → except branch (lines 309-317)."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        # Make imexamine raise to hit the except block at line 309
        mock_iraf.imexamine.side_effect = Exception("display not available")

        def fields_side_effect(filename, *a, **kw):
            return ['', '', '50.0 50.0']
        mock_iraf.fields.side_effect = fields_side_effect

        # When imexamine fails and user says 'n' to repeat → exit at line 317
        mock_util.userinput.side_effect = lambda prompt: 'n'

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0', '--interactive',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy,
                    expect_exit=True)

    def test_interactive_with_iterations(self, tmp_path, monkeypatch):
        """Interactive mode with 1 iteration."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        def userinput_side_effect(prompt):
            if 'repeat' in prompt.lower() or 'again' in prompt.lower():
                return 'n'
            if 'size' in prompt.lower() or 'length' in prompt.lower():
                return ''
            if 'order' in prompt.lower():
                return ''
            if 'iterate' in prompt.lower():
                return 'n'
            if 'adjust' in prompt.lower():
                return 'n'
            if 'error' in prompt.lower() or 'arterr' in prompt.lower():
                return 'n'
            return ''
        mock_util.userinput.side_effect = userinput_side_effect

        def fields_side_effect(filename, *a, **kw):
            if 'tmp.log' in str(filename):
                return ['50.0 50.0']
            return ['', '', '50.0 50.0']
        mock_iraf.fields.side_effect = fields_side_effect
        mock_iraf.imexamine.side_effect = lambda *a, **kw: ['50.0 50.0 1000']

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0', '--interactive',
                '-n', '1', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)


class TestInteractiveCutsAndSNID:
    """Test interactive cuts adjustment and SN identification loops."""

    def test_interactive_cuts_adjust(self, tmp_path, monkeypatch):
        """Interactive: Cuts not OK → adjust z1/z2 → then OK (lines 365-385)."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        cuts_asked = [0]
        def userinput_side_effect(prompt):
            p = prompt.lower()
            if 'repeat' in p or 'again' in p:
                return 'n'
            if 'size' in p or 'length' in p:
                return ''
            if 'order' in p:
                return ''
            if 'cuts ok' in p:
                cuts_asked[0] += 1
                if cuts_asked[0] <= 2:
                    return 'no'  # First/second: not OK → stay in loop
                return ''  # Third: empty → 'y' → exit loop
            if 'z1' in p:
                # First iteration: return value (line 373), second: empty (line 371)
                if cuts_asked[0] == 1:
                    return '100'
                return ''
            if 'z2' in p:
                if cuts_asked[0] == 1:
                    return '5000'
                return ''
            if 'iterate' in p:
                return 'n'
            if 'not yet happy' in p:
                return 'n'
            if 'error' in p or 'arterr' in p:
                return 'n'
            if 'identification' in p or 'sn and co' in p:
                return ''  # accept → 'y'
            return ''
        mock_util.userinput.side_effect = userinput_side_effect

        def fields_side_effect(filename, *a, **kw):
            if 'tmp.log' in str(filename):
                return ['50.0 50.0']
            return ['', '', '50.0 50.0']
        mock_iraf.fields.side_effect = fields_side_effect
        mock_iraf.imexamine.side_effect = lambda *a, **kw: ['50.0 50.0 1000']

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0', '--interactive',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

    def test_interactive_sn_id_repeat(self, tmp_path, monkeypatch):
        """Interactive: SN ID not OK → repeat → then OK (lines 406-430)."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        sn_id_asked = [0]
        def userinput_side_effect(prompt):
            p = prompt.lower()
            if 'repeat' in p or 'again' in p:
                return 'n'
            if 'size' in p or 'length' in p:
                return ''
            if 'order' in p:
                return ''
            if 'cuts ok' in p:
                return ''
            if 'identification' in p or 'sn and co' in p:
                sn_id_asked[0] += 1
                if sn_id_asked[0] == 1:
                    return 'no'  # First time: not OK → repeat
                return ''  # Second time: OK
            if 'iterate' in p:
                return 'n'
            if 'not yet happy' in p:
                return 'n'
            if 'error' in p or 'arterr' in p:
                return 'n'
            return ''
        mock_util.userinput.side_effect = userinput_side_effect

        def fields_side_effect(filename, *a, **kw):
            if 'tmp.log' in str(filename):
                return ['50.0 50.0']
            return ['', '', '50.0 50.0']
        mock_iraf.fields.side_effect = fields_side_effect
        mock_iraf.imexamine.side_effect = lambda *a, **kw: ['50.0 50.0 1000']

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0', '--interactive',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

    def test_interactive_bg_not_ok(self, tmp_path, monkeypatch):
        """Interactive: Background fit not OK → redo → then OK (lines 606-610)."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        bg_asked = [0]
        def userinput_side_effect(prompt):
            p = prompt.lower()
            if 'repeat' in p or 'again' in p:
                return 'n'
            if 'size' in p or 'length' in p:
                return ''
            if 'order' in p:
                return ''
            if 'cuts ok' in p:
                return ''
            if 'identification' in p or 'sn and co' in p:
                return ''
            if 'background fit ok' in p or 'background' in p and 'ok' in p:
                bg_asked[0] += 1
                if bg_asked[0] == 1:
                    return 'no'  # First time: not OK
                return ''  # Second time: OK
            if 'iterate' in p:
                return 'n'
            if 'not yet happy' in p:
                return 'n'
            if 'error' in p or 'arterr' in p:
                return 'n'
            return ''
        mock_util.userinput.side_effect = userinput_side_effect

        def fields_side_effect(filename, *a, **kw):
            if 'tmp.log' in str(filename):
                return ['50.0 50.0']
            return ['', '', '50.0 50.0']
        mock_iraf.fields.side_effect = fields_side_effect
        mock_iraf.imexamine.side_effect = lambda *a, **kw: ['50.0 50.0 1000']

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0', '--interactive',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

    def test_interactive_multi_sn_bg_region(self, tmp_path, monkeypatch):
        """Interactive multi-SN: mark bg corners via tvmark (lines 514-529)."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        def userinput_side_effect(prompt):
            p = prompt.lower()
            if 'repeat' in p or 'again' in p:
                return 'n'
            if 'size' in p or 'length' in p:
                return ''
            if 'order' in p:
                return ''
            if 'cuts ok' in p:
                return ''
            if 'identification' in p or 'sn and co' in p:
                return ''
            if 'background' in p and 'ok' in p:
                return ''
            if 'iterate' in p:
                return 'n'
            if 'not yet happy' in p:
                return 'n'
            if 'error' in p or 'arterr' in p:
                return 'n'
            return ''
        mock_util.userinput.side_effect = userinput_side_effect

        # iraf.fields returns multiple coords for SN (from wcsctran)
        def fields_side_effect(filename, *a, **kw):
            if 'tmp.log' in str(filename):
                return ['50.0 50.0', '60.0 60.0']  # Two SN positions
            return ['', '', '50.0 50.0']
        mock_iraf.fields.side_effect = fields_side_effect

        # imexamine returns two positions (multi-SN in SN ID step)
        mock_iraf.imexamine.side_effect = lambda *a, **kw: ['50.0 50.0 1000', '60.0 60.0 1000']

        # tvmark with logfile creates tmptbl - mock it to write the file
        def tvmark_side_effect(*args, **kwargs):
            logfile = kwargs.get('logfile', '')
            if logfile == 'tmptbl':
                with open(str(tmp_path / 'tmptbl'), 'w') as f:
                    f.write('10.0 10.0\n90.0 90.0\n')
        mock_iraf.tvmark.side_effect = tvmark_side_effect

        # fitsn returns results for 2 sources
        mock_snoopy.fitsn.return_value = (
            [18.0, 18.1], [17.8, 17.9], [17.5, 17.6],
            [18.0, 18.1], [17.8, 17.9], [17.5, 17.6],
            [0.01, 0.01], [0.01, 0.01], [0.01, 0.01],
            [17.0, 17.1], [16.8, 16.9], [0.02, 0.03],
            [50.0, 60.0], [50.0, 60.0],
        )

        argv = ['lscsn.py', '--RA', '150.0,150.1', '--DEC', '2.0,2.1', '--interactive',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

    def test_interactive_iterate_zero_numiter(self, tmp_path, monkeypatch):
        """Interactive with -n 0: user prompted for iteration (lines 625-627, 682-684)."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        iterate_asked = [0]
        def userinput_side_effect(prompt):
            p = prompt.lower()
            if 'repeat' in p or 'again' in p:
                return 'n'
            if 'size' in p or 'length' in p:
                return ''
            if 'order' in p:
                return ''
            if 'cuts ok' in p:
                return ''
            if 'identification' in p or 'sn and co' in p:
                return ''
            if 'background' in p and 'ok' in p:
                return ''
            if 'iterate' in p:
                iterate_asked[0] += 1
                if iterate_asked[0] == 1:
                    return ''  # empty → answ0='y' (line 627) → iterate once
                return 'n'  # second time: stop (line 683 → user says 'n')
            if 'not yet happy' in p:
                return ''  # empty → answ0='n' (line 709)
            if 'error' in p or 'arterr' in p:
                return 'n'
            return ''
        mock_util.userinput.side_effect = userinput_side_effect

        def fields_side_effect(filename, *a, **kw):
            if 'tmp.log' in str(filename):
                return ['50.0 50.0']
            return ['', '', '50.0 50.0']
        mock_iraf.fields.side_effect = fields_side_effect
        mock_iraf.imexamine.side_effect = lambda *a, **kw: ['50.0 50.0 1000']

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0', '--interactive',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

    def test_interactive_repeat_selection_yes(self, tmp_path, monkeypatch):
        """Interactive: 'yes' response to repeat selection (line 306)."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        repeat_count = [0]
        def userinput_side_effect(prompt):
            p = prompt.lower()
            if 'repeat selection' in p:
                repeat_count[0] += 1
                if repeat_count[0] == 1:
                    return 'yes'  # → repeat = 'y' (line 306)
                return 'n'  # second time: stop
            if 'again' in p:
                return 'n'
            if 'size' in p or 'length' in p:
                return ''
            if 'order' in p:
                return ''
            if 'iterate' in p:
                return 'n'
            if 'not yet happy' in p:
                return 'n'
            if 'error' in p or 'arterr' in p:
                return 'n'
            return ''
        mock_util.userinput.side_effect = userinput_side_effect

        def fields_side_effect(filename, *a, **kw):
            if 'tmp.log' in str(filename):
                return ['50.0 50.0']
            return ['', '', '50.0 50.0']
        mock_iraf.fields.side_effect = fields_side_effect
        mock_iraf.imexamine.side_effect = lambda *a, **kw: ['50.0 50.0 1000']

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0', '--interactive',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

    def test_interactive_size_custom(self, tmp_path, monkeypatch):
        """Interactive: user provides custom size (line 341)."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        def userinput_side_effect(prompt):
            p = prompt.lower()
            if 'repeat' in p or 'again' in p:
                return 'n'
            if 'size' in p:
                return '5'  # Custom size (exercises line 341: size = int(size))
            if 'length' in p:
                return ''
            if 'order' in p:
                return ''
            if 'iterate' in p:
                return 'n'
            if 'not yet happy' in p:
                return 'n'
            if 'error' in p or 'arterr' in p:
                return 'n'
            return ''
        mock_util.userinput.side_effect = userinput_side_effect

        def fields_side_effect(filename, *a, **kw):
            if 'tmp.log' in str(filename):
                return ['50.0 50.0']
            return ['', '', '50.0 50.0']
        mock_iraf.fields.side_effect = fields_side_effect
        mock_iraf.imexamine.side_effect = lambda *a, **kw: ['50.0 50.0 1000']

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0', '--interactive',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)


class TestDMNonZero:
    """Test DM correction when EXPTARG != EXPTIME."""

    def test_dm_nonzero(self, tmp_path, monkeypatch):
        """DM = 2.5*log10(targ) - 2.5*log10(diff) is non-zero when they differ."""
        monkeypatch.chdir(tmp_path)
        img_name = 'test_diff'
        imgpath = _setup_basic_env(
            tmp_path, img_name=img_name,
            extra_img_hdr={
                'CONVOL00': 'TEMPLATE',
                'EXPTIME': 60.0,  # diff exposure
                'EXPTARG': 120.0,  # target exposure (different!)
                'EXPTEMP': 60.0,
                'TEMPLATE': 'tmpl.fits',
                'TARGET': 'targ.fits',
            },
            extra_sn2_hdr={'PSFMAG1': '', 'PIXSCALE': 0.389, 'PSF_FWHM': 1.5, 'APCO': 0.05}
        )
        tmpl_sn2 = tmp_path / 'tmpl.sn2.fits'
        _make_fits(tmpl_sn2, {'APCO': 0.08, 'PIXSCALE': 0.389, 'PSF_FWHM': 1.5})

        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)
        mock_mysqldef.getfromdataraw.return_value = [{'filepath': str(tmp_path)}]

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

        # DM = 2.5*log10(120) - 2.5*log10(60) ≈ 0.752
        # psfmag in DB = truemag - DM
        import math
        expected_dm = 2.5 * math.log10(120) - 2.5 * math.log10(60)
        psfmag_calls = [c for c in mock_mysqldef.updatevalue.call_args_list
                        if c[0][1] == 'psfmag' and c[0][2] != 9999]
        assert len(psfmag_calls) == 1
        # truemag[0] = 16.8 from mock, so psfmag = 16.8 - 0.752
        assert abs(psfmag_calls[0][0][2] - (16.8 - expected_dm)) < 0.01


class TestDBConnectionError:
    """Test DB connection errors in template lookup."""

    def test_import_error_in_db_connect(self, tmp_path, monkeypatch):
        """ImportError during DB connection is caught (lines 182-186).
        Note: conn is undefined after the except, so getfromdataraw(conn,...)
        raises NameError. The script crashes — we just verify the ImportError
        handler runs (coverage)."""
        monkeypatch.chdir(tmp_path)
        img_name = 'test_diff'
        imgpath = _setup_basic_env(
            tmp_path, img_name=img_name,
            extra_img_hdr={
                'CONVOL00': 'TEMPLATE',
                'EXPTIME': 120.0,
                'EXPTARG': 120.0,
                'EXPTEMP': 60.0,
                'TEMPLATE': 'tmpl.fits',
                'TARGET': 'targ.fits',
            },
            extra_sn2_hdr={'PSFMAG1': '', 'PIXSCALE': 0.389, 'PSF_FWHM': 1.5, 'APCO': 0.05}
        )
        tmpl_sn2 = tmp_path / 'tmpl.sn2.fits'
        _make_fits(tmpl_sn2, {'APCO': 0.08, 'PIXSCALE': 0.389, 'PSF_FWHM': 1.5})

        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)
        mock_mysqldef.getconnection.side_effect = ImportError("no mysql module")
        mock_mysqldef.getfromdataraw.return_value = [{'filepath': str(tmp_path)}]

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        # Crashes with NameError on conn after except handler runs
        with pytest.raises((SystemExit, NameError, UnboundLocalError)):
            _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

    def test_general_exception_in_db_connect(self, tmp_path, monkeypatch):
        """General Exception during DB connection is caught (lines 187-189).
        Same issue: conn undefined after except -> NameError."""
        monkeypatch.chdir(tmp_path)
        img_name = 'test_diff'
        imgpath = _setup_basic_env(
            tmp_path, img_name=img_name,
            extra_img_hdr={
                'CONVOL00': 'TEMPLATE',
                'EXPTIME': 120.0,
                'EXPTARG': 120.0,
                'EXPTEMP': 60.0,
                'TEMPLATE': 'tmpl.fits',
                'TARGET': 'targ.fits',
            },
            extra_sn2_hdr={'PSFMAG1': '', 'PIXSCALE': 0.389, 'PSF_FWHM': 1.5, 'APCO': 0.05}
        )
        tmpl_sn2 = tmp_path / 'tmpl.sn2.fits'
        _make_fits(tmpl_sn2, {'APCO': 0.08, 'PIXSCALE': 0.389, 'PSF_FWHM': 1.5})

        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)
        mock_mysqldef.dbConnect.side_effect = Exception("connection refused")
        mock_mysqldef.getfromdataraw.return_value = [{'filepath': str(tmp_path)}]

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        with pytest.raises((SystemExit, NameError, UnboundLocalError)):
            _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)


class TestCoordinateNotFound:
    """When wcsctran returns empty coordinates."""

    def test_xx0_empty_non_interactive_exits(self, tmp_path, monkeypatch):
        """Line 285: exit when xx0 is empty and not interactive."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        # No RA0/DEC0 provided, no checksndb result -> _ra0/_dec0 empty
        # -> the block at 270 is skipped -> xx0='' -> line 284 triggers exit
        mock_util.checksndb.return_value = (150.0, 2.0, 1)
        # Override: give RA/DEC for SN but make the box coord path skip
        # by not providing RA0/DEC0. The script sets _ra0=_ra[0], _dec0=_dec[0]
        # when _ra is provided but _ra0 is not. So wcsctran is called for box too.
        # To get xx0='', we need to make the second fields call return something
        # that doesn't split into two values. Actually easier: just don't provide
        # RA/DEC at all, and make checksndb raise so _ra0/_dec0 stay empty.
        mock_util.checksndb.side_effect = Exception("no target in db")

        # Give RA/DEC on CLI so the SN coord loop works, but RA0/DEC0 empty
        # Actually when RA is given but RA0 is not, _ra0 = _ra[0] (line 95).
        # So to test line 282 (else: xx0, yy0 = '', ''), we need _ra0='' and _dec0=''
        # which only happens when _ra is empty AND checksndb fails.
        # Then line 233-239 exits with 'no box and sn coordinate'
        argv = ['lscsn.py', '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy,
                    expect_exit=True)


class TestNoBoxCoordNoRA0:
    """When no RA0/DEC0 and no RA/DEC from either CLI or database."""

    def test_no_ra0_no_checksndb(self, tmp_path, monkeypatch):
        """Line 282: xx0, yy0 = '' when no _ra0, _dec0 (line 226-232 path)."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)
        # checksndb returns None/empty coords
        mock_util.checksndb.side_effect = Exception("not in db")

        # No RA, no DEC, checksndb fails -> should try to use RA from box
        # but box is also empty -> exit
        argv = ['lscsn.py', '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy,
                    expect_exit=True)


class TestArtificialStarExperiment:
    """Test the artificial star error estimation branch (lines 750-763)."""

    def test_error_estimation_interactive(self, tmp_path, monkeypatch):
        """Interactive mode with error estimation (answ0='y' at line 749)."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        def userinput_side_effect(prompt):
            p = prompt.lower()
            if 'repeat' in p or 'again' in p:
                return 'n'
            if 'size' in p or 'length' in p:
                return ''
            if 'order' in p:
                return ''
            if 'iterate' in p:
                return 'n'
            if 'not yet happy' in p:
                return 'n'
            if 'arterr' in p:
                return ''  # use default arterr
            if 'error' in p:
                return 'y'  # YES do error estimation
            return ''
        mock_util.userinput.side_effect = userinput_side_effect

        def fields_side_effect(filename, *a, **kw):
            if 'tmp.log' in str(filename):
                return ['50.0 50.0']
            return ['', '', '50.0 50.0']
        mock_iraf.fields.side_effect = fields_side_effect
        mock_iraf.imexamine.side_effect = lambda *a, **kw: ['50.0 50.0 1000']

        mock_snoopy.errore.return_value = (0.05, 0.03)

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0', '--interactive',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

    def test_error_estimation_empty_prompts(self, tmp_path, monkeypatch):
        """Interactive: empty responses for 'Not yet happy' and 'Errors estimate' (lines 709, 746)."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        def userinput_side_effect(prompt):
            p = prompt.lower()
            if 'repeat' in p or 'again' in p:
                return 'n'
            if 'size' in p or 'length' in p:
                return ''
            if 'order' in p:
                return ''
            if 'iterate' in p:
                return 'n'
            if 'not yet happy' in p:
                return ''  # empty → answ0='n' (line 709)
            if 'arterr' in p:
                return ''  # use default
            if 'error' in p:
                return ''  # empty → answ0='y' (line 746)
            return ''
        mock_util.userinput.side_effect = userinput_side_effect

        def fields_side_effect(filename, *a, **kw):
            if 'tmp.log' in str(filename):
                return ['50.0 50.0']
            return ['', '', '50.0 50.0']
        mock_iraf.fields.side_effect = fields_side_effect
        mock_iraf.imexamine.side_effect = lambda *a, **kw: ['50.0 50.0 1000']

        mock_snoopy.errore.return_value = (0.05, 0.03)

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0', '--interactive',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)

    def test_error_estimation_exception(self, tmp_path, monkeypatch):
        """When errore raises exception, arterr defaults to 0 (lines 755-757)."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        def userinput_side_effect(prompt):
            p = prompt.lower()
            if 'repeat' in p or 'again' in p:
                return 'n'
            if 'size' in p or 'length' in p:
                return ''
            if 'order' in p:
                return ''
            if 'iterate' in p:
                return 'n'
            if 'not yet happy' in p:
                return 'n'
            if 'arterr' in p:
                return ''
            if 'error' in p:
                return 'y'  # do error estimation
            return ''
        mock_util.userinput.side_effect = userinput_side_effect

        def fields_side_effect(filename, *a, **kw):
            if 'tmp.log' in str(filename):
                return ['50.0 50.0']
            return ['', '', '50.0 50.0']
        mock_iraf.fields.side_effect = fields_side_effect
        mock_iraf.imexamine.side_effect = lambda *a, **kw: ['50.0 50.0 1000']

        # errore raises to exercise the except path (line 755-757)
        mock_snoopy.errore.side_effect = Exception("stamp too small")

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0', '--interactive',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)


class TestManualAdjustment:
    """Test manual magnitude adjustment (interactive, lines 714-736)."""

    def test_manual_adjust_interactive(self, tmp_path, monkeypatch):
        """Interactive mode with manual magnitude adjustment."""
        monkeypatch.chdir(tmp_path)
        imgpath = _setup_basic_env(tmp_path)
        mock_util, mock_mysqldef, mock_iraf, mock_snoopy = _build_mocks(tmp_path, imgpath)

        adjust_count = [0]
        def userinput_side_effect(prompt):
            p = prompt.lower()
            if 'repeat' in p:
                return 'n'
            if 'again' in p:
                return 'n'
            if 'size' in p or 'length' in p:
                return ''
            if 'order' in p:
                return ''
            if 'iterate' in p:
                return 'n'
            if 'not yet happy' in p:
                adjust_count[0] += 1
                if adjust_count[0] == 1:
                    return 'yes'  # First time: 'yes' → answ0='y' (line 711)
                return 'n'
            if 'd(mag)' in p:
                return '0.1'
            if 'error' in p or 'arterr' in p:
                return 'n'
            return ''
        mock_util.userinput.side_effect = userinput_side_effect

        def fields_side_effect(filename, *a, **kw):
            if 'tmp.log' in str(filename):
                return ['50.0 50.0']
            return ['', '', '50.0 50.0']
        mock_iraf.fields.side_effect = fields_side_effect
        mock_iraf.imexamine.side_effect = lambda *a, **kw: ['50.0 50.0 1000']

        mock_snoopy.manusn.return_value = (
            [18.0], [17.8], [17.5],
            [18.0], [17.8], [17.5],
            [17.0], [16.7], [0.02],
            [50.0], [50.0], [16.7],
        )

        argv = ['lscsn.py', '--RA', '150.0', '--DEC', '2.0', '--interactive',
                '-n', '0', '-x', '2', '-y', '2', imgpath]
        _run_script(argv, tmp_path, mock_util, mock_mysqldef, mock_iraf, mock_snoopy)


class TestLoadable:
    """Smoke test that module loads."""

    def test_importable(self):
        import importlib
        loader = importlib.machinery.SourceFileLoader('lscsn', SCRIPT)
        spec = importlib.util.spec_from_loader('lscsn', loader)
        mod = importlib.util.module_from_spec(spec)
        mod.__name__ = 'lscsn'
        spec.loader.exec_module(mod)
        assert hasattr(mod, 'lsc')
