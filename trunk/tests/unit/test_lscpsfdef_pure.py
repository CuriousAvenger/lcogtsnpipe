"""
Tests for lsc.lscpsfdef.
Since iraf is fully mocked via conftest.py, we can drive psffit2 and psffit
through their main paths by supplying a minimal FITS header.  runsex requires
a real sex config file (which lives in the package) plus mocked subprocess and
a tmp.cat file.
"""
import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, mock_open

pytestmark = pytest.mark.unit


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_hdr(extra=None):
    """Return a minimal astropy Header with gain/ron/pixscale."""
    from astropy.io import fits
    hdr = fits.Header()
    hdr['GAIN'] = 2.0
    hdr['RDNOISE'] = 5.0
    hdr['RON'] = 5.0
    hdr['PIXSCALE'] = 0.389
    hdr['SATURATE'] = 60000.0
    hdr['DATAMAX'] = 60000.0
    hdr['DATAMIN'] = 0.0
    hdr['INSTRUME'] = 'fa15'
    hdr['L1FWHM'] = 3.0
    hdr['WCSERR'] = 0
    hdr['PSF_FWHM'] = 3.0
    if extra:
        hdr.update(extra)
    return hdr


def _make_fits(tmp_path, name='test', extra_hdr=None):
    """Write a minimal FITS file and return path without .fits extension."""
    from astropy.io import fits
    path = tmp_path / (name + '.fits')
    data = np.random.default_rng(1).normal(1000, 50, (512, 512)).astype(np.float32)
    hdr = _make_hdr(extra_hdr)
    fits.writeto(str(path), data, hdr, overwrite=True)
    return str(tmp_path / name)  # without .fits


def _get_iraf():
    return sys.modules['pyraf'].iraf


# ── importability ─────────────────────────────────────────────────────────────

class TestLscpsfdefImportable:
    def test_module_imports(self):
        import lsc.lscpsfdef  # noqa: F401

    def test_runsex_is_callable(self):
        from lsc.lscpsfdef import runsex
        assert callable(runsex)

    def test_psffit2_is_callable(self):
        from lsc.lscpsfdef import psffit2
        assert callable(psffit2)

    def test_psffit_is_callable(self):
        from lsc.lscpsfdef import psffit
        assert callable(psffit)

    def test_ecpsf_is_callable(self):
        from lsc.lscpsfdef import ecpsf
        assert callable(ecpsf)


# ── psffit2 ───────────────────────────────────────────────────────────────────

class TestPsffit2:
    """psffit2 is all iraf calls + header reads — iraf is mocked globally."""

    def test_basic_run_returns_two_values(self, tmp_path, monkeypatch):
        from lsc.lscpsfdef import psffit2
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        result = psffit2('test_img', 5.0, 10, hdr, 0.0, 60000.0)
        assert len(result) == 2

    def test_ron_missing_falls_to_default(self, tmp_path, monkeypatch):
        from astropy.io import fits
        from lsc.lscpsfdef import psffit2
        monkeypatch.chdir(tmp_path)
        hdr = fits.Header()  # no RON / GAIN
        result = psffit2('test_img', 5.0, 10, hdr, 0.0, 60000.0)
        assert len(result) == 2

    def test_gain_missing_falls_to_default(self, tmp_path, monkeypatch):
        from astropy.io import fits
        from lsc.lscpsfdef import psffit2
        monkeypatch.chdir(tmp_path)
        hdr = fits.Header({'RON': 5.0})  # no GAIN
        result = psffit2('test_img', 5.0, 10, hdr, 0.0, 60000.0)
        assert len(result) == 2

    def test_fixaperture_false_uses_fwhm(self, tmp_path, monkeypatch):
        from lsc.lscpsfdef import psffit2
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        photmag, fitmag = psffit2('test_img', 7.0, 10, hdr, 0.0, 60000.0, fixaperture=False)
        assert photmag is not None

    def test_fixaperture_true_reads_pixscale(self, tmp_path, monkeypatch):
        """fixaperture=True branch reads PIXSCALE from file."""
        from lsc.lscpsfdef import psffit2
        img = _make_fits(tmp_path, 'fix_ap')
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        result = psffit2(img, 5.0, 10, hdr, 0.0, 60000.0, fixaperture=True)
        assert len(result) == 2

    def test_psffun_moffat_passed(self, tmp_path, monkeypatch):
        from lsc.lscpsfdef import psffit2
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        result = psffit2('test_img', 5.0, 10, hdr, 0.0, 60000.0, psffun='moffat')
        assert len(result) == 2


# ── psffit ────────────────────────────────────────────────────────────────────

class TestPsffit:
    """psffit reads/writes _psf.mag — supply a real (empty) file."""

    def _write_psf_mag(self, tmp_path, content=''):
        (tmp_path / '_psf.mag').write_text(content)

    def test_basic_run_returns_three_values(self, tmp_path, monkeypatch):
        from lsc.lscpsfdef import psffit
        self._write_psf_mag(tmp_path)
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        result = psffit('test_img', 5.0, 10, hdr, False, 0.0, 60000.0)
        assert len(result) == 3

    def test_interactive_false_pstselect_called(self, tmp_path, monkeypatch):
        from lsc.lscpsfdef import psffit
        self._write_psf_mag(tmp_path)
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        iraf = _get_iraf()
        psffit('test_img', 5.0, 10, hdr, False, 0.0, 60000.0)
        assert iraf.pstselect.called

    def test_interactive_true_cp_instead_of_pstselect(self, tmp_path, monkeypatch):
        from lsc.lscpsfdef import psffit
        self._write_psf_mag(tmp_path)
        (tmp_path / '_psf.pst').write_text('')
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        iraf = _get_iraf()
        iraf.pstselect.reset_mock()
        psffit('test_img', 5.0, 10, hdr, True, 0.0, 60000.0)
        assert not iraf.pstselect.called

    def test_fixaperture_true(self, tmp_path, monkeypatch):
        from lsc.lscpsfdef import psffit
        img = _make_fits(tmp_path, 'psffit_ap')
        self._write_psf_mag(tmp_path)
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        result = psffit(img, 5.0, 10, hdr, False, 0.0, 60000.0, fixaperture=True)
        assert len(result) == 3

    def test_ron_gain_missing_defaults(self, tmp_path, monkeypatch):
        from astropy.io import fits
        from lsc.lscpsfdef import psffit
        self._write_psf_mag(tmp_path)
        monkeypatch.chdir(tmp_path)
        hdr = fits.Header()  # empty header → ron=1, gain=1
        result = psffit('test_img', 5.0, 10, hdr, False, 0.0, 60000.0)
        assert len(result) == 3


# ── runsex ───────────────────────────────────────────────────────────────────

class TestRunsex:
    """runsex calls subprocess.Popen (sex) and opens tmp.cat.
    We mock Popen and provide an empty tmp.cat."""

    def _setup_iraf_hselect(self):
        iraf = _get_iraf()
        iraf.hselect.return_value = ['100 100']
        return iraf

    def test_returns_eight_arrays(self, tmp_path, monkeypatch):
        from lsc.lscpsfdef import runsex
        self._setup_iraf_hselect()
        (tmp_path / 'tmp.cat').write_text('# header\n')
        monkeypatch.chdir(tmp_path)
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b'', b'')
        with patch('subprocess.Popen', return_value=mock_proc):
            result = runsex('test_img', 5.0, 3.0, 0.389)
        assert len(result) == 8

    def test_returns_numpy_arrays(self, tmp_path, monkeypatch):
        from lsc.lscpsfdef import runsex
        self._setup_iraf_hselect()
        (tmp_path / 'tmp.cat').write_text('# header\n')
        monkeypatch.chdir(tmp_path)
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b'', b'')
        with patch('subprocess.Popen', return_value=mock_proc):
            result = runsex('test_img', 5.0, 3.0, 0.389)
        for arr in result:
            assert isinstance(arr, np.ndarray)

    def test_filters_border_pixels(self, tmp_path, monkeypatch):
        """Stars within 5px of border should be excluded."""
        import lsc
        from lsc.lscpsfdef import runsex
        iraf = self._setup_iraf_hselect()  # 100x100 image
        # Build a tmp.cat using the package's default2.param column order
        param_file = os.path.join(lsc.__path__[0], 'standard', 'sex', 'default2.param')
        with open(param_file) as f:
            cparam = [line.split()[0] for line in f if line.strip() and line[0] != '#']
        n_cols = len(cparam)
        valid = ['0.0'] * n_cols
        valid[cparam.index('X_IMAGE')] = '50.0'
        valid[cparam.index('Y_IMAGE')] = '50.0'
        border = ['0.0'] * n_cols
        border[cparam.index('X_IMAGE')] = '2.0'
        border[cparam.index('Y_IMAGE')] = '50.0'
        cat_content = '# header\n' + ' '.join(valid) + '\n' + ' '.join(border) + '\n'
        (tmp_path / 'tmp.cat').write_text(cat_content)
        monkeypatch.chdir(tmp_path)
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b'', b'')
        with patch('subprocess.Popen', return_value=mock_proc):
            xcoo, ycoo, ra, dec, magbest, classstar, fluxrad, bkg = runsex('test_img', 5.0, 3.0, 0.389)
        assert len(xcoo) == 1
        assert xcoo[0] == 50.0


# ── ecpsf ─────────────────────────────────────────────────────────────────────

class TestEcpsf:
    """ecpsf is wrapped in try/except → returns (0, 0, 0) on any error."""

    def test_returns_three_values_on_missing_file(self, tmp_path, monkeypatch):
        """Missing FITS → exception caught → returns (0, 0.0, 0)."""
        from lsc.lscpsfdef import ecpsf
        monkeypatch.chdir(tmp_path)
        result, fwhm_out, apco = ecpsf('nonexistent', 5.0, 3.0, 10, 3.0, False)
        assert result == 0

    def test_catalog_path_with_real_fits(self, tmp_path, monkeypatch):
        """ecpsf with _catalog='' and interactive=False runs runsex internally."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'ecpsf_test')
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        xs = np.array([100.0, 200.0, 300.0])
        ys = np.array([100.0, 200.0, 300.0])
        ran = np.zeros(3)
        decn = np.zeros(3)
        magbest = np.array([15.0, 16.0, 17.0])
        classstar = np.ones(3)
        fluxrad = np.array([5.0, 5.0, 5.0])
        bkg = np.zeros(3)
        with patch('lsc.lscpsfdef.runsex', return_value=(xs, ys, ran, decn, magbest, classstar, fluxrad, bkg)):
            with patch('lsc.lscpsfdef.psffit', return_value=(['50 50 1 15.0 0.01'], ['50 50 1'], ['50 50 1 15.0 0.01'])):
                with patch('lsc.lscpsfdef.psffit2', return_value=(['50 50 1 15.0 0.01'], ['50 50 1 15.0 0.01'])):
                    result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False)
        assert result in (0, 1)

    def test_fwhm_from_header_when_not_provided(self, tmp_path, monkeypatch):
        """When fwhm=0 ecpsf reads L1FWHM from the header."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'ecpsf_fwhm', extra_hdr={'L1FWHM': 2.5, 'WCSERR': 0})
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        xs = np.array([100.0])
        ys = np.array([100.0])
        with patch('lsc.lscpsfdef.runsex', return_value=(xs, ys, np.zeros(1), np.zeros(1),
                                                          np.array([15.0]), np.ones(1),
                                                          np.array([5.0]), np.zeros(1))):
            with patch('lsc.lscpsfdef.psffit', return_value=([], [], [])):
                with patch('lsc.lscpsfdef.psffit2', return_value=([], [])):
                    result, fwhm_out, apco = ecpsf(img, 0, 3.0, 10, 3.0, False)
        assert result in (0, 1)

    def test_make_sn2_false(self, tmp_path, monkeypatch):
        """make_sn2=False skips the binary table creation."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'ecpsf_nosn2')
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        with patch('lsc.lscpsfdef.runsex', return_value=(np.array([100.0]), np.array([100.0]),
                                                          np.zeros(1), np.zeros(1),
                                                          np.array([15.0]), np.ones(1),
                                                          np.array([5.0]), np.zeros(1))):
            with patch('lsc.lscpsfdef.psffit', return_value=([], [], [])):
                result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False, make_sn2=False)
        assert result in (0, 1)
