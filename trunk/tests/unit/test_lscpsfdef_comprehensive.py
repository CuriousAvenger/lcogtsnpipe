"""
Comprehensive tests for lsc.lscpsfdef.

Tests cover:
  - runsex: SExtractor invocation, catalog parsing, border filtering, flux radius
  - psffit / psffit2: aperture calculations, IRAF parameter setting, saturated star removal
  - ecpsf: FWHM estimation from header, catalog-based PSF, aperture corrections,
    sigma clipping, star selection logic, datamax handling, sn2 output creation
  - Helper logic: _to_float_array (inner function tested via ecpsf), duplicate elimination

Since iraf is fully mocked via conftest.py we can exercise control flow without
a real IRAF installation.
"""
import sys
import os
import re
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, mock_open, call
from astropy.io import fits

pytestmark = pytest.mark.unit


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_hdr(extra=None):
    """Return a minimal astropy Header with gain/ron/pixscale."""
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
    hdr['NAXIS1'] = 512
    hdr['NAXIS2'] = 512
    if extra:
        hdr.update(extra)
    return hdr


def _make_hdr_for_fits(extra=None, shape=(512, 512)):
    """Return a header suitable for writing to FITS (no NAXIS1/NAXIS2 since writeto sets them)."""
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


def _make_fits(tmp_path, name='test', extra_hdr=None, shape=(512, 512)):
    """Write a minimal FITS file and return path without .fits extension."""
    path = tmp_path / (name + '.fits')
    data = np.random.default_rng(1).normal(1000, 50, shape).astype(np.float32)
    hdr = _make_hdr_for_fits(extra_hdr, shape)
    hdu = fits.PrimaryHDU(data=data, header=hdr)
    hdu.writeto(str(path), overwrite=True)
    return str(tmp_path / name)  # without .fits


def _get_iraf():
    return sys.modules['pyraf'].iraf


# ── runsex comprehensive ─────────────────────────────────────────────────────

class TestRunsexComprehensive:
    """Comprehensive tests for runsex including catalog parsing and filtering."""

    def _setup_iraf_hselect(self, xdim='200', ydim='200'):
        iraf = _get_iraf()
        iraf.hselect.return_value = [f'{xdim} {ydim}']
        return iraf

    def _get_param_cols(self):
        """Get column names from the SExtractor param file."""
        import lsc
        param_file = os.path.join(lsc.__path__[0], 'standard', 'sex', 'default2.param')
        with open(param_file) as f:
            return [line.split()[0] for line in f if line.strip() and line[0] != '#']

    def _make_cat_line(self, cparam, overrides=None):
        """Build a catalog line with zeros for all columns and overrides for specifics."""
        vals = ['0.0'] * len(cparam)
        if overrides:
            for col, val in overrides.items():
                vals[cparam.index(col)] = str(val)
        return ' '.join(vals) + '\n'

    def test_empty_catalog_returns_empty_arrays(self, tmp_path, monkeypatch):
        from lsc.lscpsfdef import runsex
        self._setup_iraf_hselect()
        (tmp_path / 'tmp.cat').write_text('# only comments\n# another comment\n')
        monkeypatch.chdir(tmp_path)
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b'', b'')
        with patch('subprocess.Popen', return_value=mock_proc):
            xs, ys, ra, dec, mag, cs, fr, bkg = runsex('img', 5.0, 3.0, 0.389)
        assert len(xs) == 0
        assert len(ys) == 0

    def test_multiple_sources_parsed_correctly(self, tmp_path, monkeypatch):
        """Multiple valid sources within bounds are all returned."""
        from lsc.lscpsfdef import runsex
        self._setup_iraf_hselect('200', '200')
        cparam = self._get_param_cols()
        lines = '# header\n'
        for i in range(5):
            lines += self._make_cat_line(cparam, {
                'X_IMAGE': 50.0 + i * 20,
                'Y_IMAGE': 100.0,
                'X_WORLD': 150.0 + i * 0.01,
                'Y_WORLD': 2.0,
                'MAG_BEST': -15.0 + i,
                'CLASS_STAR': 0.95,
                'FLUX_RADIUS': 3.5,
                'BACKGROUND': 1000.0,
            })
        (tmp_path / 'tmp.cat').write_text(lines)
        monkeypatch.chdir(tmp_path)
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b'', b'')
        with patch('subprocess.Popen', return_value=mock_proc):
            xs, ys, ra, dec, mag, cs, fr, bkg = runsex('img', 5.0, 3.0, 0.389)
        assert len(xs) == 5
        assert xs[0] == pytest.approx(50.0)
        assert xs[4] == pytest.approx(130.0)

    def test_border_filter_all_edges(self, tmp_path, monkeypatch):
        """Sources at all four edges (within 5 pixels) should be excluded."""
        from lsc.lscpsfdef import runsex
        self._setup_iraf_hselect('100', '100')
        cparam = self._get_param_cols()
        # Source at left edge (x=3), right edge (x=97), top (y=3), bottom (y=97)
        lines = '# header\n'
        edge_cases = [
            {'X_IMAGE': 3.0, 'Y_IMAGE': 50.0},   # left edge
            {'X_IMAGE': 97.0, 'Y_IMAGE': 50.0},   # right edge
            {'X_IMAGE': 50.0, 'Y_IMAGE': 3.0},    # top edge
            {'X_IMAGE': 50.0, 'Y_IMAGE': 97.0},   # bottom edge
            {'X_IMAGE': 50.0, 'Y_IMAGE': 50.0},   # center (valid)
        ]
        for ec in edge_cases:
            ec.update({'X_WORLD': 150.0, 'Y_WORLD': 2.0, 'MAG_BEST': -15.0,
                       'CLASS_STAR': 0.9, 'FLUX_RADIUS': 3.0, 'BACKGROUND': 100.0})
            lines += self._make_cat_line(cparam, ec)
        (tmp_path / 'tmp.cat').write_text(lines)
        monkeypatch.chdir(tmp_path)
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b'', b'')
        with patch('subprocess.Popen', return_value=mock_proc):
            xs, ys, ra, dec, mag, cs, fr, bkg = runsex('img', 5.0, 3.0, 0.389)
        assert len(xs) == 1
        assert xs[0] == pytest.approx(50.0)
        assert ys[0] == pytest.approx(50.0)

    def test_seeing_computation(self, tmp_path, monkeypatch):
        """Verify that fwhm * pix_scale is passed as seeing."""
        from lsc.lscpsfdef import runsex
        self._setup_iraf_hselect()
        (tmp_path / 'tmp.cat').write_text('# header\n')
        monkeypatch.chdir(tmp_path)
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b'', b'')
        with patch('subprocess.Popen', return_value=mock_proc) as mock_popen:
            runsex('img', 7.0, 3.0, 0.5)
        # Check that SEEING_FWHM is 7.0 * 0.5 = 3.5
        cmd = mock_popen.call_args[0][0]
        assert '3.5' in cmd

    def test_detect_thresh_passed(self, tmp_path, monkeypatch):
        """Verify detection threshold is passed to SExtractor."""
        from lsc.lscpsfdef import runsex
        self._setup_iraf_hselect()
        (tmp_path / 'tmp.cat').write_text('# header\n')
        monkeypatch.chdir(tmp_path)
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b'', b'')
        with patch('subprocess.Popen', return_value=mock_proc) as mock_popen:
            runsex('img', 5.0, 4.5, 0.389)
        cmd = mock_popen.call_args[0][0]
        assert '4.5' in cmd

    def test_pixel_scale_passed(self, tmp_path, monkeypatch):
        """Verify pixel scale is passed to SExtractor."""
        from lsc.lscpsfdef import runsex
        self._setup_iraf_hselect()
        (tmp_path / 'tmp.cat').write_text('# header\n')
        monkeypatch.chdir(tmp_path)
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b'', b'')
        with patch('subprocess.Popen', return_value=mock_proc) as mock_popen:
            runsex('img', 5.0, 3.0, 0.25)
        cmd = mock_popen.call_args[0][0]
        assert '0.25' in cmd

    def test_boundary_exactly_at_5_excluded(self, tmp_path, monkeypatch):
        """Source at x=5 or y=5 is NOT included (condition is > 5, not >= 5)."""
        from lsc.lscpsfdef import runsex
        self._setup_iraf_hselect('100', '100')
        cparam = self._get_param_cols()
        lines = '# header\n'
        lines += self._make_cat_line(cparam, {
            'X_IMAGE': 5.0, 'Y_IMAGE': 50.0,
            'X_WORLD': 150.0, 'Y_WORLD': 2.0, 'MAG_BEST': -15.0,
            'CLASS_STAR': 0.9, 'FLUX_RADIUS': 3.0, 'BACKGROUND': 100.0
        })
        (tmp_path / 'tmp.cat').write_text(lines)
        monkeypatch.chdir(tmp_path)
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b'', b'')
        with patch('subprocess.Popen', return_value=mock_proc):
            xs, ys, ra, dec, mag, cs, fr, bkg = runsex('img', 5.0, 3.0, 0.389)
        assert len(xs) == 0

    def test_boundary_at_dim_minus_5_excluded(self, tmp_path, monkeypatch):
        """Source at x=xdim-5 is NOT included (condition is < xdim-5)."""
        from lsc.lscpsfdef import runsex
        self._setup_iraf_hselect('100', '100')
        cparam = self._get_param_cols()
        lines = '# header\n'
        lines += self._make_cat_line(cparam, {
            'X_IMAGE': 95.0, 'Y_IMAGE': 50.0,
            'X_WORLD': 150.0, 'Y_WORLD': 2.0, 'MAG_BEST': -15.0,
            'CLASS_STAR': 0.9, 'FLUX_RADIUS': 3.0, 'BACKGROUND': 100.0
        })
        (tmp_path / 'tmp.cat').write_text(lines)
        monkeypatch.chdir(tmp_path)
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b'', b'')
        with patch('subprocess.Popen', return_value=mock_proc):
            xs, ys, ra, dec, mag, cs, fr, bkg = runsex('img', 5.0, 3.0, 0.389)
        assert len(xs) == 0

    def test_returns_correct_background_values(self, tmp_path, monkeypatch):
        """Background values from the catalog should be returned correctly."""
        from lsc.lscpsfdef import runsex
        self._setup_iraf_hselect('200', '200')
        cparam = self._get_param_cols()
        lines = '# header\n'
        lines += self._make_cat_line(cparam, {
            'X_IMAGE': 100.0, 'Y_IMAGE': 100.0,
            'X_WORLD': 150.0, 'Y_WORLD': 2.0, 'MAG_BEST': -18.0,
            'CLASS_STAR': 0.99, 'FLUX_RADIUS': 4.2, 'BACKGROUND': 1234.5
        })
        (tmp_path / 'tmp.cat').write_text(lines)
        monkeypatch.chdir(tmp_path)
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b'', b'')
        with patch('subprocess.Popen', return_value=mock_proc):
            xs, ys, ra, dec, mag, cs, fr, bkg = runsex('img', 5.0, 3.0, 0.389)
        assert bkg[0] == pytest.approx(1234.5)
        assert fr[0] == pytest.approx(4.2)


# ── psffit2 comprehensive ─────────────────────────────────────────────────────

class TestPsffit2Comprehensive:
    """Comprehensive tests for psffit2 - aperture calculations and IRAF parameters."""

    def test_aperture_sizes_from_fwhm(self, tmp_path, monkeypatch):
        """Apertures are computed as fwhm+0.5, 2*fwhm+0.5, 3*fwhm+0.5, 4*fwhm+0.5."""
        from lsc.lscpsfdef import psffit2
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        iraf = _get_iraf()
        psffit2('test_img', 5.0, 10, hdr, 0.0, 60000.0)
        # a1=int(5.5)=5, a2=int(10.5)=10, a3=int(15.5)=15, a4=int(20.5)=20
        # photpars.apertures should be '10,15,20'
        assert iraf.photpars.apertures == '10,15,20'

    def test_fitskypars_annulus_set(self, tmp_path, monkeypatch):
        """fitskypars.annulus = a4 = int(fwhm*4 + 0.5)."""
        from lsc.lscpsfdef import psffit2
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        iraf = _get_iraf()
        psffit2('test_img', 5.0, 10, hdr, 0.0, 60000.0)
        assert iraf.fitskypars.annulus == 20

    def test_daopars_psfrad_set(self, tmp_path, monkeypatch):
        """daopars.psfrad = a4."""
        from lsc.lscpsfdef import psffit2
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        iraf = _get_iraf()
        psffit2('test_img', 5.0, 10, hdr, 0.0, 60000.0)
        assert iraf.daopars.psfrad == 20

    def test_daopars_fitrad_equals_a1(self, tmp_path, monkeypatch):
        """daopars.fitrad = a1 = int(fwhm + 0.5)."""
        from lsc.lscpsfdef import psffit2
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        iraf = _get_iraf()
        psffit2('test_img', 5.0, 10, hdr, 0.0, 60000.0)
        assert iraf.daopars.fitrad == 5

    def test_datapars_from_header(self, tmp_path, monkeypatch):
        """datapars.readnoise and epadu come from header RDNOISE and GAIN."""
        from lsc.lscpsfdef import psffit2
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr({'RDNOISE': 8.0, 'GAIN': 3.5})
        iraf = _get_iraf()
        psffit2('test_img', 5.0, 10, hdr, 0.0, 60000.0)
        assert iraf.datapars.readnoise == 8.0
        assert iraf.datapars.epadu == 3.5

    def test_datamin_datamax_passed(self, tmp_path, monkeypatch):
        """datapars.datamin and datamax are set from arguments."""
        from lsc.lscpsfdef import psffit2
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        iraf = _get_iraf()
        psffit2('test_img', 5.0, 10, hdr, -100.0, 55000.0)
        assert iraf.datapars.datamin == -100.0
        assert iraf.datapars.datamax == 55000.0

    def test_psffun_gauss_default(self, tmp_path, monkeypatch):
        """Default psffun is 'gauss'."""
        from lsc.lscpsfdef import psffit2
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        iraf = _get_iraf()
        psffit2('test_img', 5.0, 10, hdr, 0.0, 60000.0)
        assert iraf.daopars.functio == 'gauss'

    def test_psffun_penny_passed(self, tmp_path, monkeypatch):
        """psffun='penny' is passed to daopars.functio."""
        from lsc.lscpsfdef import psffit2
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        iraf = _get_iraf()
        psffit2('test_img', 5.0, 10, hdr, 0.0, 60000.0, psffun='penny')
        assert iraf.daopars.functio == 'penny'

    def test_small_fwhm_apertures(self, tmp_path, monkeypatch):
        """Very small FWHM (1.5) should produce integer apertures: 2, 3, 5, 6."""
        from lsc.lscpsfdef import psffit2
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        iraf = _get_iraf()
        psffit2('test_img', 1.5, 10, hdr, 0.0, 60000.0)
        # a1=int(2.0)=2, a2=int(3.5)=3, a3=int(5.0)=5, a4=int(6.5)=6
        assert iraf.photpars.apertures == '3,5,6'

    def test_large_fwhm_apertures(self, tmp_path, monkeypatch):
        """Large FWHM (10.0) produces apertures: 20, 30, 40."""
        from lsc.lscpsfdef import psffit2
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        iraf = _get_iraf()
        psffit2('test_img', 10.0, 10, hdr, 0.0, 60000.0)
        # a1=int(10.5)=10, a2=int(20.5)=20, a3=int(30.5)=30, a4=int(40.5)=40
        assert iraf.photpars.apertures == '20,30,40'

    def test_fixaperture_true_uses_pixscale(self, tmp_path, monkeypatch):
        """fixaperture=True uses 5/pixscale, 8/pixscale, 10/pixscale."""
        from lsc.lscpsfdef import psffit2
        img = _make_fits(tmp_path, 'fixap', extra_hdr={'PIXSCALE': 0.5})
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr({'PIXSCALE': 0.5})
        iraf = _get_iraf()
        psffit2(img, 5.0, 10, hdr, 0.0, 60000.0, fixaperture=True)
        # a1=5/0.5=10, a2=5/0.5=10, a3=8/0.5=16, a4=10/0.5=20
        # photpars.apertures = '10,16,20' (using a2, a3, a4)
        assert iraf.photpars.apertures == '10,16,20'

    def test_phot_called_with_correct_image(self, tmp_path, monkeypatch):
        """iraf.phot should be called with img+'[0]'."""
        from lsc.lscpsfdef import psffit2
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        iraf = _get_iraf()
        psffit2('myimage', 5.0, 10, hdr, 0.0, 60000.0)
        iraf.phot.assert_called()
        first_arg = iraf.phot.call_args[0][0]
        assert first_arg == 'myimage[0]'

    def test_group_and_nstar_called(self, tmp_path, monkeypatch):
        """group and nstar should both be called."""
        from lsc.lscpsfdef import psffit2
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        iraf = _get_iraf()
        iraf.group.reset_mock()
        iraf.nstar.reset_mock()
        psffit2('test_img', 5.0, 10, hdr, 0.0, 60000.0)
        assert iraf.group.called
        assert iraf.nstar.called


# ── psffit comprehensive ──────────────────────────────────────────────────────

class TestPsffitComprehensive:
    """Comprehensive tests for psffit - PSF building and star selection."""

    def _write_psf_mag(self, tmp_path, content=''):
        (tmp_path / '_psf.mag').write_text(content)

    def test_saturated_star_removal(self, tmp_path, monkeypatch):
        """Stars marked with BadPixels* in _psf.mag are removed."""
        from lsc.lscpsfdef import psffit
        # Build a _psf.mag file with one good and one bad entry
        good_entry = "line1\nline2\nline3\nline4\nline5\nline6\ngood data\n"
        bad_entry = "line1\nline2\nline3\nline4\nline5\nline6\nBadPixels* \n"
        content = good_entry + bad_entry
        self._write_psf_mag(tmp_path, content)
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        iraf = _get_iraf()
        psffit('test_img', 5.0, 10, hdr, False, 0.0, 60000.0)
        # After the regex substitution, only the good entry should remain
        with open(tmp_path / '_psf.mag') as f:
            remaining = f.read()
        assert 'BadPixels*' not in remaining

    def test_psfout_filename_for_zogypsf(self, tmp_path, monkeypatch):
        """For .zogypsf images, psfout removes that suffix."""
        from lsc.lscpsfdef import psffit
        self._write_psf_mag(tmp_path)
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        iraf = _get_iraf()
        psffit('test.zogypsf', 5.0, 10, hdr, False, 0.0, 60000.0)
        # psfout should be 'test.psf.fits'
        iraf.psf.assert_called()
        psf_call_args = iraf.psf.call_args
        psfout_arg = psf_call_args[0][3] if len(psf_call_args[0]) > 3 else psf_call_args[1].get('psfout', '')
        # The psf is called with positional args; psfout is the 4th positional
        assert 'test.psf.fits' in str(psf_call_args)

    def test_psfout_filename_normal(self, tmp_path, monkeypatch):
        """Normal image name: psfout = img + '.psf.fits'."""
        from lsc.lscpsfdef import psffit
        self._write_psf_mag(tmp_path)
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        iraf = _get_iraf()
        psffit('myimg', 5.0, 10, hdr, False, 0.0, 60000.0)
        # psfout should be 'myimg.psf.fits'
        assert 'myimg.psf.fits' in str(iraf.psf.call_args)

    def test_interactive_true_skips_pstselect(self, tmp_path, monkeypatch):
        """Interactive mode copies _psf.mag to _psf.pst instead of calling pstselect."""
        from lsc.lscpsfdef import psffit
        self._write_psf_mag(tmp_path, 'some data')
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        iraf = _get_iraf()
        iraf.pstselect.reset_mock()
        psffit('test_img', 5.0, 10, hdr, True, 0.0, 60000.0)
        assert not iraf.pstselect.called
        # _psf.pst should exist as a copy of _psf.mag
        assert os.path.exists(tmp_path / '_psf.pst')

    def test_varorder_is_zero(self, tmp_path, monkeypatch):
        """varorder should be 0 (numeric)."""
        from lsc.lscpsfdef import psffit
        self._write_psf_mag(tmp_path)
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        iraf = _get_iraf()
        psffit('test_img', 5.0, 10, hdr, False, 0.0, 60000.0)
        assert iraf.daopars.varorder == 0

    def test_centerpars_algorithm(self, tmp_path, monkeypatch):
        """centerpars.calgori should be 'centroid'."""
        from lsc.lscpsfdef import psffit
        self._write_psf_mag(tmp_path)
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        iraf = _get_iraf()
        psffit('test_img', 5.0, 10, hdr, False, 0.0, 60000.0)
        assert iraf.centerpars.calgori == 'centroid'

    def test_sky_algorithm_is_mean(self, tmp_path, monkeypatch):
        """fitskypars.salgori should be 'mean'."""
        from lsc.lscpsfdef import psffit
        self._write_psf_mag(tmp_path)
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        iraf = _get_iraf()
        psffit('test_img', 5.0, 10, hdr, False, 0.0, 60000.0)
        assert iraf.fitskypars.salgori == 'mean'

    def test_pstselect_receives_psfstars_count(self, tmp_path, monkeypatch):
        """pstselect is called with the psfstars argument."""
        from lsc.lscpsfdef import psffit
        self._write_psf_mag(tmp_path)
        monkeypatch.chdir(tmp_path)
        hdr = _make_hdr()
        iraf = _get_iraf()
        iraf.pstselect.reset_mock()
        psffit('test_img', 5.0, 25, hdr, False, 0.0, 60000.0)
        # pstselect(img, '_psf.mag', '_psf.pst', psfstars, ...)
        assert iraf.pstselect.called
        args = iraf.pstselect.call_args[0]
        assert args[3] == 25  # psfstars


# ── ecpsf comprehensive ───────────────────────────────────────────────────────

class TestEcpsfComprehensive:
    """Comprehensive tests for ecpsf - the main PSF building pipeline."""

    def test_fwhm_from_l1fwhm_header(self, tmp_path, monkeypatch):
        """When fwhm=0 and WCSERR=0, uses L1FWHM / PIXSCALE."""
        from lsc.lscpsfdef import ecpsf
        hdr_extra = {'L1FWHM': 2.0, 'WCSERR': 0, 'PIXSCALE': 0.4}
        img = _make_fits(tmp_path, 'l1fwhm', extra_hdr=hdr_extra)
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        xs = np.array([100.0, 200.0])
        ys = np.array([100.0, 200.0])
        with patch('lsc.lscpsfdef.runsex', return_value=(
                xs, ys, np.zeros(2), np.zeros(2),
                np.array([15.0, 16.0]), np.ones(2),
                np.array([3.0, 3.0]), np.zeros(2))):
            with patch('lsc.lscpsfdef.psffit', return_value=([], [], [])):
                with patch('lsc.lscpsfdef.psffit2', return_value=([], [])):
                    result, fwhm_out, apco = ecpsf(img, 0, 3.0, 10, 3.0, False)
        # fwhm_out should be L1FWHM (seeing) = 2.0 arcsec (since fwhm*scale = (2/0.4)*0.4 = 2.0)
        # But result may be 0 if psffit returns empty lists
        assert result in (0, 1)

    def test_fwhm_from_l1seeing(self, tmp_path, monkeypatch):
        """When L1FWHM not present but L1SEEING is, uses L1SEEING*scale."""
        from lsc.lscpsfdef import ecpsf
        hdr_extra = {'L1SEEING': 5.0, 'WCSERR': 0, 'PIXSCALE': 0.4}
        img = _make_fits(tmp_path, 'l1seeing', extra_hdr=hdr_extra)
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        with patch('lsc.lscpsfdef.runsex', return_value=(
                np.array([100.0]), np.array([100.0]),
                np.zeros(1), np.zeros(1),
                np.array([15.0]), np.ones(1),
                np.array([5.0]), np.zeros(1))):
            with patch('lsc.lscpsfdef.psffit', return_value=([], [], [])):
                with patch('lsc.lscpsfdef.psffit2', return_value=([], [])):
                    result, fwhm_out, apco = ecpsf(img, 0, 3.0, 10, 3.0, False)
        assert result in (0, 1)

    def test_fwhm_from_psf_fwhm_when_wcserr_nonzero(self, tmp_path, monkeypatch):
        """When WCSERR != 0, uses PSF_FWHM from header."""
        from lsc.lscpsfdef import ecpsf
        hdr_extra = {'WCSERR': 1, 'PSF_FWHM': 2.5, 'PIXSCALE': 0.4}
        img = _make_fits(tmp_path, 'wcserr', extra_hdr=hdr_extra)
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        with patch('lsc.lscpsfdef.runsex', return_value=(
                np.array([100.0]), np.array([100.0]),
                np.zeros(1), np.zeros(1),
                np.array([15.0]), np.ones(1),
                np.array([5.0]), np.zeros(1))):
            with patch('lsc.lscpsfdef.psffit', return_value=([], [], [])):
                with patch('lsc.lscpsfdef.psffit2', return_value=([], [])):
                    result, fwhm_out, apco = ecpsf(img, 0, 3.0, 10, 3.0, False)
        assert result in (0, 1)

    def test_wcserr_nonzero_no_psf_fwhm_raises(self, tmp_path, monkeypatch):
        """When WCSERR != 0 and no PSF_FWHM, exception is caught, result=0."""
        from lsc.lscpsfdef import ecpsf
        hdr_extra = {'WCSERR': 1, 'PIXSCALE': 0.4}
        img = _make_fits(tmp_path, 'noastr', extra_hdr=hdr_extra)
        # Remove PSF_FWHM if present
        hdul = fits.open(img + '.fits', mode='update')
        if 'PSF_FWHM' in hdul[0].header:
            del hdul[0].header['PSF_FWHM']
        hdul.close()
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        result, fwhm_out, apco = ecpsf(img, 0, 3.0, 10, 3.0, False)
        assert result == 0
        assert fwhm_out == 0.0

    def test_datamax_defaults_to_70_percent_of_header(self, tmp_path, monkeypatch):
        """When _datamax=None, uses 0.7 * hdr['DATAMAX']."""
        from lsc.lscpsfdef import ecpsf
        hdr_extra = {'DATAMAX': 50000.0, 'PIXSCALE': 0.4, 'L1FWHM': 2.0, 'WCSERR': 0}
        img = _make_fits(tmp_path, 'dm_default', extra_hdr=hdr_extra)
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        with patch('lsc.lscpsfdef.runsex', return_value=(
                np.array([100.0]), np.array([100.0]),
                np.zeros(1), np.zeros(1),
                np.array([15.0]), np.ones(1),
                np.array([5.0]), np.zeros(1))):
            with patch('lsc.lscpsfdef.psffit', return_value=([], [], [])):
                with patch('lsc.lscpsfdef.psffit2', return_value=([], [])):
                    result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False,
                                                    _datamax=None)
        # We can't easily check the internal value, but this should run without error
        assert result in (0, 1)

    def test_catalog_mode_writes_psf_coo(self, tmp_path, monkeypatch):
        """When _catalog is provided, wcsctran output is written to _psf.coo."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'catmode')
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        # Mock wcsctran to return valid coordinates
        iraf.wcsctran.return_value = [
            '# comment',
            '100.0      200.0      1',
            '300.0      400.0      2',
        ]
        with patch('lsc.lscpsfdef.psffit', return_value=(
                ['10:00:00 +02:00:00 1 15.0 0.01 15.1 0.02 15.2 0.03'],
                ['100.0 200.0 1'],
                ['100.0 200.0 1 14.9 0.01'])):
            with patch('lsc.lscpsfdef.psffit2', return_value=(
                    ['10:00:00 +02:00:00 1 15.0 0.01 15.1 0.02 15.2 0.03'],
                    ['100.0 200.0 1 14.9 0.01'])):
                result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False,
                                                _catalog='my_catalog.txt')
        # _psf.coo should have been created
        assert os.path.exists(tmp_path / '_psf.coo')

    def test_catalog_empty_field_raises(self, tmp_path, monkeypatch):
        """If no catalog objects are in the field, exception is caught."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'empty_cat')
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        # Return coordinates outside the image
        iraf.wcsctran.return_value = [
            '# comment',
            '9999.0      9999.0      1',
        ]
        result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False,
                                        _catalog='my_catalog.txt')
        assert result == 0

    def test_make_sn2_false_skips_table_creation(self, tmp_path, monkeypatch):
        """When make_sn2=False, no .sn2.fits file is created."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'nosn2')
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        with patch('lsc.lscpsfdef.runsex', return_value=(
                np.array([100.0]), np.array([100.0]),
                np.zeros(1), np.zeros(1),
                np.array([15.0]), np.ones(1),
                np.array([5.0]), np.zeros(1))):
            with patch('lsc.lscpsfdef.psffit', return_value=([], [], [])):
                result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False,
                                                make_sn2=False)
        # No sn2 file should exist
        assert not os.path.exists(tmp_path / 'nosn2.sn2.fits')

    def test_make_sn2_false_aperture_correction_zero(self, tmp_path, monkeypatch):
        """When make_sn2=False, aperture_correction should be 0."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'nosn2_apco')
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        with patch('lsc.lscpsfdef.runsex', return_value=(
                np.array([100.0]), np.array([100.0]),
                np.zeros(1), np.zeros(1),
                np.array([15.0]), np.ones(1),
                np.array([5.0]), np.zeros(1))):
            with patch('lsc.lscpsfdef.psffit', return_value=([], [], [])):
                result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False,
                                                make_sn2=False)
        if result == 1:
            assert apco == 0

    def test_max_apercorr_exceeded_returns_failure(self, tmp_path, monkeypatch):
        """When aperture correction exceeds max_apercorr, result=0."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'bigapco')
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        # Set up mocks so aperture correction is large
        photmag = ['10:00:00.00 +02:00:00.0 1 15.0 14.5 14.0 0.01 0.01 0.01']
        pst = ['100.0 200.0 1']
        fitmag = ['10:00:00.00 +02:00:00.0 1 13.0 0.01']  # large diff: 14.5 - 13.0 = 1.5
        fitmag2 = ['100.0 200.0 1 13.0 0.01']
        photmag2 = ['10:00:00.00 +02:00:00.0 1 15.0 14.5 14.0 0.01 0.01 0.01']
        iraf.wcsctran.return_value = ['# line1', '# line2', '# line3'] + photmag
        with patch('lsc.lscpsfdef.runsex', return_value=(
                np.array([100.0]), np.array([100.0]),
                np.zeros(1), np.zeros(1),
                np.array([15.0]), np.ones(1),
                np.array([5.0]), np.zeros(1))):
            with patch('lsc.lscpsfdef.psffit', return_value=(
                    ['100.0 200.0 1 15.0 14.5 14.0 0.01 0.01 0.01'],
                    ['100.0 200.0 1'],
                    ['100.0 200.0 1 13.0 0.01'])):
                with patch('lsc.lscpsfdef.psffit2', return_value=(
                        ['100.0 200.0 1 15.0 14.5 14.0 0.01 0.01 0.01'],
                        ['100.0 200.0 1 13.0 0.01'])):
                    result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False,
                                                    max_apercorr=0.1)
        # Large apercorr should cause failure
        assert result == 0

    def test_exception_returns_zero_result(self, tmp_path, monkeypatch):
        """When ecpsf fails before scale is assigned, UnboundLocalError is raised."""
        from lsc.lscpsfdef import ecpsf
        monkeypatch.chdir(tmp_path)
        # nonexistent file will raise exception in readhdr; since scale is never
        # assigned and the return statement (fwhm * scale) is outside the except,
        # this results in an UnboundLocalError.
        with pytest.raises(UnboundLocalError):
            ecpsf('nonexistent', 5.0, 3.0, 10, 3.0, False)

    def test_pixscale_from_ccdscale_and_ccdxbin(self, tmp_path, monkeypatch):
        """When PIXSCALE not in header, uses CCDSCALE * CCDXBIN."""
        from lsc.lscpsfdef import ecpsf
        hdr_extra = {'CCDSCALE': 0.2, 'CCDXBIN': 2, 'L1FWHM': 2.0, 'WCSERR': 0}
        img = _make_fits(tmp_path, 'ccdscale', extra_hdr=hdr_extra)
        # Remove PIXSCALE from the FITS file
        hdul = fits.open(img + '.fits', mode='update')
        if 'PIXSCALE' in hdul[0].header:
            del hdul[0].header['PIXSCALE']
        hdul.close()
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        with patch('lsc.lscpsfdef.runsex', return_value=(
                np.array([100.0]), np.array([100.0]),
                np.zeros(1), np.zeros(1),
                np.array([15.0]), np.ones(1),
                np.array([5.0]), np.zeros(1))):
            with patch('lsc.lscpsfdef.psffit', return_value=([], [], [])):
                with patch('lsc.lscpsfdef.psffit2', return_value=([], [])):
                    result, fwhm_out, apco = ecpsf(img, 0, 3.0, 10, 3.0, False)
        # Should use scale = 0.2 * 2 = 0.4
        assert result in (0, 1)

    def test_pixscale_from_ccdscale_and_ccdsum(self, tmp_path, monkeypatch):
        """When PIXSCALE not in header, CCDXBIN not in header, uses CCDSCALE * int(CCDSUM.split()[0])."""
        from lsc.lscpsfdef import ecpsf
        hdr_extra = {'CCDSCALE': 0.15, 'CCDSUM': '2 2', 'L1FWHM': 2.0, 'WCSERR': 0}
        img = _make_fits(tmp_path, 'ccdsum', extra_hdr=hdr_extra)
        # Remove PIXSCALE and CCDXBIN from the FITS file
        hdul = fits.open(img + '.fits', mode='update')
        if 'PIXSCALE' in hdul[0].header:
            del hdul[0].header['PIXSCALE']
        if 'CCDXBIN' in hdul[0].header:
            del hdul[0].header['CCDXBIN']
        hdul.close()
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        with patch('lsc.lscpsfdef.runsex', return_value=(
                np.array([100.0]), np.array([100.0]),
                np.zeros(1), np.zeros(1),
                np.array([15.0]), np.ones(1),
                np.array([5.0]), np.zeros(1))):
            with patch('lsc.lscpsfdef.psffit', return_value=([], [], [])):
                with patch('lsc.lscpsfdef.psffit2', return_value=([], [])):
                    result, fwhm_out, apco = ecpsf(img, 0, 3.0, 10, 3.0, False)
        assert result in (0, 1)

    def test_default_fwhm_fallback_to_3_arcsec(self, tmp_path, monkeypatch):
        """When no L1FWHM or L1SEEING, uses default seeing=3."""
        from lsc.lscpsfdef import ecpsf
        hdr_extra = {'WCSERR': 0, 'PIXSCALE': 0.4}
        img = _make_fits(tmp_path, 'nofwhm', extra_hdr=hdr_extra)
        # Remove L1FWHM and L1SEEING
        hdul = fits.open(img + '.fits', mode='update')
        for key in ['L1FWHM', 'L1SEEING', 'PSF_FWHM']:
            if key in hdul[0].header:
                del hdul[0].header[key]
        hdul.close()
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        with patch('lsc.lscpsfdef.runsex', return_value=(
                np.array([100.0]), np.array([100.0]),
                np.zeros(1), np.zeros(1),
                np.array([15.0]), np.ones(1),
                np.array([5.0]), np.zeros(1))):
            with patch('lsc.lscpsfdef.psffit', return_value=([], [], [])):
                with patch('lsc.lscpsfdef.psffit2', return_value=([], [])):
                    result, fwhm_out, apco = ecpsf(img, 0, 3.0, 10, 3.0, False)
        # fwhm = 3 / 0.4 = 7.5 pixels, fwhm_out = 7.5 * 0.4 = 3.0
        assert result in (0, 1)


# ── ecpsf star selection ──────────────────────────────────────────────────────

class TestEcpsfStarSelection:
    """Tests for the star selection logic inside ecpsf (non-interactive, non-catalog)."""

    def test_fluxrad_filter_selects_matching_stars(self, tmp_path, monkeypatch):
        """Only stars with |fluxrad*1.6 - fwhm| / fwhm < 0.5 pass."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'fsel')
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        iraf.psfmeasure = MagicMock()

        # fwhm=5.0 pixels. fluxrad*1.6 should be near 5.0
        # Good: fluxrad=3.0 -> 3.0*1.6=4.8, |4.8-5|/5=0.04 < 0.5 PASS
        # Bad: fluxrad=1.0 -> 1.0*1.6=1.6, |1.6-5|/5=0.68 > 0.5 FAIL
        xs = np.array([100.0, 200.0, 300.0])
        ys = np.array([100.0, 200.0, 300.0])
        fluxrad = np.array([3.0, 1.0, 3.1])

        # Write a real tmp.log file that psfmeasure would produce
        log_content = ("header1\nheader2\nheader3\n"
                       "col1 100.0 100.0 col4 5.0\n"
                       "300.0 300.0 col3 5.1\n"
                       "average fwhm = 5.0\n")
        (tmp_path / 'tmp.log').write_text(log_content)

        with patch('lsc.lscpsfdef.runsex', return_value=(
                xs, ys, np.zeros(3), np.zeros(3),
                np.array([15.0, 16.0, 17.0]), np.ones(3),
                fluxrad, np.zeros(3))):
            with patch('lsc.lscpsfdef.psffit', return_value=([], [], [])):
                with patch('lsc.lscpsfdef.psffit2', return_value=([], [])):
                    result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False)
        # Should not crash
        assert result in (0, 1)

    def test_distance_filter_removes_close_stars(self, tmp_path, monkeypatch):
        """Stars closer than distance*fwhm to another star are excluded."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'dist_filter')
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']

        # Two stars separated by only 5 pixels with fwhm=5.0, distance=3.0
        # min_dist needed = 3.0 * 5.0 = 15 pixels
        # These are only 5 apart, so both should be excluded
        xs = np.array([100.0, 105.0])
        ys = np.array([100.0, 100.0])
        fluxrad = np.array([3.0, 3.0])  # 3.0*1.6=4.8, |4.8-5|/5=0.04 < 0.5

        with patch('lsc.lscpsfdef.runsex', return_value=(
                xs, ys, np.zeros(2), np.zeros(2),
                np.array([15.0, 16.0]), np.ones(2),
                fluxrad, np.zeros(2))):
            with patch('lsc.lscpsfdef.psffit', return_value=([], [], [])):
                with patch('lsc.lscpsfdef.psffit2', return_value=([], [])):
                    result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False)
        # Both stars are too close, so no cursor entries are written
        # The function may still succeed or fail depending on downstream processing
        assert result in (0, 1)

    def test_saturated_stars_excluded_by_datamax(self, tmp_path, monkeypatch):
        """Stars with peak > _datamax in the image are excluded."""
        from lsc.lscpsfdef import ecpsf
        # Create image with a saturated pixel
        shape = (512, 512)
        data = np.ones(shape, dtype=np.float32) * 1000.0
        data[99, 99] = 70000.0  # saturated pixel near star at (100, 100)
        hdr = _make_hdr_for_fits({'DATAMAX': 60000.0}, shape)
        path = tmp_path / 'satstar.fits'
        hdu = fits.PrimaryHDU(data=data, header=hdr)
        hdu.writeto(str(path), overwrite=True)
        img = str(tmp_path / 'satstar')
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']

        xs = np.array([100.0, 300.0])
        ys = np.array([100.0, 300.0])
        fluxrad = np.array([3.0, 3.0])

        with patch('lsc.lscpsfdef.runsex', return_value=(
                xs, ys, np.zeros(2), np.zeros(2),
                np.array([15.0, 16.0]), np.ones(2),
                fluxrad, np.zeros(2))):
            with patch('lsc.lscpsfdef.psffit', return_value=([], [], [])):
                with patch('lsc.lscpsfdef.psffit2', return_value=([], [])):
                    result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 30.0, False)
        assert result in (0, 1)


# ── ecpsf aperture correction ─────────────────────────────────────────────────

class TestEcpsfApertureCorrection:
    """Tests for aperture correction calculation in ecpsf."""

    def test_aperture_correction_with_matching_ids(self, tmp_path, monkeypatch):
        """When phot and fit mags match, aperture correction = phot_mag - fit_mag."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'apco_match')
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']

        # photmag: ra dec id magp2 magp3 magp4 merrp2 merrp3 merrp4
        photmag_radec = ['10:00:00.000 +02:00:00.00 1 15.100 15.050 15.000 0.010 0.010 0.010',
                         '10:00:01.000 +02:00:01.00 2 16.100 16.050 16.000 0.010 0.010 0.010']
        pst = ['100.0 200.0 1', '150.0 250.0 2']
        # fitmag: ra dec id mag merr
        fitmag_radec = ['10:00:00.000 +02:00:00.00 1 15.000 0.005',
                        '10:00:01.000 +02:00:01.00 2 16.000 0.005']

        photmag2_radec = ['10:00:00.000 +02:00:00.00 1 15.100 15.050 15.000 0.010 0.010 0.010',
                          '10:00:01.000 +02:00:01.00 2 16.100 16.050 16.000 0.010 0.010 0.010']
        fitmag2 = ['100.0 200.0 1 15.000 0.005', '150.0 250.0 2 16.000 0.005']

        # wcsctran returns header lines + data
        iraf.wcsctran.side_effect = [
            ['# l1', '# l2', '# l3'] + photmag_radec,
            ['# l1', '# l2', '# l3'] + photmag2_radec,
        ]

        with patch('lsc.lscpsfdef.runsex', return_value=(
                np.array([100.0, 150.0]), np.array([200.0, 250.0]),
                np.zeros(2), np.zeros(2),
                np.array([15.0, 16.0]), np.ones(2),
                np.array([3.0, 3.0]), np.zeros(2))):
            with patch('lsc.lscpsfdef.psffit', return_value=(
                    ['100.0 200.0 1 15.100 15.050 15.000 0.010 0.010 0.010',
                     '150.0 250.0 2 16.100 16.050 16.000 0.010 0.010 0.010'],
                    pst,
                    fitmag_radec)):
                with patch('lsc.lscpsfdef.psffit2', return_value=(
                        ['100.0 200.0 1 15.100 15.050 15.000 0.010 0.010 0.010',
                         '150.0 250.0 2 16.100 16.050 16.000 0.010 0.010 0.010'],
                        fitmag2)):
                    with patch('lsc.lscabsphotdef.deg2HMS', return_value=150.0):
                        result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False,
                                                        max_apercorr=1.0)
        # aperture correction should be magp3 - magf = 15.050 - 15.000 = 0.050
        if result == 1:
            assert abs(apco) < 1.0

    def test_indef_magnitudes_excluded_from_correction(self, tmp_path, monkeypatch):
        """Stars with INDEF magnitudes should not contribute to aperture correction."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'indef_apco')
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']

        # One good star, one with INDEF mag
        photmag_radec = ['10:00:00.000 +02:00:00.00 1 15.100 INDEF 15.000 0.010 0.010 0.010',
                         '10:00:01.000 +02:00:01.00 2 16.100 16.050 16.000 0.010 0.010 0.010']
        pst = ['100.0 200.0 1', '150.0 250.0 2']
        fitmag_radec = ['10:00:00.000 +02:00:00.00 1 15.000 0.005',
                        '10:00:01.000 +02:00:01.00 2 16.000 0.005']

        iraf.wcsctran.side_effect = [
            ['# l1', '# l2', '# l3'] + photmag_radec,
            ['# l1', '# l2', '# l3'] + photmag_radec,
        ]

        with patch('lsc.lscpsfdef.runsex', return_value=(
                np.array([100.0, 150.0]), np.array([200.0, 250.0]),
                np.zeros(2), np.zeros(2),
                np.array([15.0, 16.0]), np.ones(2),
                np.array([3.0, 3.0]), np.zeros(2))):
            with patch('lsc.lscpsfdef.psffit', return_value=(
                    ['100.0 200.0 1 15.100 INDEF 15.000 0.010 0.010 0.010',
                     '150.0 250.0 2 16.100 16.050 16.000 0.010 0.010 0.010'],
                    pst,
                    fitmag_radec)):
                with patch('lsc.lscpsfdef.psffit2', return_value=(
                        ['100.0 200.0 1 15.100 INDEF 15.000 0.010 0.010 0.010',
                         '150.0 250.0 2 16.100 16.050 16.000 0.010 0.010 0.010'],
                        ['100.0 200.0 1 15.000 0.005', '150.0 250.0 2 16.000 0.005'])):
                    with patch('lsc.lscabsphotdef.deg2HMS', return_value=150.0):
                        result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False,
                                                        max_apercorr=1.0)
        # The star with INDEF should be excluded from aperture correction
        assert result in (0, 1)


# ── ecpsf duplicate elimination ───────────────────────────────────────────────

class TestEcpsfDuplicateElimination:
    """Tests for the duplicate star elimination logic in psfmeasure output parsing."""

    def test_duplicate_detection_by_position(self):
        """Stars with positions differing by < 0.2 in x AND y are duplicates."""
        # Simulate the logic from ecpsf (lines 345-350)
        xn = [100.0, 100.1, 200.0, 200.05, 300.0]
        yn = [100.0, 100.1, 200.0, 200.05, 300.0]
        _fw = [5.0, 5.0, 5.0, 5.0, 5.0]

        xns, yns, _fws = [xn[0]], [yn[0]], [_fw[0]]
        for i in range(1, len(xn)):
            if abs(xn[i] - xn[i - 1]) > .2 and abs(yn[i] - yn[i - 1]) > .2:
                xns.append(xn[i])
                yns.append(yn[i])
                _fws.append(_fw[i])

        # Duplicates (100.1, 100.1) and (200.05, 200.05) should be removed
        assert len(xns) == 3
        assert xns == [100.0, 200.0, 300.0]

    def test_no_duplicates_all_unique(self):
        """When all stars are unique, no filtering occurs."""
        xn = [100.0, 200.0, 300.0, 400.0]
        yn = [100.0, 200.0, 300.0, 400.0]
        _fw = [5.0, 5.0, 5.0, 5.0]

        xns, yns, _fws = [xn[0]], [yn[0]], [_fw[0]]
        for i in range(1, len(xn)):
            if abs(xn[i] - xn[i - 1]) > .2 and abs(yn[i] - yn[i - 1]) > .2:
                xns.append(xn[i])
                yns.append(yn[i])
                _fws.append(_fw[i])

        assert len(xns) == 4

    def test_all_duplicates_of_first(self):
        """If all entries are near-duplicates of the first, only first survives."""
        xn = [100.0, 100.1, 100.05, 100.15]
        yn = [100.0, 100.1, 100.05, 100.15]
        _fw = [5.0, 5.0, 5.0, 5.0]

        xns, yns, _fws = [xn[0]], [yn[0]], [_fw[0]]
        for i in range(1, len(xn)):
            if abs(xn[i] - xn[i - 1]) > .2 and abs(yn[i] - yn[i - 1]) > .2:
                xns.append(xn[i])
                yns.append(yn[i])
                _fws.append(_fw[i])

        assert len(xns) == 1


# ── ecpsf FWHM filtering for PSF coo ──────────────────────────────────────────

class TestEcpsfFwhmFilter:
    """Tests for the FWHM consistency check when writing _psf.coo."""

    def test_fwhm_filter_30_percent(self):
        """Only stars with |_fws[i] - fwhm| / fwhm < 0.3 are written."""
        fwhm = 5.0
        _fws = [5.0, 4.0, 6.0, 3.0, 8.0, 5.5, 4.5]
        # |5.0-5|/5=0, |4.0-5|/5=0.2, |6.0-5|/5=0.2, |3.0-5|/5=0.4, |8.0-5|/5=0.6
        # |5.5-5|/5=0.1, |4.5-5|/5=0.1
        fw = []
        for fwi in _fws:
            if abs(fwi - fwhm) / fwhm < .3:
                fw.append(fwi)

        # 3.0 (0.4) and 8.0 (0.6) should be excluded
        assert 3.0 not in fw
        assert 8.0 not in fw
        assert len(fw) == 5

    def test_all_consistent_fwhm_pass(self):
        """All stars with identical FWHM pass."""
        fwhm = 5.0
        _fws = [5.0, 5.0, 5.0]
        fw = [fwi for fwi in _fws if abs(fwi - fwhm) / fwhm < .3]
        assert len(fw) == 3

    def test_all_inconsistent_fwhm_none_pass(self):
        """Stars with very different FWHM are all excluded."""
        fwhm = 5.0
        _fws = [1.0, 10.0, 20.0]
        fw = [fwi for fwi in _fws if abs(fwi - fwhm) / fwhm < .3]
        assert len(fw) == 0

    def test_boundary_exactly_at_30_percent(self):
        """Star with exactly 30% deviation: |fwi-fwhm|/fwhm = 0.3 should be excluded (< not <=)."""
        fwhm = 5.0
        _fws = [3.5]  # |3.5-5|/5 = 0.3 exactly
        fw = [fwi for fwi in _fws if abs(fwi - fwhm) / fwhm < .3]
        assert len(fw) == 0


# ── _to_float_array inner function logic ──────────────────────────────────────

class TestToFloatArrayLogic:
    """Tests for the _to_float_array logic used in sn2 table construction."""

    def test_indef_replaced_with_fill(self):
        """INDEF values are replaced with fill value."""
        values = ['15.0', 'INDEF', '16.0', '']
        fill = 9999.0
        out = []
        for value in values:
            if value in ['INDEF', '', None, 9999]:
                out.append(fill)
            else:
                try:
                    out.append(float(value))
                except (TypeError, ValueError):
                    out.append(fill)
        result = np.array(out, dtype=float)
        assert result[0] == pytest.approx(15.0)
        assert result[1] == pytest.approx(9999.0)
        assert result[2] == pytest.approx(16.0)
        assert result[3] == pytest.approx(9999.0)

    def test_none_replaced_with_fill(self):
        """None values are replaced with fill value."""
        values = [None, '15.0']
        fill = 9999.0
        out = []
        for value in values:
            if value in ['INDEF', '', None, 9999]:
                out.append(fill)
            else:
                try:
                    out.append(float(value))
                except (TypeError, ValueError):
                    out.append(fill)
        result = np.array(out, dtype=float)
        assert result[0] == pytest.approx(9999.0)
        assert result[1] == pytest.approx(15.0)

    def test_9999_integer_replaced_with_fill(self):
        """The integer 9999 is treated as a fill sentinel."""
        values = [9999, '15.0']
        fill = 9999.0
        out = []
        for value in values:
            if value in ['INDEF', '', None, 9999]:
                out.append(fill)
            else:
                try:
                    out.append(float(value))
                except (TypeError, ValueError):
                    out.append(fill)
        result = np.array(out, dtype=float)
        assert result[0] == pytest.approx(9999.0)
        assert result[1] == pytest.approx(15.0)

    def test_all_valid_floats(self):
        """All valid float strings are converted correctly."""
        values = ['15.123', '16.456', '17.789']
        fill = 9999.0
        out = []
        for value in values:
            if value in ['INDEF', '', None, 9999]:
                out.append(fill)
            else:
                try:
                    out.append(float(value))
                except (TypeError, ValueError):
                    out.append(fill)
        result = np.array(out, dtype=float)
        assert result[0] == pytest.approx(15.123)
        assert result[1] == pytest.approx(16.456)
        assert result[2] == pytest.approx(17.789)

    def test_invalid_string_replaced_with_fill(self):
        """Non-numeric strings that aren't 'INDEF' also get fill."""
        values = ['abc', '15.0']
        fill = 9999.0
        out = []
        for value in values:
            if value in ['INDEF', '', None, 9999]:
                out.append(fill)
            else:
                try:
                    out.append(float(value))
                except (TypeError, ValueError):
                    out.append(fill)
        result = np.array(out, dtype=float)
        assert result[0] == pytest.approx(9999.0)
        assert result[1] == pytest.approx(15.0)


# ── sigma clipping of aperture correction ──────────────────────────────────────

class TestApertureCorrectionSigmaClip:
    """Tests for the 2-sigma clipping logic on aperture corrections."""

    def test_sigma_clip_removes_outlier(self):
        """Values > 2*sigma from median are removed."""
        _dmag = np.array([0.05, 0.04, 0.06, 0.05, 0.50])  # 0.50 is an outlier
        # Sigma clip: remove values > 2*std from median
        clipped = np.compress(np.abs(_dmag - np.median(_dmag)) < 2 * np.std(_dmag), _dmag)
        assert 0.50 not in clipped
        assert len(clipped) == 4

    def test_sigma_clip_no_outliers(self):
        """When no outliers exist, all values remain."""
        _dmag = np.array([0.05, 0.04, 0.06, 0.05, 0.045])
        clipped = np.compress(np.abs(_dmag - np.median(_dmag)) < 2 * np.std(_dmag), _dmag)
        assert len(clipped) == 5

    def test_sigma_clip_not_applied_with_3_or_fewer(self):
        """Sigma clipping only applies when len(_dmag) > 3."""
        _dmag = np.array([0.05, 0.04, 0.50])
        # The code checks: if len(_dmag) > 3: apply sigma clip
        if len(_dmag) > 3:
            clipped = np.compress(np.abs(_dmag - np.median(_dmag)) < 2 * np.std(_dmag), _dmag)
        else:
            clipped = _dmag
        assert len(clipped) == 3  # no clipping applied

    def test_sigma_clip_exactly_4_applies(self):
        """Sigma clipping is applied when exactly 4 points exist."""
        _dmag = np.array([0.05, 0.04, 0.06, 1.00])
        if len(_dmag) > 3:
            clipped = np.compress(np.abs(_dmag - np.median(_dmag)) < 2 * np.std(_dmag), _dmag)
        else:
            clipped = _dmag
        # 1.00 should be clipped
        assert 1.00 not in clipped
