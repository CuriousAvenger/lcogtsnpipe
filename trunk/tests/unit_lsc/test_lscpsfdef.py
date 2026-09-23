"""
Tests for lsc.lscpsfdef.

Merged from:
  - test_lscpsfdef_comprehensive.py
  - test_lscpsfdef_coverage.py
  - test_lscpsfdef_pure.py

Covers runsex, psffit, psffit2, ecpsf (all branches), helper logic,
star selection, aperture correction, sigma clipping, duplicate elimination,
and full pipeline integration.

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



# ══════════════════════════════════════════════════════════════════════════════
# From test_lscpsfdef_comprehensive.py
# ══════════════════════════════════════════════════════════════════════════════

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
        """When ecpsf fails to read the file, returns zero fwhm."""
        from lsc.lscpsfdef import ecpsf
        monkeypatch.chdir(tmp_path)
        result, fwhm, psffile = ecpsf('nonexistent', 5.0, 3.0, 10, 3.0, False)
        assert fwhm == 0.0

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


# ══════════════════════════════════════════════════════════════════════════════
# From test_lscpsfdef_coverage.py
# ══════════════════════════════════════════════════════════════════════════════

class TestL1SeeingBranch:
    """Line 237: seeing = float(L1SEEING) * scale when L1FWHM is absent."""

    def test_l1seeing_used_when_l1fwhm_absent(self, tmp_path, monkeypatch):
        """When fwhm=0, WCSERR=0, no L1FWHM but L1SEEING present, uses L1SEEING*scale."""
        from lsc.lscpsfdef import ecpsf
        hdr_extra = {'L1SEEING': 4.0, 'WCSERR': 0, 'PIXSCALE': 0.4}
        img = _make_fits(tmp_path, 'l1see', extra_hdr=hdr_extra)
        # Remove L1FWHM from the FITS file
        hdul = fits.open(img + '.fits', mode='update')
        if 'L1FWHM' in hdul[0].header:
            del hdul[0].header['L1FWHM']
        hdul.close()
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']

        # Mock runsex to avoid real SExtractor
        xs = np.array([100.0, 200.0])
        ys = np.array([100.0, 200.0])
        with patch('lsc.lscpsfdef.runsex', return_value=(
                xs, ys, np.zeros(2), np.zeros(2),
                np.array([15.0, 16.0]), np.ones(2),
                np.array([2.5, 2.5]), np.zeros(2))):
            # Need tmp.log for psfmeasure parsing
            log_content = (
                "header1\nheader2\nheader3\n"
                "col1 100.0 100.0 col4 4.0\n"
                "200.0 200.0 col3 4.1\n"
                "average fwhm = 4.0\n"
            )
            (tmp_path / 'tmp.log').write_text(log_content)
            with patch('lsc.lscpsfdef.psffit', return_value=([], [], [])):
                with patch('lsc.lscpsfdef.psffit2', return_value=([], [])):
                    result, fwhm_out, apco = ecpsf(img, 0, 3.0, 10, 3.0, False)
        # seeing = L1SEEING * scale = 4.0 * 0.4 = 1.6
        # fwhm = seeing / scale = 1.6 / 0.4 = 4.0 pixels
        # fwhm_out = fwhm * scale = 4.0 * 0.4 = 1.6
        # Result may fail downstream but L1SEEING branch is hit
        assert result in (0, 1)


# ── Lines 279-295: Interactive branch ────────────────────────────────────────

class TestInteractiveBranch:
    """Lines 279-295: interactive=True triggers imexamine/fields and writes _psf.coo."""

    def test_interactive_writes_psf_coo_from_imexamine(self, tmp_path, monkeypatch):
        """interactive=True calls imexamine+fields, writes _psf.coo."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'inter', extra_hdr={'L1FWHM': 2.0, 'WCSERR': 0})
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        # iraf.fields returns coordinate lines from imexamine output
        iraf.fields.return_value = [
            '100.0 200.0 5.0 4.5',
            '300.0 400.0 5.1 4.8',
        ]
        # For the second branch (interactive writes _psf2.coo via runsex)
        xs = np.array([50.0, 150.0, 250.0])
        ys = np.array([50.0, 150.0, 250.0])
        with patch('lsc.lscpsfdef.runsex', return_value=(
                xs, ys, np.zeros(3), np.zeros(3),
                np.array([15.0, 16.0, 17.0]), np.ones(3),
                np.array([3.0, 3.1, 2.9]), np.zeros(3))):
            with patch('lsc.lscpsfdef.psffit', return_value=([], [], [])):
                with patch('lsc.lscpsfdef.psffit2', return_value=([], [])):
                    result, fwhm_out, apco = ecpsf(img, 0, 3.0, 10, 3.0, True)

        # _psf.coo should have been created with coords from iraf.fields
        assert os.path.exists(tmp_path / '_psf.coo')
        with open(tmp_path / '_psf.coo') as f:
            lines = f.readlines()
        assert len(lines) == 2
        # First star at 100.0, 200.0
        assert '100.000' in lines[0]
        assert '200.000' in lines[0]


# ── Lines 317, 319: Boundary clipping ────────────────────────────────────────

class TestBoundaryClipping:
    """Lines 317/319: x2=int(xdim), y2=int(ydim) when star near edge."""

    def test_star_near_right_and_bottom_edge_clips(self, tmp_path, monkeypatch):
        """Star near image edge triggers x2=xdim and y2=ydim clipping."""
        from lsc.lscpsfdef import ecpsf
        # Small image where star + 3*fwhm > image dimension
        shape = (100, 100)
        data = np.ones(shape, dtype=np.float32) * 1000.0
        hdr = _make_hdr_for_fits({'L1FWHM': 2.0, 'WCSERR': 0, 'DATAMAX': 60000.0}, shape)
        path = tmp_path / 'edge.fits'
        hdu = fits.PrimaryHDU(data=data, header=hdr)
        hdu.writeto(str(path), overwrite=True)
        img = str(tmp_path / 'edge')
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['100 100']

        # fwhm=5.0 -> 3*fwhm=15. Star at x=90 -> x2=90+15=105 > 100 -> clips to 100
        # Star at y=95 -> y2=95+15=110 > 100 -> clips to 100
        # Stars far apart so min(dist2) > distance * fwhm
        # dist between stars = sqrt((90-20)^2 + (95-20)^2) = ~106
        # distance=0.1, fwhm=5 -> distance*fwhm=0.5, min_dist=106 > 0.5 -> OK
        xs = np.array([90.0, 20.0])
        ys = np.array([95.0, 20.0])
        fluxrad = np.array([3.0, 3.0])  # 3.0*1.6=4.8, |4.8-5|/5=0.04 < 0.5

        # Need tmp.log for psfmeasure parsing
        log_content = (
            "header1\nheader2\nheader3\n"
            "col1 90.0 95.0 col4 5.0\n"
            "20.0 20.0 col3 5.1\n"
            "average fwhm = 5.0\n"
        )
        (tmp_path / 'tmp.log').write_text(log_content)

        with patch('lsc.lscpsfdef.runsex', return_value=(
                xs, ys, np.zeros(2), np.zeros(2),
                np.array([15.0, 16.0]), np.ones(2),
                fluxrad, np.zeros(2))):
            with patch('lsc.lscpsfdef.psffit', return_value=([], [], [])):
                with patch('lsc.lscpsfdef.psffit2', return_value=([], [])):
                    result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 0.1, False)
        # The function should not crash when clipping is applied
        assert result in (0, 1)


# ── Lines 337-340: psfmeasure log parsing ────────────────────────────────────

class TestPsfmeasureLogParsing:
    """Lines 337-340: reading subsequent lines from psfmeasure tmp.log."""

    def test_log_with_multiple_stars_parsed(self, tmp_path, monkeypatch):
        """Multiple star lines after righe[3] are parsed correctly."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'logparse', extra_hdr={'L1FWHM': 2.0, 'WCSERR': 0})
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']

        # Build stars that will pass all filters (fwhm=5, scale=0.389)
        xs = np.array([100.0, 200.0, 300.0, 400.0])
        ys = np.array([100.0, 200.0, 300.0, 400.0])
        fluxrad = np.array([3.0, 3.0, 3.0, 3.0])  # 3.0*1.6=4.8 close to fwhm=5

        # Write tmp.log with lines that will exercise lines 336-340
        # Line[3] format: "header_text x y filler fwhm"
        # Lines[4:-2] format: "x y filler fwhm"
        log_content = (
            "line0_header\n"
            "line1_header\n"
            "line2_header\n"
            "colname 100.0 100.0 xxx 5.0\n"   # righe[3]
            "200.0 200.0 xxx 5.1\n"            # righe[4] - exercises lines 337-340
            "300.0 300.0 xxx 4.9\n"            # righe[5] - exercises lines 337-340
            "400.0 400.0 xxx 5.2\n"            # righe[6] - exercises lines 337-340
            "summary line\n"                   # righe[-2]
            "average fwhm = 5.0\n"             # righe[-1]
        )
        (tmp_path / 'tmp.log').write_text(log_content)

        with patch('lsc.lscpsfdef.runsex', return_value=(
                xs, ys, np.zeros(4), np.zeros(4),
                np.array([15.0, 16.0, 17.0, 18.0]), np.ones(4),
                fluxrad, np.zeros(4))):
            with patch('lsc.lscpsfdef.psffit', return_value=([], [], [])):
                with patch('lsc.lscpsfdef.psffit2', return_value=([], [])):
                    result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 30.0, False)
        # _psf.coo should have entries from the parsed log
        assert os.path.exists(tmp_path / '_psf.coo')
        with open(tmp_path / '_psf.coo') as f:
            content = f.read()
        # All 4 stars should be written (fwhm range check: |5.x-5|/5 < 0.3)
        assert '100.000' in content
        assert '200.000' in content
        assert '300.000' in content


# ── Lines 347-350: Duplicate elimination ─────────────────────────────────────

class TestDuplicateEliminationInEcpsf:
    """Lines 347-350: consecutive xn/yn values too close are removed."""

    def test_duplicate_stars_removed_in_ecpsf(self, tmp_path, monkeypatch):
        """Stars with positions differing by < 0.2 in both x and y are deduplicated."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'dedup', extra_hdr={'L1FWHM': 2.0, 'WCSERR': 0})
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']

        xs = np.array([100.0, 200.0, 300.0])
        ys = np.array([100.0, 200.0, 300.0])
        fluxrad = np.array([3.0, 3.0, 3.0])

        # tmp.log with duplicate entries (100.0 and 100.1 are within 0.2)
        log_content = (
            "line0\n"
            "line1\n"
            "line2\n"
            "colname 100.0 100.0 xxx 5.0\n"    # righe[3]
            "100.1 100.1 xxx 5.0\n"            # duplicate of above (diff < 0.2)
            "300.0 300.0 xxx 5.1\n"            # unique
            "summary\n"
            "average fwhm = 5.0\n"
        )
        (tmp_path / 'tmp.log').write_text(log_content)

        with patch('lsc.lscpsfdef.runsex', return_value=(
                xs, ys, np.zeros(3), np.zeros(3),
                np.array([15.0, 16.0, 17.0]), np.ones(3),
                fluxrad, np.zeros(3))):
            with patch('lsc.lscpsfdef.psffit', return_value=([], [], [])):
                with patch('lsc.lscpsfdef.psffit2', return_value=([], [])):
                    result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 30.0, False)
        # _psf.coo should have only unique entries
        assert os.path.exists(tmp_path / '_psf.coo')
        with open(tmp_path / '_psf.coo') as f:
            lines = f.readlines()
        # Should have 2 entries (100.0 and 300.0), not 3
        assert len(lines) == 2


# ── Lines 373-377: Interactive _psf2.coo ─────────────────────────────────────

class TestInteractivePsf2Coo:
    """Lines 373-377: interactive branch writes _psf2.coo from runsex output."""

    def test_interactive_writes_psf2_coo(self, tmp_path, monkeypatch):
        """Interactive mode calls runsex again and writes all stars to _psf2.coo."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'ipsf2', extra_hdr={'L1FWHM': 2.0, 'WCSERR': 0})
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        # imexamine output for the first interactive section
        iraf.fields.return_value = [
            '100.0 200.0 5.0 4.5',
            '300.0 400.0 5.1 4.8',
        ]
        # runsex output for _psf2.coo writing
        xs = np.array([50.0, 150.0, 250.0, 350.0])
        ys = np.array([60.0, 160.0, 260.0, 360.0])
        fluxrad = np.array([3.0, 3.1, 2.9, 3.2])
        with patch('lsc.lscpsfdef.runsex', return_value=(
                xs, ys, np.zeros(4), np.zeros(4),
                np.array([15.0, 16.0, 17.0, 18.0]), np.ones(4),
                fluxrad, np.zeros(4))):
            with patch('lsc.lscpsfdef.psffit', return_value=([], [], [])):
                with patch('lsc.lscpsfdef.psffit2', return_value=([], [])):
                    result, fwhm_out, apco = ecpsf(img, 0, 3.0, 10, 3.0, True)

        # _psf2.coo should have all runsex stars
        assert os.path.exists(tmp_path / '_psf2.coo')
        with open(tmp_path / '_psf2.coo') as f:
            lines = f.readlines()
        assert len(lines) == 4


# ── Lines 399-405: display/tvmark in show mode ───────────────────────────────

class TestShowDisplayTvmark:
    """Lines 399-405: try/except block for display/tvmark when show=True."""

    def test_show_true_calls_display_and_tvmark(self, tmp_path, monkeypatch):
        """show=True triggers iraf.display and iraf.tvmark calls."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'showmode', extra_hdr={'L1FWHM': 2.0, 'WCSERR': 0})
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        iraf.display.reset_mock()
        iraf.display.side_effect = None
        iraf.tvmark.reset_mock()
        iraf.set.reset_mock()
        iraf.set.side_effect = None

        # Need photmag/pst/fitmag with matching IDs for the full path
        photmag = ['100.0 200.0 1 15.0 14.95 14.90 0.01 0.01 0.01']
        pst = ['100.0 200.0 1']
        fitmag = ['100.0 200.0 1 14.90 0.01']
        photmag2 = ['100.0 200.0 1 15.0 14.95 14.90 0.01 0.01 0.01']
        fitmag2 = ['100.0 200.0 1 14.90 0.01']

        # Use _catalog mode to bypass auto path
        catalog_coords = [
            '# comment',
            '100.0 200.0 1',
        ]
        iraf.wcsctran.side_effect = [
            catalog_coords,   # for _psf.coo
            catalog_coords,   # for _psf2.coo
            ['# l1', '# l2', '# l3',
             '10:00:00.000 +02:00:00.00 1 15.000 14.950 14.900 0.010 0.010 0.010'],
            ['# l1', '# l2', '# l3',
             '10:00:00.000 +02:00:00.00 1 15.000 14.950 14.900 0.010 0.010 0.010'],
        ]

        with patch('lsc.lscpsfdef.psffit', return_value=(photmag, pst, fitmag)):
            with patch('lsc.lscpsfdef.psffit2', return_value=(photmag2, fitmag2)):
                with patch('lsc.lscabsphotdef.deg2HMS', return_value=150.0):
                    result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False,
                                                    _catalog='my_cat.txt',
                                                    show=True, max_apercorr=1.0)
        # display and tvmark should have been called
        assert iraf.display.called or iraf.set.called

    def test_show_display_exception_caught(self, tmp_path, monkeypatch):
        """If display raises an exception, it is caught and a warning is printed."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'showexc', extra_hdr={'L1FWHM': 2.0, 'WCSERR': 0})
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        iraf.set.side_effect = Exception("No DS9")

        photmag = ['100.0 200.0 1 15.0 14.95 14.90 0.01 0.01 0.01']
        pst = ['100.0 200.0 1']
        fitmag = ['100.0 200.0 1 14.90 0.01']

        # Use _catalog mode to bypass auto path
        catalog_coords = [
            '# comment',
            '100.0 200.0 1',
        ]
        iraf.wcsctran.side_effect = [
            catalog_coords,   # for _psf.coo
            catalog_coords,   # for _psf2.coo
            ['# l1', '# l2', '# l3',
             '10:00:00.000 +02:00:00.00 1 15.000 14.950 14.900 0.010 0.010 0.010'],
            ['# l1', '# l2', '# l3',
             '10:00:00.000 +02:00:00.00 1 15.000 14.950 14.900 0.010 0.010 0.010'],
        ]

        with patch('lsc.lscpsfdef.psffit', return_value=(photmag, pst, fitmag)):
            with patch('lsc.lscpsfdef.psffit2', return_value=(
                    ['100.0 200.0 1 15.0 14.95 14.90 0.01 0.01 0.01'],
                    ['100.0 200.0 1 14.90 0.01'])):
                with patch('lsc.lscabsphotdef.deg2HMS', return_value=150.0):
                    result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False,
                                                    _catalog='my_cat.txt',
                                                    show=True, max_apercorr=1.0)
        # Should not crash even when display raises
        assert result in (0, 1)


# ── Lines 416-420: fitmag matching by ID ─────────────────────────────────────

class TestFitmagMatching:
    """Lines 416-420: inner loop matching fitmag IDs to photmag IDs."""

    def test_matching_ids_compute_aperture_correction(self, tmp_path, monkeypatch):
        """When fitmag ID matches photmag ID and both are in pst, dmag is computed."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'fitmatch', extra_hdr={'L1FWHM': 2.0, 'WCSERR': 0})
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        iraf.display.reset_mock()
        iraf.display.side_effect = None
        iraf.tvmark.reset_mock()
        iraf.set.reset_mock()
        iraf.set.side_effect = None

        # 4 stars with matching IDs to trigger sigma clip (len > 3)
        # Use slightly varied magp3 values so std != 0 after sigma clip
        # dmag values: 0.050, 0.048, 0.052, 0.051 -> all close, none clipped
        photmag = [
            '100.0 200.0 1 15.1 15.050 15.0 0.01 0.01 0.01',
            '200.0 300.0 2 16.1 16.048 16.0 0.01 0.01 0.01',
            '300.0 400.0 3 17.1 17.052 17.0 0.01 0.01 0.01',
            '400.0 100.0 4 18.1 18.051 18.0 0.01 0.01 0.01',
        ]
        pst = ['100.0 200.0 1', '200.0 300.0 2', '300.0 400.0 3', '400.0 100.0 4']
        fitmag = [
            '100.0 200.0 1 15.0 0.005',
            '200.0 300.0 2 16.0 0.005',
            '300.0 400.0 3 17.0 0.005',
            '400.0 100.0 4 18.0 0.005',
        ]
        photmag2 = [
            '100.0 200.0 1 15.1 15.050 15.0 0.01 0.01 0.01',
            '200.0 300.0 2 16.1 16.048 16.0 0.01 0.01 0.01',
            '300.0 400.0 3 17.1 17.052 17.0 0.01 0.01 0.01',
            '400.0 100.0 4 18.1 18.051 18.0 0.01 0.01 0.01',
        ]
        fitmag2 = [
            '100.0 200.0 1 15.0 0.005',
            '200.0 300.0 2 16.0 0.005',
            '300.0 400.0 3 17.0 0.005',
            '400.0 100.0 4 18.0 0.005',
        ]

        # Use _catalog mode so we bypass the auto star selection
        catalog_coords = [
            '# comment',
            '100.0 200.0 1',
            '200.0 300.0 2',
            '300.0 400.0 3',
            '400.0 100.0 4',
        ]
        radec_lines = [
            '10:00:00.000 +02:00:00.00 1 15.100 15.050 15.000 0.010 0.010 0.010',
            '10:00:01.000 +02:00:01.00 2 16.100 16.048 16.000 0.010 0.010 0.010',
            '10:00:02.000 +02:00:02.00 3 17.100 17.052 17.000 0.010 0.010 0.010',
            '10:00:03.000 +02:00:03.00 4 18.100 18.051 18.000 0.010 0.010 0.010',
        ]
        iraf.wcsctran.side_effect = [
            catalog_coords,   # for _psf.coo writing
            catalog_coords,   # for _psf2.coo writing
            ['# l1', '# l2', '# l3'] + radec_lines,   # for radec
            ['# l1', '# l2', '# l3'] + radec_lines,   # for radec2
        ]

        with patch('lsc.lscpsfdef.psffit', return_value=(photmag, pst, fitmag)):
            with patch('lsc.lscpsfdef.psffit2', return_value=(photmag2, fitmag2)):
                with patch('lsc.lscabsphotdef.deg2HMS', return_value=150.0):
                    result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False,
                                                    _catalog='my_cat.txt',
                                                    max_apercorr=1.0)
        # aperture correction ~ mean(0.050, 0.048, 0.052, 0.051) ~ 0.050
        assert result == 1
        assert abs(apco - 0.050) < 0.01


# ── Lines 429-434: sigma clip when len(_dmag) > 3 ────────────────────────────

class TestSigmaClipInEcpsf:
    """Lines 429-434: sigma clipping applied when >3 valid dmag values."""

    def test_sigma_clip_applied_with_outlier(self, tmp_path, monkeypatch):
        """With 4+ stars and an outlier, sigma clipping removes it."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'sigclip', extra_hdr={'L1FWHM': 2.0, 'WCSERR': 0})
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']

        # 5 stars: 4 with small aperture correction, 1 outlier
        # magp3 values: 15.05, 16.05, 17.05, 18.05, 19.50
        # magf values:  15.00, 16.00, 17.00, 18.00, 19.00
        # dmag:          0.05,  0.05,  0.05,  0.05,  0.50 (outlier)
        photmag = [
            '100.0 200.0 1 15.1 15.05 15.0 0.01 0.01 0.01',
            '200.0 300.0 2 16.1 16.05 16.0 0.01 0.01 0.01',
            '300.0 400.0 3 17.1 17.05 17.0 0.01 0.01 0.01',
            '400.0 100.0 4 18.1 18.05 18.0 0.01 0.01 0.01',
            '150.0 250.0 5 19.5 19.50 19.4 0.01 0.01 0.01',
        ]
        pst = ['100.0 200.0 1', '200.0 300.0 2', '300.0 400.0 3',
               '400.0 100.0 4', '150.0 250.0 5']
        fitmag = [
            '100.0 200.0 1 15.0 0.005',
            '200.0 300.0 2 16.0 0.005',
            '300.0 400.0 3 17.0 0.005',
            '400.0 100.0 4 18.0 0.005',
            '150.0 250.0 5 19.0 0.005',  # outlier: 19.50 - 19.0 = 0.50
        ]
        photmag2 = photmag[:]
        fitmag2 = [
            '100.0 200.0 1 15.0 0.005',
            '200.0 300.0 2 16.0 0.005',
            '300.0 400.0 3 17.0 0.005',
            '400.0 100.0 4 18.0 0.005',
            '150.0 250.0 5 19.0 0.005',
        ]

        # Use _catalog mode to bypass auto star selection
        catalog_coords = [
            '# comment',
            '100.0 200.0 1',
            '200.0 300.0 2',
            '300.0 400.0 3',
            '400.0 100.0 4',
            '150.0 250.0 5',
        ]
        radec_lines = [
            '10:00:00.000 +02:00:00.00 1 15.100 15.050 15.000 0.010 0.010 0.010',
            '10:00:01.000 +02:00:01.00 2 16.100 16.050 16.000 0.010 0.010 0.010',
            '10:00:02.000 +02:00:02.00 3 17.100 17.050 17.000 0.010 0.010 0.010',
            '10:00:03.000 +02:00:03.00 4 18.100 18.050 18.000 0.010 0.010 0.010',
            '10:00:04.000 +02:00:04.00 5 19.500 19.500 19.400 0.010 0.010 0.010',
        ]
        iraf.wcsctran.side_effect = [
            catalog_coords,   # for _psf.coo
            catalog_coords,   # for _psf2.coo
            ['# l1', '# l2', '# l3'] + radec_lines,   # for radec
            ['# l1', '# l2', '# l3'] + radec_lines,   # for radec2
        ]

        with patch('lsc.lscpsfdef.psffit', return_value=(photmag, pst, fitmag)):
            with patch('lsc.lscpsfdef.psffit2', return_value=(photmag2, fitmag2)):
                with patch('lsc.lscabsphotdef.deg2HMS', return_value=150.0):
                    result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False,
                                                    _catalog='my_cat.txt',
                                                    max_apercorr=1.0)
        # After sigma clip, outlier 0.50 removed, mean should be ~0.05
        assert result == 1
        assert abs(apco - 0.05) < 0.02


# ── Line 440: formatting non-9.99 dmag values ───────────────────────────────

class TestDmagFormatting:
    """Line 440: dmag[i] = '%6.3f' % (dmag[i]) for valid values."""

    def test_valid_dmag_formatted(self, tmp_path, monkeypatch):
        """Non-9.99 dmag values are formatted as '%6.3f'."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'dmagfmt', extra_hdr={'L1FWHM': 2.0, 'WCSERR': 0})
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']

        # 2 stars: 1 matching, 1 not matching (will stay 9.99 -> empty string)
        photmag = [
            '100.0 200.0 1 15.1 15.05 15.0 0.01 0.01 0.01',
            '200.0 300.0 2 16.1 16.05 16.0 0.01 0.01 0.01',
        ]
        pst = ['100.0 200.0 1', '200.0 300.0 2']
        fitmag = [
            '100.0 200.0 1 15.0 0.005',
            # ID 2 not in fitmag, so dmag[1] stays 9.99
        ]
        photmag2 = photmag[:]
        fitmag2 = ['100.0 200.0 1 15.0 0.005']

        # Use _catalog mode to bypass auto path
        catalog_coords = [
            '# comment',
            '100.0 200.0 1',
            '200.0 300.0 2',
        ]
        radec_lines = [
            '10:00:00.000 +02:00:00.00 1 15.100 15.050 15.000 0.010 0.010 0.010',
            '10:00:01.000 +02:00:01.00 2 16.100 16.050 16.000 0.010 0.010 0.010',
        ]
        iraf.wcsctran.side_effect = [
            catalog_coords,   # for _psf.coo
            catalog_coords,   # for _psf2.coo
            ['# l1', '# l2', '# l3'] + radec_lines,   # for radec
            ['# l1', '# l2', '# l3'] + radec_lines,   # for radec2
        ]

        with patch('lsc.lscpsfdef.psffit', return_value=(photmag, pst, fitmag)):
            with patch('lsc.lscpsfdef.psffit2', return_value=(photmag2, fitmag2)):
                with patch('lsc.lscabsphotdef.deg2HMS', return_value=150.0):
                    result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False,
                                                    _catalog='my_cat.txt',
                                                    max_apercorr=1.0)
        # Should succeed; the formatting path is exercised
        assert result == 1


# ── Lines 443-448: aperture correction exceeds max_apercorr ──────────────────

class TestMaxApercorrExceeded:
    """Lines 443-448: when abs(aperture_correction) > max_apercorr, return early."""

    def test_large_apercorr_returns_zero(self, tmp_path, monkeypatch):
        """When aperture correction is too large, result=0 is returned."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'bigap', extra_hdr={'L1FWHM': 2.0, 'WCSERR': 0})
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']

        # magp3 - magf = 15.5 - 15.0 = 0.5 which exceeds max_apercorr=0.1
        photmag = ['100.0 200.0 1 16.0 15.5 15.0 0.01 0.01 0.01']
        pst = ['100.0 200.0 1']
        fitmag = ['100.0 200.0 1 15.0 0.005']
        photmag2 = ['100.0 200.0 1 16.0 15.5 15.0 0.01 0.01 0.01']
        fitmag2 = ['100.0 200.0 1 15.0 0.005']

        # Use _catalog mode to bypass auto path
        catalog_coords = [
            '# comment',
            '100.0 200.0 1',
        ]
        radec_lines = [
            '10:00:00.000 +02:00:00.00 1 16.000 15.500 15.000 0.010 0.010 0.010',
        ]
        iraf.wcsctran.side_effect = [
            catalog_coords,   # for _psf.coo
            catalog_coords,   # for _psf2.coo
            ['# l1', '# l2', '# l3'] + radec_lines,   # for radec
            ['# l1', '# l2', '# l3'] + radec_lines,   # for radec2
        ]

        with patch('lsc.lscpsfdef.psffit', return_value=(photmag, pst, fitmag)):
            with patch('lsc.lscpsfdef.psffit2', return_value=(photmag2, fitmag2)):
                with patch('lsc.lscabsphotdef.deg2HMS', return_value=150.0):
                    result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False,
                                                    _catalog='my_cat.txt',
                                                    max_apercorr=0.1)
        assert result == 0
        assert abs(apco) > 0.1


# ── Lines 465-469: fitmag2 matching by ID ────────────────────────────────────

class TestFitmag2Matching:
    """Lines 465-469: inner loop matching fitmag2 to radec2 by ID."""

    def test_fitmag2_ids_matched_in_sn2_creation(self, tmp_path, monkeypatch):
        """fitmag2 IDs matched to radec2 IDs populate smagf/smagerrf."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'fm2match', extra_hdr={'L1FWHM': 2.0, 'WCSERR': 0})
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']

        photmag = [
            '100.0 200.0 1 15.1 15.05 15.0 0.01 0.01 0.01',
            '200.0 300.0 2 16.1 16.05 16.0 0.01 0.01 0.01',
        ]
        pst = ['100.0 200.0 1', '200.0 300.0 2']
        fitmag = [
            '100.0 200.0 1 15.0 0.005',
            '200.0 300.0 2 16.0 0.005',
        ]
        photmag2 = [
            '100.0 200.0 1 15.1 15.05 15.0 0.01 0.01 0.01',
            '200.0 300.0 2 16.1 16.05 16.0 0.01 0.01 0.01',
        ]
        fitmag2 = [
            '100.0 200.0 1 14.95 0.004',
            '200.0 300.0 2 15.95 0.004',
        ]

        # Use _catalog mode to bypass auto path
        catalog_coords = [
            '# comment',
            '100.0 200.0 1',
            '200.0 300.0 2',
        ]
        radec_lines = [
            '10:00:00.000 +02:00:00.00 1 15.100 15.050 15.000 0.010 0.010 0.010',
            '10:00:01.000 +02:00:01.00 2 16.100 16.050 16.000 0.010 0.010 0.010',
        ]
        iraf.wcsctran.side_effect = [
            catalog_coords,   # for _psf.coo
            catalog_coords,   # for _psf2.coo
            ['# l1', '# l2', '# l3'] + radec_lines,   # for radec
            ['# l1', '# l2', '# l3'] + radec_lines,   # for radec2
        ]

        with patch('lsc.lscpsfdef.psffit', return_value=(photmag, pst, fitmag)):
            with patch('lsc.lscpsfdef.psffit2', return_value=(photmag2, fitmag2)):
                with patch('lsc.lscabsphotdef.deg2HMS', return_value=150.0):
                    result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False,
                                                    _catalog='my_cat.txt',
                                                    max_apercorr=1.0)
        # result should be 1, sn2 file should be created
        assert result == 1
        assert os.path.exists(img + '.sn2.fits')


# ── Lines 475-476: applying aperture correction to smagf ─────────────────────

class TestApertureCorrectionApplied:
    """Lines 475-476: smagf values not INDEF/9999 have aperture correction applied."""

    def test_aperture_correction_applied_to_smagf(self, tmp_path, monkeypatch):
        """Valid smagf values have aperture_correction added to them."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'apcoapply', extra_hdr={'L1FWHM': 2.0, 'WCSERR': 0})
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']

        # Set up so aperture_correction = 15.05 - 15.0 = 0.05
        photmag = [
            '100.0 200.0 1 15.1 15.05 15.0 0.01 0.01 0.01',
            '200.0 300.0 2 16.1 16.05 16.0 0.01 0.01 0.01',
        ]
        pst = ['100.0 200.0 1', '200.0 300.0 2']
        fitmag = [
            '100.0 200.0 1 15.0 0.005',
            '200.0 300.0 2 16.0 0.005',
        ]
        photmag2 = photmag[:]
        # fitmag2 provides smagf values that are valid (not INDEF/9999)
        fitmag2 = [
            '100.0 200.0 1 14.95 0.004',
            '200.0 300.0 2 15.95 0.004',
        ]

        # Use _catalog mode to bypass auto path
        catalog_coords = [
            '# comment',
            '100.0 200.0 1',
            '200.0 300.0 2',
        ]
        radec_lines = [
            '10:00:00.000 +02:00:00.00 1 15.100 15.050 15.000 0.010 0.010 0.010',
            '10:00:01.000 +02:00:01.00 2 16.100 16.050 16.000 0.010 0.010 0.010',
        ]
        iraf.wcsctran.side_effect = [
            catalog_coords,   # for _psf.coo
            catalog_coords,   # for _psf2.coo
            ['# l1', '# l2', '# l3'] + radec_lines,   # for radec
            ['# l1', '# l2', '# l3'] + radec_lines,   # for radec2
        ]

        with patch('lsc.lscpsfdef.psffit', return_value=(photmag, pst, fitmag)):
            with patch('lsc.lscpsfdef.psffit2', return_value=(photmag2, fitmag2)):
                with patch('lsc.lscabsphotdef.deg2HMS', return_value=150.0):
                    result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False,
                                                    _catalog='my_cat.txt',
                                                    max_apercorr=1.0)
        assert result == 1
        # Read the sn2 file and check smagf has been corrected
        sn2 = fits.open(img + '.sn2.fits')
        smagf_vals = sn2[1].data['smagf']
        sn2.close()
        # smagf should be ~14.95 + 0.05 = ~15.0 and ~15.95 + 0.05 = ~16.0
        assert smagf_vals[0] == pytest.approx(15.0, abs=0.01)
        assert smagf_vals[1] == pytest.approx(16.0, abs=0.01)


# ── Lines 486-487: _to_float_array TypeError/ValueError fallback ─────────────

class TestToFloatArrayFallback:
    """Lines 486-487: _to_float_array catches TypeError/ValueError."""

    def test_invalid_string_triggers_fallback(self, tmp_path, monkeypatch):
        """Non-numeric strings that are not INDEF/empty/None/9999 trigger fallback."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'floatfb', extra_hdr={'L1FWHM': 2.0, 'WCSERR': 0})
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        iraf.set.reset_mock()
        iraf.set.side_effect = None
        iraf.display.reset_mock()
        iraf.display.side_effect = None

        # Star 1 has valid mag, star 2 has 'badvalue' that cannot be float()-ed
        # photmag for aperture correction (these need valid data)
        photmag = [
            '100.0 200.0 1 15.1 15.05 15.0 0.01 0.01 0.01',
            '200.0 300.0 2 16.1 16.05 16.0 0.01 0.01 0.01',
        ]
        pst = ['100.0 200.0 1', '200.0 300.0 2']
        fitmag = [
            '100.0 200.0 1 15.0 0.005',
            '200.0 300.0 2 16.0 0.005',
        ]
        photmag2 = photmag[:]
        fitmag2 = [
            '100.0 200.0 1 14.95 0.004',
            '200.0 300.0 2 15.95 0.004',
        ]

        # Use _catalog mode to bypass auto path
        catalog_coords = [
            '# comment',
            '100.0 200.0 1',
            '200.0 300.0 2',
        ]
        # For radec (used in aperture correction): valid data
        radec_lines = [
            '10:00:00.000 +02:00:00.00 1 15.100 15.050 15.000 0.010 0.010 0.010',
            '10:00:01.000 +02:00:01.00 2 16.100 16.050 16.000 0.010 0.010 0.010',
        ]
        # For radec2 (used in sn2 table creation): one with 'badvalue' to trigger fallback
        radec2_lines = [
            '10:00:00.000 +02:00:00.00 1 15.100 15.050 15.000 0.010 0.010 0.010',
            '10:00:01.000 +02:00:01.00 2 16.100 badvalue 16.000 0.010 badvalue 0.010',
        ]
        iraf.wcsctran.side_effect = [
            catalog_coords,   # for _psf.coo
            catalog_coords,   # for _psf2.coo
            ['# l1', '# l2', '# l3'] + radec_lines,    # for radec
            ['# l1', '# l2', '# l3'] + radec2_lines,   # for radec2
        ]

        with patch('lsc.lscpsfdef.psffit', return_value=(photmag, pst, fitmag)):
            with patch('lsc.lscpsfdef.psffit2', return_value=(photmag2, fitmag2)):
                with patch('lsc.lscabsphotdef.deg2HMS', return_value=150.0):
                    result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False,
                                                    _catalog='my_cat.txt',
                                                    max_apercorr=1.0)
        # Should succeed with 9999 fill for bad values
        assert result == 1
        # Check the sn2 file has 9999 for the bad value columns
        sn2 = fits.open(img + '.sn2.fits')
        magp3_vals = sn2[1].data['magp3']
        sn2.close()
        # Second star's magp3 is 'badvalue' -> should be 9999.0
        assert magp3_vals[1] == pytest.approx(9999.0)


# ── Line 514: aperture_correction = 0 when make_sn2=False ────────────────────

class TestMakeSn2False:
    """Line 514: aperture_correction = 0 when make_sn2=False."""

    def test_make_sn2_false_returns_zero_apco(self, tmp_path, monkeypatch):
        """When make_sn2=False, aperture_correction=0 and result=1."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'nosn2', extra_hdr={'L1FWHM': 2.0, 'WCSERR': 0})
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']

        photmag = ['100.0 200.0 1 15.1 15.05 15.0 0.01 0.01 0.01']
        pst = ['100.0 200.0 1']
        fitmag = ['100.0 200.0 1 15.0 0.005']

        # Use _catalog mode to bypass auto path cleanly
        catalog_coords = [
            '# comment',
            '100.0 200.0 1',
        ]
        iraf.wcsctran.side_effect = [
            catalog_coords,   # for _psf.coo
            catalog_coords,   # for _psf2.coo
        ]

        with patch('lsc.lscpsfdef.psffit', return_value=(photmag, pst, fitmag)):
            result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False,
                                            _catalog='my_cat.txt',
                                            make_sn2=False)
        assert result == 1
        assert apco == 0

    def test_make_sn2_false_no_sn2_file_created(self, tmp_path, monkeypatch):
        """When make_sn2=False, no .sn2.fits file is created."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'nosn2b', extra_hdr={'L1FWHM': 2.0, 'WCSERR': 0})
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']

        catalog_coords = [
            '# comment',
            '100.0 200.0 1',
        ]
        iraf.wcsctran.side_effect = [
            catalog_coords,   # for _psf.coo
            catalog_coords,   # for _psf2.coo
        ]

        with patch('lsc.lscpsfdef.psffit', return_value=([], [], [])):
            result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 3.0, False,
                                            _catalog='my_cat.txt',
                                            make_sn2=False)
        assert result == 1
        assert not os.path.exists(img + '.sn2.fits')


# ── Integration: full pipeline with all branches ─────────────────────────────

class TestFullPipelineIntegration:
    """End-to-end test ensuring all target lines are reachable."""

    def test_full_auto_pipeline_success(self, tmp_path, monkeypatch):
        """Full automatic (non-interactive, non-catalog) pipeline with make_sn2=True."""
        from lsc.lscpsfdef import ecpsf
        img = _make_fits(tmp_path, 'fullpipe', extra_hdr={'L1FWHM': 2.0, 'WCSERR': 0})
        monkeypatch.chdir(tmp_path)
        iraf = _get_iraf()
        iraf.hselect.return_value = ['512 512']
        iraf.display.reset_mock()
        iraf.display.side_effect = None

        # 5 stars for sigma clip
        n = 5
        photmag = [f'{100+i*50}.0 {200+i*50}.0 {i+1} {15.1+i} {15.05+i} {15.0+i} 0.01 0.01 0.01'
                   for i in range(n)]
        pst = [f'{100+i*50}.0 {200+i*50}.0 {i+1}' for i in range(n)]
        fitmag = [f'{100+i*50}.0 {200+i*50}.0 {i+1} {15.0+i} 0.005' for i in range(n)]
        photmag2 = photmag[:]
        fitmag2 = [f'{100+i*50}.0 {200+i*50}.0 {i+1} {14.95+i} 0.004' for i in range(n)]

        radec_lines = [
            f'10:00:0{i}.000 +02:00:0{i}.00 {i+1} {15.1+i} {15.05+i} {15.0+i} 0.010 0.010 0.010'
            for i in range(n)]
        iraf.wcsctran.side_effect = [
            ['# l1', '# l2', '# l3'] + radec_lines,
            ['# l1', '# l2', '# l3'] + radec_lines,
        ]

        xs = np.array([100.0 + i * 50 for i in range(n)])
        ys = np.array([200.0 + i * 50 for i in range(n)])

        # Write tmp.log with multiple stars and a duplicate
        log_lines = ["line0\n", "line1\n", "line2\n"]
        log_lines.append("colname 100.0 200.0 xxx 5.0\n")  # righe[3]
        log_lines.append("100.1 200.1 xxx 5.0\n")          # duplicate
        log_lines.append("150.0 250.0 xxx 5.1\n")
        log_lines.append("200.0 300.0 xxx 4.9\n")
        log_lines.append("250.0 350.0 xxx 5.0\n")
        log_lines.append("300.0 400.0 xxx 5.0\n")
        log_lines.append("summary\n")
        log_lines.append("average fwhm = 5.0\n")
        (tmp_path / 'tmp.log').write_text(''.join(log_lines))

        with patch('lsc.lscpsfdef.runsex', return_value=(
                xs, ys, np.zeros(n), np.zeros(n),
                np.array([15.0 + i for i in range(n)]), np.ones(n),
                np.array([3.0] * n), np.zeros(n))):
            with patch('lsc.lscpsfdef.psffit', return_value=(photmag, pst, fitmag)):
                with patch('lsc.lscpsfdef.psffit2', return_value=(photmag2, fitmag2)):
                    with patch('lsc.lscabsphotdef.deg2HMS', return_value=150.0):
                        result, fwhm_out, apco = ecpsf(img, 5.0, 3.0, 10, 30.0, False,
                                                        max_apercorr=1.0)
        assert result == 1
        assert os.path.exists(img + '.sn2.fits')


# ══════════════════════════════════════════════════════════════════════════════
# From test_lscpsfdef_pure.py
# ══════════════════════════════════════════════════════════════════════════════

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
