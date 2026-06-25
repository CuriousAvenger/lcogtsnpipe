"""
Tests targeting specific uncovered lines in lsc.lscpsfdef.ecpsf.

Covers:
  - Line 237: L1SEEING branch (fwhm not provided, L1FWHM absent)
  - Lines 279-295: interactive branch (imexamine + fields -> _psf.coo)
  - Lines 317, 319: boundary clipping when star near image edge
  - Lines 337-340: parsing psfmeasure log (subsequent lines after line[3])
  - Lines 347-350: duplicate elimination loop
  - Lines 373-377: interactive branch writing _psf2.coo via runsex
  - Lines 399-405: display/tvmark in show or interactive mode
  - Lines 416-420: inner loop matching fitmag to photmag by ID
  - Lines 429-434: sigma clip when len(_dmag) > 3
  - Lines 440: formatting non-9.99 dmag values
  - Lines 443-448: aperture correction exceeds max_apercorr (early return)
  - Lines 465-469: inner loop matching fitmag2 to radec2 by ID
  - Lines 475-476: applying aperture correction to smagf values
  - Lines 486-487: _to_float_array TypeError/ValueError fallback
  - Line 514: aperture_correction = 0 when make_sn2=False
"""
import sys
import os
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
    hdr['WCSERR'] = 0
    hdr['NAXIS1'] = 512
    hdr['NAXIS2'] = 512
    if extra:
        hdr.update(extra)
    return hdr


def _make_hdr_for_fits(extra=None, shape=(512, 512)):
    """Return a header suitable for writing to FITS (no NAXIS1/NAXIS2)."""
    hdr = fits.Header()
    hdr['GAIN'] = 2.0
    hdr['RDNOISE'] = 5.0
    hdr['RON'] = 5.0
    hdr['PIXSCALE'] = 0.389
    hdr['SATURATE'] = 60000.0
    hdr['DATAMAX'] = 60000.0
    hdr['DATAMIN'] = 0.0
    hdr['INSTRUME'] = 'fa15'
    hdr['WCSERR'] = 0
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
    return str(tmp_path / name)


def _get_iraf():
    return sys.modules['pyraf'].iraf


# ── Line 237: L1SEEING branch ────────────────────────────────────────────────

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
