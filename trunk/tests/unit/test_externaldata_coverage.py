"""
Additional tests for lsc.externaldata to boost coverage toward 100%.

Targets uncovered lines:
- 208, 210, 212: downloadsdss removing existing output2/output3/output4
- 222-254: downloadsdss FITS processing (gain/dark, sky interpolation, weights)
- 260-462: sdss_swarp function (full execution path)
- 550-600: sloanimage function
- 639-641, 646, 650-652: downloadPS1 error handling in loop
- 21, 43, 47, 62, 71, 89, etc.: SDSS_gain_dark error prints and MJDnow verbose
"""
import datetime
import os
import sys
import tempfile

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from unittest.mock import patch, MagicMock, call, mock_open

pytestmark = pytest.mark.unit


def _make_simple_fits(path, extra_hdr=None, data=None):
    """Create a simple valid FITS file with no NAXIS1/NAXIS2 in header (let astropy handle it)."""
    if data is None:
        data = np.ones((100, 100), dtype=np.float32) * 2000.0
    hdr = fits.Header()
    if extra_hdr:
        for k, v in extra_hdr.items():
            hdr[k] = v
    fits.writeto(str(path), data, hdr, overwrite=True, output_verify='fix')
    return str(path)


# ===========================================================================
# MJDnow verbose=True (line 21)
# ===========================================================================

class TestMJDnowVerbose:
    def test_verbose_true_prints_jd(self, capsys):
        """When verbose=True, MJDnow should print JD value."""
        from lsc.externaldata import MJDnow
        fixed = datetime.datetime(2012, 1, 1, 0, 0, 0)
        MJDnow(datenow=fixed, verbose=True)
        out = capsys.readouterr().out
        assert 'JD= ' in out


# ===========================================================================
# SDSS_gain_dark - error branches for invalid ugriz (lines 43, 71, 89, 117, 140, 158)
# ===========================================================================

class TestSDSSGainDarkErrorBranches:
    def test_camcol1_invalid_ugriz(self, capsys):
        """Camcol 1 with invalid band should print error."""
        from lsc.externaldata import SDSS_gain_dark
        with pytest.raises(UnboundLocalError):
            SDSS_gain_dark(camcol=1, ugriz='x', run=500)
        out = capsys.readouterr().out
        assert 'UGRIZ not set' in out

    def test_camcol2_invalid_ugriz(self, capsys):
        """Camcol 2 with invalid band should print error."""
        from lsc.externaldata import SDSS_gain_dark
        with pytest.raises(UnboundLocalError):
            SDSS_gain_dark(camcol=2, ugriz='x', run=500)
        out = capsys.readouterr().out
        assert 'UGRIZ not set' in out

    def test_camcol3_invalid_ugriz(self, capsys):
        """Camcol 3 with invalid band should print error."""
        from lsc.externaldata import SDSS_gain_dark
        with pytest.raises(UnboundLocalError):
            SDSS_gain_dark(camcol=3, ugriz='x', run=500)
        out = capsys.readouterr().out
        assert 'UGRIZ not set' in out

    def test_camcol4_invalid_ugriz(self, capsys):
        """Camcol 4 with invalid band should print error."""
        from lsc.externaldata import SDSS_gain_dark
        with pytest.raises(UnboundLocalError):
            SDSS_gain_dark(camcol=4, ugriz='x', run=500)
        out = capsys.readouterr().out
        assert 'UGRIZ not set' in out

    def test_camcol5_invalid_ugriz(self, capsys):
        """Camcol 5 with invalid band should print error."""
        from lsc.externaldata import SDSS_gain_dark
        with pytest.raises(UnboundLocalError):
            SDSS_gain_dark(camcol=5, ugriz='x', run=500)
        out = capsys.readouterr().out
        assert 'UGRIZ not set' in out

    def test_camcol6_invalid_ugriz(self, capsys):
        """Camcol 6 with invalid band should print error."""
        from lsc.externaldata import SDSS_gain_dark
        with pytest.raises(UnboundLocalError):
            SDSS_gain_dark(camcol=6, ugriz='x', run=500)
        out = capsys.readouterr().out
        assert 'UGRIZ not set' in out

    def test_camcol2_u_run_below_1100(self):
        """Camcol 2, u, run < 1100 should return gain=1.595."""
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=2, ugriz='u', run=1000)
        assert gain == 1.595
        assert dark == 12.6025

    def test_camcol2_i_run_below_1500(self):
        """Camcol 2, i, run < 1500 should return dark=5.76."""
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=2, ugriz='i', run=1400)
        assert gain == 6.565
        assert dark == 5.76

    def test_camcol2_i_run_above_1500(self):
        """Camcol 2, i, run > 1500 should return dark=6.25."""
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=2, ugriz='i', run=1600)
        assert gain == 6.565
        assert dark == 6.25

    def test_camcol4_i_run_below_1500(self):
        """Camcol 4, i, run < 1500 should return dark=6.25."""
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=4, ugriz='i', run=1400)
        assert gain == 4.885
        assert dark == 6.25

    def test_camcol4_i_run_above_1500(self):
        """Camcol 4, i, run > 1500 should return dark=7.5625."""
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=4, ugriz='i', run=1600)
        assert gain == 4.885
        assert dark == 7.5625

    def test_camcol4_z_run_below_1500(self):
        """Camcol 4, z, run < 1500 should return dark=9.61."""
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=4, ugriz='z', run=1400)
        assert gain == 4.775
        assert dark == 9.61

    def test_camcol4_z_run_above_1500(self):
        """Camcol 4, z, run > 1500 should return dark=12.6025."""
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=4, ugriz='z', run=1600)
        assert gain == 4.775
        assert dark == 12.6025

    def test_camcol5_z_run_below_1500(self):
        """Camcol 5, z, run < 1500 should return dark=1.8225."""
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=5, ugriz='z', run=1400)
        assert gain == 3.48
        assert dark == 1.8225

    def test_camcol5_z_run_above_1500(self):
        """Camcol 5, z, run > 1500 should return dark=2.1025."""
        from lsc.externaldata import SDSS_gain_dark
        gain, dark = SDSS_gain_dark(camcol=5, ugriz='z', run=1600)
        assert gain == 3.48
        assert dark == 2.1025


# ===========================================================================
# downloadsdss - full processing path (lines 208, 210, 212, 222-254)
# ===========================================================================

class TestDownloadsdssFullProcessing:
    """Tests that exercise the FITS processing path in downloadsdss."""

    def test_full_download_and_process(self, tmp_path, monkeypatch):
        """Test the full download and processing path including FITS manipulation."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import downloadsdss

        run, camcol, field = 500, 1, 10
        band = 'r'

        xid = Table({
            'run': [run],
            'camcol': [camcol],
            'field': [field],
        })

        # Build a realistic multi-extension FITS that mimics the SDSS corrected frame
        # Frame is (NAXIS2=100, NAXIS1=200). After .transpose() -> (200, 100)
        frame_data = np.ones((100, 200), dtype=np.float32) * 1000.0
        primary_hdr = fits.Header()
        primary_hdr['CAMCOL'] = camcol
        primary_hdr['FILTER'] = band
        primary_hdr['RUN'] = run

        # Ext 1: calibration vector - length must match frame_image.shape[0] after transpose = 200
        calib = np.ones(200, dtype=np.float32) * 0.01

        # Ext 2: sky table
        # After transpose: allsky is (nx_sky, ny_sky), sky_function is built on it.
        # sky_function(yinterp, xinterp) returns shape (len(xinterp), len(yinterp))
        # For result to match frame_image (200, 100): need len(xinterp)=200, len(yinterp)=100
        allsky_data = np.ones((4, 4), dtype=np.float32) * 500.0
        # xinterp has 200 elements, yinterp has 100 elements
        n_x = 200
        n_y = 100
        xinterp_data = np.linspace(0, 3, n_x).astype(np.float32)
        yinterp_data = np.linspace(0, 3, n_y).astype(np.float32)

        sky_col1 = fits.Column(name='ALLSKY', format='16E', dim='(4,4)',
                               array=allsky_data.reshape(1, 16))
        sky_col2 = fits.Column(name='XINTERP', format=f'{n_x}E', dim=f'({n_x})',
                               array=xinterp_data.reshape(1, n_x))
        sky_col3 = fits.Column(name='YINTERP', format=f'{n_y}E', dim=f'({n_y})',
                               array=yinterp_data.reshape(1, n_y))

        primary_hdu = fits.PrimaryHDU(data=frame_data, header=primary_hdr)
        calib_hdu = fits.ImageHDU(data=calib)
        sky_hdu = fits.BinTableHDU.from_columns([sky_col1, sky_col2, sky_col3])
        hdulist = fits.HDUList([primary_hdu, calib_hdu, sky_hdu])

        output1_name = f'{band}_SDSS_{run}_{camcol}_{field}.fits'

        # Mock writeto to create the FITS file properly
        mock_hdu = MagicMock()
        def fake_writeto(fname, **kwargs):
            hdulist.writeto(str(tmp_path / fname), overwrite=True, output_verify='fix')
        mock_hdu.writeto = fake_writeto

        # Create existing output2/output3/output4 files to trigger rm (lines 208, 210, 212)
        output2_name = f'{band}_SDSS_{run}_{camcol}_{field}c.fits'
        output3_name = f'{band}_SDSS_{run}_{camcol}_{field}.weight.fits'
        output4_name = f'{band}_SDSS_{run}_{camcol}_{field}.sky.fits'
        (tmp_path / output2_name).write_text('old')
        (tmp_path / output3_name).write_text('old')
        (tmp_path / output4_name).write_text('old')

        with patch('astroquery.sdss.SDSS.query_region', return_value=xid), \
             patch('astroquery.sdss.SDSS.get_images', return_value=[mock_hdu]), \
             patch('os.system') as mock_system:
            result = downloadsdss(150.0, 2.0, band, _radius=20, force=True)

        # Verify os.system was called with rm commands for existing files
        rm_calls = [str(c) for c in mock_system.call_args_list]
        assert any(output2_name in str(c) for c in mock_system.call_args_list)
        assert any(output3_name in str(c) for c in mock_system.call_args_list)
        assert any(output4_name in str(c) for c in mock_system.call_args_list)

        # Verify output files were created
        assert isinstance(result, list)
        assert len(result) == 2
        assert any(output2_name in f for f in result)
        assert any(output3_name in f for f in result)

    def test_download_creates_weight_and_sky(self, tmp_path, monkeypatch):
        """Verify weight and sky images are correctly created."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import downloadsdss

        run, camcol, field = 600, 2, 20
        band = 'g'

        xid = Table({
            'run': [run],
            'camcol': [camcol],
            'field': [field],
        })

        frame_data = np.random.default_rng(42).normal(2000, 100, (80, 150)).astype(np.float32)
        primary_hdr = fits.Header()
        primary_hdr['CAMCOL'] = camcol
        primary_hdr['FILTER'] = band
        primary_hdr['RUN'] = run

        # After transpose, frame is 150x80. Calib needs to be length 150 (frame.T shape[0])
        calib = np.ones(150, dtype=np.float32) * 0.005

        # Sky data - need sky_function(yinterp, xinterp) to produce (150, 80)
        # So xinterp has 150 elements and yinterp has 80 elements
        allsky_data = np.ones((4, 4), dtype=np.float32) * 300.0
        n_x = 150
        n_y = 80
        xinterp_data = np.linspace(0, 3, n_x).astype(np.float32)
        yinterp_data = np.linspace(0, 3, n_y).astype(np.float32)

        sky_col1 = fits.Column(name='ALLSKY', format='16E', dim='(4,4)',
                               array=allsky_data.reshape(1, 16))
        sky_col2 = fits.Column(name='XINTERP', format=f'{n_x}E', dim=f'({n_x})',
                               array=xinterp_data.reshape(1, n_x))
        sky_col3 = fits.Column(name='YINTERP', format=f'{n_y}E', dim=f'({n_y})',
                               array=yinterp_data.reshape(1, n_y))

        primary_hdu = fits.PrimaryHDU(data=frame_data, header=primary_hdr)
        calib_hdu = fits.ImageHDU(data=calib)
        sky_hdu = fits.BinTableHDU.from_columns([sky_col1, sky_col2, sky_col3])
        hdulist = fits.HDUList([primary_hdu, calib_hdu, sky_hdu])

        output1_name = f'{band}_SDSS_{run}_{camcol}_{field}.fits'

        mock_hdu = MagicMock()
        def fake_writeto(fname, **kwargs):
            hdulist.writeto(str(tmp_path / fname), overwrite=True, output_verify='fix')
        mock_hdu.writeto = fake_writeto

        with patch('astroquery.sdss.SDSS.query_region', return_value=xid), \
             patch('astroquery.sdss.SDSS.get_images', return_value=[mock_hdu]), \
             patch('os.system'):
            result = downloadsdss(150.0, 2.0, band, _radius=20, force=True)

        output2_name = f'{band}_SDSS_{run}_{camcol}_{field}c.fits'
        output3_name = f'{band}_SDSS_{run}_{camcol}_{field}.weight.fits'
        output4_name = f'{band}_SDSS_{run}_{camcol}_{field}.sky.fits'

        assert os.path.isfile(str(tmp_path / output2_name))
        assert os.path.isfile(str(tmp_path / output3_name))
        assert os.path.isfile(str(tmp_path / output4_name))

        # Verify the count image has the right shape
        count_data = fits.getdata(str(tmp_path / output2_name))
        assert count_data.shape[0] > 0
        assert count_data.shape[1] > 0

        # Verify weight image values are positive
        weight_data = fits.getdata(str(tmp_path / output3_name))
        assert np.all(weight_data > 0)

        # Verify header has expected keys
        count_hdr = fits.getheader(str(tmp_path / output2_name))
        assert count_hdr['gain'] > 0
        assert count_hdr['BUNIT'] == 'counts'
        assert count_hdr['rdnoise'] == 2
        assert 'SKYLEVEL' in count_hdr


# ===========================================================================
# sdss_swarp - full execution (lines 260-462)
# ===========================================================================

class TestSdssSwarpExecution:
    """Test sdss_swarp function with mocked os.system for the actual swarp call."""

    def _make_sloan_fits(self, path, band='r', ra=150.0, dec=2.0, skylevel=1000.0,
                         include_saturate=True, dayobs_key='DATE-OBS', dayobs_val='2005-06-15',
                         include_airmass=True, include_dateobs=True):
        """Create a minimal FITS file mimicking a processed SDSS frame."""
        data = np.ones((100, 100), dtype=np.float32) * 2000.0
        hdr = fits.Header()
        hdr['FILTER'] = band
        hdr['GAIN'] = 4.71
        hdr['RDNOISE'] = 2
        hdr['CRVAL1'] = ra
        hdr['CRVAL2'] = dec
        if include_saturate:
            hdr['SATURATE'] = 61000
        if dayobs_key:
            hdr[dayobs_key] = dayobs_val
        if include_airmass:
            hdr['AIRMASS'] = 1.2
        if skylevel is not None:
            hdr['SKYLEVEL'] = skylevel
        hdr['CD1_1'] = -0.0001
        hdr['CD1_2'] = 0.0
        hdr['CD2_1'] = 0.0
        hdr['CD2_2'] = 0.0001
        hdr['CRPIX1'] = 50
        hdr['CRPIX2'] = 50
        hdr['DATASEC'] = '[1:100,1:100]'
        fits.writeto(str(path), data, hdr, overwrite=True, output_verify='fix')
        return str(path)

    def _make_weight_fits(self, path):
        """Create a minimal weight FITS file."""
        data = np.ones((100, 100), dtype=np.float32) * 0.5
        hdr = fits.Header()
        hdr['CD1_1'] = -0.0001
        hdr['CD1_2'] = 0.0
        hdr['CD2_1'] = 0.0
        hdr['CD2_2'] = 0.0001
        hdr['CRPIX1'] = 50
        hdr['CRPIX2'] = 50
        hdr['DATASEC'] = '[1:100,1:100]'
        fits.writeto(str(path), data, hdr, overwrite=True, output_verify='fix')
        return str(path)

    def _make_swarp_output(self, tmp_path, output_name):
        """Create the output and weight files that swarp would normally produce."""
        out_data = np.ones((100, 100), dtype=np.float32) * 1500.0
        out_hdr = fits.Header()
        out_hdr['CD1_1'] = -0.0001
        out_hdr['CD1_2'] = 0.0
        out_hdr['CD2_1'] = 0.0
        out_hdr['CD2_2'] = 0.0001
        out_hdr['CRPIX1'] = 50
        out_hdr['CRPIX2'] = 50
        out_hdr['DATASEC'] = '[1:100,1:100]'
        fits.writeto(str(tmp_path / output_name), out_data, out_hdr,
                     overwrite=True, output_verify='fix')

        weight_name = output_name.replace('.fits', '.weight.fits')
        weight_data = np.ones((100, 100), dtype=np.float32) * 2.0
        fits.writeto(str(tmp_path / weight_name), weight_data, out_hdr,
                     overwrite=True, output_verify='fix')

    def test_sdss_swarp_sloan_spectral(self, tmp_path, monkeypatch):
        """Test sdss_swarp with sloan survey, spectral telescope."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import sdss_swarp

        img1 = self._make_sloan_fits(tmp_path / 'img1c.fits', skylevel=800)
        img2 = self._make_sloan_fits(tmp_path / 'img2c.fits', skylevel=900)
        wt1 = self._make_weight_fits(tmp_path / 'img1.weight.fits')
        wt2 = self._make_weight_fits(tmp_path / 'img2.weight.fits')

        imglist = [img1, img2, wt1, wt2]
        output_name = 'output_test.fits'
        output = str(tmp_path / output_name)
        self._make_swarp_output(tmp_path, output_name)

        with patch('os.system') as mock_system:
            result_output, result_var = sdss_swarp(
                imglist, _telescope='spectral', _ra=150.0, _dec=2.0,
                output=output, objname='SN2020test', survey='sloan'
            )

        assert os.path.isfile(result_output)
        assert os.path.isfile(result_var)

        hdr = fits.getheader(result_output)
        assert hdr['FILTER'] == 'rp'
        assert hdr['RDNOISE'] == 2
        assert hdr['PIXSCALE'] == 0.30104
        assert hdr['OBJECT'] == 'SN2020test'
        assert hdr['TELESCOP'] == 'SDSS'
        assert hdr['INSTRUME'] == 'SDSS'
        assert hdr['SITEID'] == 'SDSS'
        assert 'SKYLEVEL' in hdr
        assert hdr['L1FWHM'] == 9999
        assert hdr['WCSERR'] == 0

    def test_sdss_swarp_sloan_sinistro(self, tmp_path, monkeypatch):
        """Test sdss_swarp with sinistro telescope."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import sdss_swarp

        img1 = self._make_sloan_fits(tmp_path / 'img1c.fits')
        imglist = [img1]
        output_name = 'output_sinistro.fits'
        output = str(tmp_path / output_name)
        self._make_swarp_output(tmp_path, output_name)

        with patch('os.system'):
            result_output, result_var = sdss_swarp(
                imglist, _telescope='sinistro', _ra=150.0, _dec=2.0,
                output=output, objname='SN2020x', survey='sloan'
            )

        hdr = fits.getheader(result_output)
        assert hdr['PIXSCALE'] == 0.387

    def test_sdss_swarp_sloan_sbig(self, tmp_path, monkeypatch):
        """Test sdss_swarp with sbig telescope."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import sdss_swarp

        img1 = self._make_sloan_fits(tmp_path / 'img1c.fits')
        imglist = [img1]
        output_name = 'output_sbig.fits'
        output = str(tmp_path / output_name)
        self._make_swarp_output(tmp_path, output_name)

        with patch('os.system'):
            result_output, result_var = sdss_swarp(
                imglist, _telescope='sbig', _ra=150.0, _dec=2.0,
                output=output, objname='SN2020y', survey='sloan'
            )

        hdr = fits.getheader(result_output)
        assert hdr['PIXSCALE'] == 0.467

    def test_sdss_swarp_sloan_muscat(self, tmp_path, monkeypatch):
        """Test sdss_swarp with muscat telescope."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import sdss_swarp

        img1 = self._make_sloan_fits(tmp_path / 'img1c.fits')
        imglist = [img1]
        output_name = 'output_muscat.fits'
        output = str(tmp_path / output_name)
        self._make_swarp_output(tmp_path, output_name)

        with patch('os.system'):
            result_output, result_var = sdss_swarp(
                imglist, _telescope='muscat', _ra=150.0, _dec=2.0,
                output=output, objname='SN2020z', survey='sloan'
            )

        hdr = fits.getheader(result_output)
        assert hdr['PIXSCALE'] == 0.27

    def test_sdss_swarp_sloan_qhy(self, tmp_path, monkeypatch):
        """Test sdss_swarp with QHY telescope."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import sdss_swarp

        img1 = self._make_sloan_fits(tmp_path / 'img1c.fits')
        imglist = [img1]
        output_name = 'output_qhy.fits'
        output = str(tmp_path / output_name)
        self._make_swarp_output(tmp_path, output_name)

        with patch('os.system'):
            result_output, result_var = sdss_swarp(
                imglist, _telescope='QHY', _ra=150.0, _dec=2.0,
                output=output, objname='SN2020qhy', survey='sloan'
            )

        hdr = fits.getheader(result_output)
        assert hdr['PIXSCALE'] == 0.74

    def test_sdss_swarp_unknown_telescope(self, tmp_path, monkeypatch, capsys):
        """Test sdss_swarp with an unknown telescope (prints telescope name)."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import sdss_swarp

        img1 = self._make_sloan_fits(tmp_path / 'img1c.fits')
        imglist = [img1]
        output_name = 'output_unknown.fits'
        output = str(tmp_path / output_name)
        self._make_swarp_output(tmp_path, output_name)

        with patch('os.system'):
            try:
                sdss_swarp(
                    imglist, _telescope='unknown_tel', _ra=150.0, _dec=2.0,
                    output=output, objname='SN2020q', survey='sloan'
                )
            except (NameError, UnboundLocalError):
                pass

        out = capsys.readouterr().out
        assert 'unknown_tel' in out

    def test_sdss_swarp_no_ra_dec_from_header(self, tmp_path, monkeypatch):
        """Test sdss_swarp gets RA/DEC from header when not provided."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import sdss_swarp

        img1 = self._make_sloan_fits(tmp_path / 'img1c.fits', ra=123.456, dec=-45.678)
        imglist = [img1]
        output_name = 'output_noradec.fits'
        output = str(tmp_path / output_name)
        self._make_swarp_output(tmp_path, output_name)

        with patch('os.system'):
            result_output, result_var = sdss_swarp(
                imglist, _telescope='spectral', _ra='', _dec='',
                output=output, objname='SN2020norad', survey='sloan'
            )

        hdr = fits.getheader(result_output)
        assert hdr['RA'] == 123.456
        assert hdr['DEC'] == -45.678

    def test_sdss_swarp_no_saturate_header(self, tmp_path, monkeypatch):
        """Test sdss_swarp when SATURATE not in header defaults to 61000."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import sdss_swarp

        img1 = self._make_sloan_fits(tmp_path / 'img1c.fits', include_saturate=False)
        imglist = [img1]
        output_name = 'output_nosat.fits'
        output = str(tmp_path / output_name)
        self._make_swarp_output(tmp_path, output_name)

        with patch('os.system'):
            result_output, result_var = sdss_swarp(
                imglist, _telescope='spectral', _ra=150.0, _dec=2.0,
                output=output, objname='', survey='sloan'
            )

        hdr = fits.getheader(result_output)
        assert hdr['SATURATE'] == 61000

    def test_sdss_swarp_dayobs_with_slash(self, tmp_path, monkeypatch):
        """Test sdss_swarp when day-obs has slash format (old-style)."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import sdss_swarp

        img1 = self._make_sloan_fits(tmp_path / 'img1c.fits',
                                     dayobs_key='day-obs', dayobs_val='15/06/05',
                                     include_dateobs=True)
        # Also need DATE-OBS for the dateobs output
        hdr = fits.getheader(str(tmp_path / 'img1c.fits'))
        # Add DATE-OBS as well
        data = fits.getdata(str(tmp_path / 'img1c.fits'))
        hdr['DATE-OBS'] = '2005-06-15T12:00:00'
        fits.writeto(str(tmp_path / 'img1c.fits'), data, hdr, overwrite=True, output_verify='fix')

        imglist = [str(tmp_path / 'img1c.fits')]
        output_name = 'output_slash.fits'
        output = str(tmp_path / output_name)
        self._make_swarp_output(tmp_path, output_name)

        with patch('os.system'):
            result_output, result_var = sdss_swarp(
                imglist, _telescope='spectral', _ra=150.0, _dec=2.0,
                output=output, objname='SN_slash', survey='sloan'
            )

        hdr = fits.getheader(result_output)
        assert '19' in str(hdr['DAY-OBS'])

    def test_sdss_swarp_no_output_name(self, tmp_path, monkeypatch):
        """Test sdss_swarp generates output name when not provided."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import sdss_swarp

        img1 = self._make_sloan_fits(tmp_path / 'img1c.fits')
        imglist = [img1]

        # The generated name should be: spectral_SDSS_<dayobs>_<filter>_<objname>.fits
        # DATE-OBS is 2005-06-15 -> dayobs=20050615, filter r -> rp
        expected_output = 'spectral_SDSS_20050615_rp_SN2020auto.fits'
        self._make_swarp_output(tmp_path, expected_output)

        with patch('os.system'):
            result_output, result_var = sdss_swarp(
                imglist, _telescope='spectral', _ra=150.0, _dec=2.0,
                output='', objname='SN2020auto', survey='sloan'
            )

        assert 'spectral' in result_output
        assert 'SDSS' in result_output

    def test_sdss_swarp_invalid_dayobs_exception(self, tmp_path, monkeypatch):
        """Test sdss_swarp handles invalid dayobs gracefully (exception path)."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import sdss_swarp

        img1 = self._make_sloan_fits(tmp_path / 'img1c.fits',
                                     dayobs_key='DATE-OBS', dayobs_val='invalid_date')
        imglist = [img1]
        output_name = 'output_baddate.fits'
        output = str(tmp_path / output_name)
        self._make_swarp_output(tmp_path, output_name)

        with patch('os.system'):
            result_output, result_var = sdss_swarp(
                imglist, _telescope='spectral', _ra=150.0, _dec=2.0,
                output=output, objname='SN_baddate', survey='sloan'
            )

        hdr = fits.getheader(result_output)
        assert hdr['MJD-OBS'] == 0
        assert hdr['DAY-OBS'] == '19991231'

    def test_sdss_swarp_show_flag(self, tmp_path, monkeypatch):
        """Test sdss_swarp with show=True calls display_image."""
        monkeypatch.chdir(tmp_path)
        import lsc
        from lsc.externaldata import sdss_swarp

        img1 = self._make_sloan_fits(tmp_path / 'img1c.fits')
        imglist = [img1]
        output_name = 'output_show.fits'
        output = str(tmp_path / output_name)
        self._make_swarp_output(tmp_path, output_name)

        # Set display_image on lsc module so it can be patched
        lsc.display_image = MagicMock()

        with patch('os.system'), \
             patch('lsc.display_image') as mock_display:
            result_output, result_var = sdss_swarp(
                imglist, _telescope='spectral', _ra=150.0, _dec=2.0,
                output=output, objname='SN_show', survey='sloan', show=True
            )
            mock_display.assert_called_once()

    def test_sdss_swarp_no_weight_images(self, tmp_path, monkeypatch):
        """Test sdss_swarp with no weight images in list."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import sdss_swarp

        img1 = self._make_sloan_fits(tmp_path / 'img1c.fits', skylevel=1200)
        imglist = [img1]
        output_name = 'output_nowt.fits'
        output = str(tmp_path / output_name)
        self._make_swarp_output(tmp_path, output_name)

        with patch('os.system') as mock_system:
            result_output, result_var = sdss_swarp(
                imglist, _telescope='spectral', _ra=150.0, _dec=2.0,
                output=output, objname='SN_nowt', survey='sloan'
            )

        swarp_cmd = mock_system.call_args[0][0]
        assert 'WEIGHT_IMAGE' not in swarp_cmd

    def test_sdss_swarp_ps1_survey(self, tmp_path, monkeypatch):
        """Test sdss_swarp with ps1 survey."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import sdss_swarp

        # Create a PS1-style FITS file with appropriate headers
        data = np.ones((100, 100), dtype=np.float32) * 3000.0
        hdr = fits.Header()
        hdr['HIERARCH CELL.SATURATION'] = 65000
        hdr['HIERARCH FPA.FILTER'] = 'r.00000'
        hdr['HIERARCH CELL.GAIN'] = 1.0
        hdr['HIERARCH CELL.READNOISE'] = 7.0
        hdr['MJD-OBS'] = 56000.5
        hdr['RA_DEG'] = 150.0
        hdr['DEC_DEG'] = 2.0
        hdr['AIRMASS'] = 1.3
        hdr['DATE-OBS'] = '2012-03-14T12:00:00'
        hdr['CD1_1'] = -0.0001
        hdr['CD1_2'] = 0.0
        hdr['CD2_1'] = 0.0
        hdr['CD2_2'] = 0.0001
        hdr['CRPIX1'] = 50
        hdr['CRPIX2'] = 50
        hdr['DATASEC'] = '[1:100,1:100]'

        # PS1 source image - use a short relative filename since the code does img[0][-43:]
        # The filename must be <= 43 chars so [-43:] captures the full name
        img_name = 'rings.v3.skycell.1234.056.unconv.fits'  # 38 chars
        img_path = str(tmp_path / img_name)
        fits.writeto(img_path, data, hdr, overwrite=True, output_verify='fix')

        # For ps1 survey, imglist contains tuples: [(filename,)]
        # Use just the basename since we've chdir'd to tmp_path
        imglist = [(img_name,)]
        output_name = 'output_ps1.fits'
        output = str(tmp_path / output_name)
        self._make_swarp_output(tmp_path, output_name)

        with patch('os.system'):
            result_output, result_var = sdss_swarp(
                imglist, _telescope='spectral', _ra='', _dec='',
                output=output, objname='SN_ps1', survey='ps1'
            )

        hdr_out = fits.getheader(result_output)
        assert hdr_out['TELESCOP'] == 'PS1'
        assert hdr_out['INSTRUME'] == 'PS1'
        assert hdr_out['SITEID'] == 'PS1'
        assert hdr_out['FILTER'] == 'rp'

    def test_sdss_swarp_no_airmass_header(self, tmp_path, monkeypatch):
        """Test sdss_swarp defaults airmass to 1 when not in header."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import sdss_swarp

        img1 = self._make_sloan_fits(tmp_path / 'img1c.fits', include_airmass=False)
        imglist = [img1]
        output_name = 'output_noair.fits'
        output = str(tmp_path / output_name)
        self._make_swarp_output(tmp_path, output_name)

        with patch('os.system'):
            result_output, result_var = sdss_swarp(
                imglist, _telescope='spectral', _ra=150.0, _dec=2.0,
                output=output, objname='SN_noair', survey='sloan'
            )

        hdr = fits.getheader(result_output)
        assert hdr['AIRMASS'] == 1

    def test_sdss_swarp_no_dateobs_uses_mjd(self, tmp_path, monkeypatch):
        """Test sdss_swarp when no date-obs uses jd2date from MJD."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import sdss_swarp

        # Create FITS with day-obs key but no date-obs
        data = np.ones((100, 100), dtype=np.float32) * 2000.0
        hdr = fits.Header()
        hdr['FILTER'] = 'r'
        hdr['GAIN'] = 4.71
        hdr['RDNOISE'] = 2
        hdr['CRVAL1'] = 150.0
        hdr['CRVAL2'] = 2.0
        hdr['SATURATE'] = 61000
        hdr['day-obs'] = '20050615'
        hdr['AIRMASS'] = 1.1
        hdr['CD1_1'] = -0.0001
        hdr['CD1_2'] = 0.0
        hdr['CD2_1'] = 0.0
        hdr['CD2_2'] = 0.0001
        hdr['CRPIX1'] = 50
        hdr['CRPIX2'] = 50
        hdr['DATASEC'] = '[1:100,1:100]'
        img_path = str(tmp_path / 'nodate.fits')
        fits.writeto(img_path, data, hdr, overwrite=True, output_verify='fix')

        imglist = [img_path]
        output_name = 'output_nodate.fits'
        output = str(tmp_path / output_name)
        self._make_swarp_output(tmp_path, output_name)

        with patch('os.system'):
            result_output, result_var = sdss_swarp(
                imglist, _telescope='spectral', _ra=150.0, _dec=2.0,
                output=output, objname='SN_nodate', survey='sloan'
            )

        hdr = fits.getheader(result_output)
        assert 'DATE-OBS' in hdr

    def test_sdss_swarp_with_weight_images_in_swarp_cmd(self, tmp_path, monkeypatch):
        """Test that weight images are passed to swarp command."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import sdss_swarp

        img1 = self._make_sloan_fits(tmp_path / 'img1c.fits', skylevel=900)
        wt1 = self._make_weight_fits(tmp_path / 'img1.weight.fits')
        imglist = [img1, wt1]
        output_name = 'output_wt.fits'
        output = str(tmp_path / output_name)
        self._make_swarp_output(tmp_path, output_name)

        with patch('os.system') as mock_system:
            result_output, result_var = sdss_swarp(
                imglist, _telescope='spectral', _ra=150.0, _dec=2.0,
                output=output, objname='SN_wt', survey='sloan'
            )

        swarp_cmd = mock_system.call_args[0][0]
        assert 'WEIGHT_TYPE MAP_WEIGHT' in swarp_cmd
        assert 'WEIGHT_IMAGE' in swarp_cmd

    def test_sdss_swarp_no_objname(self, tmp_path, monkeypatch):
        """Test sdss_swarp without objname."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import sdss_swarp

        img1 = self._make_sloan_fits(tmp_path / 'img1c.fits')
        imglist = [img1]
        output_name = 'output_noobj.fits'
        output = str(tmp_path / output_name)
        self._make_swarp_output(tmp_path, output_name)

        with patch('os.system'):
            result_output, result_var = sdss_swarp(
                imglist, _telescope='spectral', _ra=150.0, _dec=2.0,
                output=output, objname='', survey='sloan'
            )

        hdr = fits.getheader(result_output)
        # When objname is empty (falsy), OBJECT should not be set by the function
        # (it might exist from output_verify='fix' but should not be SN_xxx)
        assert hdr.get('OBJECT', '') == '' or 'SN' not in str(hdr.get('OBJECT', ''))

    def test_sdss_swarp_skylevel_average(self, tmp_path, monkeypatch):
        """Test that SKYLEVEL is the average of skylevel values from images."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import sdss_swarp

        img1 = self._make_sloan_fits(tmp_path / 'img1c.fits', skylevel=800)
        img2 = self._make_sloan_fits(tmp_path / 'img2c.fits', skylevel=1000)
        wt1 = self._make_weight_fits(tmp_path / 'img1.weight.fits')
        wt2 = self._make_weight_fits(tmp_path / 'img2.weight.fits')
        imglist = [img1, img2, wt1, wt2]
        output_name = 'output_sky.fits'
        output = str(tmp_path / output_name)
        self._make_swarp_output(tmp_path, output_name)

        with patch('os.system'):
            result_output, result_var = sdss_swarp(
                imglist, _telescope='spectral', _ra=150.0, _dec=2.0,
                output=output, objname='SN_sky', survey='sloan'
            )

        hdr = fits.getheader(result_output)
        assert abs(hdr['SKYLEVEL'] - 900.0) < 1.0


# ===========================================================================
# sloanimage - full execution (lines 550-600)
# ===========================================================================

class TestSloanImageExecution:
    """Test sloanimage function with mocked dependencies."""

    def _setup_lsc_mocks(self, hdr):
        """Create proper mocks for lsc.readhdr, readkey3, etc."""
        import lsc

        def mock_readhdr(img):
            return hdr

        def mock_readkey3(h, key):
            # readkey3 does case-insensitive header lookup
            for k in [key, key.upper(), key.lower()]:
                val = h.get(k)
                if val is not None:
                    return val
            return ''

        def mock_deg2HMS(ra, dec, s=''):
            return f"{ra:.4f}", f"{dec:.4f}"

        def mock_display_image(*args, **kwargs):
            pass

        # Set these as attributes on the lsc module
        lsc.readhdr = mock_readhdr
        lsc.readkey3 = mock_readkey3
        lsc.deg2HMS = mock_deg2HMS
        lsc.display_image = mock_display_image

    def test_sloanimage_sloan_survey_sinistro(self, tmp_path, monkeypatch):
        """Test sloanimage with sloan survey and sinistro instrument."""
        monkeypatch.chdir(tmp_path)
        import lsc
        from lsc.externaldata import sloanimage

        data = np.ones((100, 100), dtype=np.float32) * 2000.0
        hdr = fits.Header()
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.0
        hdr['OBJECT'] = 'SN2020test'
        hdr['INSTRUME'] = 'fl01'
        hdr['FILTER'] = 'rp'
        img_path = str(tmp_path / 'science.fits')
        fits.writeto(img_path, data, hdr, overwrite=True, output_verify='fix')

        self._setup_lsc_mocks(hdr)

        mock_frames = ['frame1c.fits', 'frame1.weight.fits']
        mock_output = ('output.fits', 'output.var.fits')

        with patch('lsc.externaldata.downloadsdss', return_value=mock_frames) as mock_dl, \
             patch('lsc.externaldata.sdss_swarp', return_value=mock_output) as mock_swarp:
            out, varimg = sloanimage(img_path, survey='sloan')

        mock_dl.assert_called_once()
        mock_swarp.assert_called_once()
        assert out == 'output.fits'
        assert varimg == 'output.var.fits'

    def test_sloanimage_sloan_spectral(self, tmp_path, monkeypatch):
        """Test sloanimage with spectral instrument."""
        monkeypatch.chdir(tmp_path)
        import lsc
        from lsc.externaldata import sloanimage

        hdr = fits.Header()
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.0
        hdr['OBJECT'] = 'SN2020spec'
        hdr['INSTRUME'] = 'fs01'
        hdr['FILTER'] = 'gp'
        data = np.ones((100, 100), dtype=np.float32)
        img_path = str(tmp_path / 'science_spec.fits')
        fits.writeto(img_path, data, hdr, overwrite=True, output_verify='fix')

        self._setup_lsc_mocks(hdr)

        mock_frames = ['frame1c.fits', 'frame1.weight.fits']
        mock_output = ('out.fits', 'out.var.fits')

        with patch('lsc.externaldata.downloadsdss', return_value=mock_frames), \
             patch('lsc.externaldata.sdss_swarp', return_value=mock_output):
            out, varimg = sloanimage(img_path, survey='sloan')

        assert out == 'out.fits'

    def test_sloanimage_sloan_sbig(self, tmp_path, monkeypatch):
        """Test sloanimage with sbig instrument."""
        monkeypatch.chdir(tmp_path)
        import lsc
        from lsc.externaldata import sloanimage

        hdr = fits.Header()
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.0
        hdr['OBJECT'] = 'SN2020sbig'
        hdr['INSTRUME'] = 'kb01'
        hdr['FILTER'] = 'ip'
        data = np.ones((100, 100), dtype=np.float32)
        img_path = str(tmp_path / 'science_sbig.fits')
        fits.writeto(img_path, data, hdr, overwrite=True, output_verify='fix')

        self._setup_lsc_mocks(hdr)

        mock_frames = ['frame1c.fits', 'frame1.weight.fits']
        mock_output = ('out.fits', 'out.var.fits')

        with patch('lsc.externaldata.downloadsdss', return_value=mock_frames), \
             patch('lsc.externaldata.sdss_swarp', return_value=mock_output):
            out, varimg = sloanimage(img_path, survey='sloan')

        assert out == 'out.fits'

    def test_sloanimage_sloan_muscat(self, tmp_path, monkeypatch):
        """Test sloanimage with muscat instrument."""
        monkeypatch.chdir(tmp_path)
        import lsc
        from lsc.externaldata import sloanimage

        hdr = fits.Header()
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.0
        hdr['OBJECT'] = 'SN2020muscat'
        hdr['INSTRUME'] = 'ep04'
        hdr['FILTER'] = 'zs'
        data = np.ones((100, 100), dtype=np.float32)
        img_path = str(tmp_path / 'science_muscat.fits')
        fits.writeto(img_path, data, hdr, overwrite=True, output_verify='fix')

        self._setup_lsc_mocks(hdr)

        mock_frames = ['frame1c.fits', 'frame1.weight.fits']
        mock_output = ('out.fits', 'out.var.fits')

        with patch('lsc.externaldata.downloadsdss', return_value=mock_frames), \
             patch('lsc.externaldata.sdss_swarp', return_value=mock_output):
            out, varimg = sloanimage(img_path, survey='sloan')

        assert out == 'out.fits'

    def test_sloanimage_sloan_qhy(self, tmp_path, monkeypatch):
        """Test sloanimage with QHY instrument."""
        monkeypatch.chdir(tmp_path)
        import lsc
        from lsc.externaldata import sloanimage

        hdr = fits.Header()
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.0
        hdr['OBJECT'] = 'SN2020qhy'
        hdr['INSTRUME'] = 'sq01'
        hdr['FILTER'] = 'up'
        data = np.ones((100, 100), dtype=np.float32)
        img_path = str(tmp_path / 'science_qhy.fits')
        fits.writeto(img_path, data, hdr, overwrite=True, output_verify='fix')

        self._setup_lsc_mocks(hdr)

        mock_frames = ['frame1c.fits', 'frame1.weight.fits']
        mock_output = ('out.fits', 'out.var.fits')

        with patch('lsc.externaldata.downloadsdss', return_value=mock_frames), \
             patch('lsc.externaldata.sdss_swarp', return_value=mock_output):
            out, varimg = sloanimage(img_path, survey='sloan')

        assert out == 'out.fits'

    def test_sloanimage_fa_instrument(self, tmp_path, monkeypatch):
        """Test sloanimage with fa instrument (sinistro)."""
        monkeypatch.chdir(tmp_path)
        import lsc
        from lsc.externaldata import sloanimage

        hdr = fits.Header()
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.0
        hdr['OBJECT'] = 'SN2020fa'
        hdr['INSTRUME'] = 'fa15'
        hdr['FILTER'] = 'rp'
        data = np.ones((100, 100), dtype=np.float32)
        img_path = str(tmp_path / 'science_fa.fits')
        fits.writeto(img_path, data, hdr, overwrite=True, output_verify='fix')

        self._setup_lsc_mocks(hdr)

        mock_frames = ['frame1c.fits', 'frame1.weight.fits']
        mock_output = ('out.fits', 'out.var.fits')

        with patch('lsc.externaldata.downloadsdss', return_value=mock_frames), \
             patch('lsc.externaldata.sdss_swarp', return_value=mock_output):
            out, varimg = sloanimage(img_path, survey='sloan')

        assert out == 'out.fits'

    def test_sloanimage_ps1_survey(self, tmp_path, monkeypatch):
        """Test sloanimage with ps1 survey."""
        monkeypatch.chdir(tmp_path)
        import lsc
        from lsc.externaldata import sloanimage

        hdr = fits.Header()
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.0
        hdr['OBJECT'] = 'SN2020ps1'
        hdr['INSTRUME'] = 'fl01'
        hdr['FILTER'] = 'rp'
        data = np.ones((100, 100), dtype=np.float32)
        img_path = str(tmp_path / 'science_ps1.fits')
        fits.writeto(img_path, data, hdr, overwrite=True, output_verify='fix')

        self._setup_lsc_mocks(hdr)

        mock_urls = ['http://example.com/img1.fits']
        mock_output = ('out.fits', 'out.var.fits')

        with patch('lsc.externaldata.geturl', return_value=mock_urls) as mock_geturl, \
             patch('lsc.externaldata.sdss_swarp', return_value=mock_output):
            out, varimg = sloanimage(img_path, survey='ps1')

        # geturl should be called 10 times (5 for +i*DD, 5 for -i*DD)
        assert mock_geturl.call_count == 10
        assert out == 'out.fits'

    def test_sloanimage_no_frames_exits(self, tmp_path, monkeypatch):
        """Test sloanimage exits when no frames are returned."""
        monkeypatch.chdir(tmp_path)
        import lsc
        from lsc.externaldata import sloanimage

        hdr = fits.Header()
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.0
        hdr['OBJECT'] = 'SN2020empty'
        hdr['INSTRUME'] = 'fl01'
        hdr['FILTER'] = 'rp'
        data = np.ones((100, 100), dtype=np.float32)
        img_path = str(tmp_path / 'science_empty.fits')
        fits.writeto(img_path, data, hdr, overwrite=True, output_verify='fix')

        self._setup_lsc_mocks(hdr)

        with patch('lsc.externaldata.downloadsdss', return_value=''):
            with pytest.raises(SystemExit):
                sloanimage(img_path, survey='sloan')

    def test_sloanimage_show_flag(self, tmp_path, monkeypatch):
        """Test sloanimage with show=True calls display_image."""
        monkeypatch.chdir(tmp_path)
        import lsc
        from lsc.externaldata import sloanimage

        hdr = fits.Header()
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.0
        hdr['OBJECT'] = 'SN2020show'
        hdr['INSTRUME'] = 'fl01'
        hdr['FILTER'] = 'rp'
        data = np.ones((100, 100), dtype=np.float32)
        img_path = str(tmp_path / 'science_show.fits')
        fits.writeto(img_path, data, hdr, overwrite=True, output_verify='fix')

        self._setup_lsc_mocks(hdr)
        display_calls = []
        lsc.display_image = lambda *a, **kw: display_calls.append(a)

        mock_frames = ['frame1c.fits', 'frame1.weight.fits']
        mock_output = ('out.fits', 'out.var.fits')

        with patch('lsc.externaldata.downloadsdss', return_value=mock_frames), \
             patch('lsc.externaldata.sdss_swarp', return_value=mock_output):
            out, varimg = sloanimage(img_path, survey='sloan', show=True)

        assert len(display_calls) == 1

    def test_sloanimage_filter_not_in_mapping(self, tmp_path, monkeypatch):
        """Test sloanimage with a filter not in the mapping (remains unchanged)."""
        monkeypatch.chdir(tmp_path)
        import lsc
        from lsc.externaldata import sloanimage

        hdr = fits.Header()
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.0
        hdr['OBJECT'] = 'SN2020V'
        hdr['INSTRUME'] = 'fl01'
        hdr['FILTER'] = 'V'
        data = np.ones((100, 100), dtype=np.float32)
        img_path = str(tmp_path / 'science_V.fits')
        fits.writeto(img_path, data, hdr, overwrite=True, output_verify='fix')

        self._setup_lsc_mocks(hdr)

        mock_frames = ['frame1c.fits', 'frame1.weight.fits']
        mock_output = ('out.fits', 'out.var.fits')

        with patch('lsc.externaldata.downloadsdss', return_value=mock_frames), \
             patch('lsc.externaldata.sdss_swarp', return_value=mock_output):
            out, varimg = sloanimage(img_path, survey='sloan')

        assert out == 'out.fits'


# ===========================================================================
# downloadPS1 - error handling within loop (lines 639-641, 646, 650-652)
# ===========================================================================

class TestDownloadPS1ErrorHandling:
    def test_urlretrieve_fails_for_individual_file(self, tmp_path, monkeypatch, capsys):
        """When urlretrieve fails for an individual file, it should print 'problem' and continue."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import downloadPS1

        homedir = str(tmp_path) + '/'
        filename = 'testdir'

        def fake_urlretrieve(url, local):
            if 'index.txt' in url:
                with open(local, 'w') as f:
                    f.write("image1.unconv.fits|12345\n")
                    f.write("image2.unconv.fits|12345\n")
            elif 'image1' in url:
                raise Exception("Download failed")
            else:
                with open(local, 'w') as f:
                    f.write("fake fits data")

        with patch('urllib.request.urlretrieve', side_effect=fake_urlretrieve), \
             patch('os.system'):
            result = downloadPS1(homedir, filename)

        out = capsys.readouterr().out
        assert 'problem' in out.lower()

    def test_mkdir_when_directory_not_exists(self, tmp_path, monkeypatch):
        """downloadPS1 should create directory when it doesn't exist (line 646)."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import downloadPS1

        homedir = str(tmp_path) + '/'
        filename = 'newdir'

        def fake_urlretrieve(url, local):
            if 'index.txt' in url:
                with open(local, 'w') as f:
                    f.write("image.unconv.fits|12345\n")
            else:
                with open(local, 'w') as f:
                    f.write("fake fits data")

        with patch('urllib.request.urlretrieve', side_effect=fake_urlretrieve), \
             patch('os.system'):
            result = downloadPS1(homedir, filename)

        assert os.path.isdir(homedir + filename)

    def test_index_file_open_fails_exits(self, tmp_path, monkeypatch, capsys):
        """When index file can't be opened (exception path), function should exit."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import downloadPS1

        homedir = str(tmp_path) + '/'
        filename = 'testdir'

        def fake_urlretrieve(url, local):
            # Don't write the file, so open() will fail
            pass

        with patch('urllib.request.urlretrieve', side_effect=fake_urlretrieve), \
             patch('os.system'):
            with pytest.raises(SystemExit):
                downloadPS1(homedir, filename)

        out = capsys.readouterr().out
        assert 'stamp_directory not found' in out or 'not found' in out


# ===========================================================================
# sdss_swarp ps1 with weight and mask images (lines 369-390)
# ===========================================================================

class TestSdssSwarpPS1WeightMask:
    """Test PS1-specific weight/mask handling in sdss_swarp."""

    def _make_swarp_output(self, tmp_path, output_name):
        """Create the output and weight files that swarp would normally produce."""
        out_data = np.ones((100, 100), dtype=np.float32) * 1500.0
        out_hdr = fits.Header()
        out_hdr['CD1_1'] = -0.0001
        out_hdr['CD1_2'] = 0.0
        out_hdr['CD2_1'] = 0.0
        out_hdr['CD2_2'] = 0.0001
        out_hdr['CRPIX1'] = 50
        out_hdr['CRPIX2'] = 50
        out_hdr['DATASEC'] = '[1:100,1:100]'
        fits.writeto(str(tmp_path / output_name), out_data, out_hdr,
                     overwrite=True, output_verify='fix')
        weight_name = output_name.replace('.fits', '.weight.fits')
        weight_data = np.ones((100, 100), dtype=np.float32) * 2.0
        fits.writeto(str(tmp_path / weight_name), weight_data, out_hdr,
                     overwrite=True, output_verify='fix')

    def test_ps1_with_weight_and_mask_files(self, tmp_path, monkeypatch):
        """Test PS1 survey with weight and mask files."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import sdss_swarp

        data = np.ones((100, 100), dtype=np.float32) * 3000.0
        hdr = fits.Header()
        hdr['HIERARCH CELL.SATURATION'] = 65000
        hdr['HIERARCH FPA.FILTER'] = 'g.00000'
        hdr['HIERARCH CELL.GAIN'] = 1.0
        hdr['HIERARCH CELL.READNOISE'] = 7.0
        hdr['MJD-OBS'] = 56000.5
        hdr['RA_DEG'] = 150.0
        hdr['DEC_DEG'] = 2.0
        hdr['AIRMASS'] = 1.3
        hdr['DATE-OBS'] = '2012-03-14T12:00:00'
        hdr['CD1_1'] = -0.0001
        hdr['CD1_2'] = 0.0
        hdr['CD2_1'] = 0.0
        hdr['CD2_2'] = 0.0001
        hdr['CRPIX1'] = 50
        hdr['CRPIX2'] = 50
        hdr['DATASEC'] = '[1:100,1:100]'

        # Science image - use short names (<=43 chars for [-43:] to work)
        img_name = 'rings.v3.skycell.1234.056.unconv.fits'
        fits.writeto(img_name, data, hdr, overwrite=True, output_verify='fix')

        # Weight image (.wt_ pattern)
        wt_data = np.ones((100, 100), dtype=np.float32) * 2.0
        wt_name = 'rings.v3.skycell.1234.056.wt_unconv.fits'
        fits.writeto(wt_name, wt_data, hdr, overwrite=True, output_verify='fix')

        # Mask image (.mk_ pattern)
        mk_data = np.zeros((100, 100), dtype=np.float32)
        mk_name = 'rings.v3.skycell.1234.056.mk_unconv.fits'
        fits.writeto(mk_name, mk_data, hdr, overwrite=True, output_verify='fix')

        # For ps1 survey, imglist starts as tuples (use relative names since we chdir'd)
        imglist = [(img_name,), (wt_name,), (mk_name,)]
        output_name = 'output_ps1wt.fits'
        output = str(tmp_path / output_name)
        self._make_swarp_output(tmp_path, output_name)

        with patch('os.system'):
            result_output, result_var = sdss_swarp(
                imglist, _telescope='spectral', _ra=150.0, _dec=2.0,
                output=output, objname='SN_ps1wt', survey='ps1'
            )

        assert os.path.isfile(result_output)
        hdr_out = fits.getheader(result_output)
        assert hdr_out['TELESCOP'] == 'PS1'

    def test_ps1_existing_unconv_1_file_removed(self, tmp_path, monkeypatch):
        """Test that existing unconv_1 file is removed before writing (line 319)."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import sdss_swarp

        data = np.ones((100, 100), dtype=np.float32) * 3000.0
        hdr = fits.Header()
        hdr['HIERARCH CELL.SATURATION'] = 65000
        hdr['HIERARCH FPA.FILTER'] = 'g.00000'
        hdr['HIERARCH CELL.GAIN'] = 1.0
        hdr['HIERARCH CELL.READNOISE'] = 7.0
        hdr['MJD-OBS'] = 56000.5
        hdr['RA_DEG'] = 150.0
        hdr['DEC_DEG'] = 2.0
        hdr['AIRMASS'] = 1.3
        hdr['DATE-OBS'] = '2012-03-14T12:00:00'
        hdr['CD1_1'] = -0.0001
        hdr['CD1_2'] = 0.0
        hdr['CD2_1'] = 0.0
        hdr['CD2_2'] = 0.0001
        hdr['CRPIX1'] = 50
        hdr['CRPIX2'] = 50
        hdr['DATASEC'] = '[1:100,1:100]'

        # Science image
        img_name = 'rings.v3.skycell.5678.012.unconv.fits'
        fits.writeto(img_name, data, hdr, overwrite=True, output_verify='fix')

        # Create the unconv_1 file that should trigger the rm on line 319
        unconv_1_name = 'rings.v3.skycell.5678.012.unconv_1.fits'
        fits.writeto(unconv_1_name, data, hdr, overwrite=True, output_verify='fix')

        imglist = [(img_name,)]
        output_name = 'output_ps1_rm.fits'
        output = str(tmp_path / output_name)
        self._make_swarp_output(tmp_path, output_name)

        with patch('os.system') as mock_system:
            result_output, result_var = sdss_swarp(
                imglist, _telescope='spectral', _ra=150.0, _dec=2.0,
                output=output, objname='SN_ps1rm', survey='ps1'
            )

        # Verify that os.system was called with rm for the existing unconv_1 file
        rm_calls = [str(c) for c in mock_system.call_args_list]
        assert any('rm ' in str(c) and 'unconv_1' in str(c) for c in mock_system.call_args_list)

    def test_ps1_weight_without_mask_uses_cp(self, tmp_path, monkeypatch):
        """Test PS1 weight file without matching mask file uses cp (line 383)."""
        monkeypatch.chdir(tmp_path)
        from lsc.externaldata import sdss_swarp

        data = np.ones((100, 100), dtype=np.float32) * 3000.0
        hdr = fits.Header()
        hdr['HIERARCH CELL.SATURATION'] = 65000
        hdr['HIERARCH FPA.FILTER'] = 'i.00000'
        hdr['HIERARCH CELL.GAIN'] = 1.0
        hdr['HIERARCH CELL.READNOISE'] = 7.0
        hdr['MJD-OBS'] = 56000.5
        hdr['RA_DEG'] = 150.0
        hdr['DEC_DEG'] = 2.0
        hdr['AIRMASS'] = 1.3
        hdr['DATE-OBS'] = '2012-03-14T12:00:00'
        hdr['CD1_1'] = -0.0001
        hdr['CD1_2'] = 0.0
        hdr['CD2_1'] = 0.0
        hdr['CD2_2'] = 0.0001
        hdr['CRPIX1'] = 50
        hdr['CRPIX2'] = 50
        hdr['DATASEC'] = '[1:100,1:100]'

        # Science image
        img_name = 'rings.v3.skycell.9999.001.unconv.fits'
        fits.writeto(img_name, data, hdr, overwrite=True, output_verify='fix')

        # Weight image (.wt_ pattern) but NO mask file (.mk_)
        wt_data = np.ones((100, 100), dtype=np.float32) * 2.0
        wt_name = 'rings.v3.skycell.9999.001.wt_unconv.fits'
        fits.writeto(wt_name, wt_data, hdr, overwrite=True, output_verify='fix')

        # Note: NO .mk_ file exists, so the else branch (line 383) should be taken
        imglist = [(img_name,), (wt_name,)]
        output_name = 'output_ps1_nosk.fits'
        output = str(tmp_path / output_name)
        self._make_swarp_output(tmp_path, output_name)

        with patch('os.system') as mock_system:
            result_output, result_var = sdss_swarp(
                imglist, _telescope='spectral', _ra=150.0, _dec=2.0,
                output=output, objname='SN_ps1nosk', survey='ps1'
            )

        # Verify cp was called for the weight file (line 383)
        cp_calls = [str(c) for c in mock_system.call_args_list]
        assert any('cp ' in str(c) for c in mock_system.call_args_list)
