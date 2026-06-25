"""
Targeted tests to boost lsc/util.py coverage from ~79% to 95%+.
Covers uncovered lines: 113, 356-411, 456-459, 535-537, 553-576, 661-708, 875-880, 912-938.
"""
import os
import sys
import re
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open, call
from astropy.io import fits

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Line 113: delete() function - the except pass branch in os.system('rm')
# ---------------------------------------------------------------------------

class TestDeleteExceptionBranch:
    """Cover line 113: the except:pass when file doesn't exist."""

    def test_delete_single_file(self, tmp_path):
        """delete a single file that exists."""
        from lsc.util import delete
        f = tmp_path / "testfile.txt"
        f.write_text("data")
        delete(str(f))
        assert not f.exists()

    def test_delete_file_with_at_sign(self, tmp_path):
        """delete with @listfile syntax."""
        from lsc.util import delete
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        listf = tmp_path / "list.txt"
        listf.write_text(str(f1) + "\n" + str(f2) + "\n")
        delete("@" + str(listf))
        assert not f1.exists()
        assert not f2.exists()

    def test_delete_comma_separated(self, tmp_path):
        """delete with comma-separated file list."""
        from lsc.util import delete
        f1 = tmp_path / "c.txt"
        f2 = tmp_path / "d.txt"
        f1.write_text("hello")
        f2.write_text("world")
        delete(str(f1) + "," + str(f2))
        assert not f1.exists()
        assert not f2.exists()

    def test_delete_nonexistent_file_does_not_raise(self, tmp_path):
        """delete on a nonexistent file should not raise (except:pass branch)."""
        from lsc.util import delete
        # This hits line 113 - the except:pass
        delete(str(tmp_path / "nonexistent_xyz.txt"))

    def test_delete_at_file_with_comments_and_blanks(self, tmp_path):
        """@listfile with comment lines and blank lines."""
        from lsc.util import delete
        f1 = tmp_path / "x.txt"
        f1.write_text("data")
        listf = tmp_path / "list2.txt"
        listf.write_text("# comment\n" + str(f1) + "\n\n")
        delete("@" + str(listf))
        assert not f1.exists()


# ---------------------------------------------------------------------------
# Lines 356-411: display_image function
# ---------------------------------------------------------------------------

class TestDisplayImage:
    """Cover lines 356-411: display_image function that uses iraf and ds9."""

    @patch('subprocess.Popen')
    @patch('glob.glob')
    def test_display_image_img_not_found(self, mock_glob, mock_popen):
        """When image not found, prints warning and returns."""
        from lsc.util import display_image
        mock_proc = MagicMock()
        mock_proc.stdout.readlines.return_value = [b'ds9 process']
        mock_popen.return_value = mock_proc
        mock_glob.return_value = []
        z1, z2, goon = display_image('nonexistent.fits', 1, None, None, False)
        assert z1 is None
        assert z2 is None

    @patch('time.sleep')
    @patch('subprocess.Popen')
    @patch('glob.glob')
    def test_display_image_ds9_not_running(self, mock_glob, mock_popen, mock_sleep):
        """When ds9 not running, starts ds9 subprocess."""
        from lsc.util import display_image
        mock_proc = MagicMock()
        mock_proc.stdout.readlines.return_value = []
        mock_popen.return_value = mock_proc
        mock_glob.return_value = []
        z1, z2, goon = display_image('nonexistent.fits', 1, None, None, False)
        # Should have called Popen at least twice
        assert mock_popen.call_count >= 2

    @patch('subprocess.Popen')
    @patch('glob.glob')
    def test_display_image_with_z2_iraf_exception(self, mock_glob, mock_popen):
        """When z2 is set and iraf.display raises, goon='False'.
        Note: There's a bug in the source - 'False' is truthy so it hits line 408
        which tries to access unbound 'sss'. We verify the exception branch is hit."""
        from lsc.util import display_image
        mock_proc = MagicMock()
        mock_proc.stdout.readlines.return_value = [b'ds9']
        mock_popen.return_value = mock_proc
        mock_glob.return_value = ['img.fits']

        mock_iraf = MagicMock()
        mock_iraf.display.side_effect = Exception("iraf error")

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            # The source code has a bug: goon='False' (string) is truthy, so
            # it tries to access `sss` which is unbound -> UnboundLocalError
            with pytest.raises(UnboundLocalError):
                display_image('img.fits', 1, 100, 500, False)

    @patch('subprocess.Popen')
    @patch('glob.glob')
    def test_display_image_with_z2_no_scale(self, mock_glob, mock_popen):
        """When z2 is set and scale=False, returns cuts from iraf output."""
        from lsc.util import display_image
        mock_proc = MagicMock()
        mock_proc.stdout.readlines.return_value = [b'ds9']
        mock_popen.return_value = mock_proc
        mock_glob.return_value = ['img.fits']

        mock_iraf = MagicMock()
        mock_iraf.display.return_value = ['z1=100 z2=500']

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            z1, z2, goon = display_image('img.fits', 1, 100, 500, False)
        assert goon == 'True'
        assert z1 == '100'
        assert z2 == '500'

    @patch('subprocess.Popen')
    @patch('glob.glob')
    def test_display_image_no_z2_iraf_exception(self, mock_glob, mock_popen):
        """When z2 is falsy and iraf.display raises, goon=False."""
        from lsc.util import display_image
        mock_proc = MagicMock()
        mock_proc.stdout.readlines.return_value = [b'ds9']
        mock_popen.return_value = mock_proc
        mock_glob.return_value = ['img.fits']

        mock_iraf = MagicMock()
        mock_iraf.display.side_effect = Exception("iraf error")

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            z1, z2, goon = display_image('img.fits', 1, 0, 0, False)
        assert goon == False

    @patch('subprocess.Popen')
    @patch('glob.glob')
    def test_display_image_no_z2_no_scale(self, mock_glob, mock_popen):
        """When z2 is falsy and scale=False, returns cuts."""
        from lsc.util import display_image
        mock_proc = MagicMock()
        mock_proc.stdout.readlines.return_value = [b'ds9']
        mock_popen.return_value = mock_proc
        mock_glob.return_value = ['img.fits']

        mock_iraf = MagicMock()
        mock_iraf.display.return_value = ['z1=200 z2=600']

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            z1, z2, goon = display_image('img.fits', 1, 0, 0, False)
        assert goon == 'True'
        assert z1 == '200'
        assert z2 == '600'

    @patch('lsc.util.userinput')
    @patch('subprocess.Popen')
    @patch('glob.glob')
    def test_display_image_with_scale_yes(self, mock_glob, mock_popen, mock_input):
        """When scale=True and user answers yes, exits loop."""
        from lsc.util import display_image
        mock_proc = MagicMock()
        mock_proc.stdout.readlines.return_value = [b'ds9']
        mock_popen.return_value = mock_proc
        mock_glob.return_value = ['img.fits']

        mock_iraf = MagicMock()
        mock_iraf.display.return_value = ['z1=100 z2=500']
        mock_input.return_value = 'y'

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            z1, z2, goon = display_image('img.fits', 1, 100, 500, True)
        assert goon == 'True'

    @patch('lsc.util.userinput')
    @patch('subprocess.Popen')
    @patch('glob.glob')
    def test_display_image_with_scale_empty_means_yes(self, mock_glob, mock_popen, mock_input):
        """When scale=True and user answers empty (default yes)."""
        from lsc.util import display_image
        mock_proc = MagicMock()
        mock_proc.stdout.readlines.return_value = [b'ds9']
        mock_popen.return_value = mock_proc
        mock_glob.return_value = ['img.fits']

        mock_iraf = MagicMock()
        mock_iraf.display.return_value = ['z1=100 z2=500']
        mock_input.return_value = ''

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            z1, z2, goon = display_image('img.fits', 1, 100, 500, True)
        assert goon == 'True'

    @patch('lsc.util.userinput')
    @patch('subprocess.Popen')
    @patch('glob.glob')
    def test_display_image_with_scale_no_then_yes(self, mock_glob, mock_popen, mock_input):
        """When scale=True and user answers n then y after adjusting cuts."""
        from lsc.util import display_image
        mock_proc = MagicMock()
        mock_proc.stdout.readlines.return_value = [b'ds9']
        mock_popen.return_value = mock_proc
        mock_glob.return_value = ['img.fits']

        mock_iraf = MagicMock()
        mock_iraf.display.return_value = ['z1=100 z2=500']
        # First: cuts ok? -> n, z1=50, z2=400, then cuts ok? -> y
        mock_input.side_effect = ['n', '50', '400', 'y']

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            z1, z2, goon = display_image('img.fits', 1, 100, 500, True)
        assert goon == 'True'

    @patch('lsc.util.userinput')
    @patch('subprocess.Popen')
    @patch('glob.glob')
    def test_display_image_with_scale_NO_means_n(self, mock_glob, mock_popen, mock_input):
        """When scale=True and user answers 'NO', it becomes 'n'."""
        from lsc.util import display_image
        mock_proc = MagicMock()
        mock_proc.stdout.readlines.return_value = [b'ds9']
        mock_popen.return_value = mock_proc
        mock_glob.return_value = ['img.fits']

        mock_iraf = MagicMock()
        mock_iraf.display.return_value = ['z1=100 z2=500']
        # 'NO' -> 'n', then empty defaults for z1/z2, then yes
        mock_input.side_effect = ['NO', '', '', 'y']

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            z1, z2, goon = display_image('img.fits', 1, 100, 500, True)
        assert goon == 'True'


# ---------------------------------------------------------------------------
# Lines 456-459: readspectrum - exception branch (data.rank)
# ---------------------------------------------------------------------------

class TestReadspectrumExceptionBranch:
    """Cover lines 456-459: fallback to data.rank when ndim raises."""

    def test_readspectrum_ndim_1d(self, tmp_path):
        """Normal 1D spectrum works."""
        from lsc.util import readspectrum
        fname = str(tmp_path / "spec1d.fits")
        data = np.arange(100, dtype=np.float32)
        hdr = fits.Header()
        hdr['NAXIS1'] = 100
        hdr['CRPIX1'] = 1
        hdr['CRVAL1'] = 4000.0
        hdr['CDELT1'] = 1.0
        fits.writeto(fname, data, hdr)
        lam, fl = readspectrum(fname)
        assert len(fl) == 100
        assert len(lam) == 100

    def test_readspectrum_2d(self, tmp_path):
        """2D spectrum data - takes [:,0] slice which has nrows elements."""
        from lsc.util import readspectrum
        fname = str(tmp_path / "spec2d.fits")
        data = np.ones((100, 3), dtype=np.float32)
        hdr = fits.Header()
        hdr['NAXIS1'] = 100
        hdr['CRPIX1'] = 1
        hdr['CRVAL1'] = 4000.0
        hdr['CDELT1'] = 1.5
        fits.writeto(fname, data, hdr)
        lam, fl = readspectrum(fname)
        # ndim==2 => fl = spec[0].data[:,0] which has shape (100,)
        assert len(fl) == 100
        assert len(lam) == 100

    def test_readspectrum_3d(self, tmp_path):
        """3D spectrum data - takes [0,0,:] slice."""
        from lsc.util import readspectrum
        fname = str(tmp_path / "spec3d.fits")
        data = np.ones((2, 3, 50), dtype=np.float32)
        hdr = fits.Header()
        hdr['NAXIS1'] = 50
        hdr['CRPIX1'] = 1
        hdr['CRVAL1'] = 5000.0
        hdr['CDELT1'] = 2.0
        fits.writeto(fname, data, hdr)
        lam, fl = readspectrum(fname)
        assert len(fl) == 50
        assert len(lam) == 50

    def test_readspectrum_no_crpix_uses_WAT(self, tmp_path):
        """When crpix1 not in header, tries WAT2_001."""
        from lsc.util import readspectrum
        fname = str(tmp_path / "specwat.fits")
        data = np.ones(50, dtype=np.float32)
        hdr = fits.Header()
        hdr['NAXIS1'] = 50
        hdr['WAT2_001'] = 'wtype=linear label "something" "1 0 0 4000.0 1.5 50 0."'
        fits.writeto(fname, data, hdr)
        lam, fl = readspectrum(fname)
        assert len(fl) == 50

    def test_readspectrum_rank_fallback_1d(self, tmp_path):
        """Cover lines 456-459 by making ndim access raise AttributeError.
        The except branch uses data.rank instead."""
        import lsc.util
        fname = str(tmp_path / "specrank.fits")
        data = np.arange(80, dtype=np.float32)
        hdr = fits.Header()
        hdr['NAXIS1'] = 80
        hdr['CRPIX1'] = 1
        hdr['CRVAL1'] = 4000.0
        hdr['CDELT1'] = 1.0
        fits.writeto(fname, data, hdr)

        # Monkey-patch to simulate the try block raising (ndim unavailable)
        original_fits_open = fits.open

        class FakeData:
            """Data that raises on ndim but has rank."""
            def __init__(self, arr):
                self._arr = arr
                self.rank = 1
            @property
            def ndim(self):
                raise TypeError("simulated ndim failure")
            def __getitem__(self, key):
                return self._arr[key]
            def __len__(self):
                return len(self._arr)

        class FakeHDU:
            def __init__(self, real_hdu):
                self.data = FakeData(real_hdu.data)
                self.header = real_hdu.header

        class FakeHDUList:
            def __init__(self, real_hdulist):
                self._hdus = [FakeHDU(real_hdulist[0])]
                self._real = real_hdulist
            def __getitem__(self, key):
                return self._hdus[key]
            def close(self):
                self._real.close()

        def patched_open(f, *a, **kw):
            real = original_fits_open(f, *a, **kw)
            return FakeHDUList(real)

        with patch.object(lsc.util.fits, 'open', patched_open):
            lam, fl = lsc.util.readspectrum(fname)
        assert len(fl) == 80

    def test_readspectrum_rank_fallback_2d(self, tmp_path):
        """Cover line 458: rank==2 branch."""
        import lsc.util
        fname = str(tmp_path / "specrank2d.fits")
        data = np.ones((80, 5), dtype=np.float32)
        hdr = fits.Header()
        hdr['NAXIS1'] = 80
        hdr['CRPIX1'] = 1
        hdr['CRVAL1'] = 4000.0
        hdr['CDELT1'] = 1.0
        fits.writeto(fname, data, hdr)

        original_fits_open = fits.open

        class FakeData2D:
            def __init__(self, arr):
                self._arr = arr
                self.rank = 2
            @property
            def ndim(self):
                raise TypeError("simulated ndim failure")
            def __getitem__(self, key):
                return self._arr[key]
            def __len__(self):
                return len(self._arr)

        class FakeHDU:
            def __init__(self, real_hdu):
                self.data = FakeData2D(real_hdu.data)
                self.header = real_hdu.header

        class FakeHDUList:
            def __init__(self, real_hdulist):
                self._hdus = [FakeHDU(real_hdulist[0])]
                self._real = real_hdulist
            def __getitem__(self, key):
                return self._hdus[key]
            def close(self):
                self._real.close()

        def patched_open(f, *a, **kw):
            real = original_fits_open(f, *a, **kw)
            return FakeHDUList(real)

        with patch.object(lsc.util.fits, 'open', patched_open):
            lam, fl = lsc.util.readspectrum(fname)
        assert len(fl) == 80

    def test_readspectrum_rank_fallback_3d(self, tmp_path):
        """Cover line 459: rank==3 branch."""
        import lsc.util
        fname = str(tmp_path / "specrank3d.fits")
        data = np.ones((2, 3, 40), dtype=np.float32)
        hdr = fits.Header()
        hdr['NAXIS1'] = 40
        hdr['CRPIX1'] = 1
        hdr['CRVAL1'] = 4000.0
        hdr['CDELT1'] = 1.0
        fits.writeto(fname, data, hdr)

        original_fits_open = fits.open

        class FakeData3D:
            def __init__(self, arr):
                self._arr = arr
                self.rank = 3
            @property
            def ndim(self):
                raise TypeError("simulated ndim failure")
            def __getitem__(self, key):
                return self._arr[key]
            def __len__(self):
                return len(self._arr)

        class FakeHDU:
            def __init__(self, real_hdu):
                self.data = FakeData3D(real_hdu.data)
                self.header = real_hdu.header

        class FakeHDUList:
            def __init__(self, real_hdulist):
                self._hdus = [FakeHDU(real_hdulist[0])]
                self._real = real_hdulist
            def __getitem__(self, key):
                return self._hdus[key]
            def close(self):
                self._real.close()

        def patched_open(f, *a, **kw):
            real = original_fits_open(f, *a, **kw)
            return FakeHDUList(real)

        with patch.object(lsc.util.fits, 'open', patched_open):
            lam, fl = lsc.util.readspectrum(fname)
        assert len(fl) == 40


# ---------------------------------------------------------------------------
# Lines 535-537: defswarp - ron parameter branch
# ---------------------------------------------------------------------------

class TestDefswarpRonBranch:
    """Cover lines 535-537: when ron is provided to defswarp."""

    def test_defswarp_with_ron_and_gain(self, tmp_path):
        """When ron and gain are provided, they get written."""
        import lsc
        from lsc.util import defswarp

        # Create a mock swarp file that contains the RDNOISE_DEFAULT and GAIN_DEFAULT lines
        swarp_content = [
            "# test swarp\n",
            "GAIN_DEFAULT    1.0  # Default gain\n",
            "RDNOISE_DEFAULT 1.0  # Default rdnoise\n",
            "IMAGEOUT_NAME   out.fits  # Output filename\n",
            "WEIGHTOUT_NAME  out.weight.fits  # Output weight-map filename\n",
            "COMBINE_TYPE    MEDIAN  # MEDIAN,AVERAGE\n",
            "PIXEL_SCALE     0.5  # pixel scale\n",
            "PIXELSCALE_TYPE MANUAL  # type\n",
            "CENTER_TYPE     MANUAL  # center\n",
            "Coordinates of the image center\n",
        ]

        mock_swarpfile = str(tmp_path / "mock_default.swarp")
        with open(mock_swarpfile, 'w') as ff:
            ff.writelines(swarp_content)

        outfile = str(tmp_path / "test.swarp")
        with patch.object(lsc, '__path__', [str(tmp_path)]):
            # need to create the directory structure
            sexdir = tmp_path / "standard" / "sex"
            sexdir.mkdir(parents=True, exist_ok=True)
            with open(sexdir / "default.swarp", 'w') as f:
                f.writelines(swarp_content)
            result = defswarp(outfile, "output.fits", "median", gain=2.5, ron=5.0)
        assert result == outfile
        content = open(outfile).read()
        assert 'GAIN_DEFAULT    2.5' in content
        assert 'RDNOISE_DEFAULT    5.0' in content

    def test_defswarp_without_ron_and_gain(self, tmp_path):
        """When ron and gain are empty, original lines are kept."""
        import lsc
        from lsc.util import defswarp

        swarp_content = [
            "# test swarp\n",
            "GAIN_DEFAULT    1.0  # Default gain\n",
            "RDNOISE_DEFAULT 1.0  # Default rdnoise\n",
            "IMAGEOUT_NAME   out.fits  # Output filename\n",
            "WEIGHTOUT_NAME  out.weight.fits  # Output weight-map filename\n",
            "COMBINE_TYPE    MEDIAN  # MEDIAN,AVERAGE\n",
            "PIXEL_SCALE     0.5  # pixel scale\n",
            "PIXELSCALE_TYPE MANUAL  # type\n",
            "CENTER_TYPE     MANUAL  # center\n",
            "Coordinates of the image center\n",
        ]

        sexdir = tmp_path / "standard" / "sex"
        sexdir.mkdir(parents=True, exist_ok=True)
        with open(sexdir / "default.swarp", 'w') as f:
            f.writelines(swarp_content)

        outfile = str(tmp_path / "test2.swarp")
        with patch.object(lsc, '__path__', [str(tmp_path)]):
            result = defswarp(outfile, "output.fits", "average", gain='', ron='')
        assert result == outfile
        content = open(outfile).read()
        # Original GAIN_DEFAULT and RDNOISE_DEFAULT lines are preserved
        assert 'GAIN_DEFAULT    1.0' in content
        assert 'RDNOISE_DEFAULT 1.0' in content


# ---------------------------------------------------------------------------
# Lines 553-576: airmass function
# ---------------------------------------------------------------------------

class TestAirmassFunction:
    """Cover lines 553-576: airmass function that uses iraf.astcalc."""

    def test_airmass_with_utc(self, tmp_path):
        """When UTC is present, computes airmass."""
        from lsc.util import airmass

        mock_hdr = {
            'UTC': 36000.0,
            'exptime': 300.0,
            'date-obs': '20200115',
            'RA': 150.0,
            'DEC': -30.0,
        }

        mock_iraf = MagicMock()
        mock_iraf.astcalc.return_value = ['1.234']

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            with patch('lsc.util.readhdr', return_value=mock_hdr):
                with patch('lsc.util.readkey3', side_effect=lambda hdr, key: hdr.get(key)):
                    with patch('lsc.util.delete') as mock_del:
                        with patch('lsc.util.updateheader') as mock_upd:
                            with patch('builtins.open', mock_open()):
                                result = airmass('test.fits', overwrite=True, _observatory='lasilla')
        assert result == 1.234
        mock_upd.assert_called_once()
        mock_del.assert_called_once_with('airmass.txt')

    def test_airmass_non_float_returns_999(self):
        """When iraf.astcalc returns non-numeric, returns 999."""
        from lsc.util import airmass

        mock_hdr = {
            'UTC': 36000.0,
            'exptime': 300.0,
            'date-obs': '20200115',
            'RA': 150.0,
            'DEC': -30.0,
        }

        mock_iraf = MagicMock()
        mock_iraf.astcalc.return_value = ['INDEF']

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            with patch('lsc.util.readhdr', return_value=mock_hdr):
                with patch('lsc.util.readkey3', side_effect=lambda hdr, key: hdr.get(key)):
                    with patch('lsc.util.delete') as mock_del:
                        with patch('lsc.util.updateheader') as mock_upd:
                            with patch('builtins.open', mock_open()):
                                result = airmass('test.fits', overwrite=False, _observatory='lasilla')
        assert result == 999
        mock_upd.assert_not_called()

    def test_airmass_no_utc_returns_empty(self):
        """When UTC not in header (readkey3 returns falsy), returns empty string."""
        from lsc.util import airmass

        mock_hdr = {'exptime': 300.0}

        mock_iraf = MagicMock()

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            with patch('lsc.util.readhdr', return_value=mock_hdr):
                with patch('lsc.util.readkey3', side_effect=lambda hdr, key: hdr.get(key)):
                    result = airmass('test.fits')
        assert result == ''

    def test_airmass_overwrite_true_large_air(self):
        """When overwrite=True but airmass >= 99, does not update header."""
        from lsc.util import airmass

        mock_hdr = {
            'UTC': 36000.0,
            'exptime': 300.0,
            'date-obs': '20200115',
            'RA': 150.0,
            'DEC': -30.0,
        }

        mock_iraf = MagicMock()
        mock_iraf.astcalc.return_value = ['100.5']

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            with patch('lsc.util.readhdr', return_value=mock_hdr):
                with patch('lsc.util.readkey3', side_effect=lambda hdr, key: hdr.get(key)):
                    with patch('lsc.util.delete'):
                        with patch('lsc.util.updateheader') as mock_upd:
                            with patch('builtins.open', mock_open()):
                                result = airmass('test.fits', overwrite=True)
        # _air >= 99 so no update
        mock_upd.assert_not_called()

    def test_airmass_overwrite_false_valid_air(self):
        """When overwrite=False with valid airmass, does not update header."""
        from lsc.util import airmass

        mock_hdr = {
            'UTC': 36000.0,
            'exptime': 300.0,
            'date-obs': '20200115',
            'RA': 150.0,
            'DEC': -30.0,
        }

        mock_iraf = MagicMock()
        mock_iraf.astcalc.return_value = ['1.5']

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            with patch('lsc.util.readhdr', return_value=mock_hdr):
                with patch('lsc.util.readkey3', side_effect=lambda hdr, key: hdr.get(key)):
                    with patch('lsc.util.delete'):
                        with patch('lsc.util.updateheader') as mock_upd:
                            with patch('builtins.open', mock_open()):
                                result = airmass('test.fits', overwrite=False)
        assert result == 1.5
        mock_upd.assert_not_called()


# ---------------------------------------------------------------------------
# Lines 661-708: marksn2 function
# ---------------------------------------------------------------------------

class TestMarksn2:
    """Cover lines 661-708: marksn2 that uses iraf for display."""

    @patch('lsc.lscabsphotdef.makecatalogue')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_marksn2_basic(self, mock_readhdr, mock_readkey3, mock_makecat):
        """Test marksn2 with mocked iraf, no fitstab2, no verbose."""
        from lsc.util import marksn2

        mock_readhdr.return_value = MagicMock()
        mock_readkey3.return_value = 'B'

        mock_makecat.return_value = {
            'B': {
                'test.sn2.fits': {
                    'ra0': np.array([10.0, 20.0]),
                    'dec0': np.array([-30.0, -40.0]),
                    'magp3': [18.0, 19.0],
                    'magp4': [17.5, 18.5],
                    'smagf': [17.0, 18.0],
                    'magp2': [18.5, 19.5],
                }
            }
        }

        mock_iraf = MagicMock()
        mock_iraf.wcsctran.return_value = ['', '', '', '100.0 200.0', '150.0 250.0']
        mock_iraf.display.return_value = ['z1=0 z2=1000']

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            marksn2('img.fits', 'test.sn2.fits', frame=1, fitstab2='', verbose=False)
        mock_iraf.tvmark.assert_called_once()

    @patch('lsc.lscabsphotdef.makecatalogue')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_marksn2_with_fitstab2(self, mock_readhdr, mock_readkey3, mock_makecat):
        """Test marksn2 with a second fits table."""
        from lsc.util import marksn2

        mock_readhdr.return_value = MagicMock()
        mock_readkey3.side_effect = ['B', 'B', 300.0]

        mock_makecat.side_effect = [
            {'B': {'tab1.fits': {'ra0': np.array([10.0]), 'dec0': np.array([-30.0]),
                                  'magp3': [18.0], 'magp4': [17.5], 'smagf': [17.0], 'magp2': [18.5]}}},
            {'B': {'tab2.fits': {'ra0': np.array([11.0]), 'dec0': np.array([-31.0])}}}
        ]

        mock_iraf = MagicMock()
        mock_iraf.wcsctran.return_value = ['', '', '', '100.0 200.0']
        mock_iraf.display.return_value = ['z1=0 z2=1000']

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            marksn2('img.fits', 'tab1.fits', frame=1, fitstab2='tab2.fits', verbose=False)
        assert mock_iraf.tvmark.call_count == 2

    @patch('lsc.lscabsphotdef.makecatalogue')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_marksn2_verbose(self, mock_readhdr, mock_readkey3, mock_makecat):
        """Test marksn2 verbose=True prints info."""
        from lsc.util import marksn2

        mock_readhdr.return_value = MagicMock()
        mock_readkey3.return_value = 'V'

        mock_makecat.return_value = {
            'V': {
                'test.sn2.fits': {
                    'ra0': [10.0],
                    'dec0': [-30.0],
                    'magp3': [18.0],
                    'magp4': [17.5],
                    'smagf': [17.0],
                    'magp2': [18.5],
                }
            }
        }

        mock_iraf = MagicMock()
        mock_iraf.wcsctran.return_value = ['', '', '', '100.0 200.0']
        mock_iraf.display.return_value = ['z1=0 z2=1000']

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            marksn2('img.fits', 'test.sn2.fits', frame=1, fitstab2='', verbose=True)


# ---------------------------------------------------------------------------
# Lines 875-880: Docosmic - pyversion < 3 branch
# ---------------------------------------------------------------------------

class TestDocosmicPyversion2Branch:
    """Cover lines 875-880: pyversion < 3 uses lsc.cosmics directly."""

    def test_docosmic_pyversion2_branch(self, tmp_path):
        """Test the python 2 branch of Docosmic by mocking sys.version_info."""
        from lsc.util import Docosmic

        # Create a test FITS file
        fname = str(tmp_path / "test_cosmic.fits")
        data = np.random.normal(1000, 50, (100, 100)).astype(np.float32)
        hdr = fits.Header()
        hdr['GAIN'] = 1.5
        hdr['SATURATE'] = 60000
        hdr['RDNOISE'] = 5.0
        hdr['telescop'] = 'test'
        fits.writeto(fname, data, hdr)

        # Mock lsc.cosmics for pyversion < 3 branch
        mock_cosmics_image = MagicMock()
        mock_cosmics_image.cleanarray = data.copy()
        mock_cosmics_image.rawarray = data.copy()
        mock_cosmics_image.getsatstars.return_value = np.zeros_like(data)

        # The function re-imports sys inside, so we need to mock at the right level
        # Actually it does: pyversion = sys.version_info[0], so we mock sys.version_info
        mock_version = MagicMock()
        mock_version.__getitem__ = lambda self, key: 2 if key == 0 else 0

        with patch('sys.version_info', mock_version):
            with patch('lsc.cosmics.cosmicsimage', return_value=mock_cosmics_image):
                out, outmask, outsat = Docosmic(fname)
        assert out.endswith('.clean.fits')
        assert outmask.endswith('.mask.fits')
        assert outsat.endswith('.sat.fits')
        # Cleanup
        for f in [out, outmask, outsat]:
            if os.path.isfile(f):
                os.remove(f)


# ---------------------------------------------------------------------------
# Lines 912-938: checksnlist function
# ---------------------------------------------------------------------------

class TestChecksnlist:
    """Cover lines 912-938: checksnlist using readstandard + iraf.wcsctran."""

    @patch('lsc.util.readstandard')
    @patch('lsc.util.readhdr')
    @patch('lsc.util.readkey3')
    def test_checksnlist_sn_in_field(self, mock_readkey3, mock_readhdr, mock_readstd):
        """When SN is within image bounds."""
        from lsc.util import checksnlist

        mock_readhdr.return_value = MagicMock()
        # readkey3 calls: RA, DEC, object, XDIM, YDIM, NAXIS1, NAXIS2
        mock_readkey3.side_effect = [150.0, -30.0, 'SN2020abc', None, None, 2048, 2048]

        mock_readstd.return_value = (
            np.array(['SN2020abc']),
            np.array([150.001]),
            np.array([-30.001]),
            np.array([18.0])
        )

        mock_iraf = MagicMock()
        mock_iraf.wcsctran.return_value = ['', '', '', '1024.0 1024.0']

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            ra, dec, sn = checksnlist('test.fits', 'supernovaelist.txt')

        assert ra == 150.001
        assert dec == -30.001
        assert sn == 'SN2020abc'

    @patch('lsc.util.readstandard')
    @patch('lsc.util.readhdr')
    @patch('lsc.util.readkey3')
    def test_checksnlist_sn_out_of_field(self, mock_readkey3, mock_readhdr, mock_readstd):
        """When SN is outside image bounds, returns empty strings."""
        from lsc.util import checksnlist

        mock_readhdr.return_value = MagicMock()
        mock_readkey3.side_effect = [150.0, -30.0, 'SN2020abc', 2048, 2048]

        mock_readstd.return_value = (
            np.array(['SN2020abc']),
            np.array([150.001]),
            np.array([-30.001]),
            np.array([18.0])
        )

        mock_iraf = MagicMock()
        mock_iraf.wcsctran.return_value = ['', '', '', '-100.0 -100.0']

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            ra, dec, sn = checksnlist('test.fits', 'supernovaelist.txt')

        assert ra == ''
        assert dec == ''
        assert sn == ''

    @patch('lsc.util.readstandard')
    @patch('lsc.util.readhdr')
    @patch('lsc.util.readkey3')
    def test_checksnlist_indef_result(self, mock_readkey3, mock_readhdr, mock_readstd):
        """When wcsctran returns INDEF."""
        from lsc.util import checksnlist

        mock_readhdr.return_value = MagicMock()
        mock_readkey3.side_effect = [150.0, -30.0, 'SN2020abc', 2048, 2048]

        mock_readstd.return_value = (
            np.array(['SN2020abc']),
            np.array([150.001]),
            np.array([-30.001]),
            np.array([18.0])
        )

        mock_iraf = MagicMock()
        mock_iraf.wcsctran.return_value = ['', '', '', 'INDEF INDEF']

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            ra, dec, sn = checksnlist('test.fits', 'supernovaelist.txt')

        assert ra == ''
        assert dec == ''
        assert sn == ''

    @patch('lsc.util.readstandard')
    @patch('lsc.util.readhdr')
    @patch('lsc.util.readkey3')
    def test_checksnlist_uses_naxis_fallback(self, mock_readkey3, mock_readhdr, mock_readstd):
        """When XDIM/YDIM not set, falls back to NAXIS1/NAXIS2."""
        from lsc.util import checksnlist

        mock_readhdr.return_value = MagicMock()
        # RA, DEC, object, XDIM=None, YDIM=None, NAXIS1, NAXIS2
        mock_readkey3.side_effect = [150.0, -30.0, 'SN2020abc', None, None, 1024, 1024]

        mock_readstd.return_value = (
            np.array(['SN2020abc']),
            np.array([150.001]),
            np.array([-30.001]),
            np.array([18.0])
        )

        mock_iraf = MagicMock()
        mock_iraf.wcsctran.return_value = ['', '', '', '500.0 500.0']

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            ra, dec, sn = checksnlist('test.fits', 'supernovaelist.txt')

        assert ra == 150.001
        assert dec == -30.001
        assert sn == 'SN2020abc'

    @patch('lsc.util.readstandard')
    @patch('lsc.util.readhdr')
    @patch('lsc.util.readkey3')
    def test_checksnlist_x_exceeds_bounds(self, mock_readkey3, mock_readhdr, mock_readstd):
        """When pixel x exceeds dimension bounds."""
        from lsc.util import checksnlist

        mock_readhdr.return_value = MagicMock()
        mock_readkey3.side_effect = [150.0, -30.0, 'SN2020abc', 512, 512]

        mock_readstd.return_value = (
            np.array(['SN2020abc']),
            np.array([150.001]),
            np.array([-30.001]),
            np.array([18.0])
        )

        mock_iraf = MagicMock()
        # x > _xdimen
        mock_iraf.wcsctran.return_value = ['', '', '', '600.0 256.0']

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            ra, dec, sn = checksnlist('test.fits', 'supernovaelist.txt')

        assert ra == ''
        assert dec == ''
        assert sn == ''
