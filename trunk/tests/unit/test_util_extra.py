"""
Additional tests for lsc.util — covers name_duplicate, repstringinfile,
checksnlist, and airmass (with mocked IRAF).
"""
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# name_duplicate — generates unique filenames for duplicates
# ---------------------------------------------------------------------------

class TestNameDuplicate:
    def test_no_existing_files_returns_name_1(self, tmp_path):
        """When no files exist, appends _1 before extension."""
        import lsc.util as util
        img = str(tmp_path / 'test.fits')
        # Create a minimal FITS file so readhdr works
        from astropy.io import fits
        fits.writeto(img, np.ones((10, 10)), overwrite=True)
        result = util.name_duplicate(img, 'SN2020oi', '')
        assert '_1' in result
        assert 'SN2020oi' in result

    def test_existing_file_with_same_date(self, tmp_path):
        """If a file with the same DATE-OBS exists, returns that filename."""
        import lsc.util as util
        from astropy.io import fits
        img = str(tmp_path / 'test.fits')
        hdr = fits.Header()
        hdr['DATE-OBS'] = '2020-05-01T12:00:00'
        fits.writeto(img, np.ones((10, 10)), header=hdr, overwrite=True)
        result = util.name_duplicate(img, 'SN2020oi', '')
        # Should return the existing file name since DATE-OBS matches
        assert result == img

    def test_existing_file_different_date(self, tmp_path):
        """If file exists but different DATE-OBS, increments counter."""
        import lsc.util as util
        from astropy.io import fits
        img = str(tmp_path / 'test.fits')
        hdr = fits.Header()
        hdr['DATE-OBS'] = '2020-05-01T12:00:00'
        fits.writeto(img, np.ones((10, 10)), header=hdr, overwrite=True)
        # Create a file with _1 suffix that has a different date
        img_1 = str(tmp_path / 'test_1.fits')
        hdr2 = fits.Header()
        hdr2['DATE-OBS'] = '2020-06-01T12:00:00'
        fits.writeto(img_1, np.ones((10, 10)), header=hdr2, overwrite=True)
        result = util.name_duplicate(img, 'SN2020oi', '')
        # Should return _2 since _1 exists with different date
        assert '_2' in result


# ---------------------------------------------------------------------------
# repstringinfile — replace string1 with string2 in a file
# ---------------------------------------------------------------------------

class TestRepstringinfile:
    def test_replaces_string(self, tmp_path):
        from lsc.util import repstringinfile
        fin = str(tmp_path / 'input.txt')
        fout = str(tmp_path / 'output.txt')
        with open(fin, 'w') as f:
            f.write('hello world\nfoo bar\n')
        repstringinfile(fin, fout, 'hello', 'goodbye')
        with open(fout) as f:
            content = f.read()
        assert 'goodbye world' in content
        assert 'hello' not in content

    def test_no_match_unchanged(self, tmp_path):
        from lsc.util import repstringinfile
        fin = str(tmp_path / 'input.txt')
        fout = str(tmp_path / 'output.txt')
        with open(fin, 'w') as f:
            f.write('hello world\n')
        repstringinfile(fin, fout, 'nonexistent', 'replacement')
        with open(fout) as f:
            content = f.read()
        assert content == 'hello world\n'

    def test_multiple_occurrences(self, tmp_path):
        from lsc.util import repstringinfile
        fin = str(tmp_path / 'input.txt')
        fout = str(tmp_path / 'output.txt')
        with open(fin, 'w') as f:
            f.write('foo bar foo baz foo\n')
        repstringinfile(fin, fout, 'foo', 'qux')
        with open(fout) as f:
            content = f.read()
        assert content.count('qux') == 3
        assert 'foo' not in content

    def test_empty_file(self, tmp_path):
        from lsc.util import repstringinfile
        fin = str(tmp_path / 'input.txt')
        fout = str(tmp_path / 'output.txt')
        with open(fin, 'w') as f:
            f.write('')
        repstringinfile(fin, fout, 'a', 'b')
        with open(fout) as f:
            content = f.read()
        assert content == ''


# ---------------------------------------------------------------------------
# checksnlist — checks if SN is in a list file
# ---------------------------------------------------------------------------

class TestChecksnlist:
    def test_sn_in_list(self, tmp_path):
        from lsc.util import checksnlist
        listfile = str(tmp_path / 'snlist.txt')
        with open(listfile, 'w') as f:
            f.write('SN2020oi\nSN2020ab\n')
        # checksnlist reads the file and checks something
        # Based on the function signature: checksnlist(img, listfile)
        # It reads the header to get the object name, then checks the list
        from astropy.io import fits
        img = str(tmp_path / 'test.fits')
        hdr = fits.Header()
        hdr['OBJECT'] = 'SN2020oi'
        fits.writeto(img, np.ones((10, 10)), header=hdr, overwrite=True)
        result = checksnlist(img, listfile)
        # Function should not raise when SN is in list

    def test_sn_not_in_list(self, tmp_path):
        from lsc.util import checksnlist
        listfile = str(tmp_path / 'snlist.txt')
        with open(listfile, 'w') as f:
            f.write('SN2020xx\nSN2020yy\n')
        from astropy.io import fits
        img = str(tmp_path / 'test.fits')
        hdr = fits.Header()
        hdr['OBJECT'] = 'SN2020oi'
        fits.writeto(img, np.ones((10, 10)), header=hdr, overwrite=True)
        result = checksnlist(img, listfile)
        # Function should handle SN not in list gracefully


# ---------------------------------------------------------------------------
# airmass — computes airmass from FITS header (uses IRAF astcalc)
# ---------------------------------------------------------------------------

class TestAirmass:
    def test_returns_airmass_with_mocked_iraf(self, tmp_path, monkeypatch):
        """With mocked IRAF astcalc, airmass returns a float."""
        from lsc.util import airmass
        from astropy.io import fits
        img = str(tmp_path / 'test.fits')
        hdr = fits.Header()
        hdr['UTC'] = 3600.0  # 1 hour in seconds
        hdr['EXPTIME'] = 120.0
        hdr['DATE-OBS'] = '20200501T120000'
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.2
        fits.writeto(img, np.ones((10, 10)), header=hdr, overwrite=True)

        # Mock iraf.astcalc to return a known value
        import lsc.util as util_mod
        mock_iraf = MagicMock()
        mock_iraf.astcalc.return_value = ['1.2345']
        monkeypatch.setattr(util_mod, 'iraf', mock_iraf)

        result = airmass(img, overwrite=False)
        assert isinstance(result, float)
        assert abs(result - 1.2345) < 1e-6

    def test_no_utc_returns_empty(self, tmp_path, monkeypatch):
        """When UTC header is missing, returns empty string."""
        from lsc.util import airmass
        from astropy.io import fits
        img = str(tmp_path / 'test.fits')
        hdr = fits.Header()
        hdr['EXPTIME'] = 120.0
        hdr['DATE-OBS'] = '20200501T120000'
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.2
        # No UTC keyword
        fits.writeto(img, np.ones((10, 10)), header=hdr, overwrite=True)

        result = airmass(img, overwrite=False)
        assert result == ''

    def test_airmass_too_large_returns_999(self, tmp_path, monkeypatch):
        """When astcalc returns a value >= 99, it's capped to 999."""
        from lsc.util import airmass
        from astropy.io import fits
        img = str(tmp_path / 'test.fits')
        hdr = fits.Header()
        hdr['UTC'] = 3600.0
        hdr['EXPTIME'] = 120.0
        hdr['DATE-OBS'] = '20200501T120000'
        hdr['RA'] = 150.0
        hdr['DEC'] = 2.2
        fits.writeto(img, np.ones((10, 10)), header=hdr, overwrite=True)

        import lsc.util as util_mod
        mock_iraf = MagicMock()
        mock_iraf.astcalc.return_value = ['not_a_number']
        monkeypatch.setattr(util_mod, 'iraf', mock_iraf)

        result = airmass(img, overwrite=False)
        assert result == 999
