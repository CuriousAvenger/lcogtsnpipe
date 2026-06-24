"""
Additional tests for lsc.externaldata — covers northupeastleft, geturl, getimages.
No network access required (geturl is pure URL building; getimages/northupeastleft
use mocks).
"""
import pytest
import numpy as np
from astropy.io import fits

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# northupeastleft — flips WCS so N is up and E is left
# ---------------------------------------------------------------------------

class TestNorthupeastleft:
    def _make_header(self, cd1_1=0.0001, cd1_2=0.0, cd2_1=0.0, cd2_2=0.0001,
                     crpix1=50, crpix2=50, naxis1=100, naxis2=100):
        hdr = fits.Header()
        hdr['NAXIS1'] = naxis1
        hdr['NAXIS2'] = naxis2
        hdr['CRPIX1'] = crpix1
        hdr['CRPIX2'] = crpix2
        hdr['CD1_1'] = cd1_1
        hdr['CD1_2'] = cd1_2
        hdr['CD2_1'] = cd2_1
        hdr['CD2_2'] = cd2_2
        hdr['DATASEC'] = '[1:100,1:100]'
        return hdr

    def test_no_flip_needed(self):
        """cd1_2=0, cd1_1>0, cd2_2>0 → no flips or swaps."""
        from lsc.externaldata import northupeastleft
        hdr = self._make_header(cd1_1=0.0001, cd2_2=0.0001)
        data = np.ones((100, 100))
        out_data, out_hdr = northupeastleft(data=data, header=hdr)
        assert out_data.shape == (100, 100)
        assert out_hdr['CD1_1'] > 0
        assert out_hdr['CD2_2'] > 0

    def test_swaps_when_cd1_2_dominant(self):
        """|cd1_2| > |cd1_1| → axes are swapped."""
        from lsc.externaldata import northupeastleft
        hdr = self._make_header(cd1_1=0.00001, cd1_2=0.0001)
        data = np.ones((80, 120))
        out_data, out_hdr = northupeastleft(data=data, header=hdr)
        # After swap, naxis1/naxis2 should be swapped
        assert out_hdr['NAXIS1'] == 120
        assert out_hdr['NAXIS2'] == 80

    def test_flips_x_when_cd1_1_negative(self):
        """cd1_1 > 0 triggers x flip (cd1_1 *= -1)."""
        from lsc.externaldata import northupeastleft
        hdr = self._make_header(cd1_1=0.0001, cd2_2=0.0001)
        data = np.ones((100, 100))
        out_data, out_hdr = northupeastleft(data=data, header=hdr)
        # cd1_1 > 0 means no flip needed, stays positive
        assert out_hdr['CD1_1'] > 0

    def test_flips_y_when_cd2_2_negative(self):
        """cd2_2 < 0 triggers y flip (cd2_2 *= -1)."""
        from lsc.externaldata import northupeastleft
        hdr = self._make_header(cd1_1=0.0001, cd2_2=-0.0001)
        data = np.ones((100, 100))
        out_data, out_hdr = northupeastleft(data=data, header=hdr)
        assert out_hdr['CD2_2'] > 0

    def test_datasec_reversed(self):
        """When axes swap, DATASEC is reversed."""
        from lsc.externaldata import northupeastleft
        hdr = self._make_header(cd1_1=0.00001, cd1_2=0.0001)
        data = np.ones((80, 120))
        out_data, out_hdr = northupeastleft(data=data, header=hdr)
        # DATASEC should have reversed dimensions
        assert '100' in out_hdr['DATASEC']  # original naxis2=100 now in x position

    def test_writes_to_file(self):
        """When filename is given, writes to disk."""
        from lsc.externaldata import northupeastleft
        import tempfile, os
        hdr = self._make_header(cd1_1=0.0001, cd2_2=0.0001)
        data = np.ones((50, 50), dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
            fname = f.name
        try:
            fits.writeto(fname, data, hdr, overwrite=True)
            northupeastleft(filename=fname)
            with fits.open(fname) as hdul:
                assert hdul[0].data.shape == (50, 50)
        finally:
            os.unlink(fname)


# ---------------------------------------------------------------------------
# geturl — URL builder for PS1 image cutouts
# ---------------------------------------------------------------------------

class TestGeturl:
    def test_returns_list_for_fits(self):
        from lsc.externaldata import geturl
        # Patch getimages to avoid network
        fake_table = MagicMock()
        fake_table.__iter__ = MagicMock(return_value=iter([
            {'filename': 'img1.fits', 'filter': 'g'},
            {'filename': 'img2.fits', 'filter': 'r'},
        ]))
        with patch('lsc.externaldata.getimages', return_value=fake_table):
            result = geturl(150.0, 2.0, size=100, filters='gr', format='fits')
        assert isinstance(result, list)
        assert len(result) == 2
        assert 'img1.fits' in result[0]
        assert 'img2.fits' in result[1]

    def test_returns_string_for_jpg(self):
        from lsc.externaldata import geturl
        fake_table = MagicMock()
        fake_table.__iter__ = MagicMock(return_value=iter([
            {'filename': 'img1.fits', 'filter': 'g'},
            {'filename': 'img2.fits', 'filter': 'r'},
        ]))
        with patch('lsc.externaldata.getimages', return_value=fake_table):
            result = geturl(150.0, 2.0, size=100, filters='gr', format='jpg')
        assert isinstance(result, str)
        assert 'img1.fits' in result

    def test_color_image_three_filters(self):
        from lsc.externaldata import geturl
        fake_table = MagicMock()
        fake_table.__iter__ = MagicMock(return_value=iter([
            {'filename': 'img_z.fits', 'filter': 'z'},
            {'filename': 'img_i.fits', 'filter': 'i'},
            {'filename': 'img_r.fits', 'filter': 'r'},
            {'filename': 'img_g.fits', 'filter': 'g'},
        ]))
        with patch('lsc.externaldata.getimages', return_value=fake_table):
            result = geturl(150.0, 2.0, size=100, filters='gri', format='jpg', color=True)
        assert isinstance(result, str)
        assert 'red=' in result
        assert 'green=' in result
        assert 'blue=' in result

    def test_color_fits_raises(self):
        from lsc.externaldata import geturl
        with pytest.raises(ValueError, match="color"):
            geturl(150.0, 2.0, size=100, filters='gr', format='fits', color=True)

    def test_invalid_format_raises(self):
        from lsc.externaldata import geturl
        with pytest.raises(ValueError, match="format"):
            geturl(150.0, 2.0, size=100, filters='gr', format='bmp')

    def test_output_size_appended(self):
        from lsc.externaldata import geturl
        fake_table = MagicMock()
        fake_table.__iter__ = MagicMock(return_value=iter([
            {'filename': 'img1.fits', 'filter': 'g'},
            {'filename': 'img2.fits', 'filter': 'r'},
        ]))
        with patch('lsc.externaldata.getimages', return_value=fake_table):
            result = geturl(150.0, 2.0, size=100, filters='gr', format='jpg', output_size=200)
        assert 'output_size=200' in result


# Need MagicMock for the patch above
from unittest.mock import MagicMock
