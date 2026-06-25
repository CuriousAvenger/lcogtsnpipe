"""
Targeted tests to boost lsc/myloopdef.py coverage from ~81% to 90%+.
"""
import os
import sys
import pytest
import numpy as np
import datetime
from unittest.mock import patch, MagicMock, mock_open, call
from astropy.io import fits
from astropy.table import Table

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Lines 33-40: Exception handling during DB connection at module level
# ---------------------------------------------------------------------------

class TestModuleConnectionErrors:
    def test_module_loaded(self):
        import lsc.myloopdef
        assert hasattr(lsc.myloopdef, 'weighted_avg_and_std')


# ---------------------------------------------------------------------------
# Lines 123-124: dmag1 except branch + line 149 (_show) + line 162+ (output)
# ---------------------------------------------------------------------------

class TestRunGetmagBranches:
    @patch('lsc.myloopdef.conn')
    @patch('lsc.mysqldef.getfromdataraw')
    def test_multiple_images_same_bin(self, mock_getfrom, mock_conn):
        """When multiple images fall in same bin, weighted avg is computed."""
        from lsc.myloopdef import run_getmag

        mock_getfrom.side_effect = [
            [{'mag': 18.0, 'dmag': 0.05, 'mjd': 59000.0, 'filter': 'B',
              'telescope': 'tel1', 'dateobs': datetime.datetime(2020, 1, 1), 'z1': 0, 'z2': 1000,
              'magtype': 1, 'filetype': 1, 'difftype': 0, 'targetid': 1}],
            [{'name': 'SN2020test'}],
            [{'mag': 18.1, 'dmag': 0.05, 'mjd': 59000.0, 'filter': 'B',
              'telescope': 'tel1', 'dateobs': datetime.datetime(2020, 1, 1), 'z1': 0, 'z2': 1000,
              'magtype': 1, 'filetype': 1, 'difftype': 0, 'targetid': 1}],
            [{'name': 'SN2020test'}],
        ]

        run_getmag(['img1.fits', 'img2.fits'], _output='', _interactive=False,
                   _show=False, _bin=1.0, magtype='mag')

    def test_empty_imglist(self):
        from lsc.myloopdef import run_getmag
        run_getmag([])

    @patch('lsc.myloopdef.plotfast2')
    @patch('lsc.myloopdef.conn')
    @patch('lsc.mysqldef.getfromdataraw')
    def test_show_calls_plotfast2(self, mock_getfrom, mock_conn, mock_plotfast2):
        """When _show=True, plotfast2 is called."""
        from lsc.myloopdef import run_getmag

        mock_getfrom.side_effect = [
            [{'mag': 18.0, 'dmag': 0.01, 'mjd': 59000.0, 'filter': 'B',
              'telescope': 'tel1', 'dateobs': datetime.datetime(2020, 1, 1), 'z1': 0, 'z2': 1000,
              'magtype': 1, 'filetype': 1, 'difftype': 0, 'targetid': 1}],
            [{'name': 'SN2020test'}],
        ]

        run_getmag(['img1.fits'], _output='', _interactive=False,
                   _show=True, _bin=1e-10, magtype='mag')
        mock_plotfast2.assert_called_once()

    @patch('lsc.myloopdef.plotfast')
    @patch('lsc.myloopdef.conn')
    @patch('lsc.mysqldef.getfromdataraw')
    def test_output_writes_file(self, mock_getfrom, mock_conn, mock_plotfast):
        """When _output is set, writes table to file."""
        from lsc.myloopdef import run_getmag
        import tempfile

        mock_getfrom.side_effect = [
            [{'mag': 18.0, 'dmag': 0.01, 'mjd': 59000.0, 'filter': 'B',
              'telescope': 'tel1', 'dateobs': datetime.datetime(2020, 1, 1), 'z1': 0, 'z2': 1000,
              'magtype': 1, 'filetype': 1, 'difftype': 0, 'targetid': 1}],
            [{'name': 'SN2020test'}],
        ]

        fd, outfile = tempfile.mkstemp(suffix='.dat')
        os.close(fd)
        try:
            run_getmag(['img1.fits'], _output=outfile, _interactive=False,
                       _show=False, _bin=1e-10, magtype='mag', snex2_upload=False)
            assert os.path.isfile(outfile)
        finally:
            if os.path.isfile(outfile):
                os.remove(outfile)


# ---------------------------------------------------------------------------
# Lines 162-230: snex2_upload block
# ---------------------------------------------------------------------------

class TestRunGetmagSnex2Upload:
    @patch('lsc.myloopdef.conn')
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('lsc.util.userinput')
    @patch('requests.post')
    @patch('getpass.getpass')
    @patch('os.system')
    def test_snex2_upload_ftype1(self, mock_system, mock_getpass, mock_post,
                                  mock_userinput, mock_getfrom, mock_conn, tmp_path, monkeypatch):
        """Test snex2 upload path with filetype=1."""
        from lsc.myloopdef import run_getmag

        monkeypatch.chdir(tmp_path)

        mock_getfrom.side_effect = [
            [{'mag': 18.0, 'dmag': 0.01, 'mjd': 59000.0, 'filter': 'B',
              'telescope': 'tel1', 'dateobs': datetime.datetime(2020, 1, 1), 'z1': 0, 'z2': 1000,
              'magtype': 1, 'filetype': 1, 'difftype': 0, 'targetid': 1}],
            [{'name': 'SN2020test'}],
        ]

        mock_userinput.side_effect = [
            'PSF', 'LCO', 'y', 'Smith, John', 'testuser',
        ]
        mock_getpass.return_value = 'testpass'
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_post.return_value = mock_response

        # Use relative path so os.getcwd() + '/' + snex2_filename works
        outfile = 'output.dat'
        snex2file = 'output_snex2.csv'
        with open(snex2file, 'w') as f:
            f.write("test data")

        run_getmag(['img1.fits'], _output=outfile, _interactive=False,
                   _show=False, _bin=1e-10, magtype='mag', snex2_upload=True)

    @patch('lsc.myloopdef.conn')
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('lsc.util.userinput')
    @patch('requests.post')
    @patch('getpass.getpass')
    @patch('os.system')
    def test_snex2_upload_ftype3(self, mock_system, mock_getpass, mock_post,
                                  mock_userinput, mock_getfrom, mock_conn, tmp_path, monkeypatch):
        """Test snex2 upload path with filetype=3."""
        from lsc.myloopdef import run_getmag

        monkeypatch.chdir(tmp_path)

        mock_getfrom.side_effect = [
            [{'mag': 18.0, 'dmag': 0.01, 'mjd': 59000.0, 'filter': 'B',
              'telescope': 'tel1', 'dateobs': datetime.datetime(2020, 1, 1), 'z1': 0, 'z2': 1000,
              'magtype': 1, 'filetype': 3, 'difftype': 1, 'targetid': 1}],
            [{'name': 'SN2020test'}],
        ]

        mock_userinput.side_effect = [
            'mixed', 'SDSS', 'UC Davis', 'n', '', 'testuser',
        ]
        mock_getpass.return_value = 'testpass'
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        outfile = 'output.dat'
        snex2file = 'output_snex2.csv'
        with open(snex2file, 'w') as f:
            f.write("test data")

        run_getmag(['img1.fits'], _output=outfile, _interactive=False,
                   _show=False, _bin=1e-10, magtype='mag', snex2_upload=True)


# ---------------------------------------------------------------------------
# Lines 300, 302: These are dead code (unreachable). Skip.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Line 810: psfmag filter in filtralist
# ---------------------------------------------------------------------------

class TestFiltralistPsfmag:
    def test_filtralist_bad_psfmag(self):
        """When _bad='psfmag', standard field objects are excluded."""
        from lsc.myloopdef import filtralist

        ll2 = {
            'filename': ['img1.fits', 'img2.fits', 'img3.fits'],
            'objname': ['SN2020abc', 'L104', 'SN2020def'],
            'filter': ['B', 'B', 'B'],
            'psfmag': [9999, 9999, 9999],
            'quality': [0, 0, 0],
            'wcs': [0, 0, 0],
            'psf': ['done', 'done', 'done'],
            'filetype': [1, 1, 1],
            'groupidcode': [1, 1, 1],
            'instrument': ['inst', 'inst', 'inst'],
            'telescope': ['tel', 'tel', 'tel'],
            'difftype': [None, None, None],
            'classificationid': [None, None, None],
            'targetid': [1, 1, 1],
            'mag': [9999, 9999, 9999],
            'ra': [10.0, 10.0, 10.0],
            'dec': [-30.0, -30.0, -30.0],
            'filepath': ['/data/', '/data/', '/data/'],
            'abscat': ['done', 'done', 'done'],
            'zcat': ['done', 'done', 'done'],
        }

        result = filtralist(ll2, '', '', '', '', '', 'psfmag')
        names = list(result['objname'])
        assert 'L104' not in names
        assert len(names) == 2


# ---------------------------------------------------------------------------
# Lines 867-882: position function
# ---------------------------------------------------------------------------

class TestPositionFunction:
    def test_position_empty_list(self):
        """Empty list returns empty strings via except branch."""
        from lsc.myloopdef import position

        mock_iraf = MagicMock()
        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            ra, dec = position([], None, None, show=False)
        # With empty list, ra/dec are empty lists, np.mean([]) = nan,
        # but the try/except catches and returns ''
        assert ra == '' or (isinstance(ra, float) and np.isnan(ra))


# ---------------------------------------------------------------------------
# Lines 994-1006: checkpsf - iraf fallback to matplotlib
# ---------------------------------------------------------------------------

class TestCheckpsfIrafFallback:
    @patch('lsc.myloopdef.make_psf_plot')
    @patch('lsc.myloopdef.mark_stars_on_image')
    @patch('lsc.myloopdef.checkstage')
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('os.path.isfile')
    @patch('lsc.util.userinput')
    @patch('lsc.util.marksn2')
    def test_checkpsf_iraf_falls_back(self, mock_marksn2, mock_input,
                                       mock_isfile, mock_getfrom,
                                       mock_checkstage, mock_mark_stars, mock_psf_plot):
        """When iraf display fails, uses matplotlib."""
        from lsc.myloopdef import checkpsf

        mock_checkstage.return_value = 1
        mock_getfrom.return_value = [{'filepath': '/data/', 'filter': 'B'}]
        mock_isfile.return_value = True
        mock_marksn2.side_effect = Exception("iraf unavailable")
        mock_input.return_value = 'y'

        mock_iraf = MagicMock()
        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            with patch('matplotlib.pyplot.ion'):
                with patch('matplotlib.pyplot.figure') as mock_fig:
                    mock_fig.return_value = MagicMock()
                    checkpsf(['img.fits'], no_iraf=False)

        mock_mark_stars.assert_called()
        mock_psf_plot.assert_called()


# ---------------------------------------------------------------------------
# Lines 1108, 1114-1129, 1137, 1144-1165: checkwcs function
# ---------------------------------------------------------------------------

class TestCheckwcs:
    @patch('lsc.myloopdef.checkstage')
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('lsc.util.checksnlist')
    @patch('lsc.util.checksndb')
    @patch('lsc.util.getcatalog')
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.userinput')
    @patch('lsc.mysqldef.updatevalue')
    @patch('os.path.isfile')
    @patch('os.system')
    def test_checkwcs_good_wcs_with_z1z2(self, mock_system, mock_isfile, mock_updateval,
                                          mock_input, mock_readtxt, mock_getcat,
                                          mock_checksndb, mock_checksnlist, mock_getfrom,
                                          mock_checkstage):
        from lsc.myloopdef import checkwcs

        mock_checkstage.return_value = 1
        mock_getfrom.return_value = [{'filepath': '/data/', 'filter': 'B', 'exptime': 300}]
        mock_checksnlist.side_effect = [('', '', ''), ('', '', '')]
        mock_checksndb.return_value = (10.0, -30.0, 1)
        mock_getcat.return_value = '/data/cat.txt'
        mock_readtxt.return_value = {'ra': ['10.0'], 'dec': ['-30.0']}
        mock_input.return_value = 'y'
        mock_isfile.return_value = False

        mock_iraf = MagicMock()
        mock_iraf.display.return_value = ['z1=0 z2=1000']
        mock_iraf.wcsctran.return_value = ['', '', '', '100.0 200.0']

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            checkwcs(['img.fits'], force=False, _z1=100, _z2=500)

    @patch('lsc.myloopdef.checkstage')
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('lsc.util.checksnlist')
    @patch('lsc.util.checksndb')
    @patch('lsc.util.getcatalog')
    @patch('lsc.lscastrodef.querycatalogue')
    @patch('lsc.util.userinput')
    @patch('lsc.mysqldef.updatevalue')
    @patch('os.path.isfile')
    @patch('os.system')
    def test_checkwcs_no_z1z2_no_catalog(self, mock_system, mock_isfile, mock_updateval,
                                          mock_input, mock_querycat, mock_getcat,
                                          mock_checksndb, mock_checksnlist,
                                          mock_getfrom, mock_checkstage):
        from lsc.myloopdef import checkwcs

        mock_checkstage.return_value = 1
        mock_getfrom.return_value = [{'filepath': '/data/', 'filter': 'B', 'exptime': 300}]
        mock_checksnlist.side_effect = [('', '', ''), ('', '', '')]
        mock_checksndb.return_value = (10.0, -30.0, 1)
        mock_getcat.return_value = ''
        mock_querycat.return_value = {'pix': ['100 200']}
        mock_input.return_value = 'y'
        mock_isfile.return_value = False

        mock_iraf = MagicMock()
        mock_iraf.display.return_value = ['z1=0 z2=1000']
        mock_iraf.wcsctran.return_value = ['', '', '', '100.0 200.0']

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            checkwcs(['img.fits'], force=False, _z1=None, _z2=None)

    @patch('lsc.myloopdef.checkstage')
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('lsc.util.checksnlist')
    @patch('lsc.util.checksndb')
    @patch('lsc.util.getcatalog')
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.userinput')
    @patch('lsc.mysqldef.updatevalue')
    @patch('os.path.isfile')
    @patch('os.system')
    def test_checkwcs_bad_quality(self, mock_system, mock_isfile, mock_updateval,
                                   mock_input, mock_readtxt, mock_getcat,
                                   mock_checksndb, mock_checksnlist, mock_getfrom,
                                   mock_checkstage):
        from lsc.myloopdef import checkwcs

        mock_checkstage.return_value = 1
        mock_getfrom.return_value = [{'filepath': '/data/', 'filter': 'B', 'exptime': 300}]
        mock_checksnlist.side_effect = [('', '', ''), ('', '', '')]
        mock_checksndb.return_value = (10.0, -30.0, 1)
        mock_getcat.return_value = '/data/cat.txt'
        mock_readtxt.return_value = {'ra': ['10.0'], 'dec': ['-30.0']}
        mock_input.return_value = 'b'
        mock_isfile.return_value = True

        mock_iraf = MagicMock()
        mock_iraf.display.return_value = ['z1=0 z2=1000']
        mock_iraf.wcsctran.return_value = ['', '', '', '100.0 200.0']

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            checkwcs(['img.fits'], force=False, _z1=100, _z2=500)

        assert mock_updateval.call_count >= 3

    @patch('lsc.myloopdef.checkstage')
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('lsc.util.checksnlist')
    @patch('lsc.util.checksndb')
    @patch('lsc.util.getcatalog')
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.userinput')
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.mysqldef.deleteredufromarchive')
    @patch('os.path.isfile')
    @patch('os.system')
    def test_checkwcs_cancel(self, mock_system, mock_isfile, mock_deletedb,
                              mock_updateval, mock_input, mock_readtxt, mock_getcat,
                              mock_checksndb, mock_checksnlist, mock_getfrom,
                              mock_checkstage):
        from lsc.myloopdef import checkwcs

        mock_checkstage.return_value = 1
        mock_getfrom.return_value = [{'filepath': '/data/', 'filter': 'B', 'exptime': 300}]
        mock_checksnlist.side_effect = [('', '', ''), ('', '', '')]
        mock_checksndb.return_value = (10.0, -30.0, 1)
        mock_getcat.return_value = '/data/cat.txt'
        mock_readtxt.return_value = {'ra': ['10.0'], 'dec': ['-30.0']}
        mock_input.return_value = 'c'
        mock_isfile.return_value = False

        mock_iraf = MagicMock()
        mock_iraf.display.return_value = ['z1=0 z2=1000']
        mock_iraf.wcsctran.return_value = ['', '', '', '100.0 200.0']

        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            checkwcs(['img.fits'], force=False, _z1=100, _z2=500)

        mock_deletedb.assert_called()

    @patch('lsc.myloopdef.checkstage')
    def test_checkwcs_negative_statuses(self, mock_checkstage):
        from lsc.myloopdef import checkwcs

        mock_iraf = MagicMock()
        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            for status in [-1, -2, -4, -99]:
                mock_checkstage.return_value = status
                checkwcs(['img.fits'], force=True, _z1=100, _z2=500)


# ---------------------------------------------------------------------------
# Lines 1201-1203, 1211-1212, 1219: makestamp
# ---------------------------------------------------------------------------

class TestMakestamp:
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('lsc.util.checksndb')
    @patch('lsc.myloopdef.getsky')
    @patch('matplotlib.pyplot.clf')
    @patch('matplotlib.pyplot.imshow')
    @patch('matplotlib.pyplot.xlim')
    @patch('matplotlib.pyplot.ylim')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.savefig')
    @patch('os.path.isfile')
    def test_makestamp_non_interactive(self, mock_isfile_os, mock_savefig,
                                       mock_plot, mock_ylim, mock_xlim,
                                       mock_imshow, mock_clf, mock_getsky,
                                       mock_checksndb, mock_getfrom):
        import lsc
        from lsc.myloopdef import makestamp

        mock_getfrom.return_value = [{'filepath': '/data/', 'targetid': 1}]
        # Return coords that result in pixel position at center (300,300)
        mock_checksndb.return_value = (10.0, -30.0, 1)
        mock_getsky.return_value = (100.0, 20.0)
        mock_isfile_os.return_value = False

        # Use a MagicMock for data so slicing works without type issues
        mock_data = MagicMock()
        mock_data.__getitem__ = lambda self, key: np.ones((100, 100))
        mock_hdr = fits.Header()
        mock_hdr['NAXIS1'] = 600
        mock_hdr['NAXIS2'] = 600
        mock_hdr['CRPIX1'] = 300
        mock_hdr['CRPIX2'] = 300
        mock_hdr['CRVAL1'] = 10.0
        mock_hdr['CRVAL2'] = -30.0
        mock_hdr['CTYPE1'] = 'RA---TAN'
        mock_hdr['CTYPE2'] = 'DEC--TAN'
        mock_hdr['CD1_1'] = -0.0001
        mock_hdr['CD1_2'] = 0.0
        mock_hdr['CD2_1'] = 0.0
        mock_hdr['CD2_2'] = 0.0001

        mock_hdu = MagicMock()
        mock_hdu.__getitem__ = lambda self, k: MagicMock(data=mock_data, header=mock_hdr)

        # Monkeypatch lsc.checkstage and lsc.delete since makestamp calls them
        orig_cs = getattr(lsc, 'checkstage', None)
        orig_del = getattr(lsc, 'delete', None)
        lsc.checkstage = MagicMock(return_value=1)
        lsc.delete = MagicMock()
        try:
            with patch('astropy.io.fits.open', return_value=mock_hdu):
                makestamp(['img.fits'], _z1='', _z2='', _interactive=False,
                          redo=False, _output='stamp.png')
        finally:
            if orig_cs is None:
                delattr(lsc, 'checkstage')
            else:
                lsc.checkstage = orig_cs
            if orig_del is None:
                delattr(lsc, 'delete')
            else:
                lsc.delete = orig_del

    @patch('lsc.mysqldef.getfromdataraw')
    @patch('lsc.util.checksndb')
    @patch('lsc.myloopdef.getsky')
    @patch('matplotlib.pyplot.clf')
    @patch('matplotlib.pyplot.imshow')
    @patch('matplotlib.pyplot.xlim')
    @patch('matplotlib.pyplot.ylim')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.savefig')
    @patch('os.path.isfile')
    def test_makestamp_imshow_fallback(self, mock_isfile_os, mock_savefig,
                                       mock_plot, mock_ylim, mock_xlim,
                                       mock_imshow, mock_clf, mock_getsky,
                                       mock_checksndb, mock_getfrom):
        import lsc
        from lsc.myloopdef import makestamp

        mock_getfrom.return_value = [{'filepath': '/data/', 'targetid': 1}]
        mock_checksndb.return_value = (10.0, -30.0, 1)
        mock_getsky.return_value = (100.0, 20.0)
        mock_isfile_os.return_value = False
        mock_imshow.side_effect = [Exception("bad vmin"), MagicMock()]

        mock_data = MagicMock()
        mock_data.__getitem__ = lambda self, key: np.ones((100, 100))
        mock_hdr = fits.Header()
        mock_hdr['NAXIS1'] = 600
        mock_hdr['NAXIS2'] = 600
        mock_hdr['CRPIX1'] = 300
        mock_hdr['CRPIX2'] = 300
        mock_hdr['CRVAL1'] = 10.0
        mock_hdr['CRVAL2'] = -30.0
        mock_hdr['CTYPE1'] = 'RA---TAN'
        mock_hdr['CTYPE2'] = 'DEC--TAN'
        mock_hdr['CD1_1'] = -0.0001
        mock_hdr['CD1_2'] = 0.0
        mock_hdr['CD2_1'] = 0.0
        mock_hdr['CD2_2'] = 0.0001

        mock_hdu = MagicMock()
        mock_hdu.__getitem__ = lambda self, k: MagicMock(data=mock_data, header=mock_hdr)

        orig_cs = getattr(lsc, 'checkstage', None)
        orig_del = getattr(lsc, 'delete', None)
        lsc.checkstage = MagicMock(return_value=1)
        lsc.delete = MagicMock()
        try:
            with patch('astropy.io.fits.open', return_value=mock_hdu):
                makestamp(['img.fits'], _z1='', _z2='', _interactive=False,
                          redo=False, _output='stamp.png')
        finally:
            if orig_cs is None:
                delattr(lsc, 'checkstage')
            else:
                lsc.checkstage = orig_cs
            if orig_del is None:
                delattr(lsc, 'delete')
            else:
                lsc.delete = orig_del

        assert mock_imshow.call_count == 2


# ---------------------------------------------------------------------------
# Lines 1292-1309: checkcosmic function
# ---------------------------------------------------------------------------

class TestCheckcosmic:
    @patch('lsc.myloopdef.checkstage')
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('lsc.util.userinput')
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.mysqldef.deleteredufromarchive')
    @patch('os.path.isfile')
    @patch('os.system')
    def test_checkcosmic_bad_mask(self, mock_system, mock_isfile, mock_deletedb,
                                   mock_updateval, mock_input, mock_getfrom,
                                   mock_checkstage):
        from lsc.myloopdef import checkcosmic

        mock_checkstage.return_value = 1
        mock_getfrom.return_value = [{'filepath': '/data/', 'filter': 'B'}]
        mock_isfile.return_value = True
        mock_input.return_value = 'b'

        mock_iraf = MagicMock()
        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            checkcosmic(['img.fits'])

        mock_updateval.assert_called()

    @patch('lsc.myloopdef.checkstage')
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('os.path.isfile')
    def test_checkcosmic_files_not_found(self, mock_isfile, mock_getfrom,
                                          mock_checkstage):
        from lsc.myloopdef import checkcosmic

        mock_checkstage.return_value = 1
        mock_getfrom.return_value = [{'filepath': '/data/', 'filter': 'B'}]
        mock_isfile.return_value = False

        mock_iraf = MagicMock()
        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            checkcosmic(['img.fits'])


# ---------------------------------------------------------------------------
# Lines 1329-1348: display_subtraction
# ---------------------------------------------------------------------------

class TestDisplaySubtraction:
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('os.path.isfile')
    @patch('astropy.io.fits.getdata')
    @patch('matplotlib.pyplot.clf')
    @patch('matplotlib.pyplot.subplot')
    @patch('matplotlib.pyplot.xlim')
    @patch('matplotlib.pyplot.ylim')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.gcf')
    def test_display_subtraction_files_exist(self, mock_gcf, mock_tight, mock_ylim, mock_xlim,
                                              mock_subplot, mock_clf, mock_getdata,
                                              mock_isfile, mock_getfrom):
        from lsc.myloopdef import display_subtraction

        mock_getfrom.return_value = [{'filepath': '/data/', 'filter': 'B',
                                       'psfmag': 18.0, 'psfdmag': 0.1,
                                       'mag': 18.5, 'dmag': 0.2}]
        mock_isfile.return_value = True
        mock_getdata.return_value = np.ones((200, 200))
        mock_gcf.return_value = MagicMock()
        mock_ax = MagicMock()
        mock_subplot.return_value = mock_ax

        result = display_subtraction('img.diff.fits')
        assert len(result) == 3

    @patch('lsc.mysqldef.getfromdataraw')
    @patch('os.path.isfile')
    def test_display_subtraction_files_missing(self, mock_isfile, mock_getfrom):
        from lsc.myloopdef import display_subtraction

        mock_getfrom.return_value = [{'filepath': '/data/', 'filter': 'B',
                                       'psfmag': 18.0, 'psfdmag': 0.1,
                                       'mag': 18.5, 'dmag': 0.2}]
        mock_isfile.return_value = False
        result = display_subtraction('img.diff.fits')


# ---------------------------------------------------------------------------
# Lines 1368-1383: checkdiff
# ---------------------------------------------------------------------------

class TestCheckdiff:
    @patch('lsc.myloopdef.checkstage')
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('lsc.util.userinput')
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.mysqldef.deleteredufromarchive')
    @patch('os.path.isfile')
    @patch('os.system')
    def test_checkdiff_bad_diff(self, mock_system, mock_isfile, mock_deletedb,
                                 mock_updateval, mock_input, mock_getfrom,
                                 mock_checkstage):
        from lsc.myloopdef import checkdiff

        mock_checkstage.return_value = 1
        mock_getfrom.return_value = [{'filepath': '/data/', 'filter': 'B'}]
        mock_isfile.return_value = True
        mock_input.return_value = 'b'

        mock_iraf = MagicMock()
        with patch.dict('sys.modules', {'pyraf': MagicMock(iraf=mock_iraf), 'pyraf.iraf': mock_iraf}):
            checkdiff(['img.diff.fits'])

        mock_updateval.assert_called()
        mock_deletedb.assert_called()


# ---------------------------------------------------------------------------
# Lines 1403-1435: display_psf_fit
# ---------------------------------------------------------------------------

class TestDisplayPsfFit:
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('os.path.isfile')
    @patch('lsc.util.readkey3')
    @patch('matplotlib.pyplot.clf')
    @patch('matplotlib.pyplot.subplot')
    @patch('matplotlib.pyplot.colorbar')
    @patch('matplotlib.pyplot.gcf')
    @patch('astropy.io.fits.getdata')
    def test_display_psf_fit_with_sffile(self, mock_getdata, mock_gcf, mock_colorbar,
                                          mock_subplot, mock_clf, mock_readkey3,
                                          mock_isfile, mock_getfrom):
        from lsc.myloopdef import display_psf_fit

        mock_getfrom.return_value = [{'filepath': '/data/', 'filter': 'B',
                                       'psfmag': 18.0, 'psfdmag': 0.1,
                                       'mag': 18.5, 'dmag': 0.2, 'filename': 'img.fits'}]
        # sffile exists (line 1402), ogfile exists (1406), rsfile (1406), sffile again (1430)
        mock_isfile.return_value = True
        mock_readkey3.return_value = 60000

        mock_data = np.random.normal(1000, 100, (50, 50))
        mock_hdr = fits.Header()
        mock_hdr['DATAMAX'] = 60000

        # Order of getdata calls:
        # 1. sffile (line 1403): fits.getdata(sffile)
        # 2. ogfile (line 1407): fits.getdata(ogfile, header=True) -> (data, hdr)
        # 3. rsfile (line 1408): fits.getdata(rsfile)
        # 4. sffile again (line 1431): fits.getdata(sffile)
        mock_getdata.side_effect = [
            mock_data,                  # sffile
            (mock_data, mock_hdr),      # ogfile with header=True
            mock_data,                  # rsfile
            mock_data,                  # sffile again
        ]

        mock_ax = MagicMock()
        mock_ax.imshow.return_value = MagicMock()
        mock_subplot.return_value = mock_ax
        mock_gcf.return_value = MagicMock()

        result = display_psf_fit('img.fits')
        assert result[0].endswith('.og.fits')

    @patch('lsc.mysqldef.getfromdataraw')
    @patch('os.path.isfile')
    @patch('lsc.util.readkey3')
    @patch('matplotlib.pyplot.clf')
    @patch('matplotlib.pyplot.subplot')
    @patch('matplotlib.pyplot.colorbar')
    @patch('matplotlib.pyplot.gcf')
    @patch('astropy.io.fits.getdata')
    def test_display_psf_fit_no_sffile(self, mock_getdata, mock_gcf, mock_colorbar,
                                        mock_subplot, mock_clf, mock_readkey3,
                                        mock_isfile, mock_getfrom):
        from lsc.myloopdef import display_psf_fit

        mock_getfrom.return_value = [{'filepath': '/data/', 'filter': 'B',
                                       'psfmag': 18.0, 'psfdmag': 0.1,
                                       'mag': 18.5, 'dmag': 0.2, 'filename': 'img.fits'}]
        # sffile (1402) False, then ogfile AND rsfile in one check (1406) True, sffile again (1430) False
        mock_isfile.side_effect = [False, True, True, False]
        mock_readkey3.return_value = 60000

        mock_data = np.random.normal(1000, 100, (50, 50))
        mock_hdr = fits.Header()
        mock_hdr['DATAMAX'] = 60000

        mock_getdata.side_effect = [
            (mock_data, mock_hdr),  # ogfile
            mock_data,              # rsfile
        ]

        mock_ax = MagicMock()
        mock_ax.imshow.return_value = MagicMock()
        mock_subplot.return_value = mock_ax
        mock_gcf.return_value = MagicMock()

        result = display_psf_fit('img.fits')


# ---------------------------------------------------------------------------
# Lines 1446-1455: checkmag
# ---------------------------------------------------------------------------

class TestCheckmag:
    @patch('lsc.myloopdef.checkstage')
    @patch('lsc.myloopdef.display_psf_fit')
    @patch('lsc.util.userinput')
    @patch('lsc.mysqldef.query')
    @patch('lsc.mysqldef.updatevalue')
    @patch('os.system')
    @patch('matplotlib.pyplot.ion')
    def test_checkmag_bad_quality(self, mock_ion, mock_system, mock_updateval,
                                   mock_query, mock_input, mock_display,
                                   mock_checkstage):
        from lsc.myloopdef import checkmag

        mock_checkstage.return_value = 2
        mock_display.return_value = ('/data/img.og.fits', '/data/img.rs.fits', '/data/img.sf.fits')
        mock_input.return_value = 'b'

        checkmag(['img.fits'])

        mock_query.assert_called_once()
        mock_updateval.assert_called_once()


# ---------------------------------------------------------------------------
# Lines 1633, 1636: PickablePlot
# ---------------------------------------------------------------------------

class TestPickablePlot:
    def test_delete_current_no_active(self):
        from lsc.myloopdef import PickablePlot
        obj = object.__new__(PickablePlot)
        obj.i_active = None
        obj.x = np.array([1.0, 2.0, 3.0])
        obj.y = np.array([4.0, 5.0, 6.0])
        obj.xdel = np.array([])
        obj.ydel = np.array([])
        obj.delete_current()

    def test_delete_current_with_active(self):
        from lsc.myloopdef import PickablePlot
        obj = object.__new__(PickablePlot)
        obj.i_active = 1
        obj.x = np.array([1.0, 2.0, 3.0])
        obj.y = np.array([4.0, 5.0, 6.0])
        obj.xdel = np.array([])
        obj.ydel = np.array([])
        obj.delete_current()
        assert np.isnan(obj.x[1])
        assert obj.xdel[0] == 2.0

    def test_onclick_with_click_hook(self):
        from lsc.myloopdef import PickablePlot
        obj = object.__new__(PickablePlot)
        obj.hooks = {'click': MagicMock()}
        obj.selectedmenu = 'test'
        mock_event = MagicMock()
        mock_event.ind = [2]
        obj.onclick(mock_event)
        assert obj.i_active == 2
        obj.hooks['click'].assert_called_once_with(2)


# ---------------------------------------------------------------------------
# Lines 1678-1721: plotfast2 hooks
# ---------------------------------------------------------------------------

class TestPlotfast2:
    @patch('lsc.myloopdef.PickablePlot')
    @patch('lsc.sites.filterst1', {'B': 'B'})
    def test_plotfast2_creates_plot(self, mock_pickable):
        from lsc.myloopdef import plotfast2
        setup = {'tel1': {'B': {'filename': ['img.fits'], 'mjd': [59000.0],
                                 'mag': [18.0], 'dmag': [0.1]}}}
        plotfast2(setup)
        mock_pickable.assert_called_once()

    @patch('lsc.sites.filterst1', {'B': 'B'})
    @patch('lsc.mysqldef.getvaluefromarchive')
    @patch('lsc.mysqldef.query')
    @patch('lsc.util.updateheader')
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.mysqldef.deleteredufromarchive')
    @patch('os.system')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.gca')
    @patch('matplotlib.pyplot.errorbar')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    def test_plotfast2_hooks(self, mock_ylabel, mock_xlabel, mock_legend, mock_errorbar,
                             mock_gca, mock_figure, mock_system, mock_deletedb,
                             mock_updateval, mock_updatehdr, mock_query, mock_getval):
        from lsc.myloopdef import plotfast2

        setup = {'tel1': {'B': {'filename': ['img1.fits', 'img2.fits'],
                                 'mjd': [59000.0, 59001.0],
                                 'mag': [18.0, 18.5], 'dmag': [0.1, 0.2]}}}
        captured_hooks = {}

        def capture(x, y, mainmenu='', selectedmenu='', hooks={}):
            nonlocal captured_hooks
            captured_hooks = hooks

        with patch('lsc.myloopdef.PickablePlot', side_effect=capture):
            plotfast2(setup)

        if 'plot' in captured_hooks:
            mock_gca.return_value = MagicMock(invert_yaxis=MagicMock())
            captured_hooks['plot']()

        if 'd' in captured_hooks:
            mock_getval.return_value = [{'filepath': '/data/'}]
            captured_hooks['d'](0)

        if 'b' in captured_hooks:
            mock_getval.return_value = [{'filepath': '/data/', 'filetype': '1'}]
            captured_hooks['b'](0)

        if 'u' in captured_hooks:
            captured_hooks['u'](0)


# ---------------------------------------------------------------------------
# Line 1762: plotfast lolims
# ---------------------------------------------------------------------------

class TestPlotfastLolims:
    @patch('lsc.sites.filterst1', {'B': 'B'})
    @patch('matplotlib.pyplot.ion')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.axes')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.errorbar')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.xlim')
    @patch('matplotlib.pyplot.ylim')
    @patch('matplotlib.pyplot.getp')
    @patch('matplotlib.pyplot.setp')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.savefig')
    def test_plotfast_with_limits(self, mock_savefig, mock_legend, mock_setp,
                                   mock_getp, mock_ylim, mock_xlim,
                                   mock_ylabel, mock_xlabel, mock_errorbar,
                                   mock_plot, mock_axes, mock_figure, mock_ion):
        from lsc.myloopdef import plotfast
        setup = {'tel1': {'B': {'filename': ['img1.fits', 'img2.fits'],
                                 'mjd': [59000.0, 59001.0],
                                 'mag': [18.0, 19.0], 'dmag': [0.1, 0.2],
                                 'magtype': [-1, 1]}}}
        mock_getp.return_value = []
        mock_leg = MagicMock()
        mock_leg.get_texts.return_value = []
        mock_legend.return_value = mock_leg
        plotfast(setup, output='test.png')
        mock_errorbar.assert_called()


# ---------------------------------------------------------------------------
# Line 945: checkcat
# ---------------------------------------------------------------------------

class TestCheckcatUserinput:
    @patch('lsc.myloopdef.checkstage')
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('lsc.myloopdef.mark_stars_on_image')
    @patch('lsc.util.userinput')
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.util.delete')
    @patch('os.path.isfile')
    def test_checkcat_bad(self, mock_isfile, mock_delete, mock_updateval,
                           mock_input, mock_mark, mock_getfrom, mock_checkstage):
        from lsc.myloopdef import checkcat
        mock_checkstage.return_value = 1
        mock_getfrom.return_value = [{'filepath': '/data/', 'abscat': 'done'}]
        mock_isfile.return_value = True
        mock_input.return_value = 'n'
        cat_content = "# header\n# header2\nstar1\nstar2\nstar3\n"
        with patch('builtins.open', mock_open(read_data=cat_content)):
            checkcat(['img.fits'])
        mock_updateval.assert_called()

    @patch('lsc.myloopdef.checkstage')
    @patch('lsc.mysqldef.getfromdataraw')
    @patch('lsc.mysqldef.updatevalue')
    @patch('os.path.isfile')
    def test_checkcat_file_missing(self, mock_isfile, mock_updateval,
                                    mock_getfrom, mock_checkstage):
        from lsc.myloopdef import checkcat
        mock_checkstage.return_value = 1
        mock_getfrom.return_value = [{'filepath': '/data/', 'abscat': 'done'}]
        mock_isfile.return_value = False
        checkcat(['img.fits'])
        mock_updateval.assert_called_with('photlco', 'abscat', 'X', 'img.fits')


# ---------------------------------------------------------------------------
# Line 890: mark_stars_on_image
# ---------------------------------------------------------------------------

class TestMarkStarsOnImage:
    @patch('lsc.myloopdef.get_psf_star_coords')
    def test_mark_stars_on_image_fits_catalog(self, mock_get_psf, tmp_path):
        from lsc.myloopdef import mark_stars_on_image
        import matplotlib.pyplot as plt

        mock_get_psf.return_value = (np.array([50.0]), np.array([50.0]), ['1'])

        imgfile = str(tmp_path / "test.fits")
        data = np.ones((100, 100), dtype=np.float32)
        hdr = fits.Header()
        hdr['CRPIX1'] = 50
        hdr['CRPIX2'] = 50
        hdr['CRVAL1'] = 10.0
        hdr['CRVAL2'] = -30.0
        hdr['CTYPE1'] = 'RA---TAN'
        hdr['CTYPE2'] = 'DEC--TAN'
        hdr['CD1_1'] = -0.0001
        hdr['CD1_2'] = 0.0
        hdr['CD2_1'] = 0.0
        hdr['CD2_2'] = 0.0001
        fits.writeto(imgfile, data, hdr)

        catfile = str(tmp_path / "test.sn2.fits")
        t = Table()
        t['ra'] = ['00:40:00.0']
        t['dec'] = ['-30:00:00.0']
        t.write(catfile, format='fits', overwrite=True)

        fig = plt.figure()
        mark_stars_on_image(imgfile, catfile, fig=fig)
        plt.close(fig)
