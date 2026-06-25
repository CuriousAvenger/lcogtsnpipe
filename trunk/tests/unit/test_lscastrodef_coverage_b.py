"""
Tests for lsc.lscastrodef.zeropoint (lines 482-795).

Exercises all branches:
- Filter in BVRI (landolt path)
- Filter in SDSS-U/G/R/I/Pan-Starrs-Z (sloan path)
- _field='sloan' with BVRI filter (dummy path)
- _field='landolt' with SDSS filter (dummy path)
- catalogue provided vs not
- xstdL>=1 vs xstdS>=1 vs neither
- len(xstd0)>1 vs <=1 (standard stars in field or not)
- photometry success vs exception
- len(colore)==0, ==1, >1
- verbose=True and verbose=False
"""
import os
import sys
import math
import types
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open, call

pytestmark = pytest.mark.unit


# =============================================================================
# Helpers
# =============================================================================

def _make_mock_hdr(filter_val='V', siteid='lsc', ra=180.0, dec=30.0,
                   naxis1=2048, naxis2=2048, airmass=1.2, exptime=100,
                   instrume='fl03', date_night='20200101', obj='test_field',
                   datamax=60000, datamin=-100, ron=10.0, gain=1.5):
    """Create a mock FITS header dict-like object."""
    data = {
        'AIRMASS': airmass,
        'airmass': airmass,
        'exptime': exptime,
        'filter': filter_val,
        'instrume': instrume,
        'date-night': date_night,
        'object': obj,
        'datamax': datamax,
        'datamin': datamin,
        'SITEID': siteid,
        'RA': ra,
        'DEC': dec,
        'naxis1': naxis1,
        'naxis2': naxis2,
        'ron': ron,
        'gain': gain,
    }
    hdr = MagicMock()
    hdr.__getitem__ = lambda self, key: data[key]
    hdr.__contains__ = lambda self, key: key in data
    hdr.get = lambda key, default=None: data.get(key, default)
    return hdr, data


def _readkey3_factory(data):
    """Return a readkey3 mock function that looks up keys in data dict."""
    def _readkey3(hdr, keyword):
        return data.get(keyword, None)
    return _readkey3


def _make_standard_stdcoo(field='landolt', n=5):
    """Create fake stdcoo dict with photometry data."""
    stdcoo = {
        'ra': [str(180.0 + i * 0.001) for i in range(n)],
        'dec': [str(30.0 + i * 0.001) for i in range(n)],
        'id': [f'star_{i}' for i in range(n)],
    }
    if field == 'landolt':
        for f in 'UBVRI':
            stdcoo[f] = [str(12.0 + i * 0.1) for i in range(n)]
        stdcoo['BV'] = [str(0.5 + i * 0.05) for i in range(n)]
        stdcoo['UB'] = [str(0.3 + i * 0.02) for i in range(n)]
        stdcoo['VR'] = [str(0.4 + i * 0.03) for i in range(n)]
        stdcoo['RI'] = [str(0.2 + i * 0.01) for i in range(n)]
        stdcoo['VI'] = [str(0.6 + i * 0.04) for i in range(n)]
    elif field == 'sloan':
        for f in 'ugriz':
            stdcoo[f] = [str(14.0 + i * 0.1) for i in range(n)]
        stdcoo['ug'] = [str(0.5 + i * 0.05) for i in range(n)]
        stdcoo['gr'] = [str(0.4 + i * 0.03) for i in range(n)]
        stdcoo['ri'] = [str(0.3 + i * 0.02) for i in range(n)]
        stdcoo['iz'] = [str(0.2 + i * 0.01) for i in range(n)]
        stdcoo['rz'] = [str(0.4 + i * 0.03) for i in range(n)]
    return stdcoo


def _make_standardpix(n=5, xrange=(100, 1900), yrange=(100, 1900)):
    """Create fake standard pixel coordinates (within frame)."""
    xs = [str(xrange[0] + i * (xrange[1] - xrange[0]) / max(n - 1, 1)) for i in range(n)]
    ys = [str(yrange[0] + i * (yrange[1] - yrange[0]) / max(n - 1, 1)) for i in range(n)]
    return {
        'ra': xs,
        'dec': ys,
        'id': [f'star_{i}' for i in range(n)],
    }


def _make_outside_pix(n=3):
    """Create pixel coords outside field (negative)."""
    return {
        'ra': [str(-100 * (i + 1)) for i in range(n)],
        'dec': [str(-100 * (i + 1)) for i in range(n)],
        'id': [f's{i}' for i in range(n)],
    }


def _setup_iraf_mock():
    """Setup comprehensive iraf mock."""
    iraf = MagicMock()
    iraf.noao = MagicMock()
    iraf.digiphot = MagicMock()
    iraf.daophot = MagicMock()
    iraf.images = MagicMock()
    iraf.imcoords = MagicMock()
    iraf.proto = MagicMock()
    iraf.wcsctran = MagicMock()
    iraf.noao.digiphot.daophot.phot = MagicMock(
        return_value=['img 500.0 600.0 3.0 18.5 0.02']
    )
    return iraf


def _make_sex_fields_mock(n_sex, phot_fail=False):
    """Create fields mock that returns numeric arrays for sextractor output.

    iraf.proto.fields returns values; in production these are numeric.
    The code does array(flags)==0, so flags must be integers.
    """
    xsex = [100.0 + i * 200.0 for i in range(n_sex)]
    ysex = [100.0 + i * 200.0 for i in range(n_sex)]
    fw = [3.5] * n_sex
    flags = [0] * n_sex
    ra_wcs = [180.0 + i * 0.001 for i in range(n_sex)]
    dec_wcs = [30.0 + i * 0.001 for i in range(n_sex)]

    def mock_fields(filename, fields='1', Stdout=None):
        fname = str(filename)
        if 'detections.cat' in fname:
            if fields == '2':
                return xsex
            elif fields == '3':
                return ysex
            elif fields == '8':
                return fw
            elif fields == '6':
                return flags
        elif 'detection_sex.coo' in fname:
            if fields == '1':
                return ra_wcs
            elif fields == '2':
                return dec_wcs
        return []

    return mock_fields, xsex, ysex


def _run_zeropoint(filter_val, field, siteid, iraf_mock, mock_readtxt_fn,
                   mock_readhdr, mock_readkey3, mock_sloan2file,
                   n_sex, crossmatch_result, phot_return, verbose=False,
                   catalogue='', readtxt_side=None, extra_patches=None):
    """Common runner for zeropoint calls."""
    hdr, data = _make_mock_hdr(filter_val=filter_val, siteid=siteid)
    mock_readhdr.return_value = hdr
    mock_readkey3.side_effect = _readkey3_factory(data)

    if readtxt_side:
        mock_readtxt_fn.side_effect = readtxt_side

    mock_fields, _, _ = _make_sex_fields_mock(n_sex)
    iraf_mock.proto.fields = MagicMock(side_effect=mock_fields)

    if phot_return == 'fail':
        iraf_mock.noao.digiphot.daophot.phot = MagicMock(
            side_effect=Exception("INDEF")
        )
    else:
        iraf_mock.noao.digiphot.daophot.phot = MagicMock(
            return_value=[phot_return]
        )

    modules = {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}
    if verbose:
        modules['pylab'] = MagicMock()
        modules['matplotlib'] = MagicMock()
        modules['matplotlib.pyplot'] = MagicMock()

    if extra_patches:
        with extra_patches:
            with patch.dict('sys.modules', modules):
                sys.modules['pyraf'].iraf = iraf_mock
                import lsc.lscastrodef
                return lsc.lscastrodef.zeropoint('test_img.fits', field,
                                                  verbose=verbose, catalogue=catalogue)
    else:
        with patch.dict('sys.modules', modules):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            return lsc.lscastrodef.zeropoint('test_img.fits', field,
                                              verbose=verbose, catalogue=catalogue)


# =============================================================================
# Tests: Landolt field with standard stars in frame
# =============================================================================

class TestZeropointLandoltWithStars:
    """Test zeropoint with landolt field, BVRI filters, standards in field."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.lscastrodef.half_total_flux_radius_to_fwhm', return_value=4.0)
    @patch('lsc.lscastrodef.crossmatch')
    @patch('lsc.lscastrodef.linreg', return_value=(0.05, 25.0, 0.99))
    @patch('lsc.lscastrodef.sloan2file', return_value=3)
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.util.defsex', return_value='/tmp/default.sex')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_landolt_filter_V_no_catalogue(self, mock_readhdr, mock_readkey3,
                                            mock_delete, mock_defsex,
                                            mock_updateheader, mock_readtxt_fn,
                                            mock_sloan2file, mock_linreg,
                                            mock_crossmatch, mock_htfr2fwhm,
                                            mock_os_system, mock_file_open):
        """Landolt path: filter=V, no catalogue, landolt stars in field."""
        hdr, data = _make_mock_hdr(filter_val='V', siteid='lsc')
        mock_readhdr.return_value = hdr
        mock_readkey3.side_effect = _readkey3_factory(data)

        iraf_mock = _setup_iraf_mock()
        n_std = 5
        stdcoo_landolt = _make_standard_stdcoo('landolt', n_std)
        stdcoo_sloan = _make_standard_stdcoo('sloan', n_std)
        standardpix_landolt = _make_standardpix(n_std)
        standardpix_sloan = _make_standardpix(n_std)

        def readtxt_side(filepath):
            if 'landolt.cat' in filepath:
                return stdcoo_landolt
            elif '_tmpsloan.cat' in filepath:
                return stdcoo_sloan
            elif 'tmp.stdL.pix' in filepath:
                return standardpix_landolt
            elif 'tmp.stdS.pix' in filepath:
                return standardpix_sloan
            return {'ra': [], 'dec': [], 'id': []}

        mock_readtxt_fn.side_effect = readtxt_side

        pos0 = np.array([0, 1, 2])
        pos1 = np.array([0, 1, 2])
        mock_crossmatch.return_value = (np.array([0.1, 0.1, 0.1]), pos0, pos1)

        mock_fields, _, _ = _make_sex_fields_mock(n_std)
        iraf_mock.proto.fields = MagicMock(side_effect=mock_fields)
        iraf_mock.noao.digiphot.daophot.phot = MagicMock(
            return_value=['img 500.0 600.0 3.0 18.5 0.02']
        )

        with patch.dict('sys.modules', {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            result = lsc.lscastrodef.zeropoint('test_img.fits', 'landolt',
                                               verbose=False, catalogue='')

        assert isinstance(result, dict)
        mock_linreg.assert_called()
        assert mock_updateheader.call_count >= 1

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.lscastrodef.half_total_flux_radius_to_fwhm', return_value=4.0)
    @patch('lsc.lscastrodef.crossmatch')
    @patch('lsc.lscastrodef.linreg', return_value=(0.05, 25.0, 0.99))
    @patch('lsc.lscastrodef.sloan2file', return_value=3)
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.util.defsex', return_value='/tmp/default.sex')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_landolt_filter_B_with_catalogue(self, mock_readhdr, mock_readkey3,
                                              mock_delete, mock_defsex,
                                              mock_updateheader, mock_readtxt_fn,
                                              mock_sloan2file, mock_linreg,
                                              mock_crossmatch, mock_htfr2fwhm,
                                              mock_os_system, mock_file_open):
        """Landolt path: filter=B, with catalogue."""
        hdr, data = _make_mock_hdr(filter_val='B', siteid='coj')
        mock_readhdr.return_value = hdr
        mock_readkey3.side_effect = _readkey3_factory(data)

        iraf_mock = _setup_iraf_mock()
        n_std = 4
        stdcoo_cat = _make_standard_stdcoo('landolt', n_std)
        stdcoo_landolt = _make_standard_stdcoo('landolt', 2)
        stdcoo_sloan = _make_standard_stdcoo('sloan', 2)
        standardpix_cat = _make_standardpix(n_std)
        standardpix_landolt = _make_standardpix(2)
        standardpix_sloan = _make_standardpix(2)

        def readtxt_side(filepath):
            if 'landolt.cat' in filepath:
                return stdcoo_landolt
            elif '_tmpsloan.cat' in filepath:
                return stdcoo_sloan
            elif 'tmp.stdL.pix' in filepath:
                return standardpix_landolt
            elif 'tmp.stdS.pix' in filepath:
                return standardpix_sloan
            elif 'tmp.stdC.pix' in filepath:
                return standardpix_cat
            elif '.cat' in filepath:
                return stdcoo_cat
            return {'ra': [], 'dec': [], 'id': []}

        mock_readtxt_fn.side_effect = readtxt_side

        pos0 = np.array([0, 1, 2])
        pos1 = np.array([0, 1, 2])
        mock_crossmatch.return_value = (np.array([0.1, 0.1, 0.1]), pos0, pos1)

        mock_fields, _, _ = _make_sex_fields_mock(n_std)
        iraf_mock.proto.fields = MagicMock(side_effect=mock_fields)
        iraf_mock.noao.digiphot.daophot.phot = MagicMock(
            return_value=['img 500.0 600.0 3.0 17.0 0.02']
        )

        with patch.dict('sys.modules', {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            result = lsc.lscastrodef.zeropoint('test_img.fits', 'landolt',
                                               verbose=False, catalogue='mycatalog')

        assert isinstance(result, dict)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.lscastrodef.half_total_flux_radius_to_fwhm', return_value=4.0)
    @patch('lsc.lscastrodef.crossmatch')
    @patch('lsc.lscastrodef.linreg', return_value=(0.05, 25.0, 0.99))
    @patch('lsc.lscastrodef.sloan2file', return_value=3)
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.lscastrodef.transformsloanlandolt')
    @patch('lsc.util.updateheader')
    @patch('lsc.util.defsex', return_value='/tmp/default.sex')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_landolt_filter_V_sloan_transform(self, mock_readhdr, mock_readkey3,
                                               mock_delete, mock_defsex,
                                               mock_updateheader,
                                               mock_transform,
                                               mock_readtxt_fn,
                                               mock_sloan2file, mock_linreg,
                                               mock_crossmatch, mock_htfr2fwhm,
                                               mock_os_system, mock_file_open):
        """Landolt path: filter=V, no landolt stars, uses sloan->landolt transform."""
        hdr, data = _make_mock_hdr(filter_val='V', siteid='lsc')
        mock_readhdr.return_value = hdr
        mock_readkey3.side_effect = _readkey3_factory(data)

        iraf_mock = _setup_iraf_mock()
        n_std = 4
        stdcoo_sloan = _make_standard_stdcoo('sloan', n_std)
        standardpix_landolt = _make_outside_pix(n_std)  # landolt outside
        standardpix_sloan = _make_standardpix(n_std)     # sloan inside

        transformed = _make_standard_stdcoo('landolt', n_std)
        mock_transform.return_value = transformed

        def readtxt_side(filepath):
            if 'landolt.cat' in filepath:
                return _make_standard_stdcoo('landolt', n_std)
            elif '_tmpsloan.cat' in filepath:
                return stdcoo_sloan
            elif 'tmp.stdL.pix' in filepath:
                return standardpix_landolt
            elif 'tmp.stdS.pix' in filepath:
                return standardpix_sloan
            return {'ra': [], 'dec': [], 'id': []}

        mock_readtxt_fn.side_effect = readtxt_side

        pos0 = np.array([0, 1, 2])
        pos1 = np.array([0, 1, 2])
        mock_crossmatch.return_value = (np.array([0.1, 0.1, 0.1]), pos0, pos1)

        mock_fields, _, _ = _make_sex_fields_mock(n_std)
        iraf_mock.proto.fields = MagicMock(side_effect=mock_fields)
        iraf_mock.noao.digiphot.daophot.phot = MagicMock(
            return_value=['img 500.0 600.0 3.0 18.5 0.02']
        )

        with patch.dict('sys.modules', {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            result = lsc.lscastrodef.zeropoint('test_img.fits', '',
                                               verbose=False, catalogue='')

        assert isinstance(result, dict)
        mock_transform.assert_called()

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.lscastrodef.half_total_flux_radius_to_fwhm', return_value=4.0)
    @patch('lsc.lscastrodef.crossmatch')
    @patch('lsc.lscastrodef.linreg', return_value=(0.04, 22.0, 0.95))
    @patch('lsc.lscastrodef.sloan2file', return_value=3)
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.util.defsex', return_value='/tmp/default.sex')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_filter_I_landolt(self, mock_readhdr, mock_readkey3,
                               mock_delete, mock_defsex,
                               mock_updateheader, mock_readtxt_fn,
                               mock_sloan2file, mock_linreg,
                               mock_crossmatch, mock_htfr2fwhm,
                               mock_os_system, mock_file_open):
        """Filter I has two color terms ['VI', 'RI']."""
        hdr, data = _make_mock_hdr(filter_val='I', siteid='lsc')
        mock_readhdr.return_value = hdr
        mock_readkey3.side_effect = _readkey3_factory(data)

        iraf_mock = _setup_iraf_mock()
        n_std = 4
        stdcoo_landolt = _make_standard_stdcoo('landolt', n_std)
        stdcoo_sloan = _make_standard_stdcoo('sloan', n_std)
        standardpix_landolt = _make_standardpix(n_std)
        standardpix_sloan = _make_standardpix(n_std)

        def readtxt_side(filepath):
            if 'landolt.cat' in filepath:
                return stdcoo_landolt
            elif '_tmpsloan.cat' in filepath:
                return stdcoo_sloan
            elif 'tmp.stdL.pix' in filepath:
                return standardpix_landolt
            elif 'tmp.stdS.pix' in filepath:
                return standardpix_sloan
            return {'ra': [], 'dec': [], 'id': []}

        mock_readtxt_fn.side_effect = readtxt_side

        pos0 = np.array([0, 1, 2])
        pos1 = np.array([0, 1, 2])
        mock_crossmatch.return_value = (np.array([0.1, 0.1, 0.1]), pos0, pos1)

        mock_fields, _, _ = _make_sex_fields_mock(n_std)
        iraf_mock.proto.fields = MagicMock(side_effect=mock_fields)
        iraf_mock.noao.digiphot.daophot.phot = MagicMock(
            return_value=['img 500.0 600.0 3.0 18.0 0.02']
        )

        with patch.dict('sys.modules', {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            result = lsc.lscastrodef.zeropoint('test_img.fits', 'landolt',
                                               verbose=False, catalogue='')

        assert isinstance(result, dict)
        # Filter I has colors ['VI', 'RI'] -> linreg called twice
        assert mock_linreg.call_count >= 2


# =============================================================================
# Tests: Sloan field with SDSS filters
# =============================================================================

class TestZeropointSloanPath:
    """Test zeropoint with sloan field, SDSS filters.

    NOTE: The sloan deep-photometry path (lines 662-678, 708-710, 736-740) is
    unreachable due to a code bug: the `filters` dict at line 582 has keys
    {u,i,g,r,z} but _filter values are {SDSS-U,SDSS-G,SDSS-R,SDSS-I,Pan-Starrs-Z}.
    So filters[_filter] at line 644 always raises KeyError for SDSS filters.

    We test the SDSS filter selection logic (lines 578-597) and verify the
    KeyError behavior. For the deep photometry, only the landolt path is reachable.
    """

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.lscastrodef.sloan2file', return_value=5)
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.util.defsex', return_value='/tmp/default.sex')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_sloan_filter_SDSS_R_sloan_stars_in_field(self, mock_readhdr, mock_readkey3,
                                                       mock_delete, mock_defsex,
                                                       mock_updateheader, mock_readtxt_fn,
                                                       mock_sloan2file,
                                                       mock_os_system, mock_file_open):
        """SDSS-R with sloan stars in field: exercises line 588-590 selection
        but hits KeyError at line 644 (code bug). Verify branch is entered."""
        hdr, data = _make_mock_hdr(filter_val='SDSS-R', siteid='elp')
        mock_readhdr.return_value = hdr
        mock_readkey3.side_effect = _readkey3_factory(data)

        iraf_mock = _setup_iraf_mock()
        n_std = 5
        stdcoo_sloan = _make_standard_stdcoo('sloan', n_std)
        standardpix_sloan = _make_standardpix(n_std)

        def readtxt_side(filepath):
            if 'landolt.cat' in filepath:
                return _make_standard_stdcoo('landolt', 2)
            elif '_tmpsloan.cat' in filepath:
                return stdcoo_sloan
            elif 'tmp.stdL.pix' in filepath:
                return _make_standardpix(2)
            elif 'tmp.stdS.pix' in filepath:
                return standardpix_sloan
            return {'ra': [], 'dec': [], 'id': []}

        mock_readtxt_fn.side_effect = readtxt_side

        mock_fields, _, _ = _make_sex_fields_mock(n_std)
        iraf_mock.proto.fields = MagicMock(side_effect=mock_fields)

        with patch.dict('sys.modules', {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            # Line 644 raises KeyError('SDSS-R') because filters dict has
            # wrong keys. This confirms we reached the sloan deep path.
            with pytest.raises(KeyError, match='SDSS-R'):
                lsc.lscastrodef.zeropoint('test_img.fits', 'sloan',
                                           verbose=False, catalogue='')

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.lscastrodef.sloan2file', return_value=5)
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.lscastrodef.transformlandoltsloan')
    @patch('lsc.util.updateheader')
    @patch('lsc.util.defsex', return_value='/tmp/default.sex')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_sloan_filter_SDSS_G_landolt_transform(self, mock_readhdr, mock_readkey3,
                                                     mock_delete, mock_defsex,
                                                     mock_updateheader,
                                                     mock_transform,
                                                     mock_readtxt_fn,
                                                     mock_sloan2file,
                                                     mock_os_system, mock_file_open):
        """SDSS-G with no sloan stars, landolt present: exercises transform path
        (lines 591-595) then hits KeyError at 644."""
        hdr, data = _make_mock_hdr(filter_val='SDSS-G', siteid='ogg')
        mock_readhdr.return_value = hdr
        mock_readkey3.side_effect = _readkey3_factory(data)

        iraf_mock = _setup_iraf_mock()
        n_std = 4
        stdcoo_landolt = _make_standard_stdcoo('landolt', n_std)
        standardpix_landolt = _make_standardpix(n_std)

        transformed = _make_standard_stdcoo('sloan', n_std)
        mock_transform.return_value = transformed

        def readtxt_side(filepath):
            if 'landolt.cat' in filepath:
                return stdcoo_landolt
            elif '_tmpsloan.cat' in filepath:
                return _make_standard_stdcoo('sloan', n_std)
            elif 'tmp.stdL.pix' in filepath:
                return standardpix_landolt
            elif 'tmp.stdS.pix' in filepath:
                return _make_outside_pix(n_std)
            return {'ra': [], 'dec': [], 'id': []}

        mock_readtxt_fn.side_effect = readtxt_side

        mock_fields, _, _ = _make_sex_fields_mock(n_std)
        iraf_mock.proto.fields = MagicMock(side_effect=mock_fields)

        with patch.dict('sys.modules', {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            with pytest.raises(KeyError, match='SDSS-G'):
                lsc.lscastrodef.zeropoint('test_img.fits', '',
                                           verbose=False, catalogue='')

        # transformlandoltsloan was called (line 594 reached)
        mock_transform.assert_called()

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.lscastrodef.sloan2file', return_value=5)
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.util.defsex', return_value='/tmp/default.sex')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_sloan_filter_SDSS_I_with_catalogue(self, mock_readhdr, mock_readkey3,
                                                  mock_delete, mock_defsex,
                                                  mock_updateheader, mock_readtxt_fn,
                                                  mock_sloan2file,
                                                  mock_os_system, mock_file_open):
        """SDSS-I with catalogue: exercises line 584-586 then hits KeyError at 644."""
        hdr, data = _make_mock_hdr(filter_val='SDSS-I', siteid='cpt')
        mock_readhdr.return_value = hdr
        mock_readkey3.side_effect = _readkey3_factory(data)

        iraf_mock = _setup_iraf_mock()
        n_std = 5
        stdcoo_cat = _make_standard_stdcoo('sloan', n_std)
        standardpix_cat = _make_standardpix(n_std)

        def readtxt_side(filepath):
            if 'landolt.cat' in filepath:
                return _make_standard_stdcoo('landolt', 2)
            elif '_tmpsloan.cat' in filepath:
                return _make_standard_stdcoo('sloan', 2)
            elif 'tmp.stdL.pix' in filepath:
                return _make_standardpix(2)
            elif 'tmp.stdS.pix' in filepath:
                return _make_standardpix(2)
            elif 'tmp.stdC.pix' in filepath:
                return standardpix_cat
            elif '.cat' in filepath:
                return stdcoo_cat
            return {'ra': [], 'dec': [], 'id': []}

        mock_readtxt_fn.side_effect = readtxt_side

        mock_fields, _, _ = _make_sex_fields_mock(n_std)
        iraf_mock.proto.fields = MagicMock(side_effect=mock_fields)

        with patch.dict('sys.modules', {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            with pytest.raises(KeyError, match='SDSS-I'):
                lsc.lscastrodef.zeropoint('test_img.fits', 'sloan',
                                           verbose=False, catalogue='sloan_cat')

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.lscastrodef.sloan2file', return_value=5)
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.util.defsex', return_value='/tmp/default.sex')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_panstarrs_z_filter(self, mock_readhdr, mock_readkey3,
                                 mock_delete, mock_defsex,
                                 mock_updateheader, mock_readtxt_fn,
                                 mock_sloan2file,
                                 mock_os_system, mock_file_open):
        """Pan-Starrs-Z exercises line 578 condition, hits KeyError at 644."""
        hdr, data = _make_mock_hdr(filter_val='Pan-Starrs-Z', siteid='ogg')
        mock_readhdr.return_value = hdr
        mock_readkey3.side_effect = _readkey3_factory(data)

        iraf_mock = _setup_iraf_mock()
        n_std = 5
        stdcoo_sloan = _make_standard_stdcoo('sloan', n_std)
        standardpix_sloan = _make_standardpix(n_std)

        def readtxt_side(filepath):
            if 'landolt.cat' in filepath:
                return _make_standard_stdcoo('landolt', 2)
            elif '_tmpsloan.cat' in filepath:
                return stdcoo_sloan
            elif 'tmp.stdL.pix' in filepath:
                return _make_standardpix(2)
            elif 'tmp.stdS.pix' in filepath:
                return standardpix_sloan
            return {'ra': [], 'dec': [], 'id': []}

        mock_readtxt_fn.side_effect = readtxt_side

        mock_fields, _, _ = _make_sex_fields_mock(n_std)
        iraf_mock.proto.fields = MagicMock(side_effect=mock_fields)

        with patch.dict('sys.modules', {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            with pytest.raises(KeyError, match='Pan-Starrs-Z'):
                lsc.lscastrodef.zeropoint('test_img.fits', 'sloan',
                                           verbose=False, catalogue='')


# =============================================================================
# Tests: Edge cases
# =============================================================================

class TestZeropointEdgeCases:
    """Test edge cases: no stars, phot failure, field mismatch."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.lscastrodef.sloan2file', return_value=0)
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.util.defsex', return_value='/tmp/default.sex')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_no_standards_in_field(self, mock_readhdr, mock_readkey3,
                                    mock_delete, mock_defsex,
                                    mock_updateheader, mock_readtxt_fn,
                                    mock_sloan2file,
                                    mock_os_system, mock_file_open):
        """No standard stars in field -> returns empty string result."""
        hdr, data = _make_mock_hdr(filter_val='V', siteid='lsc')
        mock_readhdr.return_value = hdr
        mock_readkey3.side_effect = _readkey3_factory(data)

        iraf_mock = _setup_iraf_mock()

        def readtxt_side(filepath):
            if 'landolt.cat' in filepath:
                return _make_standard_stdcoo('landolt', 3)
            elif '_tmpsloan.cat' in filepath:
                return _make_standard_stdcoo('sloan', 3)
            elif 'tmp.stdL.pix' in filepath:
                return _make_outside_pix(3)
            elif 'tmp.stdS.pix' in filepath:
                return _make_outside_pix(3)
            return {'ra': [], 'dec': [], 'id': []}

        mock_readtxt_fn.side_effect = readtxt_side

        with patch.dict('sys.modules', {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            result = lsc.lscastrodef.zeropoint('test_img.fits', 'landolt',
                                               verbose=False, catalogue='')

        assert result == ''

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.lscastrodef.sloan2file', return_value=0)
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.util.defsex', return_value='/tmp/default.sex')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_field_sloan_with_bvri_filter(self, mock_readhdr, mock_readkey3,
                                           mock_delete, mock_defsex,
                                           mock_updateheader, mock_readtxt_fn,
                                           mock_sloan2file,
                                           mock_os_system, mock_file_open):
        """_field='sloan' with filter in BVRI -> dummy standardpix."""
        hdr, data = _make_mock_hdr(filter_val='R', siteid='lsc')
        mock_readhdr.return_value = hdr
        mock_readkey3.side_effect = _readkey3_factory(data)

        iraf_mock = _setup_iraf_mock()

        def readtxt_side(filepath):
            if 'landolt.cat' in filepath:
                return _make_standard_stdcoo('landolt', 3)
            elif '_tmpsloan.cat' in filepath:
                return _make_standard_stdcoo('sloan', 3)
            elif 'tmp.stdL.pix' in filepath:
                return _make_standardpix(3)
            elif 'tmp.stdS.pix' in filepath:
                return _make_standardpix(3)
            return {'ra': [], 'dec': [], 'id': []}

        mock_readtxt_fn.side_effect = readtxt_side

        with patch.dict('sys.modules', {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            result = lsc.lscastrodef.zeropoint('test_img.fits', 'sloan',
                                               verbose=False, catalogue='')

        # standardpix set to dummy {9999} -> xstd0 empty -> result ''
        assert result == ''

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.lscastrodef.sloan2file', return_value=0)
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.util.defsex', return_value='/tmp/default.sex')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_field_landolt_with_sdss_filter(self, mock_readhdr, mock_readkey3,
                                             mock_delete, mock_defsex,
                                             mock_updateheader, mock_readtxt_fn,
                                             mock_sloan2file,
                                             mock_os_system, mock_file_open):
        """_field='landolt' with SDSS filter -> dummy standardpix."""
        hdr, data = _make_mock_hdr(filter_val='SDSS-G', siteid='lsc')
        mock_readhdr.return_value = hdr
        mock_readkey3.side_effect = _readkey3_factory(data)

        iraf_mock = _setup_iraf_mock()

        def readtxt_side(filepath):
            if 'landolt.cat' in filepath:
                return _make_standard_stdcoo('landolt', 3)
            elif '_tmpsloan.cat' in filepath:
                return _make_standard_stdcoo('sloan', 3)
            elif 'tmp.stdL.pix' in filepath:
                return _make_standardpix(3)
            elif 'tmp.stdS.pix' in filepath:
                return _make_standardpix(3)
            return {'ra': [], 'dec': [], 'id': []}

        mock_readtxt_fn.side_effect = readtxt_side

        with patch.dict('sys.modules', {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            result = lsc.lscastrodef.zeropoint('test_img.fits', 'landolt',
                                               verbose=False, catalogue='')

        assert result == ''

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.lscastrodef.half_total_flux_radius_to_fwhm', return_value=4.0)
    @patch('lsc.lscastrodef.crossmatch')
    @patch('lsc.lscastrodef.linreg', return_value=(0.05, 25.0, 0.99))
    @patch('lsc.lscastrodef.sloan2file', return_value=3)
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.util.defsex', return_value='/tmp/default.sex')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_photometry_failure(self, mock_readhdr, mock_readkey3,
                                 mock_delete, mock_defsex,
                                 mock_updateheader, mock_readtxt_fn,
                                 mock_sloan2file, mock_linreg,
                                 mock_crossmatch, mock_htfr2fwhm,
                                 mock_os_system, mock_file_open):
        """Photometry fails -> mag=999 -> zero>50 -> colore empty -> no calibration."""
        hdr, data = _make_mock_hdr(filter_val='V', siteid='lsc')
        mock_readhdr.return_value = hdr
        mock_readkey3.side_effect = _readkey3_factory(data)

        iraf_mock = _setup_iraf_mock()
        n_std = 3
        stdcoo_landolt = _make_standard_stdcoo('landolt', n_std)
        standardpix_landolt = _make_standardpix(n_std)

        def readtxt_side(filepath):
            if 'landolt.cat' in filepath:
                return stdcoo_landolt
            elif '_tmpsloan.cat' in filepath:
                return _make_standard_stdcoo('sloan', n_std)
            elif 'tmp.stdL.pix' in filepath:
                return standardpix_landolt
            elif 'tmp.stdS.pix' in filepath:
                return _make_standardpix(n_std)
            return {'ra': [], 'dec': [], 'id': []}

        mock_readtxt_fn.side_effect = readtxt_side

        pos0 = np.array([0, 1, 2])
        pos1 = np.array([0, 1, 2])
        mock_crossmatch.return_value = (np.array([0.1, 0.1, 0.1]), pos0, pos1)

        mock_fields, _, _ = _make_sex_fields_mock(n_std)
        iraf_mock.proto.fields = MagicMock(side_effect=mock_fields)
        # phot always fails
        iraf_mock.noao.digiphot.daophot.phot = MagicMock(
            side_effect=Exception("INDEF")
        )

        with patch.dict('sys.modules', {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            result = lsc.lscastrodef.zeropoint('test_img.fits', 'landolt',
                                               verbose=False, catalogue='')

        # All phot failed -> mag=999, zero values will be large (|zero|>50)
        # compress removes them -> colore empty -> result=''
        assert result == ''

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.lscastrodef.half_total_flux_radius_to_fwhm', return_value=4.0)
    @patch('lsc.lscastrodef.crossmatch')
    @patch('lsc.lscastrodef.sloan2file', return_value=3)
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.util.defsex', return_value='/tmp/default.sex')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_single_star_calibration(self, mock_readhdr, mock_readkey3,
                                      mock_delete, mock_defsex,
                                      mock_updateheader, mock_readtxt_fn,
                                      mock_sloan2file,
                                      mock_crossmatch, mock_htfr2fwhm,
                                      mock_os_system, mock_file_open):
        """Single star match -> len(colore)==1 branch."""
        hdr, data = _make_mock_hdr(filter_val='V', siteid='lsc')
        mock_readhdr.return_value = hdr
        mock_readkey3.side_effect = _readkey3_factory(data)

        iraf_mock = _setup_iraf_mock()
        n_std = 3
        stdcoo_landolt = _make_standard_stdcoo('landolt', n_std)
        standardpix_landolt = _make_standardpix(n_std)

        def readtxt_side(filepath):
            if 'landolt.cat' in filepath:
                return stdcoo_landolt
            elif '_tmpsloan.cat' in filepath:
                return _make_standard_stdcoo('sloan', n_std)
            elif 'tmp.stdL.pix' in filepath:
                return standardpix_landolt
            elif 'tmp.stdS.pix' in filepath:
                return _make_standardpix(n_std)
            return {'ra': [], 'dec': [], 'id': []}

        mock_readtxt_fn.side_effect = readtxt_side

        # Only one match
        pos0 = np.array([0])
        pos1 = np.array([0])
        mock_crossmatch.return_value = (np.array([0.1]), pos0, pos1)

        mock_fields, _, _ = _make_sex_fields_mock(n_std)
        iraf_mock.proto.fields = MagicMock(side_effect=mock_fields)
        iraf_mock.noao.digiphot.daophot.phot = MagicMock(
            return_value=['img 500.0 600.0 3.0 18.5 0.02']
        )

        with patch.dict('sys.modules', {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            result = lsc.lscastrodef.zeropoint('test_img.fits', 'landolt',
                                               verbose=False, catalogue='')

        # Single star -> result dict with [0, zero[0], 0]
        assert isinstance(result, dict)
        mock_updateheader.assert_called()

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.lscastrodef.sloan2file', return_value=0)
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.util.defsex', return_value='/tmp/default.sex')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_no_landolt_no_sloan_landolt_field(self, mock_readhdr, mock_readkey3,
                                                mock_delete, mock_defsex,
                                                mock_updateheader, mock_readtxt_fn,
                                                mock_sloan2file,
                                                mock_os_system, mock_file_open):
        """Filter I, neither landolt nor sloan stars -> dummy stdpix."""
        hdr, data = _make_mock_hdr(filter_val='I', siteid='lsc')
        mock_readhdr.return_value = hdr
        mock_readkey3.side_effect = _readkey3_factory(data)

        iraf_mock = _setup_iraf_mock()

        def readtxt_side(filepath):
            if 'landolt.cat' in filepath:
                return _make_standard_stdcoo('landolt', 3)
            elif '_tmpsloan.cat' in filepath:
                return _make_standard_stdcoo('sloan', 3)
            elif 'tmp.stdL.pix' in filepath:
                return _make_outside_pix(3)
            elif 'tmp.stdS.pix' in filepath:
                return _make_outside_pix(3)
            return {'ra': [], 'dec': [], 'id': []}

        mock_readtxt_fn.side_effect = readtxt_side

        with patch.dict('sys.modules', {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            result = lsc.lscastrodef.zeropoint('test_img.fits', '',
                                               verbose=False, catalogue='')

        assert result == ''

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.lscastrodef.sloan2file', return_value=0)
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.util.defsex', return_value='/tmp/default.sex')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_no_landolt_no_sloan_sdss_filter(self, mock_readhdr, mock_readkey3,
                                              mock_delete, mock_defsex,
                                              mock_updateheader, mock_readtxt_fn,
                                              mock_sloan2file,
                                              mock_os_system, mock_file_open):
        """SDSS-U filter, neither sloan nor landolt stars -> dummy."""
        hdr, data = _make_mock_hdr(filter_val='SDSS-U', siteid='lsc')
        mock_readhdr.return_value = hdr
        mock_readkey3.side_effect = _readkey3_factory(data)

        iraf_mock = _setup_iraf_mock()

        def readtxt_side(filepath):
            if 'landolt.cat' in filepath:
                return _make_standard_stdcoo('landolt', 3)
            elif '_tmpsloan.cat' in filepath:
                return _make_standard_stdcoo('sloan', 3)
            elif 'tmp.stdL.pix' in filepath:
                return _make_outside_pix(3)
            elif 'tmp.stdS.pix' in filepath:
                return _make_outside_pix(3)
            return {'ra': [], 'dec': [], 'id': []}

        mock_readtxt_fn.side_effect = readtxt_side

        with patch.dict('sys.modules', {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            result = lsc.lscastrodef.zeropoint('test_img.fits', '',
                                               verbose=False, catalogue='')

        assert result == ''


# =============================================================================
# Tests: verbose and airmass fallback
# =============================================================================

class TestZeropointVerboseAndAirmass:
    """Test verbose=True plotting path and airmass fallback."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.lscastrodef.half_total_flux_radius_to_fwhm', return_value=4.0)
    @patch('lsc.lscastrodef.crossmatch')
    @patch('lsc.lscastrodef.linreg', return_value=(0.05, 25.0, 0.99))
    @patch('lsc.lscastrodef.sloan2file', return_value=3)
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.util.defsex', return_value='/tmp/default.sex')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_verbose_with_multiple_stars(self, mock_readhdr, mock_readkey3,
                                          mock_delete, mock_defsex,
                                          mock_updateheader, mock_readtxt_fn,
                                          mock_sloan2file, mock_linreg,
                                          mock_crossmatch, mock_htfr2fwhm,
                                          mock_os_system, mock_file_open):
        """verbose=True exercises the plotting code path."""
        hdr, data = _make_mock_hdr(filter_val='V', siteid='lsc')
        mock_readhdr.return_value = hdr
        mock_readkey3.side_effect = _readkey3_factory(data)

        iraf_mock = _setup_iraf_mock()
        n_std = 5
        stdcoo_landolt = _make_standard_stdcoo('landolt', n_std)
        standardpix_landolt = _make_standardpix(n_std)

        def readtxt_side(filepath):
            if 'landolt.cat' in filepath:
                return stdcoo_landolt
            elif '_tmpsloan.cat' in filepath:
                return _make_standard_stdcoo('sloan', n_std)
            elif 'tmp.stdL.pix' in filepath:
                return standardpix_landolt
            elif 'tmp.stdS.pix' in filepath:
                return _make_standardpix(n_std)
            return {'ra': [], 'dec': [], 'id': []}

        mock_readtxt_fn.side_effect = readtxt_side

        pos0 = np.array([0, 1, 2])
        pos1 = np.array([0, 1, 2])
        mock_crossmatch.return_value = (np.array([0.1, 0.1, 0.1]), pos0, pos1)

        mock_fields, _, _ = _make_sex_fields_mock(n_std)
        iraf_mock.proto.fields = MagicMock(side_effect=mock_fields)
        iraf_mock.noao.digiphot.daophot.phot = MagicMock(
            return_value=['img 500.0 600.0 3.0 18.5 0.02']
        )

        mock_pylab = MagicMock()
        with patch.dict('sys.modules', {
            'pyraf': MagicMock(),
            'pyraf.iraf': iraf_mock,
            'pylab': mock_pylab,
            'matplotlib': MagicMock(),
            'matplotlib.pyplot': mock_pylab,
        }):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            result = lsc.lscastrodef.zeropoint('test_img.fits', 'landolt',
                                               verbose=True, catalogue='')

        assert isinstance(result, dict)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.lscastrodef.sloan2file', return_value=0)
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.util.defsex', return_value='/tmp/default.sex')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_airmass_fallback(self, mock_readhdr, mock_readkey3,
                               mock_delete, mock_defsex,
                               mock_updateheader, mock_readtxt_fn,
                               mock_sloan2file,
                               mock_os_system, mock_file_open):
        """AIRMASS key not found initially, falls back to 'airmass'."""
        hdr, data = _make_mock_hdr(filter_val='V', siteid='lsc')
        mock_readhdr.return_value = hdr

        def custom_readkey3(hdr, keyword):
            if keyword == 'AIRMASS':
                return None  # First AIRMASS returns None -> fallback
            return data.get(keyword, None)

        mock_readkey3.side_effect = custom_readkey3

        iraf_mock = _setup_iraf_mock()

        def readtxt_side(filepath):
            return _make_outside_pix(1)

        mock_readtxt_fn.side_effect = readtxt_side

        with patch.dict('sys.modules', {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            result = lsc.lscastrodef.zeropoint('test_img.fits', 'landolt',
                                               verbose=False, catalogue='')

        assert result == ''

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.system')
    @patch('lsc.util.delete')
    @patch('lsc.util.readkey3')
    @patch('lsc.util.readhdr')
    def test_unknown_siteid_exits(self, mock_readhdr, mock_readkey3,
                                   mock_delete,
                                   mock_os_system, mock_file_open):
        """Unknown SITEID triggers sys.exit (lines 512-513)."""
        hdr, data = _make_mock_hdr(filter_val='V', siteid='UNKNOWN_SITE')
        mock_readhdr.return_value = hdr
        mock_readkey3.side_effect = _readkey3_factory(data)

        iraf_mock = _setup_iraf_mock()

        with patch.dict('sys.modules', {'pyraf': MagicMock(), 'pyraf.iraf': iraf_mock}):
            sys.modules['pyraf'].iraf = iraf_mock
            import lsc.lscastrodef
            with pytest.raises(SystemExit, match='siteid not in lsc.sites.extinction'):
                lsc.lscastrodef.zeropoint('test_img.fits', 'landolt',
                                           verbose=False, catalogue='')
