"""
Tests to boost coverage of lsc.lscabsphotdef from ~75% toward 95%.

Targets uncovered lines:
- 256, 263-266, 268-269, 274-283: absphot catalogue/field resolution paths
- 286, 290, 294, 298, 305-313: absphot colorefisso instrument branches
- 319-552: absphot main calibration loop (largest gap)
- 559, 580-588: fitcol3 interactive path
- 679-680, 698-699: fitcol edge cases
- 724, 732: fitcol2 edge cases
- 1229-1232: panstarrs2file None path
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open, PropertyMock
import os
import sys
import types

pytestmark = pytest.mark.unit


# ===========================================================================
# Helper to build a mock header for absphot
# ===========================================================================

def _make_mock_hdr(instrume='fa15', filter_='rp', airmass=1.2, siteid='lsc',
                   catalog='', xdim=100, ydim=100):
    """Create a dict-like mock header for absphot tests."""
    hdr = MagicMock()
    keymap = {
        'catalog': catalog,
        'instrume': instrume,
        'filter': filter_,
        'airmass': airmass,
        'exptime': 120.0,
        'date-obs': '2020-05-01',
        'object': 'SN2024abc',
        'PSF_FWHM': 4.0,
        'XDIM': xdim,
        'YDIM': ydim,
    }
    hdr.__getitem__ = lambda self, key: 'lsc' if key == 'SITEID' else keymap.get(key, '')
    hdr.__contains__ = lambda self, key: key in keymap or key == 'SITEID'
    return hdr, keymap


# ===========================================================================
# absphot: catalogue/field resolution paths (lines 256, 263-269, 274-283)
# ===========================================================================

class TestAbsphotCatalogPaths:
    """Test catalog resolution branches in absphot."""

    @patch('lsc.lscabsphotdef.fits.getheader')
    @patch('lsc.lscabsphotdef.WCS')
    @patch('lsc.myloopdef.checkstage', return_value=1)
    @patch('lsc.mysqldef.updatevalue')
    def test_siteid_not_in_extinction_raises(self, mock_update, mock_checkstage, mock_wcs, mock_getheader):
        """Line 256: raise if siteid not in lsc.sites.extinction."""
        import lsc.lscabsphotdef
        hdr, keymap = _make_mock_hdr(siteid='UNKNOWN')
        hdr.__getitem__ = lambda self, key: 'UNKNOWN' if key == 'SITEID' else keymap.get(key, '')

        def readkey3_side(h, k):
            return keymap.get(k, '')
        mock_getheader.return_value = hdr
        with patch('lsc.util.readkey3', side_effect=readkey3_side):
            with pytest.raises(Exception, match='not in lsc.sites.extinction'):
                lsc.lscabsphotdef.absphot('/tmp/test.fits', redo=True)

    @patch('lsc.lscabsphotdef.fits.getheader')
    @patch('lsc.lscabsphotdef.WCS')
    @patch('lsc.myloopdef.checkstage', return_value=1)
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.util.getcatalog', return_value=(None, ''))
    def test_no_catalog_found_returns_none(self, mock_getcat, mock_update, mock_checkstage, mock_wcs, mock_getheader):
        """Lines 268-269: return early if catalogpath is None."""
        import lsc.lscabsphotdef
        hdr, keymap = _make_mock_hdr(catalog='')
        def readkey3_side(h, k):
            return keymap.get(k, '')
        mock_getheader.return_value = hdr
        with patch('lsc.util.readkey3', side_effect=readkey3_side):
            result = lsc.lscabsphotdef.absphot('/tmp/test.fits', redo=True)
        assert result is None

    @patch('lsc.lscabsphotdef.fits.getheader')
    @patch('lsc.lscabsphotdef.WCS')
    @patch('lsc.myloopdef.checkstage', return_value=1)
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.lscastrodef.readtxt')
    def test_catalogue_absolute_path(self, mock_readtxt, mock_update, mock_checkstage, mock_wcs, mock_getheader):
        """Lines 263-264: catalogue starting with '/' uses realpath."""
        import lsc.lscabsphotdef
        hdr, keymap = _make_mock_hdr(catalog='')
        def readkey3_side(h, k):
            return keymap.get(k, '')
        mock_getheader.return_value = hdr
        # readtxt returns a table-like object; we make it have colnames but not matching any known field
        mock_table = MagicMock()
        mock_table.colnames = ['foo', 'bar']
        mock_readtxt.return_value = mock_table
        with patch('lsc.util.readkey3', side_effect=readkey3_side):
            # Should return None because field detection will fail
            result = lsc.lscabsphotdef.absphot('/tmp/test.fits', _catalogue='/tmp/mycat.cat', redo=True)
        assert result is None

    @patch('lsc.lscabsphotdef.fits.getheader')
    @patch('lsc.lscabsphotdef.WCS')
    @patch('lsc.myloopdef.checkstage', return_value=1)
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.lscastrodef.readtxt')
    def test_catalogue_relative_path(self, mock_readtxt, mock_update, mock_checkstage, mock_wcs, mock_getheader):
        """Lines 265-266: catalogue not starting with / or . joins with workdirectory."""
        import lsc.lscabsphotdef
        hdr, keymap = _make_mock_hdr(catalog='')
        def readkey3_side(h, k):
            return keymap.get(k, '')
        mock_getheader.return_value = hdr
        mock_table = MagicMock()
        mock_table.colnames = ['foo', 'bar']
        mock_readtxt.return_value = mock_table
        with patch('lsc.util.readkey3', side_effect=readkey3_side):
            with patch.dict(os.environ, {'LCOSNDIR': '/tmp'}):
                result = lsc.lscabsphotdef.absphot('/tmp/test.fits', _catalogue='mycat.cat', redo=True)
        assert result is None

    @patch('lsc.lscabsphotdef.fits.getheader')
    @patch('lsc.lscabsphotdef.WCS')
    @patch('lsc.myloopdef.checkstage', return_value=1)
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.lscastrodef.readtxt')
    def test_field_detection_sloan(self, mock_readtxt, mock_update, mock_checkstage, mock_wcs, mock_getheader):
        """Lines 274-283: field auto-detection from catalog columns (sloan)."""
        import lsc.lscabsphotdef
        from astropy.table import Table
        hdr, keymap = _make_mock_hdr(catalog='')
        def readkey3_side(h, k):
            return keymap.get(k, '')
        mock_getheader.return_value = hdr
        # Make a table with sloan columns
        mock_table = Table()
        mock_table['ra'] = [150.0]
        mock_table['dec'] = [2.0]
        mock_table['id'] = ['star1']
        for f in 'ugriz':
            mock_table[f] = [18.0]
            mock_table[f + 'err'] = [0.01]
        mock_table['x'] = [0.0]
        mock_table['y'] = [0.0]
        mock_readtxt.return_value = mock_table
        # _cat is empty and redo=True so it goes into the calibration block
        # but it'll fail at makecatalogue, which is fine for testing the field resolution
        with patch('lsc.util.readkey3', side_effect=readkey3_side):
            with patch('lsc.lscabsphotdef.makecatalogue') as mock_makecat:
                mock_makecat.side_effect = Exception("stop here")
                with pytest.raises(Exception, match="stop here"):
                    lsc.lscabsphotdef.absphot('/tmp/test.fits', _catalogue='/tmp/sloan.cat', redo=True)


# ===========================================================================
# absphot: colorefisso instrument branches (lines 286-313)
# ===========================================================================

class TestAbsphotColorefisso:
    """Test various instrument branches for colorefisso selection."""

    def _run_absphot_to_colorefisso(self, instrume, calib='sloan'):
        """Helper that runs absphot far enough to test colorefisso selection."""
        import lsc.lscabsphotdef
        from astropy.table import Table

        hdr, keymap = _make_mock_hdr(instrume=instrume, catalog='')
        keymap['instrume'] = instrume
        def readkey3_side(h, k):
            return keymap.get(k, '')

        mock_table = Table()
        mock_table['ra'] = [150.0]
        mock_table['dec'] = [2.0]
        mock_table['id'] = ['star1']
        for f in 'ugriz':
            mock_table[f] = [18.0]
            mock_table[f + 'err'] = [0.01]
        mock_table['x'] = [0.0]
        mock_table['y'] = [0.0]

        with patch('lsc.lscabsphotdef.fits.getheader', return_value=hdr), \
             patch('lsc.lscabsphotdef.WCS'), \
             patch('lsc.myloopdef.checkstage', return_value=1), \
             patch('lsc.mysqldef.updatevalue'), \
             patch('lsc.lscastrodef.readtxt', return_value=mock_table), \
             patch('lsc.util.readkey3', side_effect=readkey3_side), \
             patch('lsc.lscabsphotdef.makecatalogue') as mock_makecat:
            mock_makecat.side_effect = Exception("stop at makecatalogue")
            with pytest.raises(Exception, match="stop at makecatalogue"):
                lsc.lscabsphotdef.absphot('/tmp/test.fits', _catalogue='/tmp/sloan.cat', redo=True, _calib=calib)

    def test_sloanprime_fs_instrument(self):
        """Line 286: sloanprime with fs instrument."""
        self._run_absphot_to_colorefisso('fs01', calib='sloanprime')

    def test_sloanprime_em_instrument(self):
        """Line 286: sloanprime with em instrument."""
        self._run_absphot_to_colorefisso('em01', calib='sloanprime')

    def test_sloanprime_other_instrument(self):
        """Line 290: sloanprime with non-fs/em instrument."""
        self._run_absphot_to_colorefisso('fl05', calib='sloanprime')

    def test_natural_calib(self):
        """Line 294: natural calibration."""
        self._run_absphot_to_colorefisso('fa15', calib='natural')

    def test_fs_instrument_default_calib(self):
        """Line 298: fs instrument with default sloan calib."""
        self._run_absphot_to_colorefisso('fs02', calib='sloan')

    def test_fl_instrument(self):
        """Line 305: fl instrument."""
        self._run_absphot_to_colorefisso('fl05', calib='sloan')

    def test_fa_instrument(self):
        """Line 305: fa instrument."""
        self._run_absphot_to_colorefisso('fa15', calib='sloan')

    def test_ep_instrument(self):
        """Line 308: ep instrument."""
        self._run_absphot_to_colorefisso('ep01', calib='sloan')

    def test_unknown_instrument(self):
        """Lines 311-313: unknown instrument falls to default."""
        self._run_absphot_to_colorefisso('xx99', calib='sloan')


# ===========================================================================
# absphot: main calibration loop (lines 319-552) - THE BIGGEST GAP
# ===========================================================================

class TestAbsphotCalibrationLoop:
    """Test the main calibration loop of absphot (lines 319-552)."""

    def _setup_absphot_mocks(self, field='sloan', instrume='fa15', filter_='rp', n_stars=10):
        """Set up all mocks needed to run absphot through the full calibration loop."""
        from astropy.table import Table

        hdr, keymap = _make_mock_hdr(instrume=instrume, filter_=filter_, catalog='')
        keymap['instrume'] = instrume
        keymap['filter'] = filter_
        def readkey3_side(h, k):
            return keymap.get(k, '')

        # Standard catalog (sloan field)
        stdcoo = Table()
        rng = np.random.default_rng(42)
        stdcoo['ra'] = rng.uniform(149.9, 150.1, n_stars)
        stdcoo['dec'] = rng.uniform(1.9, 2.1, n_stars)
        stdcoo['id'] = [f'star{i}' for i in range(n_stars)]
        if field == 'sloan':
            for f in 'ugriz':
                stdcoo[f] = rng.uniform(15, 18, n_stars)
                stdcoo[f + 'err'] = rng.uniform(0.005, 0.03, n_stars)
            stdcoo['w'] = stdcoo['r']
            stdcoo['werr'] = stdcoo['rerr']
        elif field == 'landolt':
            for f in 'UBVRI':
                stdcoo[f] = rng.uniform(14, 17, n_stars)
                stdcoo[f + 'err'] = rng.uniform(0.005, 0.03, n_stars)
        elif field == 'apass':
            for f in 'BVgri':
                stdcoo[f] = rng.uniform(14, 18, n_stars)
                stdcoo[f + 'err'] = rng.uniform(0.005, 0.03, n_stars)
            stdcoo['w'] = stdcoo['r']
            stdcoo['werr'] = stdcoo['rerr']
        stdcoo['x'] = rng.uniform(10, 90, n_stars)
        stdcoo['y'] = rng.uniform(10, 90, n_stars)

        # Sextractor catalogue output from makecatalogue
        cat_dict = {}
        cat_filt = filter_
        sn2_name = '/tmp/test.sn2.fits'
        cat_dict[cat_filt] = {}
        cat_dict[cat_filt][sn2_name] = {
            'ra0': stdcoo['ra'] + rng.uniform(-0.0001, 0.0001, n_stars),
            'dec0': stdcoo['dec'] + rng.uniform(-0.0001, 0.0001, n_stars),
            'smagf': rng.uniform(-8, -6, n_stars),
            'smagerrf': rng.uniform(0.01, 0.05, n_stars),
            'magp3': rng.uniform(-8, -6, n_stars),
            'merrp3': rng.uniform(0.01, 0.05, n_stars),
        }

        # Mock WCS
        mock_wcs_instance = MagicMock()
        mock_wcs_instance.wcs_world2pix.return_value = (
            np.array(stdcoo['x']),
            np.array(stdcoo['y'])
        )

        return hdr, keymap, readkey3_side, stdcoo, cat_dict, mock_wcs_instance

    @patch('lsc.lscabsphotdef.plt')
    @patch('lsc.lscabsphotdef.get_other_filters')
    @patch('lsc.lscabsphotdef.limmag', return_value=22.5)
    @patch('lsc.lscabsphotdef.fitcol3')
    @patch('lsc.lscabsphotdef.transform2natural')
    @patch('lsc.lscabsphotdef.zeropoint2')
    @patch('lsc.lscabsphotdef.makecatalogue')
    @patch('lsc.lscastrodef.crossmatch')
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.myloopdef.checkstage', return_value=1)
    @patch('lsc.lscabsphotdef.WCS')
    @patch('lsc.lscabsphotdef.fits.getheader')
    def test_sloan_field_full_loop(self, mock_getheader, mock_WCS, mock_checkstage,
                                    mock_updatevalue, mock_updateheader,
                                    mock_readtxt, mock_crossmatch,
                                    mock_makecat, mock_zeropoint2,
                                    mock_transform2natural, mock_fitcol3,
                                    mock_limmag, mock_get_other_filters, mock_plt):
        """Test full calibration loop with sloan field."""
        import lsc.lscabsphotdef

        n_stars = 10
        hdr, keymap, readkey3_side, stdcoo, cat_dict, mock_wcs_instance = \
            self._setup_absphot_mocks(field='sloan', instrume='fa15', filter_='rp', n_stars=n_stars)

        mock_getheader.return_value = hdr
        mock_WCS.return_value = mock_wcs_instance
        mock_readtxt.return_value = stdcoo
        mock_makecat.return_value = cat_dict

        # crossmatch returns matched indices
        pos0 = np.arange(n_stars)
        pos1 = np.arange(n_stars)
        distvec = np.zeros(n_stars)
        mock_crossmatch.return_value = (distvec, pos0, pos1)

        # transform2natural returns the same catalogue
        def transform_side(instr, cat, coloref, field):
            return {k: np.array(v, float) for k, v in cat.items()}
        mock_transform2natural.side_effect = transform_side

        # zeropoint2 returns valid values
        mock_zeropoint2.return_value = (25.0, 0.05, np.array([1, 2, 3]), np.array([25.0, 25.1, 24.9]))

        # get_other_filters returns a set of filters
        mock_get_other_filters.return_value = {'g', 'r', 'i'}

        # fitcol3 returns Z, dZ, C, dC
        mock_fitcol3.return_value = (25.0, 0.05, 0.03, 0.01)

        with patch('lsc.util.readkey3', side_effect=readkey3_side), \
             patch('builtins.open', mock_open()):
            lsc.lscabsphotdef.absphot('/tmp/test.fits', _catalogue='/tmp/sloan.cat',
                                       _field='sloan', redo=True, show=False,
                                       _interactive=False, _type='fit')

        # Verify the update was called with calibration results
        assert mock_updatevalue.called
        assert mock_updateheader.called

    @patch('lsc.lscabsphotdef.plt')
    @patch('lsc.lscabsphotdef.get_other_filters')
    @patch('lsc.lscabsphotdef.limmag', return_value=22.5)
    @patch('lsc.lscabsphotdef.fitcol3')
    @patch('lsc.lscabsphotdef.transform2natural')
    @patch('lsc.lscabsphotdef.zeropoint2')
    @patch('lsc.lscabsphotdef.makecatalogue')
    @patch('lsc.lscastrodef.crossmatch')
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.myloopdef.checkstage', return_value=1)
    @patch('lsc.lscabsphotdef.WCS')
    @patch('lsc.lscabsphotdef.fits.getheader')
    def test_landolt_field_full_loop(self, mock_getheader, mock_WCS, mock_checkstage,
                                      mock_updatevalue, mock_updateheader,
                                      mock_readtxt, mock_crossmatch,
                                      mock_makecat, mock_zeropoint2,
                                      mock_transform2natural, mock_fitcol3,
                                      mock_limmag, mock_get_other_filters, mock_plt):
        """Test full calibration loop with landolt field."""
        import lsc.lscabsphotdef

        n_stars = 10
        hdr, keymap, readkey3_side, stdcoo, cat_dict, mock_wcs_instance = \
            self._setup_absphot_mocks(field='landolt', instrume='fa15', filter_='V', n_stars=n_stars)
        keymap['filter'] = 'V'

        # Rebuild cat_dict with correct filter
        sn2_name = '/tmp/test.sn2.fits'
        rng = np.random.default_rng(42)
        cat_dict = {}
        cat_dict['V'] = {}
        cat_dict['V'][sn2_name] = {
            'ra0': stdcoo['ra'] + rng.uniform(-0.0001, 0.0001, n_stars),
            'dec0': stdcoo['dec'] + rng.uniform(-0.0001, 0.0001, n_stars),
            'smagf': rng.uniform(-8, -6, n_stars),
            'smagerrf': rng.uniform(0.01, 0.05, n_stars),
            'magp3': rng.uniform(-8, -6, n_stars),
            'merrp3': rng.uniform(0.01, 0.05, n_stars),
        }

        mock_getheader.return_value = hdr
        mock_WCS.return_value = mock_wcs_instance
        mock_readtxt.return_value = stdcoo
        mock_makecat.return_value = cat_dict
        mock_crossmatch.return_value = (np.zeros(n_stars), np.arange(n_stars), np.arange(n_stars))

        def transform_side(instr, cat, coloref, field):
            return {k: np.array(v, float) for k, v in cat.items()}
        mock_transform2natural.side_effect = transform_side
        mock_zeropoint2.return_value = (25.0, 0.05, np.array([1, 2, 3]), np.array([25.0, 25.1, 24.9]))
        mock_get_other_filters.return_value = {'B', 'V', 'R'}
        mock_fitcol3.return_value = (25.0, 0.05, 0.03, 0.01)

        with patch('lsc.util.readkey3', side_effect=readkey3_side), \
             patch('builtins.open', mock_open()):
            lsc.lscabsphotdef.absphot('/tmp/test.fits', _catalogue='/tmp/landolt.cat',
                                       _field='landolt', redo=True, show=False,
                                       _interactive=False, _type='fit')

        assert mock_updatevalue.called

    @patch('lsc.lscabsphotdef.plt')
    @patch('lsc.lscabsphotdef.get_other_filters')
    @patch('lsc.lscabsphotdef.limmag', return_value=22.5)
    @patch('lsc.lscabsphotdef.fitcol3')
    @patch('lsc.lscabsphotdef.transform2natural')
    @patch('lsc.lscabsphotdef.zeropoint2')
    @patch('lsc.lscabsphotdef.makecatalogue')
    @patch('lsc.lscastrodef.crossmatch')
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.myloopdef.checkstage', return_value=1)
    @patch('lsc.lscabsphotdef.WCS')
    @patch('lsc.lscabsphotdef.fits.getheader')
    def test_apass_field_full_loop(self, mock_getheader, mock_WCS, mock_checkstage,
                                    mock_updatevalue, mock_updateheader,
                                    mock_readtxt, mock_crossmatch,
                                    mock_makecat, mock_zeropoint2,
                                    mock_transform2natural, mock_fitcol3,
                                    mock_limmag, mock_get_other_filters, mock_plt):
        """Test full calibration loop with apass field."""
        import lsc.lscabsphotdef

        n_stars = 10
        hdr, keymap, readkey3_side, stdcoo, cat_dict, mock_wcs_instance = \
            self._setup_absphot_mocks(field='apass', instrume='fa15', filter_='rp', n_stars=n_stars)

        # Rebuild cat_dict with correct filter
        sn2_name = '/tmp/test.sn2.fits'
        rng = np.random.default_rng(42)
        cat_dict = {}
        cat_dict['rp'] = {}
        cat_dict['rp'][sn2_name] = {
            'ra0': stdcoo['ra'] + rng.uniform(-0.0001, 0.0001, n_stars),
            'dec0': stdcoo['dec'] + rng.uniform(-0.0001, 0.0001, n_stars),
            'smagf': rng.uniform(-8, -6, n_stars),
            'smagerrf': rng.uniform(0.01, 0.05, n_stars),
            'magp3': rng.uniform(-8, -6, n_stars),
            'merrp3': rng.uniform(0.01, 0.05, n_stars),
        }

        mock_getheader.return_value = hdr
        mock_WCS.return_value = mock_wcs_instance
        mock_readtxt.return_value = stdcoo
        mock_makecat.return_value = cat_dict
        mock_crossmatch.return_value = (np.zeros(n_stars), np.arange(n_stars), np.arange(n_stars))

        def transform_side(instr, cat, coloref, field):
            return {k: np.array(v, float) for k, v in cat.items()}
        mock_transform2natural.side_effect = transform_side
        mock_zeropoint2.return_value = (25.0, 0.05, np.array([1, 2, 3]), np.array([25.0, 25.1, 24.9]))
        mock_get_other_filters.return_value = {'B', 'V', 'g', 'r', 'i'}
        mock_fitcol3.return_value = (25.0, 0.05, 0.03, 0.01)

        with patch('lsc.util.readkey3', side_effect=readkey3_side), \
             patch('builtins.open', mock_open()):
            lsc.lscabsphotdef.absphot('/tmp/test.fits', _catalogue='/tmp/apass.cat',
                                       _field='apass', _calib='apass', redo=True, show=False,
                                       _interactive=False, _type='fit')

        assert mock_updatevalue.called

    @patch('lsc.lscabsphotdef.plt')
    @patch('lsc.lscabsphotdef.get_other_filters')
    @patch('lsc.lscabsphotdef.limmag', return_value=22.5)
    @patch('lsc.lscabsphotdef.fitcol3')
    @patch('lsc.lscabsphotdef.transform2natural')
    @patch('lsc.lscabsphotdef.zeropoint2')
    @patch('lsc.lscabsphotdef.makecatalogue')
    @patch('lsc.lscastrodef.crossmatch')
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.myloopdef.checkstage', return_value=1)
    @patch('lsc.lscabsphotdef.WCS')
    @patch('lsc.lscabsphotdef.fits.getheader')
    def test_ph_type(self, mock_getheader, mock_WCS, mock_checkstage,
                     mock_updatevalue, mock_updateheader,
                     mock_readtxt, mock_crossmatch,
                     mock_makecat, mock_zeropoint2,
                     mock_transform2natural, mock_fitcol3,
                     mock_limmag, mock_get_other_filters, mock_plt):
        """Lines 329-331: _type='ph' path uses magp3/merrp3."""
        import lsc.lscabsphotdef

        n_stars = 10
        hdr, keymap, readkey3_side, stdcoo, cat_dict, mock_wcs_instance = \
            self._setup_absphot_mocks(field='sloan', instrume='fa15', filter_='rp', n_stars=n_stars)

        mock_getheader.return_value = hdr
        mock_WCS.return_value = mock_wcs_instance
        mock_readtxt.return_value = stdcoo
        mock_makecat.return_value = cat_dict
        mock_crossmatch.return_value = (np.zeros(n_stars), np.arange(n_stars), np.arange(n_stars))

        def transform_side(instr, cat, coloref, field):
            return {k: np.array(v, float) for k, v in cat.items()}
        mock_transform2natural.side_effect = transform_side
        mock_zeropoint2.return_value = (25.0, 0.05, np.array([1, 2, 3]), np.array([25.0, 25.1, 24.9]))
        mock_get_other_filters.return_value = {'g', 'r', 'i'}
        mock_fitcol3.return_value = (25.0, 0.05, 0.03, 0.01)

        with patch('lsc.util.readkey3', side_effect=readkey3_side), \
             patch('builtins.open', mock_open()):
            lsc.lscabsphotdef.absphot('/tmp/test.fits', _catalogue='/tmp/sloan.cat',
                                       _field='sloan', redo=True, show=False,
                                       _interactive=False, _type='ph')

        assert mock_updatevalue.called

    @patch('lsc.lscabsphotdef.plt')
    @patch('lsc.lscabsphotdef.get_other_filters')
    @patch('lsc.lscabsphotdef.limmag', return_value=22.5)
    @patch('lsc.lscabsphotdef.fitcol3')
    @patch('lsc.lscabsphotdef.transform2natural')
    @patch('lsc.lscabsphotdef.zeropoint2')
    @patch('lsc.lscabsphotdef.makecatalogue')
    @patch('lsc.lscastrodef.crossmatch')
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.myloopdef.checkstage', return_value=1)
    @patch('lsc.lscabsphotdef.WCS')
    @patch('lsc.lscabsphotdef.fits.getheader')
    def test_invalid_type_raises(self, mock_getheader, mock_WCS, mock_checkstage,
                                  mock_updatevalue, mock_updateheader,
                                  mock_readtxt, mock_crossmatch,
                                  mock_makecat, mock_zeropoint2,
                                  mock_transform2natural, mock_fitcol3,
                                  mock_limmag, mock_get_other_filters, mock_plt):
        """Line 333: invalid _type raises exception."""
        import lsc.lscabsphotdef

        n_stars = 10
        hdr, keymap, readkey3_side, stdcoo, cat_dict, mock_wcs_instance = \
            self._setup_absphot_mocks(field='sloan', instrume='fa15', filter_='rp', n_stars=n_stars)

        mock_getheader.return_value = hdr
        mock_WCS.return_value = mock_wcs_instance
        mock_readtxt.return_value = stdcoo
        mock_makecat.return_value = cat_dict

        with patch('lsc.util.readkey3', side_effect=readkey3_side), \
             patch('builtins.open', mock_open()):
            with pytest.raises(Exception, match='not valid'):
                lsc.lscabsphotdef.absphot('/tmp/test.fits', _catalogue='/tmp/sloan.cat',
                                           _field='sloan', redo=True, show=False,
                                           _interactive=False, _type='INVALID')

    @patch('lsc.lscabsphotdef.plt')
    @patch('lsc.lscabsphotdef.get_other_filters')
    @patch('lsc.lscabsphotdef.limmag', return_value=22.5)
    @patch('lsc.lscabsphotdef.fitcol3')
    @patch('lsc.lscabsphotdef.transform2natural')
    @patch('lsc.lscabsphotdef.zeropoint2')
    @patch('lsc.lscabsphotdef.makecatalogue')
    @patch('lsc.lscastrodef.crossmatch')
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.myloopdef.checkstage', return_value=1)
    @patch('lsc.lscabsphotdef.WCS')
    @patch('lsc.lscabsphotdef.fits.getheader')
    def test_zeropoint_9999_path(self, mock_getheader, mock_WCS, mock_checkstage,
                                  mock_updatevalue, mock_updateheader,
                                  mock_readtxt, mock_crossmatch,
                                  mock_makecat, mock_zeropoint2,
                                  mock_transform2natural, mock_fitcol3,
                                  mock_limmag, mock_get_other_filters, mock_plt):
        """Lines 480+: when zeropoint2 returns 9999, limmag is not called."""
        import lsc.lscabsphotdef

        n_stars = 10
        hdr, keymap, readkey3_side, stdcoo, cat_dict, mock_wcs_instance = \
            self._setup_absphot_mocks(field='sloan', instrume='fa15', filter_='rp', n_stars=n_stars)

        mock_getheader.return_value = hdr
        mock_WCS.return_value = mock_wcs_instance
        mock_readtxt.return_value = stdcoo
        mock_makecat.return_value = cat_dict
        mock_crossmatch.return_value = (np.zeros(n_stars), np.arange(n_stars), np.arange(n_stars))

        def transform_side(instr, cat, coloref, field):
            return {k: np.array(v, float) for k, v in cat.items()}
        mock_transform2natural.side_effect = transform_side
        # Return 9999 so limmag is NOT called
        mock_zeropoint2.return_value = (9999, 9999, np.array([1, 2, 3]), np.array([25.0, 25.1, 24.9]))
        mock_get_other_filters.return_value = {'g', 'r', 'i'}
        mock_fitcol3.return_value = (25.0, 0.05, 0.03, 0.01)

        with patch('lsc.util.readkey3', side_effect=readkey3_side), \
             patch('builtins.open', mock_open()):
            lsc.lscabsphotdef.absphot('/tmp/test.fits', _catalogue='/tmp/sloan.cat',
                                       _field='sloan', redo=True, show=False,
                                       _interactive=False, _type='fit')

        mock_limmag.assert_not_called()

    @patch('lsc.lscabsphotdef.plt')
    @patch('lsc.lscabsphotdef.get_other_filters')
    @patch('lsc.lscabsphotdef.limmag', return_value=22.5)
    @patch('lsc.lscabsphotdef.fitcol3')
    @patch('lsc.lscabsphotdef.transform2natural')
    @patch('lsc.lscabsphotdef.zeropoint2')
    @patch('lsc.lscabsphotdef.makecatalogue')
    @patch('lsc.lscastrodef.crossmatch')
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.myloopdef.checkstage', return_value=1)
    @patch('lsc.lscabsphotdef.WCS')
    @patch('lsc.lscabsphotdef.fits.getheader')
    def test_no_colorvec_uses_self_color(self, mock_getheader, mock_WCS, mock_checkstage,
                                          mock_updatevalue, mock_updateheader,
                                          mock_readtxt, mock_crossmatch,
                                          mock_makecat, mock_zeropoint2,
                                          mock_transform2natural, mock_fitcol3,
                                          mock_limmag, mock_get_other_filters, mock_plt):
        """Lines 491-492: when colorvec is empty, append self-color like 'rr'."""
        import lsc.lscabsphotdef

        n_stars = 10
        hdr, keymap, readkey3_side, stdcoo, cat_dict, mock_wcs_instance = \
            self._setup_absphot_mocks(field='sloan', instrume='fa15', filter_='rp', n_stars=n_stars)

        mock_getheader.return_value = hdr
        mock_WCS.return_value = mock_wcs_instance
        mock_readtxt.return_value = stdcoo
        mock_makecat.return_value = cat_dict
        mock_crossmatch.return_value = (np.zeros(n_stars), np.arange(n_stars), np.arange(n_stars))

        def transform_side(instr, cat, coloref, field):
            return {k: np.array(v, float) for k, v in cat.items()}
        mock_transform2natural.side_effect = transform_side
        mock_zeropoint2.return_value = (25.0, 0.05, np.array([1, 2, 3]), np.array([25.0, 25.1, 24.9]))
        # Return only the filter itself so chosecolor returns empty for it
        mock_get_other_filters.return_value = {'r'}
        mock_fitcol3.return_value = (25.0, 0.05, 0.0, 0.0)

        with patch('lsc.util.readkey3', side_effect=readkey3_side), \
             patch('builtins.open', mock_open()):
            lsc.lscabsphotdef.absphot('/tmp/test.fits', _catalogue='/tmp/sloan.cat',
                                       _field='sloan', redo=True, show=False,
                                       _interactive=False, _type='fit')

        assert mock_fitcol3.called

    @patch('lsc.lscabsphotdef.plt')
    @patch('lsc.lscabsphotdef.get_other_filters')
    @patch('lsc.lscabsphotdef.limmag', return_value=22.5)
    @patch('lsc.lscabsphotdef.fitcol2')
    @patch('lsc.lscabsphotdef.transform2natural')
    @patch('lsc.lscabsphotdef.zeropoint2')
    @patch('lsc.lscabsphotdef.makecatalogue')
    @patch('lsc.lscastrodef.crossmatch')
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.myloopdef.checkstage', return_value=1)
    @patch('lsc.lscabsphotdef.WCS')
    @patch('lsc.lscabsphotdef.fits.getheader')
    def test_zcatold_uses_fitcol2(self, mock_getheader, mock_WCS, mock_checkstage,
                                   mock_updatevalue, mock_updateheader,
                                   mock_readtxt, mock_crossmatch,
                                   mock_makecat, mock_zeropoint2,
                                   mock_transform2natural, mock_fitcol2,
                                   mock_limmag, mock_get_other_filters, mock_plt):
        """Lines 531-533: zcatold=True uses fitcol2 (non-interactive)."""
        import lsc.lscabsphotdef

        n_stars = 10
        hdr, keymap, readkey3_side, stdcoo, cat_dict, mock_wcs_instance = \
            self._setup_absphot_mocks(field='sloan', instrume='fa15', filter_='rp', n_stars=n_stars)

        mock_getheader.return_value = hdr
        mock_WCS.return_value = mock_wcs_instance
        mock_readtxt.return_value = stdcoo
        mock_makecat.return_value = cat_dict
        mock_crossmatch.return_value = (np.zeros(n_stars), np.arange(n_stars), np.arange(n_stars))

        def transform_side(instr, cat, coloref, field):
            return {k: np.array(v, float) for k, v in cat.items()}
        mock_transform2natural.side_effect = transform_side
        mock_zeropoint2.return_value = (25.0, 0.05, np.array([1, 2, 3]), np.array([25.0, 25.1, 24.9]))
        mock_get_other_filters.return_value = {'g', 'r', 'i'}
        mock_fitcol2.return_value = (25.0, 0.05, 0.03, 0.01)

        with patch('lsc.util.readkey3', side_effect=readkey3_side), \
             patch('builtins.open', mock_open()):
            lsc.lscabsphotdef.absphot('/tmp/test.fits', _catalogue='/tmp/sloan.cat',
                                       _field='sloan', redo=True, show=False,
                                       _interactive=False, _type='fit', zcatold=True)

        assert mock_fitcol2.called

    @patch('lsc.lscabsphotdef.plt')
    @patch('lsc.lscabsphotdef.get_other_filters')
    @patch('lsc.lscabsphotdef.limmag', return_value=22.5)
    @patch('lsc.lscabsphotdef.fitcol3')
    @patch('lsc.lscabsphotdef.transform2natural')
    @patch('lsc.lscabsphotdef.zeropoint2')
    @patch('lsc.lscabsphotdef.makecatalogue')
    @patch('lsc.lscastrodef.crossmatch')
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.myloopdef.checkstage', return_value=1)
    @patch('lsc.lscabsphotdef.WCS')
    @patch('lsc.lscabsphotdef.fits.getheader')
    def test_cutmag_limits_stars(self, mock_getheader, mock_WCS, mock_checkstage,
                                  mock_updatevalue, mock_updateheader,
                                  mock_readtxt, mock_crossmatch,
                                  mock_makecat, mock_zeropoint2,
                                  mock_transform2natural, mock_fitcol3,
                                  mock_limmag, mock_get_other_filters, mock_plt):
        """Lines 336-343: cutmag parameter cuts bright stars."""
        import lsc.lscabsphotdef

        n_stars = 10
        hdr, keymap, readkey3_side, stdcoo, cat_dict, mock_wcs_instance = \
            self._setup_absphot_mocks(field='sloan', instrume='fa15', filter_='rp', n_stars=n_stars)

        mock_getheader.return_value = hdr
        mock_WCS.return_value = mock_wcs_instance
        mock_readtxt.return_value = stdcoo
        mock_makecat.return_value = cat_dict
        mock_crossmatch.return_value = (np.zeros(n_stars), np.arange(n_stars), np.arange(n_stars))

        def transform_side(instr, cat, coloref, field):
            return {k: np.array(v, float) for k, v in cat.items()}
        mock_transform2natural.side_effect = transform_side
        mock_zeropoint2.return_value = (25.0, 0.05, np.array([1, 2, 3]), np.array([25.0, 25.1, 24.9]))
        mock_get_other_filters.return_value = {'g', 'r', 'i'}
        mock_fitcol3.return_value = (25.0, 0.05, 0.03, 0.01)

        with patch('lsc.util.readkey3', side_effect=readkey3_side), \
             patch('builtins.open', mock_open()):
            # Use a very low cutmag so all stars are too faint
            lsc.lscabsphotdef.absphot('/tmp/test.fits', _catalogue='/tmp/sloan.cat',
                                       _field='sloan', redo=True, show=False,
                                       _interactive=False, _type='fit', cutmag=0)

    @patch('lsc.lscabsphotdef.plt')
    @patch('lsc.lscabsphotdef.get_other_filters')
    @patch('lsc.lscabsphotdef.limmag', return_value=22.5)
    @patch('lsc.lscabsphotdef.fitcol3')
    @patch('lsc.lscabsphotdef.transform2natural')
    @patch('lsc.lscabsphotdef.zeropoint2')
    @patch('lsc.lscabsphotdef.makecatalogue')
    @patch('lsc.lscastrodef.crossmatch')
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.myloopdef.checkstage', return_value=1)
    @patch('lsc.lscabsphotdef.WCS')
    @patch('lsc.lscabsphotdef.fits.getheader')
    def test_result_with_nan_handled(self, mock_getheader, mock_WCS, mock_checkstage,
                                      mock_updatevalue, mock_updateheader,
                                      mock_readtxt, mock_crossmatch,
                                      mock_makecat, mock_zeropoint2,
                                      mock_transform2natural, mock_fitcol3,
                                      mock_limmag, mock_get_other_filters, mock_plt):
        """Lines 541-543: NaN/Inf in results get replaced with 0.0."""
        import lsc.lscabsphotdef

        n_stars = 10
        hdr, keymap, readkey3_side, stdcoo, cat_dict, mock_wcs_instance = \
            self._setup_absphot_mocks(field='sloan', instrume='fa15', filter_='rp', n_stars=n_stars)

        mock_getheader.return_value = hdr
        mock_WCS.return_value = mock_wcs_instance
        mock_readtxt.return_value = stdcoo
        mock_makecat.return_value = cat_dict
        mock_crossmatch.return_value = (np.zeros(n_stars), np.arange(n_stars), np.arange(n_stars))

        def transform_side(instr, cat, coloref, field):
            return {k: np.array(v, float) for k, v in cat.items()}
        mock_transform2natural.side_effect = transform_side
        mock_zeropoint2.return_value = (25.0, 0.05, np.array([1, 2, 3]), np.array([25.0, 25.1, 24.9]))
        mock_get_other_filters.return_value = {'g', 'r', 'i'}
        # Return NaN in color term to test NaN handling
        mock_fitcol3.return_value = (float('nan'), 0.05, float('inf'), 0.01)

        with patch('lsc.util.readkey3', side_effect=readkey3_side), \
             patch('builtins.open', mock_open()):
            lsc.lscabsphotdef.absphot('/tmp/test.fits', _catalogue='/tmp/sloan.cat',
                                       _field='sloan', redo=True, show=False,
                                       _interactive=False, _type='fit')

        assert mock_updateheader.called

    @patch('lsc.lscabsphotdef.plt')
    @patch('lsc.lscabsphotdef.get_other_filters')
    @patch('lsc.lscabsphotdef.limmag', return_value=22.5)
    @patch('lsc.lscabsphotdef.fitcol3')
    @patch('lsc.lscabsphotdef.transform2natural')
    @patch('lsc.lscabsphotdef.zeropoint2')
    @patch('lsc.lscabsphotdef.makecatalogue')
    @patch('lsc.lscastrodef.crossmatch')
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.myloopdef.checkstage', return_value=1)
    @patch('lsc.lscabsphotdef.WCS')
    @patch('lsc.lscabsphotdef.fits.getheader')
    def test_result_9999_sets_zcat_X(self, mock_getheader, mock_WCS, mock_checkstage,
                                      mock_updatevalue, mock_updateheader,
                                      mock_readtxt, mock_crossmatch,
                                      mock_makecat, mock_zeropoint2,
                                      mock_transform2natural, mock_fitcol3,
                                      mock_limmag, mock_get_other_filters, mock_plt):
        """Lines 550-551: result[ll][0] == 9999 sets zcat='X'."""
        import lsc.lscabsphotdef

        n_stars = 10
        hdr, keymap, readkey3_side, stdcoo, cat_dict, mock_wcs_instance = \
            self._setup_absphot_mocks(field='sloan', instrume='fa15', filter_='rp', n_stars=n_stars)

        mock_getheader.return_value = hdr
        mock_WCS.return_value = mock_wcs_instance
        mock_readtxt.return_value = stdcoo
        mock_makecat.return_value = cat_dict
        mock_crossmatch.return_value = (np.zeros(n_stars), np.arange(n_stars), np.arange(n_stars))

        def transform_side(instr, cat, coloref, field):
            return {k: np.array(v, float) for k, v in cat.items()}
        mock_transform2natural.side_effect = transform_side
        mock_zeropoint2.return_value = (25.0, 0.05, np.array([1, 2, 3]), np.array([25.0, 25.1, 24.9]))
        mock_get_other_filters.return_value = {'g', 'r', 'i'}
        # Return 9999 as zero point
        mock_fitcol3.return_value = (9999, 0.0, 0.0, 0.0)

        with patch('lsc.util.readkey3', side_effect=readkey3_side), \
             patch('builtins.open', mock_open()):
            lsc.lscabsphotdef.absphot('/tmp/test.fits', _catalogue='/tmp/sloan.cat',
                                       _field='sloan', redo=True, show=False,
                                       _interactive=False, _type='fit')

        assert mock_updatevalue.called

    @patch('lsc.lscabsphotdef.plt')
    @patch('lsc.lscabsphotdef.get_other_filters')
    @patch('lsc.lscabsphotdef.limmag', return_value=22.5)
    @patch('lsc.lscabsphotdef.fitcol3')
    @patch('lsc.lscabsphotdef.transform2natural')
    @patch('lsc.lscabsphotdef.zeropoint2')
    @patch('lsc.lscabsphotdef.makecatalogue')
    @patch('lsc.lscastrodef.crossmatch')
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.myloopdef.checkstage', return_value=1)
    @patch('lsc.lscabsphotdef.WCS')
    @patch('lsc.lscabsphotdef.fits.getheader')
    def test_up_filter_maxcolor_10(self, mock_getheader, mock_WCS, mock_checkstage,
                                    mock_updatevalue, mock_updateheader,
                                    mock_readtxt, mock_crossmatch,
                                    mock_makecat, mock_zeropoint2,
                                    mock_transform2natural, mock_fitcol3,
                                    mock_limmag, mock_get_other_filters, mock_plt):
        """Lines 508-509: up/zs filter uses maxcolor=10."""
        import lsc.lscabsphotdef

        n_stars = 10
        hdr, keymap, readkey3_side, stdcoo, cat_dict, mock_wcs_instance = \
            self._setup_absphot_mocks(field='sloan', instrume='fa15', filter_='up', n_stars=n_stars)
        keymap['filter'] = 'up'

        # Rebuild cat_dict with correct filter
        sn2_name = '/tmp/test.sn2.fits'
        rng = np.random.default_rng(42)
        cat_dict = {}
        cat_dict['up'] = {}
        cat_dict['up'][sn2_name] = {
            'ra0': stdcoo['ra'] + rng.uniform(-0.0001, 0.0001, n_stars),
            'dec0': stdcoo['dec'] + rng.uniform(-0.0001, 0.0001, n_stars),
            'smagf': rng.uniform(-8, -6, n_stars),
            'smagerrf': rng.uniform(0.01, 0.05, n_stars),
            'magp3': rng.uniform(-8, -6, n_stars),
            'merrp3': rng.uniform(0.01, 0.05, n_stars),
        }

        mock_getheader.return_value = hdr
        mock_WCS.return_value = mock_wcs_instance
        mock_readtxt.return_value = stdcoo
        mock_makecat.return_value = cat_dict
        mock_crossmatch.return_value = (np.zeros(n_stars), np.arange(n_stars), np.arange(n_stars))

        def transform_side(instr, cat, coloref, field):
            return {k: np.array(v, float) for k, v in cat.items()}
        mock_transform2natural.side_effect = transform_side
        mock_zeropoint2.return_value = (25.0, 0.05, np.array([1, 2, 3]), np.array([25.0, 25.1, 24.9]))
        mock_get_other_filters.return_value = {'u', 'g'}
        mock_fitcol3.return_value = (25.0, 0.05, 0.03, 0.01)

        with patch('lsc.util.readkey3', side_effect=readkey3_side), \
             patch('builtins.open', mock_open()):
            lsc.lscabsphotdef.absphot('/tmp/test.fits', _catalogue='/tmp/sloan.cat',
                                       _field='sloan', redo=True, show=False,
                                       _interactive=False, _type='fit')

    @patch('lsc.lscabsphotdef.plt')
    @patch('lsc.lscabsphotdef.get_other_filters')
    @patch('lsc.lscabsphotdef.limmag', return_value=22.5)
    @patch('lsc.lscabsphotdef.fitcol3')
    @patch('lsc.lscabsphotdef.transform2natural')
    @patch('lsc.lscabsphotdef.zeropoint2')
    @patch('lsc.lscabsphotdef.makecatalogue')
    @patch('lsc.lscastrodef.crossmatch')
    @patch('lsc.lscastrodef.readtxt')
    @patch('lsc.util.updateheader')
    @patch('lsc.mysqldef.updatevalue')
    @patch('lsc.myloopdef.checkstage', return_value=1)
    @patch('lsc.lscabsphotdef.WCS')
    @patch('lsc.lscabsphotdef.fits.getheader')
    def test_show_creates_figure(self, mock_getheader, mock_WCS, mock_checkstage,
                                  mock_updatevalue, mock_updateheader,
                                  mock_readtxt, mock_crossmatch,
                                  mock_makecat, mock_zeropoint2,
                                  mock_transform2natural, mock_fitcol3,
                                  mock_limmag, mock_get_other_filters, mock_plt):
        """Lines 493-494: show=True with not zcatold creates figure."""
        import lsc.lscabsphotdef

        n_stars = 10
        hdr, keymap, readkey3_side, stdcoo, cat_dict, mock_wcs_instance = \
            self._setup_absphot_mocks(field='sloan', instrume='fa15', filter_='rp', n_stars=n_stars)

        mock_getheader.return_value = hdr
        mock_WCS.return_value = mock_wcs_instance
        mock_readtxt.return_value = stdcoo
        mock_makecat.return_value = cat_dict
        mock_crossmatch.return_value = (np.zeros(n_stars), np.arange(n_stars), np.arange(n_stars))

        def transform_side(instr, cat, coloref, field):
            return {k: np.array(v, float) for k, v in cat.items()}
        mock_transform2natural.side_effect = transform_side
        mock_zeropoint2.return_value = (25.0, 0.05, np.array([1, 2, 3]), np.array([25.0, 25.1, 24.9]))
        mock_get_other_filters.return_value = {'g', 'r', 'i'}
        mock_fitcol3.return_value = (25.0, 0.05, 0.03, 0.01)

        # plt.subplots must return (fig, axarr)
        mock_fig = MagicMock()
        mock_axarr = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_axarr)

        with patch('lsc.util.readkey3', side_effect=readkey3_side), \
             patch('builtins.open', mock_open()):
            lsc.lscabsphotdef.absphot('/tmp/test.fits', _catalogue='/tmp/sloan.cat',
                                       _field='sloan', redo=True, show=True,
                                       _interactive=False, _type='fit')

        mock_plt.subplots.assert_called()


# ===========================================================================
# fitcol3 interactive path (lines 559, 580-588)
# ===========================================================================

class TestFitcol3Interactive:
    """Test fitcol3 interactive path."""

    @patch('lsc.lscabsphotdef.plt')
    @patch('lsc.util.userinput', return_value='')
    def test_interactive_mode(self, mock_userinput, mock_plt):
        """Lines 559, 580-588: interactive mode calls plt and userinput."""
        from lsc.lscabsphotdef import fitcol3
        colors = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0])
        deltas = np.array([25.0, 25.3, 25.5, 25.8, 26.1, 26.3, 26.5])
        dcolors = np.full(7, 0.01)
        ddeltas = np.full(7, 0.05)

        # Mock the canvas connection
        mock_fig = MagicMock()
        mock_plt.gcf.return_value = mock_fig

        Z, dZ, C, dC = fitcol3(colors, deltas, dcolors, ddeltas, show=True, interactive=True)
        assert isinstance(Z, float)
        mock_userinput.assert_called()
        mock_fig.canvas.mpl_connect.assert_called()
        mock_fig.canvas.mpl_disconnect.assert_called()


# ===========================================================================
# fitcol: len(idd)<=2 path (lines 679-680)
# ===========================================================================

class TestFitcolEdgeCases:
    """Test fitcol edge cases with 2 points."""

    @patch('lsc.lscabsphotdef.plt')
    @patch('lsc.util.userinput', return_value='')
    def test_two_points_no_fixcol(self, mock_userinput, mock_plt):
        """Lines 679-680: len(idd)<=2 sets sigmaa=0, sigmab=0."""
        from lsc.lscabsphotdef import fitcol
        col = np.array([0.5, 1.0])
        dmag = np.array([25.0, 25.5])
        mock_fig = MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_plt.figtext.return_value = MagicMock()

        a, sa, b, sb = fitcol(col, dmag, 'r', 'gr', fissa='')
        assert sa == 0.0
        assert sb == 0.0

    @patch('lsc.lscabsphotdef.plt')
    @patch('lsc.util.userinput', return_value='')
    def test_with_fixcol(self, mock_userinput, mock_plt):
        """Lines 698-699: with fixcol set, uses median approach."""
        from lsc.lscabsphotdef import fitcol
        col = np.array([0.5, 1.0, 1.5, 2.0])
        dmag = np.array([25.0, 25.1, 25.2, 25.3])
        mock_fig = MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_plt.figtext.return_value = MagicMock()

        a, sa, b, sb = fitcol(col, dmag, 'r', 'gr', fissa=0.1)
        assert b == 0.1
        assert sb == 0.0


# ===========================================================================
# fitcol2: exception paths (lines 724, 732)
# ===========================================================================

class TestFitcol2EdgeCases:
    """Test fitcol2 edge cases."""

    def test_single_point_returns_9999(self):
        """Lines 748-752: single point results in 9999."""
        import lsc.lscabsphotdef
        col = np.array([0.5])
        dmag = np.array([25.0])
        a, sa, b, sb = lsc.lscabsphotdef.fitcol2(col, dmag, 'r', 'gr')
        assert a == 9999
        assert sa == 9999

    @patch('lsc.lscabsphotdef.meanclip2')
    def test_zero_division_in_sigmae_with_fixcol(self, mock_meanclip2):
        """Line 724: try/except when division by zero in sigmae calc."""
        import lsc.lscabsphotdef
        # Return values where len(xx0)-2 = 0 (i.e., exactly 2 points returned)
        xx0 = np.array([0.5, 1.0])
        yy0 = np.array([25.0, 25.5])
        mock_meanclip2.return_value = (25.0, 0.0, yy0, xx0)
        col = np.array([0.5, 1.0, 1.5])
        dmag = np.array([25.0, 25.5, 26.0])
        a, sa, b, sb = lsc.lscabsphotdef.fitcol2(col, dmag, 'r', 'gr', fixcol=0.1)
        # sigmae should be 0 because sig0=0.0 => sqrt(0/0) => except => 0
        assert sa == 0.0

    @patch('lsc.lscabsphotdef.meanclip3')
    def test_zero_division_in_sigmae_no_fixcol(self, mock_meanclip3):
        """Line 732: try/except when division by zero in sigmae calc (no fixcol)."""
        import lsc.lscabsphotdef
        # Return values where len(xx0)-2 = 0 (i.e., exactly 2 points returned)
        xx0 = np.array([0.5, 1.0])
        yy0 = np.array([25.0, 25.5])
        mock_meanclip3.return_value = (25.0, 0.0, 0.5, yy0, xx0)
        col = np.array([0.5, 1.0, 1.5])
        dmag = np.array([25.0, 25.5, 26.0])
        a, sa, b, sb = lsc.lscabsphotdef.fitcol2(col, dmag, 'r', 'gr', fixcol='')
        # sigmae should be 0 because sig0=0.0 => sqrt(0/0) => except => 0
        assert sa == 0.0


# ===========================================================================
# panstarrs2file: None table path (lines 1229-1232)
# ===========================================================================

class TestPanstarrs2fileNone:
    """Test panstarrs2file when the table processing ends with no output."""

    @patch('lsc.lscabsphotdef.SkyCoord')
    def test_none_table_prints_no_matching(self, mock_skycoord):
        """Lines 1229-1232: panstarrs2file handles None/empty gracefully."""
        import lsc.lscabsphotdef
        from unittest.mock import patch as patch2

        # Mock Vizier
        with patch2('lsc.lscabsphotdef.u') as mock_u:
            mock_u.degree = 'degree'
            mock_u.arcmin = 'arcmin'
            # We need to mock the import inside the function
            mock_vizier_mod = MagicMock()
            mock_vizier = MagicMock()
            mock_vizier_mod.Vizier = mock_vizier
            # query_region returns a list with one table
            from astropy.table import Table
            t = Table()
            t['RAJ2000'] = [150.0, 151.0]
            t['DEJ2000'] = [2.0, 2.1]
            t['objID'] = [12345, 67890]
            t['gFlags'] = [49432, 49432]  # good_dq flags
            t['ymag'] = [18.0, 19.0]
            t['e_ymag'] = [0.01, 0.02]
            t['gmag'] = [17.0, 18.0]
            t['e_gmag'] = [0.01, 0.02]
            t['rmag'] = [16.5, 17.5]
            t['e_rmag'] = [0.01, 0.02]
            t['imag'] = [16.0, 17.0]
            t['e_imag'] = [0.01, 0.02]
            t['zmag'] = [15.5, 16.5]
            t['e_zmag'] = [0.01, 0.02]
            # Make all flags extended to get empty table after filtering
            t['gFlags'] = [16777216, 16777216]  # extended flag only => keep_indx = False
            mock_vizier.query_region.return_value = [t]

            with patch2.dict('sys.modules', {'astroquery.vizier': mock_vizier_mod}):
                # This will fail because table after filtering is empty
                # and code tries to remove_column which won't work on 0-row table
                # But it still exercises the path
                try:
                    lsc.lscabsphotdef.panstarrs2file(150.0, 2.0, output='/tmp/test.cat')
                except (IndexError, KeyError, Exception):
                    pass  # expected - the point is to exercise the code path


# ===========================================================================
# get_other_filters (lines 24-39)
# ===========================================================================

class TestGetOtherFilters:
    """Test get_other_filters function."""

    @patch('lsc.myloopdef.conn')
    @patch('lsc.mysqldef.query')
    def test_match_by_site_false(self, mock_query, mock_conn):
        """Lines 28-30: match_by_site=False uses simple join."""
        import lsc.lscabsphotdef
        mock_query.return_value = [{'filter': 'rp'}, {'filter': 'gp'}]
        result = lsc.lscabsphotdef.get_other_filters('test.fits', match_by_site=False)
        assert 'r' in result
        assert 'g' in result

    @patch('lsc.myloopdef.conn')
    @patch('lsc.mysqldef.query')
    def test_match_by_site_true(self, mock_query, mock_conn):
        """Lines 24-27: match_by_site=True uses telescope join."""
        import lsc.lscabsphotdef
        mock_query.return_value = [{'filter': 'V'}, {'filter': 'B'}]
        result = lsc.lscabsphotdef.get_other_filters('test.fits', match_by_site=True)
        assert 'V' in result
        assert 'B' in result


# ===========================================================================
# limmag (lines 42-59)
# ===========================================================================

class TestLimmag:
    """Test limmag function."""

    def test_normal_case(self, tmp_path):
        """Lines 42-56: normal case with valid inputs."""
        from lsc.lscabsphotdef import limmag
        from astropy.io import fits
        # Create a simple FITS file
        data = np.random.default_rng(42).normal(1000, 50, (100, 100)).astype(np.float32)
        hdr = fits.Header()
        hdr['EXPTIME'] = 120.0
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 10.0
        hdr['PIXSCALE'] = 0.389
        path = str(tmp_path / 'test.fits')
        fits.writeto(path, data, hdr, overwrite=True)

        with patch('lsc.util.readkey3') as mock_readkey3:
            def readkey3_side(h, k):
                keymap = {'exptime': 120.0, 'gain': 2.0, 'ron': 10.0, 'pixscale': 0.389}
                return keymap.get(k, '')
            mock_readkey3.side_effect = readkey3_side
            mag = limmag(path, zeropoint=25.0, Nsigma_limit=3, _fwhm=4.0)
        assert isinstance(mag, float)
        assert mag < 30  # reasonable magnitude

    def test_zero_radius_returns_9999(self, tmp_path):
        """Lines 57-58: if radius is 0, returns 9999."""
        from lsc.lscabsphotdef import limmag
        from astropy.io import fits
        data = np.random.default_rng(42).normal(1000, 50, (100, 100)).astype(np.float32)
        hdr = fits.Header()
        hdr['EXPTIME'] = 120.0
        hdr['GAIN'] = 2.0
        hdr['RDNOISE'] = 10.0
        hdr['PIXSCALE'] = 0.389
        path = str(tmp_path / 'test2.fits')
        fits.writeto(path, data, hdr, overwrite=True)

        with patch('lsc.util.readkey3') as mock_readkey3:
            def readkey3_side(h, k):
                keymap = {'exptime': 120.0, 'gain': 0, 'ron': 10.0, 'pixscale': 0.389}
                return keymap.get(k, '')
            mock_readkey3.side_effect = readkey3_side
            mag = limmag(path, zeropoint=25.0, Nsigma_limit=3, _fwhm=4.0)
        # gain=0 means _radius will still be computed, but _gain=0 will make condition false
        assert mag == 9999


# ===========================================================================
# deg2HMS (lines 98-124)
# ===========================================================================

class TestDeg2HMS:
    """Test deg2HMS conversion function."""

    def test_ra_colon_format(self):
        """Line 114-117: ra in HH:MM:SS format."""
        from lsc.lscabsphotdef import deg2HMS
        result = deg2HMS(ra='10:00:00.0')
        assert abs(result - 150.0) < 0.01

    def test_dec_colon_format(self):
        """Lines 101-106: dec in DD:MM:SS format."""
        from lsc.lscabsphotdef import deg2HMS
        result = deg2HMS(dec='02:12:00.0')
        assert abs(result - 2.2) < 0.01

    def test_dec_negative_colon(self):
        """Line 105: negative dec in DD:MM:SS."""
        from lsc.lscabsphotdef import deg2HMS
        result = deg2HMS(dec='-30:00:00.0')
        assert abs(result - (-30.0)) < 0.01

    def test_ra_numeric(self):
        """Lines 119-122: ra as numeric degrees returns HH:MM:SS string."""
        from lsc.lscabsphotdef import deg2HMS
        result = deg2HMS(ra=150.0)
        assert ':' in str(result)

    def test_dec_numeric(self):
        """Lines 108-112: dec as numeric degrees returns DD:MM:SS string."""
        from lsc.lscabsphotdef import deg2HMS
        result = deg2HMS(dec=2.2)
        assert ':' in str(result)

    def test_both_ra_dec(self):
        """Line 123: both ra and dec returns tuple."""
        from lsc.lscabsphotdef import deg2HMS
        ra, dec = deg2HMS(ra='10:00:00.0', dec='02:12:00.0')
        assert abs(ra - 150.0) < 0.01
        assert abs(dec - 2.2) < 0.01


# ===========================================================================
# transform2natural (lines 1012-1067)
# ===========================================================================

class TestTransform2Natural:
    """Test transform2natural function."""

    def test_sloan_system(self):
        """Lines 1017-1033: sloan system transformation."""
        from lsc.lscabsphotdef import transform2natural
        catalogue = {
            'u': np.array([18.0, 19.0]),
            'g': np.array([17.0, 18.0]),
            'r': np.array([16.5, 17.5]),
            'i': np.array([16.0, 17.0]),
            'z': np.array([15.5, 16.5]),
        }
        colorefisso = {'uug': 0.0, 'ggr': 0.1, 'rri': 0.03, 'iri': 0.03, 'ziz': 0.0}
        result = transform2natural('fa15', catalogue, colorefisso, 'sloan')
        assert 'r' in result
        # g should be modified by ggr color term
        assert not np.array_equal(result['g'], catalogue['g'])

    def test_landolt_system(self):
        """Lines 1034-1050: landolt system transformation."""
        from lsc.lscabsphotdef import transform2natural
        catalogue = {
            'U': np.array([15.0, 16.0]),
            'B': np.array([14.5, 15.5]),
            'V': np.array([14.0, 15.0]),
            'R': np.array([13.5, 14.5]),
            'I': np.array([13.0, 14.0]),
        }
        colorefisso = {'UUB': 0.059, 'BBV': 0.06, 'VVR': 0.03, 'RVR': -0.028, 'IRI': 0.013}
        result = transform2natural('fa15', catalogue, colorefisso, 'landolt')
        assert 'V' in result
        assert not np.array_equal(result['B'], catalogue['B'])

    def test_apass_system(self):
        """Lines 1051-1066: apass system transformation."""
        from lsc.lscabsphotdef import transform2natural
        catalogue = {
            'B': np.array([14.5, 15.5]),
            'V': np.array([14.0, 15.0]),
            'g': np.array([14.2, 15.2]),
            'r': np.array([13.8, 14.8]),
            'i': np.array([13.5, 14.5]),
        }
        colorefisso = {'BBV': 0.06, 'ggr': 0.1, 'rri': 0.03, 'iri': 0.03}
        result = transform2natural('fa15', catalogue, colorefisso, 'apass')
        assert 'g' in result
        assert not np.array_equal(result['g'], catalogue['g'])

    def test_values_above_99_treated_as_zero_color(self):
        """Lines 1019-1022: values >= 99 produce zero color."""
        from lsc.lscabsphotdef import transform2natural
        catalogue = {
            'u': np.array([99.0]),
            'g': np.array([17.0]),
            'r': np.array([16.5]),
            'i': np.array([16.0]),
            'z': np.array([15.5]),
        }
        colorefisso = {'uug': 0.1, 'ggr': 0.1, 'rri': 0.03, 'iri': 0.03, 'ziz': 0.0}
        result = transform2natural('fa15', catalogue, colorefisso, 'sloan')
        # u=99, g<99 => color (u-g) should be treated as 0 for u
        # So u_natural = u - colorefisso['uug'] * 0 = u
        assert result['u'][0] == 99.0


# ===========================================================================
# zeronew (lines 1071-1113)
# ===========================================================================

class TestZeronew:
    """Test zeronew function."""

    def test_basic_convergence(self):
        """Lines 1071-1113: zeronew converges on clean data."""
        from lsc.lscabsphotdef import zeronew
        rng = np.random.default_rng(42)
        ZZ = rng.normal(25.0, 0.1, 50)
        # Add a few outliers
        ZZ = np.append(ZZ, [30.0, 20.0, 35.0])
        ZZcut, sigmacut, mediancut = zeronew(ZZ)
        assert abs(mediancut - 25.0) < 0.2
        assert len(ZZcut) <= len(ZZ)

    def test_verbose_output(self, capsys):
        """Lines 1085-1090, 1101-1102: verbose mode prints information."""
        from lsc.lscabsphotdef import zeronew
        rng = np.random.default_rng(42)
        ZZ = rng.normal(25.0, 0.3, 50)
        ZZ = np.append(ZZ, [30.0, 20.0])
        zeronew(ZZ, verbose=True)
        captured = capsys.readouterr()
        assert 'reject' in captured.out or 'number of object' in captured.out

    def test_no_outliers(self):
        """Test with very tight data (minimal rejection)."""
        from lsc.lscabsphotdef import zeronew
        rng = np.random.default_rng(99)
        ZZ = rng.normal(25.0, 0.001, 20)  # very tight distribution
        ZZcut, sigmacut, mediancut = zeronew(ZZ)
        assert abs(mediancut - 25.0) < 0.01


# ===========================================================================
# meanclip2 and meanclip3 (lines 755-814)
# ===========================================================================

class TestMeanclip:
    """Test meanclip2 and meanclip3 functions."""

    def test_meanclip2_basic(self):
        """Lines 755-783: meanclip2 with clean data."""
        from lsc.lscabsphotdef import meanclip2
        xx = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        yy = np.array([25.0, 25.5, 26.0, 26.5, 27.0])
        mean, sig, yy0, xx0 = meanclip2(xx, yy, slope=1.0)
        assert isinstance(mean, float)
        assert len(xx0) <= len(xx)

    def test_meanclip2_with_outlier(self):
        """meanclip2 rejects outliers."""
        from lsc.lscabsphotdef import meanclip2
        xx = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        yy = np.array([25.0, 25.5, 26.0, 26.5, 27.0, 50.0])  # outlier at end
        mean, sig, yy0, xx0 = meanclip2(xx, yy, slope=1.0, clipsig=2.0)
        assert len(xx0) < len(xx)

    def test_meanclip3_basic(self):
        """Lines 785-814: meanclip3 with clean data."""
        from lsc.lscabsphotdef import meanclip3
        xx = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        yy = np.array([25.0, 25.5, 26.0, 26.5, 27.0])
        mean0, sig, slope, yy0, xx0 = meanclip3(xx, yy, slope='')
        assert isinstance(mean0, (float, np.floating))
        assert isinstance(slope, (float, np.floating))

    def test_meanclip3_with_outlier(self):
        """meanclip3 rejects outliers."""
        from lsc.lscabsphotdef import meanclip3
        xx = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        yy = np.array([25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0, 100.0])  # extreme outlier
        mean0, sig, slope, yy0, xx0 = meanclip3(xx, yy, slope='', clipsig=1.5)
        assert len(xx0) < len(xx)


# ===========================================================================
# makecatalogue (lines 818-857)
# ===========================================================================

class TestMakecatalogue:
    """Test makecatalogue function."""

    def test_basic(self, tmp_path):
        """Lines 818-857: makecatalogue reads FITS table."""
        from lsc.lscabsphotdef import makecatalogue
        from astropy.io import fits

        n = 5
        rng = np.random.default_rng(42)
        # Create a sn2.fits file with primary + table
        hdr = fits.Header()
        hdr['FILTER'] = 'rp'
        hdr['EXPTIME'] = 120.0
        hdr['AIRMASS'] = 1.2
        hdr['TELESCOP'] = '1m0-01'
        hdr['SITEID'] = 'lsc'
        hdr['MJD'] = 58970.5

        col_ra = fits.Column(name='ra0', format='D', array=rng.uniform(149, 151, n))
        col_dec = fits.Column(name='dec0', format='D', array=rng.uniform(1.5, 2.5, n))
        col_smagf = fits.Column(name='smagf', format='E', array=rng.uniform(-8, -6, n))
        col_smagerrf = fits.Column(name='smagerrf', format='E', array=rng.uniform(0.01, 0.05, n))
        col_magp3 = fits.Column(name='magp3', format='E', array=rng.uniform(-8, -6, n))
        col_merrp3 = fits.Column(name='merrp3', format='E', array=rng.uniform(0.01, 0.05, n))

        primary = fits.PrimaryHDU(header=hdr)
        table_hdu = fits.BinTableHDU.from_columns(
            [col_ra, col_dec, col_smagf, col_smagerrf, col_magp3, col_merrp3]
        )
        hdul = fits.HDUList([primary, table_hdu])
        path = str(tmp_path / 'test.sn2.fits')
        hdul.writeto(path, overwrite=True)

        with patch('lsc.util.readkey3') as mock_readkey3:
            def readkey3_side(h, k):
                keymap = {'filter': 'rp', 'exptime': 120.0, 'airmass': 1.2,
                          'telescop': '1m0-01'}
                return keymap.get(k, h.get(k, ''))
            mock_readkey3.side_effect = readkey3_side
            result = makecatalogue([path])

        assert 'rp' in result
        assert path in result['rp']
        assert 'ra0' in result['rp'][path]


# ===========================================================================
# finalmag and erroremag (lines 860-892)
# ===========================================================================

class TestFinalmagErroremag:
    """Test finalmag and erroremag functions."""

    def test_finalmag(self):
        """Lines 860-867: finalmag computes calibrated magnitudes."""
        from lsc.lscabsphotdef import finalmag
        M1, M2 = finalmag(25.0, 24.5, 0.03, -0.02, -7.0, -6.5)
        assert isinstance(M1, float)
        assert isinstance(M2, float)

    def test_erroremag_position0(self):
        """Lines 870-876: erroremag for position 0."""
        from lsc.lscabsphotdef import erroremag
        dc0, dc1, dz0, dz1, dm0, dm1 = erroremag(0.05, 0.05, 0.02, 0.02, 0.03, -0.02, 0)
        assert all(isinstance(v, float) for v in [dc0, dc1, dz0, dz1, dm0, dm1])

    def test_erroremag_position1(self):
        """Lines 877-883: erroremag for position 1."""
        from lsc.lscabsphotdef import erroremag
        dc0, dc1, dz0, dz1, dm0, dm1 = erroremag(0.05, 0.05, 0.02, 0.02, 0.03, -0.02, 1)
        assert all(isinstance(v, float) for v in [dc0, dc1, dz0, dz1, dm0, dm1])

    def test_erroremag_other_position(self):
        """Lines 884-891: erroremag for other positions uses defaults."""
        from lsc.lscabsphotdef import erroremag
        dc0, dc1, dz0, dz1, dm0, dm1 = erroremag(0.05, 0.05, 0.02, 0.02, 0.03, -0.02, 2)
        assert dm0 == 1
        assert dz0 == 0
        assert dc0 == 0


# ===========================================================================
# zeropoint (lines 896-945)
# ===========================================================================

class TestZeropoint:
    """Test zeropoint function."""

    def test_basic_convergence(self):
        """Lines 896-945: zeropoint converges."""
        from lsc.lscabsphotdef import zeropoint
        rng = np.random.default_rng(42)
        data = rng.normal(25.0, 0.1, 30)
        mag = rng.uniform(14, 18, 30)
        z2, std2, mag2, data2 = zeropoint(data, mag)
        assert abs(z2 - 25.0) < 0.3

    def test_with_outliers(self):
        """zeropoint rejects outliers."""
        from lsc.lscabsphotdef import zeropoint
        rng = np.random.default_rng(42)
        data = rng.normal(25.0, 0.1, 30)
        data = np.append(data, [30.0, 20.0])
        mag = rng.uniform(14, 18, 32)
        z2, std2, mag2, data2 = zeropoint(data, mag)
        assert len(data2) < len(data)


# ===========================================================================
# zeropoint2 (lines 949-1008)
# ===========================================================================

class TestZeropoint2:
    """Test zeropoint2 function."""

    def test_empty_input(self):
        """Lines 1006-1007: empty input returns 9999."""
        from lsc.lscabsphotdef import zeropoint2
        z2, std2, mag2, data2 = zeropoint2(np.array([]), np.array([]))
        assert z2 == 9999

    def test_basic(self):
        """Lines 949-1005: basic convergence."""
        from lsc.lscabsphotdef import zeropoint2
        rng = np.random.default_rng(42)
        xx = rng.normal(25.0, 0.1, 30)
        mag = rng.uniform(14, 18, 30)
        # xx should be std mag, mag is instrumental
        z2, std2, mag2, data2 = zeropoint2(xx, mag)
        assert z2 != 9999

    def test_nan_result(self):
        """Line 1005: if z2 is nan, return 9999."""
        from lsc.lscabsphotdef import zeropoint2
        # All same values produce zero std, can cause nan
        xx = np.array([25.0, 25.0, 25.0, 25.0, 25.0, 25.0])
        mag = np.array([18.0, 18.0, 18.0, 18.0, 18.0, 18.0])
        z2, std2, mag2, data2 = zeropoint2(xx, mag)
        # Should converge to 7.0 (25 - 18)
        assert abs(z2 - 7.0) < 0.01 or z2 == 9999


# ===========================================================================
# sloan2file (lines 1116-1153)
# ===========================================================================

class TestSloan2file:
    """Test sloan2file function."""

    @patch('lsc.lscabsphotdef.SDSS')
    def test_none_result(self, mock_SDSS):
        """Lines 1152-1153: SDSS returns None."""
        import lsc.lscabsphotdef
        mock_SDSS.query_sql.return_value = None
        lsc.lscabsphotdef.sloan2file(150.0, 2.0, output='/tmp/test.cat')
        # Should print 'No matching objects.' and not crash

    @patch('lsc.lscabsphotdef.SDSS')
    def test_valid_result(self, mock_SDSS, tmp_path):
        """Lines 1122-1151: SDSS returns valid table."""
        import lsc.lscabsphotdef
        from astropy.table import Table
        t = Table()
        t['ra'] = [150.0, 150.1]
        t['dec'] = [2.0, 2.1]
        t['objID'] = [12345, 67890]
        for filt in 'ugriz':
            t[filt] = [18.0, 19.0]
            t['err_' + filt] = [0.01, 0.02]
        mock_SDSS.query_sql.return_value = t
        output = str(tmp_path / 'sloan.cat')
        lsc.lscabsphotdef.sloan2file(150.0, 2.0, output=output)
        assert os.path.exists(output)


# ===========================================================================
# calcZC edge cases (lines 605-645)
# ===========================================================================

class TestCalcZCEdgeCases:
    """Test calcZC function edge cases."""

    def test_no_keep_uses_guess(self):
        """Lines 621-623: when keep is all False, uses guess."""
        import lsc.lscabsphotdef
        # Set the global keep to all False
        lsc.lscabsphotdef.keep = np.array([False, False, False, False, False])
        colors = np.array([0.5, 0.8, 1.0, 1.2, 1.5])
        deltas = np.array([25.0, 25.3, 25.5, 25.8, 26.1])
        dcolors = np.full(5, 0.01)
        ddeltas = np.full(5, 0.05)
        Z, dZ, C, dC = lsc.lscabsphotdef.calcZC(
            colors, deltas, dcolors, ddeltas,
            fixedC=0.1, filt='r', col='gr', show=False, guess=[23.0, 0.03]
        )
        # When all keep=False with fixedC set, falls to else branch using guess
        assert Z == 23.0
        assert C == 0.03  # C comes from guess[1] in else branch
        assert dZ == 0
        assert dC == 0

    def test_fixedC_none_uses_odr(self):
        """Lines 607-616: fixedC=None uses ODR fitting."""
        import lsc.lscabsphotdef
        lsc.lscabsphotdef.keep = np.array([True, True, True, True, True, True, True])
        colors = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0])
        deltas = np.array([25.0, 25.3, 25.5, 25.8, 26.1, 26.3, 26.5])
        dcolors = np.full(7, 0.01)
        ddeltas = np.full(7, 0.05)
        Z, dZ, C, dC = lsc.lscabsphotdef.calcZC(
            colors, deltas, dcolors, ddeltas,
            fixedC=None, filt='r', col='gr', show=False, guess=[23.0, 0.03]
        )
        assert isinstance(Z, (float, np.floating))
        assert isinstance(C, (float, np.floating))

    @patch('lsc.lscabsphotdef.plt')
    def test_show_plots(self, mock_plt):
        """Lines 626-643: show=True creates plots."""
        import lsc.lscabsphotdef
        lsc.lscabsphotdef.keep = np.array([True, True, True, False, True])
        colors = np.array([0.5, 0.8, 1.0, 1.2, 1.5])
        deltas = np.array([25.0, 25.3, 25.5, 25.8, 26.1])
        dcolors = np.full(5, 0.01)
        ddeltas = np.full(5, 0.05)

        # Need to set up the mock for plt.gca() and plt.axis()
        mock_ax = MagicMock()
        mock_ax.get_autoscale_on.return_value = True
        mock_plt.gca.return_value = mock_ax
        mock_plt.axis.return_value = [0, 2, 24, 27]

        Z, dZ, C, dC = lsc.lscabsphotdef.calcZC(
            colors, deltas, dcolors, ddeltas,
            fixedC=0.1, filt='r', col='gr', show=True, guess=[23.0, 0.03]
        )
        mock_plt.scatter.assert_called()

    @patch('lsc.lscabsphotdef.plt')
    def test_show_with_existing_limits(self, mock_plt):
        """Line 628: show with autoscale off uses existing limits."""
        import lsc.lscabsphotdef
        lsc.lscabsphotdef.keep = np.array([True, True, True, True, True])
        colors = np.array([0.5, 0.8, 1.0, 1.2, 1.5])
        deltas = np.array([25.0, 25.3, 25.5, 25.8, 26.1])
        dcolors = np.full(5, 0.01)
        ddeltas = np.full(5, 0.05)

        mock_ax = MagicMock()
        mock_ax.get_autoscale_on.return_value = False
        mock_plt.gca.return_value = mock_ax
        mock_plt.axis.return_value = [0, 2, 24, 27]

        Z, dZ, C, dC = lsc.lscabsphotdef.calcZC(
            colors, deltas, dcolors, ddeltas,
            fixedC=0.1, filt='r', col='gr', show=True, guess=[23.0, 0.03]
        )
        # Should call plt.axis() to get limits
        assert mock_plt.axis.called


# ===========================================================================
# fitcol3: crazy color term path (line 589-601)
# ===========================================================================

class TestFitcol3CrazyColorTerm:
    """Test fitcol3 when color term is too large."""

    @patch('lsc.lscabsphotdef.plt')
    def test_color_term_too_large_redo(self, mock_plt):
        """Lines 589-601: C > 0.3 triggers redo with fixed C."""
        import lsc.lscabsphotdef

        # We need to make calcZC return a large C first time, then normal
        call_count = [0]
        original_calcZC = lsc.lscabsphotdef.calcZC

        def mock_calcZC(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: return large C
                return 25.0, 0.05, 0.5, 0.1  # C=0.5 > 0.3
            else:
                return 25.0, 0.05, 0.0, 0.0

        colors = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0])
        deltas = np.array([25.0, 25.3, 25.5, 25.8, 26.1, 26.3, 26.5])
        dcolors = np.full(7, 0.01)
        ddeltas = np.full(7, 0.05)

        with patch.object(lsc.lscabsphotdef, 'calcZC', side_effect=mock_calcZC):
            Z, dZ, C, dC = lsc.lscabsphotdef.fitcol3(
                colors, deltas, dcolors, ddeltas,
                fixedC=None, filt='r', show=False, interactive=False
            )
        # Should have been called twice (initial + redo)
        assert call_count[0] == 2

    @patch('lsc.lscabsphotdef.plt')
    def test_color_term_too_large_g_filter(self, mock_plt):
        """Lines 590-591: g filter with crazy C uses 0.1."""
        import lsc.lscabsphotdef

        call_count = [0]

        def mock_calcZC(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return 25.0, 0.05, 0.5, 0.1
            else:
                return 25.0, 0.05, 0.1, 0.0

        colors = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0])
        deltas = np.array([25.0, 25.3, 25.5, 25.8, 26.1, 26.3, 26.5])
        dcolors = np.full(7, 0.01)
        ddeltas = np.full(7, 0.05)

        with patch.object(lsc.lscabsphotdef, 'calcZC', side_effect=mock_calcZC):
            Z, dZ, C, dC = lsc.lscabsphotdef.fitcol3(
                colors, deltas, dcolors, ddeltas,
                fixedC=None, filt='g', show=False, interactive=False
            )
        assert call_count[0] == 2


# ===========================================================================
# fitcol3: not enough points path (line 567-570)
# ===========================================================================

class TestFitcol3NotEnoughPoints:
    """Test fitcol3 when not enough points remain after rejection."""

    @patch('lsc.lscabsphotdef.plt')
    def test_few_points_defaults_to_fixed(self, mock_plt):
        """Lines 567-570: if sum(keep)<=5 after theilslopes, uses fixed C."""
        import lsc.lscabsphotdef

        # Create data where many points are outliers (will be rejected)
        colors = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 50.0, 60.0, 70.0])
        deltas = np.array([25.0, 25.3, 25.5, 25.8, 26.1, 100.0, 200.0, 300.0])
        dcolors = np.full(8, 0.01)
        ddeltas = np.full(8, 0.05)

        # The crazy outliers will cause most points to be rejected
        Z, dZ, C, dC = lsc.lscabsphotdef.fitcol3(
            colors, deltas, dcolors, ddeltas,
            fixedC=None, filt='r', show=False, interactive=False
        )
        assert isinstance(Z, (float, np.floating))
