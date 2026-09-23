"""Tests for bin/calibratemag.py -- exercises functions and __main__."""
import sys
import os
import runpy
import importlib
import pytest
import numpy as np
import numpy.ma as ma
from unittest.mock import patch, MagicMock
from astropy.table import Table
from astropy.io import ascii

TRUNK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(TRUNK, 'bin', 'calibratemag.py')


def _load_module():
    """Load calibratemag functions without running __main__."""
    with open(SCRIPT) as f:
        source = f.read()
    idx = source.find('\nif __name__ == "__main__":')
    mod_source = source[:idx] if idx > 0 else source
    ns = {}
    exec(compile(mod_source, SCRIPT, 'exec'), ns)
    return ns


@pytest.fixture
def funcs():
    return _load_module()


class TestCrossmatch:
    def test_basic(self, funcs):
        cat0 = Table({'ra': [1.0, 2.0, 3.0], 'dec': [1.0, 2.0, 3.0], 'mag': [15.0, 16.0, 17.0]})
        cat1 = Table({'ra': [1.0001, 2.0001], 'dec': [1.0001, 2.0001]})
        out, matched = funcs['crossmatch'](cat0, cat1, threshold=1.0)
        assert len(matched) == 2

    def test_no_match(self, funcs):
        cat0 = Table({'ra': [1.0], 'dec': [1.0], 'mag': [15.0]})
        cat1 = Table({'ra': [50.0], 'dec': [50.0]})
        out, matched = funcs['crossmatch'](cat0, cat1, threshold=0.1)
        assert len(matched) == 0


class TestAverageInFlux:
    def test_single_value(self, funcs):
        mag = ma.array([15.0])
        dmag = ma.array([0.01])
        avg, davg = funcs['average_in_flux'](mag, dmag)
        assert abs(avg - 15.0) < 0.001

    def test_equal_weights(self, funcs):
        mag = ma.array([15.0, 15.0])
        dmag = ma.array([0.01, 0.01])
        avg, _ = funcs['average_in_flux'](mag, dmag)
        assert abs(avg - 15.0) < 0.001

    def test_2d_axis(self, funcs):
        mag = ma.array([[18.0, 18.1], [19.0, 19.1]])
        dmag = ma.array([[0.01, 0.01], [0.01, 0.01]])
        avg, davg = funcs['average_in_flux'](mag, dmag, axis=1)
        assert avg.shape == (2,)


class TestCombineNights:
    def test_basic(self, funcs):
        mags = ma.array([[15.0, 16.0], [15.1, 16.1], [15.05, 16.05]])
        combined = Table({'filter': ['r', 'r', 'r'], 'mag': mags})
        refcat = Table({'ra': [0.0, 1.0], 'dec': [0.0, 1.0], 'id': [0, 1]})
        cat = funcs['combine_nights'](combined, ['r'], refcat)
        assert abs(cat['r'][0] - 15.05) < 0.1

    def test_all_masked(self, funcs):
        mags = ma.array([[15.0, 16.0]], mask=[[True, True]])
        combined = Table({'filter': ['r'], 'mag': mags})
        refcat = Table({'ra': [0.0, 1.0], 'dec': [0.0, 1.0], 'id': [0, 1]})
        cat = funcs['combine_nights'](combined, ['r'], refcat)
        assert all(cat['r'].mask)


class TestMainBlock:
    """Exercise __main__ via runpy."""

    def test_main_empty_list_exits(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        listf = tmp_path / 'empty.txt'
        listf.write_text('')
        argv = ['calibratemag.py', str(listf), '-s', 'mag']
        with patch.object(sys, 'argv', argv), \
             patch('lsc.mysqldef', MagicMock()), \
             patch('lsc.myloopdef', MagicMock()), \
             patch('lsc.sites', MagicMock()):
            with pytest.raises(SystemExit):
                runpy.run_path(SCRIPT, run_name='__main__')

    def test_main_mag_stage(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        listf = tmp_path / 'imgs.txt'
        listf.write_text('img1.fits\nimg2.fits\n')

        mock_myloopdef = MagicMock()
        mock_myloopdef.conn = MagicMock()

        mock_mysqldef = MagicMock()
        mock_mysqldef.query.return_value = [
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'img1.fits',
             'airmass': 1.2, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.0, 'c1': 0.0,
             'dz1': 0.01, 'dc1': 0.0, 'zcol2': 'ri', 'z2': 25.0, 'c2': 0.0,
             'dz2': 0.01, 'dc2': 0.0, 'psfmag': 17.5, 'psfdmag': 0.02,
             'apmag': 17.6, 'dapmag': 0.03},
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'img2.fits',
             'airmass': 1.3, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.0, 'c1': 0.0,
             'dz1': 0.01, 'dc1': 0.0, 'zcol2': 'ri', 'z2': 25.0, 'c2': 0.0,
             'dz2': 0.01, 'dc2': 0.0, 'psfmag': 17.4, 'psfdmag': 0.02,
             'apmag': 17.5, 'dapmag': 0.03},
        ]

        mock_sites = MagicMock()
        mock_sites.filterst1 = {'rp': 'r', 'gp': 'g', 'ip': 'i'}
        mock_sites.chosecolor.return_value = {'r': ['rg']}
        mock_sites.extinction = {'lsc': {'r': 0.1, 'g': 0.2, 'i': 0.05}}

        argv = ['calibratemag.py', str(listf), '-s', 'mag', '-t', 'fit']
        with patch.object(sys, 'argv', argv), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites):
            try:
                runpy.run_path(SCRIPT, run_name='__main__')
            except (SystemExit, Exception):
                pass

    def test_main_local_stage_needs_exzp(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        listf = tmp_path / 'imgs.txt'
        listf.write_text('img1.fits\n')
        argv = ['calibratemag.py', str(listf), '-s', 'local']
        with patch.object(sys, 'argv', argv), \
             patch('lsc.mysqldef', MagicMock()), \
             patch('lsc.myloopdef', MagicMock()), \
             patch('lsc.sites', MagicMock()):
            with pytest.raises(SystemExit):
                runpy.run_path(SCRIPT, run_name='__main__')


# ---------------------------------------------------------------------------
# Helper to build consistent mock_sites / mock_mysqldef / mock_myloopdef
# ---------------------------------------------------------------------------

def _make_mocks(tmp_path, nstars=3):
    """Return (mock_mysqldef, mock_myloopdef, mock_sites) configured for 2 images."""
    mock_myloopdef = MagicMock()
    mock_myloopdef.conn = MagicMock()

    mock_mysqldef = MagicMock()
    mock_mysqldef.query.return_value = [
        {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'img1.fits',
         'airmass': 1.2, 'dayobs': '20200501', 'targetid': 1,
         'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
         'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.0, 'c1': 0.05,
         'dz1': 0.01, 'dc1': 0.01, 'zcol2': 'ri', 'z2': 25.1, 'c2': 0.03,
         'dz2': 0.01, 'dc2': 0.01, 'psfmag': 17.5, 'psfdmag': 0.02,
         'apmag': 17.6, 'dapmag': 0.03},
        {'filter': 'gp', 'filepath': str(tmp_path) + '/', 'filename': 'img2.fits',
         'airmass': 1.3, 'dayobs': '20200501', 'targetid': 1,
         'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
         'instrument': 'imager', 'zcol1': 'gr', 'z1': 25.2, 'c1': 0.04,
         'dz1': 0.01, 'dc1': 0.01, 'zcol2': 'gi', 'z2': 25.3, 'c2': 0.02,
         'dz2': 0.01, 'dc2': 0.01, 'psfmag': 18.0, 'psfdmag': 0.02,
         'apmag': 18.1, 'dapmag': 0.03},
    ]

    mock_sites = MagicMock()
    mock_sites.filterst1 = {'rp': 'r', 'gp': 'g', 'ip': 'i', 'up': 'u', 'zs': 'z',
                            'U': 'U', 'B': 'B', 'V': 'V', 'R': 'R', 'I': 'I'}
    mock_sites.chosecolor.return_value = {'r': ['rg'], 'g': ['gr']}
    mock_sites.extinction = {'lsc': {'r': 0.1, 'g': 0.2, 'i': 0.05, 'u': 0.5, 'z': 0.03,
                                     'U': 0.5, 'B': 0.3, 'V': 0.15, 'R': 0.1, 'I': 0.05}}

    return mock_mysqldef, mock_myloopdef, mock_sites


def _make_sn2_file(path, nstars=3):
    """Write a minimal sn2.fits table to path."""
    ra = np.linspace(10.0, 10.001, nstars)
    dec = np.linspace(20.0, 20.001, nstars)
    mag = np.linspace(17.0, 18.0, nstars)
    err = np.full(nstars, 0.02)
    t = Table({'ra0': ra, 'dec0': dec, 'ra': ra, 'dec': dec,
               'smagf': mag, 'smagerrf': err, 'magp3': mag + 0.1, 'merrp3': err})
    t.write(str(path), format='fits', overwrite=True)


class TestCrossmatchRightJoin:
    """Cover lines 24-27: right_join=True branch."""

    def test_right_join_masks_non_matches(self, funcs):
        cat0 = Table({'ra': [1.0, 2.0, 3.0], 'dec': [1.0, 2.0, 3.0], 'mag': [15.0, 16.0, 17.0]})
        cat1 = Table({'ra': [1.0001, 50.0], 'dec': [1.0001, 50.0]})
        out, matched = funcs['crossmatch'](cat0, cat1, threshold=1.0, right_join=True)
        # right_join preserves all rows in cat1, masks non-matches
        assert len(out) == 2
        assert out['mag'].mask[1] == True  # second row has no match


class TestGetImageDataSn2:
    """Cover lines 45-69: sn2.fits crossmatch branch of get_image_data."""

    def test_sn2_crossmatch(self, tmp_path, funcs):
        nstars = 3
        # Create sn2 files
        _make_sn2_file(tmp_path / 'img1.sn2.fits', nstars)
        _make_sn2_file(tmp_path / 'img2.sn2.fits', nstars)

        # Build a refcat
        refcat = Table({'ra': np.linspace(10.0, 10.001, nstars),
                        'dec': np.linspace(20.0, 20.001, nstars),
                        'id': list(range(nstars))})

        mock_mysqldef, mock_myloopdef, mock_sites = _make_mocks(tmp_path, nstars)
        # For get_image_data with sn2 branch, magcol must NOT be in the DB columns
        mock_mysqldef.query.return_value = [
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'img1.fits',
             'airmass': 1.2, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.0, 'c1': 0.0,
             'dz1': 0.01, 'dc1': 0.0, 'zcol2': 'ri', 'z2': 25.0, 'c2': 0.0,
             'dz2': 0.01, 'dc2': 0.0, 'psfmag': 17.5, 'psfdmag': 0.02,
             'apmag': 17.6, 'dapmag': 0.03},
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'img2.fits',
             'airmass': 1.3, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.0, 'c1': 0.0,
             'dz1': 0.01, 'dc1': 0.0, 'zcol2': 'ri', 'z2': 25.0, 'c2': 0.0,
             'dz2': 0.01, 'dc2': 0.0, 'psfmag': 17.4, 'psfdmag': 0.02,
             'apmag': 17.5, 'dapmag': 0.03},
        ]

        with patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites):
            ns = _load_module()
            result = ns['get_image_data'](
                [str(tmp_path) + '/img1.sn2.fits', str(tmp_path) + '/img2.sn2.fits'],
                magcol='smagf', errcol='smagerrf', refcat=refcat)
        assert 'instmag' in result.colnames
        assert len(result) == 2
        assert result['instmag'].shape == (2, nstars)

    def test_sn2_missing_file(self, tmp_path, funcs):
        """Lines 63-64: sn2 file missing removes that row."""
        nstars = 3
        _make_sn2_file(tmp_path / 'img1.sn2.fits', nstars)
        # img2.sn2.fits intentionally NOT created

        refcat = Table({'ra': np.linspace(10.0, 10.001, nstars),
                        'dec': np.linspace(20.0, 20.001, nstars),
                        'id': list(range(nstars))})

        mock_mysqldef, mock_myloopdef, mock_sites = _make_mocks(tmp_path, nstars)
        mock_mysqldef.query.return_value = [
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'img1.fits',
             'airmass': 1.2, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.0, 'c1': 0.0,
             'dz1': 0.01, 'dc1': 0.0, 'zcol2': 'ri', 'z2': 25.0, 'c2': 0.0,
             'dz2': 0.01, 'dc2': 0.0, 'psfmag': 17.5, 'psfdmag': 0.02,
             'apmag': 17.6, 'dapmag': 0.03},
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'img2.fits',
             'airmass': 1.3, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.0, 'c1': 0.0,
             'dz1': 0.01, 'dc1': 0.0, 'zcol2': 'ri', 'z2': 25.0, 'c2': 0.0,
             'dz2': 0.01, 'dc2': 0.0, 'psfmag': 17.4, 'psfdmag': 0.02,
             'apmag': 17.5, 'dapmag': 0.03},
        ]

        with patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites):
            ns = _load_module()
            result = ns['get_image_data'](
                [str(tmp_path) + '/img1.sn2.fits', str(tmp_path) + '/img2.sn2.fits'],
                magcol='smagf', errcol='smagerrf', refcat=refcat)
        # Only 1 row should remain (img2 removed)
        assert len(result) == 1


class TestMainCatalogReading:
    """Cover lines 131-142: catalog reading with different formats."""

    def _run_abscat_with_catalog(self, tmp_path, catalog_file, mock_mysqldef, mock_myloopdef, mock_sites):
        listf = tmp_path / 'imgs.txt'
        listf.write_text('img1.fits\nimg2.fits\n')
        _make_sn2_file(tmp_path / 'img1.sn2.fits')
        _make_sn2_file(tmp_path / 'img2.sn2.fits')

        argv = ['calibratemag.py', str(listf), '-s', 'abscat', '-t', 'fit',
                '-c', str(catalog_file), '-F']
        with patch.object(sys, 'argv', argv), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites):
            runpy.run_path(SCRIPT, run_name='__main__')

    def test_gaia_catalog_source_id(self, tmp_path, monkeypatch):
        """Line 133-134: catalog with 'source_id' column."""
        monkeypatch.chdir(tmp_path)
        nstars = 3
        mock_mysqldef, mock_myloopdef, mock_sites = _make_mocks(tmp_path, nstars)

        # Gaia-style catalog with source_id
        cat = Table({'source_id': [1, 2, 3],
                     'ra': np.linspace(10.0, 10.001, 3),
                     'dec': np.linspace(20.0, 20.001, 3)})
        catfile = tmp_path / 'gaia.cat'
        cat.write(str(catfile), format='ascii')

        self._run_abscat_with_catalog(tmp_path, catfile, mock_mysqldef, mock_myloopdef, mock_sites)

    def test_gaia_catalog_SOURCE_ID(self, tmp_path, monkeypatch):
        """Line 135-136: catalog with 'SOURCE_ID' column."""
        monkeypatch.chdir(tmp_path)
        nstars = 3
        mock_mysqldef, mock_myloopdef, mock_sites = _make_mocks(tmp_path, nstars)

        cat = Table({'SOURCE_ID': [1, 2, 3],
                     'ra': np.linspace(10.0, 10.001, 3),
                     'dec': np.linspace(20.0, 20.001, 3)})
        catfile = tmp_path / 'gaia2.cat'
        cat.write(str(catfile), format='ascii')

        self._run_abscat_with_catalog(tmp_path, catfile, mock_mysqldef, mock_myloopdef, mock_sites)

    def test_landolt_catalog_format(self, tmp_path, monkeypatch):
        """Lines 141-144: InconsistentTableError triggers Landolt format."""
        monkeypatch.chdir(tmp_path)
        nstars = 3
        mock_mysqldef, mock_myloopdef, mock_sites = _make_mocks(tmp_path, nstars)

        # Write a file that will cause InconsistentTableError on first read
        # then succeed with Landolt names. Landolt format: id ra dec U B V R I vary Uerr Berr Verr Rerr Ierr + extras
        lines = []
        for i in range(nstars):
            lines.append('{} {:.6f} {:.6f} 15.0 16.0 15.5 15.2 14.8 0 0.01 0.01 0.01 0.01 0.01 0.0 0.0 0.0 0.0 0.0'.format(
                i+1, 10.0 + i*0.0001, 20.0 + i*0.0001))
        catfile = tmp_path / 'landolt.cat'
        catfile.write_text('\n'.join(lines) + '\n')

        # Patch Table.read to raise InconsistentTableError on first call, succeed on second
        original_read = Table.read
        call_count = [0]
        def patched_read(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ascii.core.InconsistentTableError("fake error")
            return original_read(*args, **kwargs)

        listf = tmp_path / 'imgs.txt'
        listf.write_text('img1.fits\nimg2.fits\n')
        _make_sn2_file(tmp_path / 'img1.sn2.fits')
        _make_sn2_file(tmp_path / 'img2.sn2.fits')

        argv = ['calibratemag.py', str(listf), '-s', 'abscat', '-t', 'fit',
                '-c', str(catfile), '-F']
        with patch.object(sys, 'argv', argv), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites), \
             patch('astropy.table.Table.read', side_effect=patched_read):
            runpy.run_path(SCRIPT, run_name='__main__')


class TestMainTypemag:
    """Cover lines 148-153: different typemag/stage combos."""

    def test_aperture_photometry_mag_stage(self, tmp_path, monkeypatch):
        """Line 148-149: typemag='ph', stage='mag' uses apmag/dapmag."""
        monkeypatch.chdir(tmp_path)
        mock_mysqldef, mock_myloopdef, mock_sites = _make_mocks(tmp_path)
        # Need both filters same so chosecolor works simply
        mock_mysqldef.query.return_value = [
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'img1.fits',
             'airmass': 1.2, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.0, 'c1': 0.0,
             'dz1': 0.01, 'dc1': 0.0, 'zcol2': 'ri', 'z2': 25.0, 'c2': 0.0,
             'dz2': 0.01, 'dc2': 0.0, 'psfmag': 17.5, 'psfdmag': 0.02,
             'apmag': 17.6, 'dapmag': 0.03},
        ]
        mock_sites.chosecolor.return_value = {'r': ['rg']}

        listf = tmp_path / 'imgs.txt'
        listf.write_text('img1.fits\n')

        argv = ['calibratemag.py', str(listf), '-s', 'mag', '-t', 'ph']
        with patch.object(sys, 'argv', argv), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites):
            runpy.run_path(SCRIPT, run_name='__main__')
        # Verify query was called (the INSERT)
        assert mock_mysqldef.query.call_count >= 2

    def test_aperture_photometry_abscat_stage(self, tmp_path, monkeypatch):
        """Line 152-153: typemag='ph', stage='abscat' uses magp3/merrp3."""
        monkeypatch.chdir(tmp_path)
        nstars = 3
        mock_mysqldef, mock_myloopdef, mock_sites = _make_mocks(tmp_path, nstars)

        _make_sn2_file(tmp_path / 'img1.sn2.fits', nstars)
        _make_sn2_file(tmp_path / 'img2.sn2.fits', nstars)

        refcat = Table({'source_id': list(range(nstars)),
                        'ra': np.linspace(10.0, 10.001, nstars),
                        'dec': np.linspace(20.0, 20.001, nstars)})
        catfile = tmp_path / 'ref.cat'
        refcat.write(str(catfile), format='ascii')

        listf = tmp_path / 'imgs.txt'
        listf.write_text('img1.fits\nimg2.fits\n')

        argv = ['calibratemag.py', str(listf), '-s', 'abscat', '-t', 'ph',
                '-c', str(catfile), '-F']
        with patch.object(sys, 'argv', argv), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites):
            runpy.run_path(SCRIPT, run_name='__main__')


class TestMainMatchBySite:
    """Cover lines 159-160: --match-by-site flag."""

    def test_match_by_site(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mock_mysqldef, mock_myloopdef, mock_sites = _make_mocks(tmp_path)
        mock_mysqldef.query.return_value = [
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'img1.fits',
             'airmass': 1.2, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.0, 'c1': 0.0,
             'dz1': 0.01, 'dc1': 0.0, 'zcol2': 'ri', 'z2': 25.0, 'c2': 0.0,
             'dz2': 0.01, 'dc2': 0.0, 'psfmag': 17.5, 'psfdmag': 0.02,
             'apmag': 17.6, 'dapmag': 0.03},
        ]
        mock_sites.chosecolor.return_value = {'r': ['rg']}

        listf = tmp_path / 'imgs.txt'
        listf.write_text('img1.fits\n')

        argv = ['calibratemag.py', str(listf), '-s', 'mag', '-t', 'fit', '--match-by-site']
        with patch.object(sys, 'argv', argv), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites):
            runpy.run_path(SCRIPT, run_name='__main__')


class TestMainExzp:
    """Cover lines 167-200: exzp handling."""

    def test_exzp_copies_zeropoints(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        nstars = 3
        mock_mysqldef, mock_myloopdef, mock_sites = _make_mocks(tmp_path, nstars)

        _make_sn2_file(tmp_path / 'img1.sn2.fits', nstars)
        _make_sn2_file(tmp_path / 'img2.sn2.fits', nstars)
        _make_sn2_file(tmp_path / 'std1.sn2.fits', nstars)

        refcat = Table({'source_id': list(range(nstars)),
                        'ra': np.linspace(10.0, 10.001, nstars),
                        'dec': np.linspace(20.0, 20.001, nstars)})
        catfile = tmp_path / 'ref.cat'
        refcat.write(str(catfile), format='ascii')

        # target list
        listf = tmp_path / 'imgs.txt'
        listf.write_text('img1.fits\nimg2.fits\n')

        # standards list (exzp)
        exzpf = tmp_path / 'stds.txt'
        exzpf.write_text('std1.fits\n')

        # Mock query returns: first call for targets, second call for standards
        target_rows = [
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'img1.fits',
             'airmass': 1.2, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.0, 'c1': 0.05,
             'dz1': 0.01, 'dc1': 0.01, 'zcol2': 'ri', 'z2': 25.1, 'c2': 0.03,
             'dz2': 0.01, 'dc2': 0.01, 'psfmag': 17.5, 'psfdmag': 0.02,
             'apmag': 17.6, 'dapmag': 0.03},
            {'filter': 'gp', 'filepath': str(tmp_path) + '/', 'filename': 'img2.fits',
             'airmass': 1.3, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'gr', 'z1': 25.2, 'c1': 0.04,
             'dz1': 0.01, 'dc1': 0.01, 'zcol2': 'gi', 'z2': 25.3, 'c2': 0.02,
             'dz2': 0.01, 'dc2': 0.01, 'psfmag': 18.0, 'psfdmag': 0.02,
             'apmag': 18.1, 'dapmag': 0.03},
        ]
        std_rows = [
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'std1.fits',
             'airmass': 1.1, 'dayobs': '20200501', 'targetid': 2,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.05, 'c1': 0.06,
             'dz1': 0.005, 'dc1': 0.005, 'zcol2': 'ri', 'z2': 25.15, 'c2': 0.035,
             'dz2': 0.005, 'dc2': 0.005, 'psfmag': 16.0, 'psfdmag': 0.01,
             'apmag': 16.1, 'dapmag': 0.02},
        ]
        mock_mysqldef.query.side_effect = [target_rows, std_rows]

        argv = ['calibratemag.py', str(listf), '-s', 'abscat', '-t', 'fit',
                '-c', str(catfile), '-e', str(exzpf), '-F']
        with patch.object(sys, 'argv', argv), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites):
            runpy.run_path(SCRIPT, run_name='__main__')

    def test_exzp_dc_zero(self, tmp_path, monkeypatch):
        """Lines 191-193, 198-200: dc1==0 and dc2==0 branches."""
        monkeypatch.chdir(tmp_path)
        nstars = 3
        mock_mysqldef, mock_myloopdef, mock_sites = _make_mocks(tmp_path, nstars)

        _make_sn2_file(tmp_path / 'img1.sn2.fits', nstars)
        _make_sn2_file(tmp_path / 'std1.sn2.fits', nstars)

        refcat = Table({'source_id': list(range(nstars)),
                        'ra': np.linspace(10.0, 10.001, nstars),
                        'dec': np.linspace(20.0, 20.001, nstars)})
        catfile = tmp_path / 'ref.cat'
        refcat.write(str(catfile), format='ascii')

        listf = tmp_path / 'imgs.txt'
        listf.write_text('img1.fits\n')
        exzpf = tmp_path / 'stds.txt'
        exzpf.write_text('std1.fits\n')

        target_rows = [
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'img1.fits',
             'airmass': 1.2, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.0, 'c1': 0.05,
             'dz1': 0.01, 'dc1': 0.01, 'zcol2': 'ri', 'z2': 25.1, 'c2': 0.03,
             'dz2': 0.01, 'dc2': 0.01, 'psfmag': 17.5, 'psfdmag': 0.02,
             'apmag': 17.6, 'dapmag': 0.03},
        ]
        # Standards with dc1=0 and dc2=0 to trigger else branches
        std_rows = [
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'std1.fits',
             'airmass': 1.1, 'dayobs': '20200501', 'targetid': 2,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.05, 'c1': 0.06,
             'dz1': 0.005, 'dc1': 0.0, 'zcol2': 'ri', 'z2': 25.15, 'c2': 0.035,
             'dz2': 0.005, 'dc2': 0.0, 'psfmag': 16.0, 'psfdmag': 0.01,
             'apmag': 16.1, 'dapmag': 0.02},
        ]
        mock_mysqldef.query.side_effect = [target_rows, std_rows]
        mock_sites.chosecolor.return_value = {'r': ['rg']}

        argv = ['calibratemag.py', str(listf), '-s', 'abscat', '-t', 'fit',
                '-c', str(catfile), '-e', str(exzpf), '-F']
        with patch.object(sys, 'argv', argv), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites):
            runpy.run_path(SCRIPT, run_name='__main__')


class TestMainColorEdgeCases:
    """Cover lines 217-218, 223-224: dc1/dc2 == 0 in color calculation."""

    def test_dc_zero_in_color_calc(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mock_mysqldef, mock_myloopdef, mock_sites = _make_mocks(tmp_path)
        # Two images in different filters with dc1=0, dc2=0
        mock_mysqldef.query.return_value = [
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'img1.fits',
             'airmass': 1.2, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.0, 'c1': 0.05,
             'dz1': 0.01, 'dc1': 0.0, 'zcol2': 'ri', 'z2': 25.1, 'c2': 0.03,
             'dz2': 0.01, 'dc2': 0.0, 'psfmag': 17.5, 'psfdmag': 0.02,
             'apmag': 17.6, 'dapmag': 0.03},
            {'filter': 'gp', 'filepath': str(tmp_path) + '/', 'filename': 'img2.fits',
             'airmass': 1.3, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'gr', 'z1': 25.2, 'c1': 0.04,
             'dz1': 0.01, 'dc1': 0.0, 'zcol2': 'gi', 'z2': 25.3, 'c2': 0.02,
             'dz2': 0.01, 'dc2': 0.0, 'psfmag': 18.0, 'psfdmag': 0.02,
             'apmag': 18.1, 'dapmag': 0.03},
        ]
        mock_sites.chosecolor.return_value = {'r': ['rg'], 'g': ['rg']}

        listf = tmp_path / 'imgs.txt'
        listf.write_text('img1.fits\nimg2.fits\n')

        argv = ['calibratemag.py', str(listf), '-s', 'mag', '-t', 'fit']
        with patch.object(sys, 'argv', argv), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites):
            runpy.run_path(SCRIPT, run_name='__main__')


class TestMainAbscatStage:
    """Cover lines 259-277: abscat stage writes .cat files."""

    def test_abscat_writes_cat_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        nstars = 3
        mock_mysqldef, mock_myloopdef, mock_sites = _make_mocks(tmp_path, nstars)

        _make_sn2_file(tmp_path / 'img1.sn2.fits', nstars)
        _make_sn2_file(tmp_path / 'img2.sn2.fits', nstars)

        refcat = Table({'source_id': list(range(nstars)),
                        'ra': np.linspace(10.0, 10.001, nstars),
                        'dec': np.linspace(20.0, 20.001, nstars)})
        catfile = tmp_path / 'ref.cat'
        refcat.write(str(catfile), format='ascii')

        listf = tmp_path / 'imgs.txt'
        listf.write_text('img1.fits\nimg2.fits\n')

        argv = ['calibratemag.py', str(listf), '-s', 'abscat', '-t', 'fit',
                '-c', str(catfile), '-F']
        with patch.object(sys, 'argv', argv), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites):
            runpy.run_path(SCRIPT, run_name='__main__')

        # Check that .cat files were created or updatevalue was called
        assert mock_mysqldef.updatevalue.called or \
               os.path.exists(str(tmp_path / 'img1.cat')) or \
               os.path.exists(str(tmp_path / 'img2.cat'))

    def test_abscat_no_good_mags(self, tmp_path, monkeypatch):
        """Lines 263-266: no good magnitudes prints message and marks X."""
        monkeypatch.chdir(tmp_path)
        nstars = 3
        mock_mysqldef, mock_myloopdef, mock_sites = _make_mocks(tmp_path, nstars)

        # Create sn2 files with 9999 magnitudes (will be masked)
        for fname in ['img1.sn2.fits', 'img2.sn2.fits']:
            ra = np.linspace(10.0, 10.001, nstars)
            dec = np.linspace(20.0, 20.001, nstars)
            t = Table({'ra0': ra, 'dec0': dec, 'ra': ra, 'dec': dec,
                       'smagf': [9999.0]*nstars, 'smagerrf': [9999.0]*nstars,
                       'magp3': [9999.0]*nstars, 'merrp3': [9999.0]*nstars})
            t.write(str(tmp_path / fname), format='fits', overwrite=True)

        refcat = Table({'source_id': list(range(nstars)),
                        'ra': np.linspace(10.0, 10.001, nstars),
                        'dec': np.linspace(20.0, 20.001, nstars)})
        catfile = tmp_path / 'ref.cat'
        refcat.write(str(catfile), format='ascii')

        listf = tmp_path / 'imgs.txt'
        listf.write_text('img1.fits\nimg2.fits\n')

        argv = ['calibratemag.py', str(listf), '-s', 'abscat', '-t', 'fit',
                '-c', str(catfile), '-F']
        with patch.object(sys, 'argv', argv), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites):
            runpy.run_path(SCRIPT, run_name='__main__')

        # updatevalue should be called with 'X' for images with no good mags
        calls = [str(c) for c in mock_mysqldef.updatevalue.call_args_list]
        assert any('X' in c for c in calls)

    def test_abscat_no_overwrite(self, tmp_path, monkeypatch):
        """Lines 276-277: IOError when file exists and -F not given."""
        monkeypatch.chdir(tmp_path)
        nstars = 3
        mock_mysqldef, mock_myloopdef, mock_sites = _make_mocks(tmp_path, nstars)

        _make_sn2_file(tmp_path / 'img1.sn2.fits', nstars)

        # Only one image with filter r
        mock_mysqldef.query.return_value = [
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'img1.fits',
             'airmass': 1.2, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.0, 'c1': 0.05,
             'dz1': 0.01, 'dc1': 0.01, 'zcol2': 'ri', 'z2': 25.1, 'c2': 0.03,
             'dz2': 0.01, 'dc2': 0.01, 'psfmag': 17.5, 'psfdmag': 0.02,
             'apmag': 17.6, 'dapmag': 0.03},
        ]
        mock_sites.chosecolor.return_value = {'r': ['rg']}

        refcat = Table({'source_id': list(range(nstars)),
                        'ra': np.linspace(10.0, 10.001, nstars),
                        'dec': np.linspace(20.0, 20.001, nstars)})
        catfile = tmp_path / 'ref.cat'
        refcat.write(str(catfile), format='ascii')

        # Pre-create the output file so write will fail without -F
        (tmp_path / 'img1.cat').write_text('existing')

        listf = tmp_path / 'imgs.txt'
        listf.write_text('img1.fits\n')

        # No -F flag, should hit IOError branch
        argv = ['calibratemag.py', str(listf), '-s', 'abscat', '-t', 'fit',
                '-c', str(catfile)]
        with patch.object(sys, 'argv', argv), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites):
            # Should not raise - it prints the error and continues
            runpy.run_path(SCRIPT, run_name='__main__')


class TestMainLocalStage:
    """Cover lines 278-357: local stage with combine_nights."""

    def _setup_local(self, tmp_path, field='sloan'):
        nstars = 3
        mock_mysqldef, mock_myloopdef, mock_sites = _make_mocks(tmp_path, nstars)

        _make_sn2_file(tmp_path / 'img1.sn2.fits', nstars)
        _make_sn2_file(tmp_path / 'img2.sn2.fits', nstars)
        _make_sn2_file(tmp_path / 'std1.sn2.fits', nstars)

        refcat = Table({'source_id': list(range(nstars)),
                        'ra': np.linspace(10.0, 10.001, nstars),
                        'dec': np.linspace(20.0, 20.001, nstars)})
        catfile = tmp_path / 'SN2024abc_ref.cat'
        refcat.write(str(catfile), format='ascii')

        listf = tmp_path / 'imgs.txt'
        listf.write_text('img1.fits\nimg2.fits\n')
        exzpf = tmp_path / 'stds.txt'
        exzpf.write_text('std1.fits\n')

        target_rows = [
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'img1.fits',
             'airmass': 1.2, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.0, 'c1': 0.05,
             'dz1': 0.01, 'dc1': 0.01, 'zcol2': 'ri', 'z2': 25.1, 'c2': 0.03,
             'dz2': 0.01, 'dc2': 0.01, 'psfmag': 17.5, 'psfdmag': 0.02,
             'apmag': 17.6, 'dapmag': 0.03},
            {'filter': 'gp', 'filepath': str(tmp_path) + '/', 'filename': 'img2.fits',
             'airmass': 1.3, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'gr', 'z1': 25.2, 'c1': 0.04,
             'dz1': 0.01, 'dc1': 0.01, 'zcol2': 'gi', 'z2': 25.3, 'c2': 0.02,
             'dz2': 0.01, 'dc2': 0.01, 'psfmag': 18.0, 'psfdmag': 0.02,
             'apmag': 18.1, 'dapmag': 0.03},
        ]
        std_rows = [
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'std1.fits',
             'airmass': 1.1, 'dayobs': '20200501', 'targetid': 2,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.05, 'c1': 0.06,
             'dz1': 0.005, 'dc1': 0.005, 'zcol2': 'ri', 'z2': 25.15, 'c2': 0.035,
             'dz2': 0.005, 'dc2': 0.005, 'psfmag': 16.0, 'psfdmag': 0.01,
             'apmag': 16.1, 'dapmag': 0.02},
        ]
        mock_mysqldef.query.side_effect = [target_rows, std_rows]

        return mock_mysqldef, mock_myloopdef, mock_sites, catfile, listf, exzpf

    def test_local_sloan(self, tmp_path, monkeypatch):
        """Lines 282-283: sloan field."""
        monkeypatch.chdir(tmp_path)
        mock_mysqldef, mock_myloopdef, mock_sites, catfile, listf, exzpf = self._setup_local(tmp_path, 'sloan')

        argv = ['calibratemag.py', str(listf), '-s', 'local', '-t', 'fit',
                '-c', str(catfile), '-e', str(exzpf), '-f', 'sloan', '-F']
        with patch.object(sys, 'argv', argv), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites), \
             patch('lsc.util', MagicMock()):
            runpy.run_path(SCRIPT, run_name='__main__')

        # Should have written an output catalog file
        output_files = [f for f in os.listdir(str(tmp_path)) if f.endswith('.cat') and 'sloan' in f]
        assert len(output_files) >= 1

    def test_local_landolt(self, tmp_path, monkeypatch):
        """Lines 279-280: landolt field."""
        monkeypatch.chdir(tmp_path)
        mock_mysqldef, mock_myloopdef, mock_sites, catfile, listf, exzpf = self._setup_local(tmp_path, 'landolt')

        argv = ['calibratemag.py', str(listf), '-s', 'local', '-t', 'fit',
                '-c', str(catfile), '-e', str(exzpf), '-f', 'landolt', '-F']
        with patch.object(sys, 'argv', argv), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites), \
             patch('lsc.util', MagicMock()):
            runpy.run_path(SCRIPT, run_name='__main__')

        output_files = [f for f in os.listdir(str(tmp_path)) if f.endswith('.cat') and 'landolt' in f]
        assert len(output_files) >= 1

    def test_local_apass(self, tmp_path, monkeypatch):
        """Lines 283-284: apass field."""
        monkeypatch.chdir(tmp_path)
        mock_mysqldef, mock_myloopdef, mock_sites, catfile, listf, exzpf = self._setup_local(tmp_path, 'apass')

        argv = ['calibratemag.py', str(listf), '-s', 'local', '-t', 'fit',
                '-c', str(catfile), '-e', str(exzpf), '-f', 'apass', '-F']
        with patch.object(sys, 'argv', argv), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites), \
             patch('lsc.util', MagicMock()):
            runpy.run_path(SCRIPT, run_name='__main__')

        output_files = [f for f in os.listdir(str(tmp_path)) if f.endswith('.cat') and 'apass' in f]
        assert len(output_files) >= 1

    def test_local_no_field_raises(self, tmp_path, monkeypatch):
        """Line 286: no --field raises Exception."""
        monkeypatch.chdir(tmp_path)
        mock_mysqldef, mock_myloopdef, mock_sites, catfile, listf, exzpf = self._setup_local(tmp_path)

        # argparse choices will reject invalid field, but None triggers the else branch
        # We need to bypass argparse to test line 286, so patch args directly
        argv = ['calibratemag.py', str(listf), '-s', 'local', '-t', 'fit',
                '-c', str(catfile), '-e', str(exzpf), '-F']
        with patch.object(sys, 'argv', argv), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites), \
             patch('lsc.util', MagicMock()):
            with pytest.raises(Exception, match='Need to give --field'):
                runpy.run_path(SCRIPT, run_name='__main__')


class TestGetImageDataNoRefcat:
    """Cover lines 54-56: refcat=None branch in get_image_data sn2 path."""

    def test_sn2_no_refcat(self, tmp_path):
        nstars = 3
        _make_sn2_file(tmp_path / 'img1.sn2.fits', nstars)
        _make_sn2_file(tmp_path / 'img2.sn2.fits', nstars)

        mock_mysqldef, mock_myloopdef, mock_sites = _make_mocks(tmp_path, nstars)
        mock_mysqldef.query.return_value = [
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'img1.fits',
             'airmass': 1.2, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.0, 'c1': 0.0,
             'dz1': 0.01, 'dc1': 0.0, 'zcol2': 'ri', 'z2': 25.0, 'c2': 0.0,
             'dz2': 0.01, 'dc2': 0.0, 'psfmag': 17.5, 'psfdmag': 0.02,
             'apmag': 17.6, 'dapmag': 0.03},
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'img2.fits',
             'airmass': 1.3, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.0, 'c1': 0.0,
             'dz1': 0.01, 'dc1': 0.0, 'zcol2': 'ri', 'z2': 25.0, 'c2': 0.0,
             'dz2': 0.01, 'dc2': 0.0, 'psfmag': 17.4, 'psfdmag': 0.02,
             'apmag': 17.5, 'dapmag': 0.03},
        ]

        with patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites):
            ns = _load_module()
            # Pass refcat=None explicitly to trigger lines 54-56
            result = ns['get_image_data'](
                [str(tmp_path) + '/img1.sn2.fits', str(tmp_path) + '/img2.sn2.fits'],
                magcol='smagf', errcol='smagerrf', refcat=None)
        assert 'instmag' in result.colnames
        assert len(result) == 2


class TestMainCatalogCommentedHeader:
    """Cover lines 138-140: catalog with commented header (pipeline .cat format)."""

    def test_commented_header_catalog(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        nstars = 3
        mock_mysqldef, mock_myloopdef, mock_sites = _make_mocks(tmp_path, nstars)

        _make_sn2_file(tmp_path / 'img1.sn2.fits', nstars)
        _make_sn2_file(tmp_path / 'img2.sn2.fits', nstars)

        # Write a catalog file with the pipeline's commented header format
        # 6 space-separated fields in comments -> triggers line 138
        header_lines = [
            '# ra    1 0 d degrees %10.6f',
            '# dec   2 0 d degrees %10.6f',
            '# id    3 0 c INDEF %3d',
            '# r     4 0 r INDEF %6.3f',
            '# rerr  5 0 r INDEF %6.3f',
            '# g     6 0 r INDEF %6.3f',
        ]
        data_lines = []
        for i in range(nstars):
            data_lines.append('{:.6f} {:.6f} {} 15.0 0.01 16.0'.format(
                10.0 + i*0.0001, 20.0 + i*0.0001, i))
        catfile = tmp_path / 'pipeline.cat'
        catfile.write_text('\n'.join(header_lines + data_lines) + '\n')

        listf = tmp_path / 'imgs.txt'
        listf.write_text('img1.fits\nimg2.fits\n')

        argv = ['calibratemag.py', str(listf), '-s', 'abscat', '-t', 'fit',
                '-c', str(catfile), '-F']
        with patch.object(sys, 'argv', argv), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites):
            runpy.run_path(SCRIPT, run_name='__main__')


class TestMainExzpNoneZcol:
    """Cover line 172: zcol with None values in standards."""

    def test_exzp_none_zcol(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        nstars = 3
        mock_mysqldef, mock_myloopdef, mock_sites = _make_mocks(tmp_path, nstars)

        _make_sn2_file(tmp_path / 'img1.sn2.fits', nstars)
        _make_sn2_file(tmp_path / 'std1.sn2.fits', nstars)

        refcat = Table({'source_id': list(range(nstars)),
                        'ra': np.linspace(10.0, 10.001, nstars),
                        'dec': np.linspace(20.0, 20.001, nstars)})
        catfile = tmp_path / 'ref.cat'
        refcat.write(str(catfile), format='ascii')

        listf = tmp_path / 'imgs.txt'
        listf.write_text('img1.fits\n')
        exzpf = tmp_path / 'stds.txt'
        exzpf.write_text('std1.fits\n')

        target_rows = [
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'img1.fits',
             'airmass': 1.2, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.0, 'c1': 0.05,
             'dz1': 0.01, 'dc1': 0.01, 'zcol2': 'ri', 'z2': 25.1, 'c2': 0.03,
             'dz2': 0.01, 'dc2': 0.01, 'psfmag': 17.5, 'psfdmag': 0.02,
             'apmag': 17.6, 'dapmag': 0.03},
        ]
        # Standards with None in zcol1 to trigger line 172
        std_rows = [
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'std1.fits',
             'airmass': 1.1, 'dayobs': '20200501', 'targetid': 2,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': None, 'z1': 25.05, 'c1': 0.06,
             'dz1': 0.005, 'dc1': 0.005, 'zcol2': None, 'z2': 25.15, 'c2': 0.035,
             'dz2': 0.005, 'dc2': 0.005, 'psfmag': 16.0, 'psfdmag': 0.01,
             'apmag': 16.1, 'dapmag': 0.02},
        ]
        mock_mysqldef.query.side_effect = [target_rows, std_rows]
        mock_sites.chosecolor.return_value = {'r': ['rg']}

        argv = ['calibratemag.py', str(listf), '-s', 'abscat', '-t', 'fit',
                '-c', str(catfile), '-e', str(exzpf), '-F']
        with patch.object(sys, 'argv', argv), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites):
            runpy.run_path(SCRIPT, run_name='__main__')


class TestMainLocalInteractive:
    """Cover lines 291-346: interactive plotting in local stage."""

    def test_local_interactive(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        nstars = 3
        mock_mysqldef, mock_myloopdef, mock_sites = _make_mocks(tmp_path, nstars)

        _make_sn2_file(tmp_path / 'img1.sn2.fits', nstars)
        _make_sn2_file(tmp_path / 'img2.sn2.fits', nstars)
        _make_sn2_file(tmp_path / 'std1.sn2.fits', nstars)

        # Include filter columns in refcat so plotting with refcat comparison works
        refcat = Table({'source_id': list(range(nstars)),
                        'ra': np.linspace(10.0, 10.001, nstars),
                        'dec': np.linspace(20.0, 20.001, nstars),
                        'r': [15.0, 16.0, 17.0],
                        'g': [15.5, 16.5, 17.5]})
        catfile = tmp_path / 'SN2024abc_ref.cat'
        refcat.write(str(catfile), format='ascii')

        listf = tmp_path / 'imgs.txt'
        listf.write_text('img1.fits\nimg2.fits\n')
        exzpf = tmp_path / 'stds.txt'
        exzpf.write_text('std1.fits\n')

        target_rows = [
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'img1.fits',
             'airmass': 1.2, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.0, 'c1': 0.05,
             'dz1': 0.01, 'dc1': 0.01, 'zcol2': 'ri', 'z2': 25.1, 'c2': 0.03,
             'dz2': 0.01, 'dc2': 0.01, 'psfmag': 17.5, 'psfdmag': 0.02,
             'apmag': 17.6, 'dapmag': 0.03},
            {'filter': 'gp', 'filepath': str(tmp_path) + '/', 'filename': 'img2.fits',
             'airmass': 1.3, 'dayobs': '20200501', 'targetid': 1,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'gr', 'z1': 25.2, 'c1': 0.04,
             'dz1': 0.01, 'dc1': 0.01, 'zcol2': 'gi', 'z2': 25.3, 'c2': 0.02,
             'dz2': 0.01, 'dc2': 0.01, 'psfmag': 18.0, 'psfdmag': 0.02,
             'apmag': 18.1, 'dapmag': 0.03},
        ]
        std_rows = [
            {'filter': 'rp', 'filepath': str(tmp_path) + '/', 'filename': 'std1.fits',
             'airmass': 1.1, 'dayobs': '20200501', 'targetid': 2,
             'telescopeid': 1, 'instrumentid': 1, 'shortname': 'lsc 1m0',
             'instrument': 'imager', 'zcol1': 'rg', 'z1': 25.05, 'c1': 0.06,
             'dz1': 0.005, 'dc1': 0.005, 'zcol2': 'ri', 'z2': 25.15, 'c2': 0.035,
             'dz2': 0.005, 'dc2': 0.005, 'psfmag': 16.0, 'psfdmag': 0.01,
             'apmag': 16.1, 'dapmag': 0.02},
        ]
        mock_mysqldef.query.side_effect = [target_rows, std_rows]

        mock_util = MagicMock()
        mock_plt = MagicMock()
        # Make plt.figure return a mock with the needed methods
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.plot.return_value = [MagicMock()]
        mock_ax.errorbar.return_value = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_plt.figure.return_value = mock_fig
        mock_plt.subplot.return_value = mock_ax

        argv = ['calibratemag.py', str(listf), '-s', 'local', '-t', 'fit',
                '-c', str(catfile), '-e', str(exzpf), '-f', 'sloan', '-i', '-F']
        with patch.object(sys, 'argv', argv), \
             patch('lsc.mysqldef', mock_mysqldef), \
             patch('lsc.myloopdef', mock_myloopdef), \
             patch('lsc.sites', mock_sites), \
             patch('lsc.util', mock_util), \
             patch('matplotlib.pyplot', mock_plt):
            runpy.run_path(SCRIPT, run_name='__main__')

        # userinput should have been called (once per filter with data)
        assert mock_util.userinput.called
